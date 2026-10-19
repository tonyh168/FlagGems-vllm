# GLM-5.3-Flash on MetaX C550: two Triton lowering fixes

## Summary

Two operator-level fixes that unblock GLM-5.3-Flash inference on MetaX C550
(TP16, 2 nodes × 8 cards). Both problems show up as Triton compilation or launch
failures that are specific to the MACA backend.

| # | File | Problem | Fix |
|---|---|---|---|
| 1 | `ops/causal_conv1d.py` | `tl.gather(axis=1)` lowers to `maca.shfl.sync`, which the MACA backend does not implement | Replace the 4×4 register rotation with an equivalent `where` + `sum` |
| 2 | `ops/flashmla_sparse.py` | The fallback sparse-MLA kernel's only autotune configs need ~160 KB shared memory; C550 has 64 KB | Add smaller tile presets; Triton already skips configs that raise `OutOfResources` |

With both applied the service reaches `Application startup complete`
(KV cache 6,389,760 tokens) and answers correctly:

```
问:中国的首都是哪个城市呢？答:应该是  ->  北京。北京。北京是中国的首都
```

## Problem 1 — `tl.gather` on axis 1 needs a warp shuffle

GLM-5.3-Flash's KDA (linear attention) uses a short causal convolution. The
Triton kernel keeps a sliding window in registers and rotates it every token:

```python
nxt = tl.broadcast_to(((j + 1) % BW)[None, :], (BLOCK_D, BW))
history = tl.gather(window, nxt, 1)
```

On the MACA backend this fails to compile:

```
Internal Triton mlir opt error:
  error: custom op 'maca.shfl.sync' is unknown
    %history_128 = maca.shfl.sync idx %29, %history_127, ...
  at flagtree/triton/backends/metax/compiler.py:471 make_llir
```

`BW` is `next_power_of_2(width)` which is 4 for these models, so a dense
`where`/`sum` over a 4×4 tile costs nothing:

```python
_src = (j + 1) % BW
# history[i] = window[(i + 1) % BW] -- the index has to live on the *output*
# slot axis. Writing _src[None, :] == j[:, None] yields (k + 1) % BW == i,
# i.e. history[i] = window[(i - 1) % BW], rotating the wrong way (element-wise
# equality is symmetric, so swapping the operands does not fix it).
_match = _src[:, None] == j[None, :]
history = tl.sum(tl.where(_match[None, :, :], window[:, None, :], 0.0), axis=2)
```

Note the axis order in `_match`: getting it backwards produces a shift in the
wrong direction, which silently corrupts the convolution history rather than
failing loudly.

## Problem 2 — the fallback sparse-MLA kernel does not fit in 64 KB

`flash_mla_sparse_fwd` switches to a pure-Triton path when the TLE variant is
unavailable. Its only two autotune configs are `BK=64, BH=64` with
`num_stages` 2 and 4, which need roughly:

```
q_blk0/q_blk1   [BH, 256] bf16 = 32 KB each
kv_blk0/kv_blk1 [256, BK] bf16 = 32 KB each × num_stages
```

That is ~160 KB, sized for H100-class parts (228 KB). On C550
(`shared_memory_per_block = 65536`) both configs raise:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 163840, Hardware limit: 65536
```

Triton's autotuner does handle this correctly: `_bench` catches
`OutOfResources` and scores the config as `inf`, so a viable config would be
picked — but with every config at `inf`, `min()` falls back to the first one and
the failure surfaces at launch time.

The fix adds `32x32` and `16x32` presets while keeping the large tiles, so
large-shared devices still get the fast configuration:

| Preset | Approx shared | Notes |
|---|---|---|
| `64x64` stages 2/4 | ~160 KB | H100 / large-shared parts |
| `32x32` stages 1 | ~48 KB | fits C550 with headroom |
| `16x32` stages 1 | ~32 KB | most conservative |

`BH` must divide `HQ` (64 for GLM-5.3-Flash), and the grid is
`cdiv(HQ, BH) * SQ`, so changing `BH` is safe.

`GLM53_FLASHMLA_TILE` can pin a single preset for debugging.

## Verification

Sparse-MLA numerics were checked against a PyTorch reference at the shapes the
model actually uses:

```
shape SQ=8 HQ=64 DQK=576 SKV=256 TOPK=128
cos(output[1:], ref)  = 0.999998
cos(lse, ref_lse)     = 0.999998
```

(The full-tensor cosine is NaN because the test deliberately pads one row with
invalid `-1` indices, which the reference leaves at zero.)

End-to-end, after both fixes:

| Prompt | Output | |
|---|---|---|
| `问:中国的首都是哪个城市呢？答:应该是` | `北京。北京。北京是中国的首都` | ✅ 8/8 runs |
| `What is the capital of France? Answer:` | ` Paris` | ✅ |
| `问:1+1等于几？答:` | `1+1等于2,1+1等于3,` | ⚠️ first answer right, then degrades |

The model passes the day-0 smoke check but degrades after a few tokens, so it is
not fully healthy yet. That is a model/kernel-accuracy question separate from
these two compile/launch fixes.

## Branch / commit

| | |
|---|---|
| Repository | `tonyh168/FlagGems-vllm` |
| Branch | `fix/glm5-metax-causal-conv-tile` |
| Base | `origin/main` (flagos-ai) |
| Files | `src/flaggems_vllm/ops/causal_conv1d.py`, `src/flaggems_vllm/ops/flashmla_sparse.py` |

Related fixes live in `tonyh168/vllm-plugin-FL`
(`fix/glm5-metax-indexer-block-size-mccl-abort`): GLM5 compressed-indexer
block-size alignment and an `ncclCommAbort` fallback for the MCCL wrapper.
