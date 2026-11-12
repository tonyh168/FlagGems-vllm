# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thead-only sorted MoE down projection with sparse MMA16/dense MMA32.

One/two-row expert groups compact their original routed row IDs into the
minimum valid m16 MMA tile. Denser groups retain captured m32. Both keep
BN64/BK128/FP32 accumulation and router multiplication before BF16 store.
Original sorted groups remain32 rows; no assignment or router change.
The rejected C49 FP32 vector reduction branch is absent. No tile below m16,
workspace, atomics, copies, synchronization or quantization is introduced.
Input/output may share disjoint workspace; pointers include their offsets.
Fixed launch parameters are a Thead-only captured-schedule exemption;
no new NVIDIA path or NVIDIA autotune support. Full model gates mandatory.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def moe_down_mma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    weights_ptr,
    sorted_ptr,
    experts_ptr,
    padded_ptr,
    rows: tl.constexpr,
    n: tl.constexpr,
    em: tl.constexpr,
    stride_be: tl.constexpr,
):
    group = tl.program_id(0)
    col_tile = tl.program_id(1)
    padded = tl.load(padded_ptr)
    if group * 32 >= padded:
        return
    positions = group * 32 + tl.arange(0, 32)
    routes = tl.load(sorted_ptr + positions, mask=positions < em, other=rows)
    valid = (routes >= 0) & (routes < rows)
    count = tl.sum(valid.to(tl.int32), 0)
    if count == 0:
        return
    expert = tl.load(experts_ptr + group).to(tl.int64)
    cols = col_tile * 64 + tl.arange(0, 64)
    if expert == -1:
        zeros = tl.zeros((32, 64), tl.float32)
        tl.store(
            c_ptr + routes[:, None].to(tl.int64) * n + cols[None, :],
            zeros,
            mask=valid[:, None] & (cols[None, :] < n),
        )
        return

    if count <= 2:
        first = tl.min(tl.where(valid, routes, rows), 0)
        last = tl.max(tl.where(valid, routes, -1), 0)
        lanes = tl.arange(0, 16)
        small_valid = lanes < count
        small_routes = tl.where(lanes == 0, first, tl.where(lanes == 1, last, rows))
        small_acc = tl.zeros((16, 64), tl.float32)
        for small_block in tl.static_range(0, 2):
            small_k = small_block * 128 + tl.arange(0, 128)
            small_a = tl.load(
                a_ptr + small_routes[:, None].to(tl.int64) * 256 + small_k[None, :],
                mask=small_valid[:, None],
                other=0.0,
            )
            small_b = tl.load(
                b_ptr + expert * stride_be + small_k[:, None] + cols[None, :] * 256,
                mask=cols[None, :] < n,
                other=0.0,
            )
            small_acc += tl.dot(small_a, small_b, allow_tf32=False)
        small_weights = tl.load(weights_ptr + small_routes, mask=small_valid, other=0.0)
        small_acc *= small_weights[:, None]
        tl.store(
            c_ptr + small_routes[:, None].to(tl.int64) * n + cols[None, :],
            small_acc.to(tl.bfloat16),
            mask=small_valid[:, None] & (cols[None, :] < n),
        )

    else:
        # Keep the frozen generic BF16 down-GEMM K grouping and router rounding.
        acc = tl.zeros((32, 64), tl.float32)
        for block in tl.static_range(0, 2):
            kk = block * 128 + tl.arange(0, 128)
            a = tl.load(
                a_ptr + routes[:, None].to(tl.int64) * 256 + kk[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            b = tl.load(
                b_ptr + expert * stride_be + kk[:, None] + cols[None, :] * 256,
                mask=cols[None, :] < n,
                other=0.0,
            )
            acc += tl.dot(a, b, allow_tf32=False)
        weights = tl.load(weights_ptr + routes, mask=valid, other=0.0)
        acc *= weights[:, None]
        tl.store(
            c_ptr + routes[:, None].to(tl.int64) * n + cols[None, :],
            acc.to(tl.bfloat16),
            mask=valid[:, None] & (cols[None, :] < n),
        )


def storage_spans_overlap(left, right):
    """Exact byte-span comparison for the contiguous supported views only."""
    if left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr():
        return False
    ls = left.storage_offset() * left.element_size()
    rs = right.storage_offset() * right.element_size()
    return max(ls, rs) < min(
        ls + left.numel() * left.element_size(),
        rs + right.numel() * right.element_size(),
    )


def supports_moe_down_mma(a, b, out, weights, sorted_ids, expert_ids, padded, config):
    """Metadata guard; quant/bias/activation flags stay the public caller's gate."""
    if any(x is None for x in [a, b, out, weights, sorted_ids, expert_ids, padded]):
        return False
    if not isinstance(config, dict):
        return False
    required = dict(
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=1,
        num_warps=4,
        num_stages=3,
    )
    if any(config.get(key) != value for key, value in required.items()):
        return False
    if config.get("SPLIT_K", 1) != 1 or config.get("SWAP_AB", False):
        return False
    if not (a.ndim == 2 and b.ndim == out.ndim == 3 and weights.ndim == 2):
        return False
    if not (
        a.dtype == b.dtype == out.dtype == torch.bfloat16
        and weights.dtype == torch.float32
        and all(x.dtype == torch.int32 for x in [sorted_ids, expert_ids, padded])
    ):
        return False
    tensors = [a, b, out, weights, sorted_ids, expert_ids, padded]
    if a.device.type != "cuda" or any(
        x.device != a.device or not x.is_contiguous() for x in tensors
    ):
        return False
    r, k = a.shape
    e, n, bk = b.shape
    if not (
        1 <= weights.shape[0] <= 64
        and weights.shape[1] == 8
        and r == weights.numel()
        and k == bk == 256
        and e > 0
        and 1024 <= n <= 4096
        and n % 64 == 0
        and tuple(out.shape) == tuple(weights.shape) + (n,)
    ):
        return False
    if not (
        sorted_ids.ndim == expert_ids.ndim == 1
        and padded.numel() == 1
        and r <= sorted_ids.numel() <= r + e * 31 + 31
        and expert_ids.numel() >= (sorted_ids.numel() + 31) // 32
    ):
        return False
    # Shared allocator storage is valid; overlapping read/write spans are not.
    return not any(
        storage_spans_overlap(out, x)
        for x in [a, b, weights, sorted_ids, expert_ids, padded]
    )


def moe_down_mma_out(a, b, out, weights, sorted_ids, expert_ids, padded, config):
    """Internal out primitive; unsupported metadata never falls back to torch."""
    if not supports_moe_down_mma(
        a, b, out, weights, sorted_ids, expert_ids, padded, config
    ):
        raise NotImplementedError(
            "Unsupported sorted BF16 MoE down-projection contract"
        )
    grid = (triton.cdiv(sorted_ids.numel(), 32), triton.cdiv(b.shape[1], 64))
    with torch_device_fn.device(a.device):
        moe_down_mma_kernel[grid](
            a,
            b,
            out,
            weights,
            sorted_ids,
            expert_ids,
            padded,
            a.shape[0],
            b.shape[1],
            sorted_ids.numel(),
            b.stride(0),
            num_warps=4,
            num_stages=3,
        )
    return out
