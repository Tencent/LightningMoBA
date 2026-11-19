"""Single-pass dynamic-threshold sparse-pattern builder for LightningMoBA.

Phase-1 scope: batch size 1, causal packed Q/K summaries, fp16/bf16, GQA.
The selector never materializes the dense score matrix. It compacts threshold
hits per Q/head into a fixed-capacity workspace, records the true row NNZ and
overflow, and emits the CSC tensors consumed by LightningMoBA attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
import triton
import triton.language as tl


@triton.jit
def _threshold_select_count_kernel(
    q_ptr,
    km_ptr,
    threshold_ptr,
    row_indices_ptr,
    row_nnz_ptr,
    row_overflow_ptr,
    col_nnz_ptr,
    seq_len,
    num_k_blocks,
    stride_q_row,
    stride_q_head,
    stride_km_row,
    stride_km_head,
    stride_rowidx_row,
    stride_rowidx_head,
    stride_rowidx_slot,
    stride_meta_row,
    stride_meta_head,
    stride_col_head,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_block = tl.program_id(0)
    q_head = tl.program_id(1)
    rows = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    valid_rows = rows < seq_len
    kv_head = q_head // (num_q_heads // num_kv_heads)

    q = tl.load(
        q_ptr
        + rows[:, None] * stride_q_row
        + q_head * stride_q_head
        + dims[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    )
    threshold = tl.load(threshold_ptr + q_head).to(tl.float32)
    # Normalize out per-query magnitude so one scalar/per-head policy does not
    # create pathological NNZ tails merely because Q norms differ.
    q_f32 = q.to(tl.float32)
    q_rms = tl.sqrt(tl.sum(q_f32 * q_f32, axis=1) / HEAD_DIM)
    effective_threshold = threshold * q_rms
    selected_count = tl.zeros((BLOCK_M,), dtype=tl.int32)

    # A runtime loop avoids unrolling 1024/32 score tiles into enormous code.
    for k_start in range(0, num_k_blocks, BLOCK_N):
        blocks = k_start + tl.arange(0, BLOCK_N)
        valid_blocks = blocks < num_k_blocks
        km = tl.load(
            km_ptr
            + blocks[:, None] * stride_km_row
            + kv_head * stride_km_head
            + dims[None, :],
            mask=valid_blocks[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(km), out_dtype=tl.float32)

        local_block = rows // CHUNK_SIZE
        # This is the same block-causal convention used by fused top-k:
        # completed earlier blocks are scored and the current local block is
        # forced in, including its not-yet-visible tokens at attention time.
        completed = (blocks[None, :] + 1) * CHUNK_SIZE <= rows[:, None]
        local = blocks[None, :] == local_block[:, None]
        keep = (
            valid_rows[:, None]
            & valid_blocks[None, :]
            & ((completed & (scores >= effective_threshold[:, None])) | local)
        )
        keep_i32 = keep.to(tl.int32)
        local_rank = tl.cumsum(keep_i32, axis=1) - 1
        positions = selected_count[:, None] + local_rank
        stored = keep & (positions < CAPACITY)
        tl.store(
            row_indices_ptr
            + rows[:, None] * stride_rowidx_row
            + q_head * stride_rowidx_head
            + positions * stride_rowidx_slot,
            blocks[None, :],
            mask=stored,
        )
        # Count only entries that are actually represented in the capacity
        # workspace, keeping CSC NNZ exactly consistent under overflow.
        # Explicitly broadcast the column pointer across Q rows. Triton does
        # not broadcast a [1, BLOCK_N] atomic pointer to a
        # [BLOCK_M, BLOCK_N] mask.
        col_ptrs = (
            col_nnz_ptr
            + q_head * stride_col_head
            + blocks[None, :]
            + rows[:, None] * 0
        )
        tl.atomic_add(col_ptrs, keep_i32, mask=stored)
        selected_count += tl.sum(keep_i32, axis=1)

    clipped = tl.minimum(selected_count, CAPACITY)
    overflow = tl.maximum(selected_count - CAPACITY, 0)
    tl.store(
        row_nnz_ptr + rows * stride_meta_row + q_head * stride_meta_head,
        clipped,
        mask=valid_rows,
    )
    tl.store(
        row_overflow_ptr + rows * stride_meta_row + q_head * stride_meta_head,
        overflow,
        mask=valid_rows,
    )


@triton.jit
def _csc_offset_kernel(
    col_nnz_ptr,
    col_offsets_ptr,
    seq_len,
    num_k_blocks,
    stride_col_head,
    capacity: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_head = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_N)
    valid = blocks < num_k_blocks
    counts = tl.load(
        col_nnz_ptr + q_head * stride_col_head + blocks,
        mask=valid,
        other=0,
    ).to(tl.int32)
    exclusive = tl.cumsum(counts, axis=0) - counts
    base = q_head.to(tl.int64) * seq_len * capacity
    tl.store(
        col_offsets_ptr + q_head * stride_col_head + blocks,
        base + exclusive.to(tl.int64),
        mask=valid,
    )


@triton.jit
def _csc_scatter_kernel(
    row_indices_ptr,
    row_nnz_ptr,
    col_offsets_ptr,
    col_cursor_ptr,
    csc_indices_ptr,
    seq_len,
    stride_rowidx_row,
    stride_rowidx_head,
    stride_rowidx_slot,
    stride_meta_row,
    stride_meta_head,
    stride_col_head,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_block = tl.program_id(0)
    q_head = tl.program_id(1)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    slots = tl.arange(0, CAPACITY)
    valid_rows = rows < seq_len
    nnz = tl.load(
        row_nnz_ptr + rows * stride_meta_row + q_head * stride_meta_head,
        mask=valid_rows,
        other=0,
    )
    valid = valid_rows[:, None] & (slots[None, :] < nnz[:, None])
    columns = tl.load(
        row_indices_ptr
        + rows[:, None] * stride_rowidx_row
        + q_head * stride_rowidx_head
        + slots[None, :] * stride_rowidx_slot,
        mask=valid,
        other=0,
    )
    cursor_ptrs = col_cursor_ptr + q_head * stride_col_head + columns
    positions = tl.atomic_add(cursor_ptrs, 1, mask=valid)
    offsets = tl.load(
        col_offsets_ptr + q_head * stride_col_head + columns,
        mask=valid,
        other=0,
    )
    tl.store(csc_indices_ptr + offsets + positions, rows[:, None], mask=valid)


@dataclass
class DynamicThresholdPattern:
    col_offsets: torch.Tensor
    col_nnz: torch.Tensor
    csc_indices: torch.Tensor
    row_indices: torch.Tensor
    row_nnz: torch.Tensor
    row_overflow: torch.Tensor

    @property
    def total_nnz(self) -> torch.Tensor:
        return self.row_nnz.sum(dtype=torch.int64)

    @property
    def total_overflow(self) -> torch.Tensor:
        return self.row_overflow.sum(dtype=torch.int64)


def flash_dynamic_threshold_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    threshold: Union[float, torch.Tensor],
    chunk_size: int,
    *,
    capacity: int = 256,
    causal: bool = True,
    block_m: int = 64,
    block_n: int = 32,
    build_csc: bool = True,
) -> DynamicThresholdPattern:
    """Build a threshold-selected CSC pattern without materializing scores.

    This phase-1 implementation intentionally supports batch size 1 only. The
    restriction keeps the performance experiment honest while the downstream
    variable-NNZ contract is still being validated.
    """
    if not (q.is_cuda and km.is_cuda):
        raise ValueError("q and km must be CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16) or km.dtype != q.dtype:
        raise TypeError("q and km must have the same fp16/bf16 dtype")
    if q.ndim != 3 or km.ndim != 3 or q.shape[2] != km.shape[2]:
        raise ValueError("expected q=[tokens,q_heads,d], km=[blocks,kv_heads,d]")
    if q.shape[1] % km.shape[1]:
        raise ValueError("KV heads must divide Q heads")
    if q.shape[2] not in (64, 128, 256):
        raise ValueError("phase-1 kernel supports head dimensions 64/128/256")
    if cu_seqlens_q.numel() != 2 or cu_seqlens_km.numel() != 2:
        raise NotImplementedError("phase-1 dynamic threshold supports batch size 1")
    if int(q.shape[0]) != max_seqlen_q or int(km.shape[0]) != (max_seqlen_k + chunk_size - 1) // chunk_size:
        raise ValueError("phase-1 requires one unpadded full-prefill sequence")
    if not causal:
        raise NotImplementedError("phase-1 kernel currently supports causal=True only")
    if capacity not in (64, 128, 256):
        raise ValueError("capacity must be 64, 128, or 256")
    if block_m not in (32, 64, 128) or block_n not in (16, 32, 64):
        raise ValueError("block_m must be 32/64/128 and block_n 16/32/64")

    q = q.contiguous()
    km = km.contiguous()
    seq_len, q_heads, head_dim = q.shape
    num_k_blocks, kv_heads, _ = km.shape
    if isinstance(threshold, torch.Tensor):
        if threshold.numel() == 1:
            thresholds = threshold.to(device=q.device, dtype=torch.float32).expand(q_heads).contiguous()
        elif threshold.numel() == q_heads:
            thresholds = threshold.to(device=q.device, dtype=torch.float32).reshape(q_heads).contiguous()
        else:
            raise ValueError("threshold tensor must be scalar or have q_heads elements")
    else:
        thresholds = torch.full((q_heads,), float(threshold), device=q.device, dtype=torch.float32)

    row_indices = torch.empty((seq_len, q_heads, capacity), device=q.device, dtype=torch.int32)
    row_nnz = torch.empty((seq_len, q_heads), device=q.device, dtype=torch.int32)
    row_overflow = torch.empty_like(row_nnz)
    col_nnz = torch.zeros((1, q_heads, num_k_blocks), device=q.device, dtype=torch.int32)
    col_offsets = torch.empty((1, q_heads, num_k_blocks), device=q.device, dtype=torch.int64)

    _threshold_select_count_kernel[(triton.cdiv(seq_len, block_m), q_heads)](
        q,
        km,
        thresholds,
        row_indices,
        row_nnz,
        row_overflow,
        col_nnz,
        seq_len,
        num_k_blocks,
        q.stride(0),
        q.stride(1),
        km.stride(0),
        km.stride(1),
        row_indices.stride(0),
        row_indices.stride(1),
        row_indices.stride(2),
        row_nnz.stride(0),
        row_nnz.stride(1),
        col_nnz.stride(1),
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        HEAD_DIM=head_dim,
        CHUNK_SIZE=chunk_size,
        CAPACITY=capacity,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8 if block_m >= 64 else 4,
        num_stages=2,
    )

    if not build_csc:
        return DynamicThresholdPattern(
            col_offsets=col_offsets,
            col_nnz=col_nnz,
            csc_indices=torch.empty(0, device=q.device, dtype=torch.int32),
            row_indices=row_indices,
            row_nnz=row_nnz,
            row_overflow=row_overflow,
        )

    csc_indices = torch.empty(seq_len * q_heads * capacity, device=q.device, dtype=torch.int32)
    offset_block = triton.next_power_of_2(num_k_blocks)
    if offset_block > 2048:
        raise ValueError("phase-1 CSC offset supports at most 2048 K blocks")
    _csc_offset_kernel[(q_heads,)](
        col_nnz,
        col_offsets,
        seq_len,
        num_k_blocks,
        col_nnz.stride(1),
        capacity=capacity,
        BLOCK_N=offset_block,
        num_warps=8,
    )

    col_cursor = torch.zeros_like(col_nnz)
    _csc_scatter_kernel[(triton.cdiv(seq_len, 32), q_heads)](
        row_indices,
        row_nnz,
        col_offsets,
        col_cursor,
        csc_indices,
        seq_len,
        row_indices.stride(0),
        row_indices.stride(1),
        row_indices.stride(2),
        row_nnz.stride(0),
        row_nnz.stride(1),
        col_nnz.stride(1),
        CAPACITY=capacity,
        BLOCK_M=32,
        num_warps=8,
    )
    return DynamicThresholdPattern(
        col_offsets=col_offsets,
        col_nnz=col_nnz,
        csc_indices=csc_indices,
        row_indices=row_indices,
        row_nnz=row_nnz,
        row_overflow=row_overflow,
    )
