"""Bounded KV-centric prefill with the validated online FP32 O/LSE merge."""
from __future__ import annotations

import os

import torch


def streaming_chunk_edges(value: int | None = None) -> int:
    value = int(os.environ.get("LIGHTNING_MOBA_STREAM_CHUNK_EDGES", "2000000")) if value is None else value
    if not isinstance(value, int) or not 0 <= value <= 2147483647:
        raise ValueError("LIGHTNING_MOBA_STREAM_CHUNK_EDGES must be in [0, INT_MAX]")
    return value


def build_streaming_q_csr(col_offsets, col_nnz, indices, cu_seqlens_q, total_q, num_heads):
    """Stable edge grouping makes each Q-CSR row increasing in CSC edge ID."""
    counts = col_nnz.reshape(-1).to(torch.int64)
    columns = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=torch.int64), counts
    )
    if columns.numel() > indices.numel():
        raise ValueError("CSC storage is smaller than its column counts")
    indices = indices[:columns.numel()]
    expected = counts.cumsum(0) - counts
    if not torch.equal(col_offsets.reshape(-1), expected):
        raise ValueError("streaming requires exclusive-prefix CSC offsets")
    max_columns = col_nnz.shape[-1]
    head = torch.div(columns, max_columns, rounding_mode="floor").remainder(num_heads)
    batch = torch.div(columns, max_columns * num_heads, rounding_mode="floor")
    local_q = indices.to(torch.int64)
    lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    if bool(((local_q < 0) | (local_q >= lengths[batch])).any().item()):
        raise ValueError("CSC contains an out-of-range query index")
    qh = (local_q + cu_seqlens_q[batch].to(torch.int64)) * num_heads + head
    permutation = torch.argsort(qh, stable=True)
    row_counts = torch.bincount(qh, minlength=total_q * num_heads)
    q_offsets = torch.cat((row_counts.new_zeros(1), row_counts.cumsum(0)))
    return q_offsets.contiguous(), permutation.to(torch.int32).contiguous()


def compact_streaming_csc(col_offsets, col_nnz, indices):
    """Remove fixed-TopK per-head padding without changing selected rows."""
    counts = col_nnz.reshape(-1).to(torch.int64)
    logical_nnz = int(counts.sum().item())
    if logical_nnz < 0 or logical_nnz > indices.numel():
        raise ValueError("CSC counts exceed available storage")
    compact_offsets = counts.cumsum(0) - counts
    if torch.equal(col_offsets.reshape(-1), compact_offsets):
        return col_offsets, indices[:logical_nnz]
    columns = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=torch.int64), counts
    )
    source_edges = torch.arange(logical_nnz, device=counts.device, dtype=torch.int64)
    source_edges += col_offsets.reshape(-1)[columns] - compact_offsets[columns]
    if bool(((source_edges < 0) | (source_edges >= indices.numel())).any().item()):
        raise ValueError("CSC column offsets exceed available storage")
    return compact_offsets.reshape_as(col_nnz), indices.index_select(0, source_edges)


def lightning_moba_streaming_kv_centric_func(
    q, k, v, col_offsets, col_nnz, indices, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, softmax_scale=None, causal=True, chunk_size=128,
    *, chunk_edges=None, restrict_grid=None, q_offsets=None, q_csc=None,
):
    """Return final O and head-major global LSE; eager-only, no dropout.

    The CSC pattern is unchanged. Partial O/LSE use at most ``chunk_edges``
    entries; CSC and Q-CSR metadata remain proportional to total NNZ.
    """
    import lightning_moba_cuda
    from lightning_moba.cuda_dynamic_threshold import build_compact_kvcentric_msplit_worklist

    capacity = streaming_chunk_edges(chunk_edges)
    if capacity == 0:
        raise ValueError("the explicit streaming API requires chunk_edges > 0")
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("streaming prefill is eager-only; use the existing decode graph path")
    if not hasattr(lightning_moba_cuda, "moba_varlen_fwd_kvcentric_stream"):
        raise RuntimeError("lightning_moba_cuda lacks streaming support; rebuild the selected source with setup.py")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("expected packed Q/K/V tensors with matching K/V shapes")
    if col_nnz.ndim != 3 or col_offsets.shape != col_nnz.shape:
        raise ValueError("expected CSC metadata [batch, heads, columns]")
    if cu_seqlens_q.numel() != col_nnz.shape[0] + 1 or col_nnz.shape[1] != q.shape[1]:
        raise ValueError("CSC batch/head axes do not match Q")
    if (q_offsets is None) != (q_csc is None):
        raise ValueError("provide both sorted Q-CSR tensors or neither")
    original_offsets = col_offsets
    col_offsets, indices = compact_streaming_csc(col_offsets, col_nnz, indices)
    if q_offsets is not None and col_offsets is not original_offsets:
        raise ValueError("prebuilt Q-CSR requires compact CSC storage")
    if q_offsets is None:
        q_offsets, q_csc = build_streaming_q_csr(
            col_offsets, col_nnz, indices, cu_seqlens_q, q.shape[0], q.shape[1]
        )
    work = build_compact_kvcentric_msplit_worklist(col_nnz)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if restrict_grid is None:
        restrict_grid = os.environ.get("LIGHTNING_MOBA_STREAM_RANGED", "1") != "0"
    return tuple(lightning_moba_cuda.moba_varlen_fwd_kvcentric_stream(
        q.contiguous(), k.contiguous(), v.contiguous(), col_offsets.contiguous(),
        col_nnz.contiguous(), indices.contiguous(), work, cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(), q_offsets, q_csc, max_seqlen_q, max_seqlen_k,
        softmax_scale, causal, chunk_size, capacity, restrict_grid,
    ))
