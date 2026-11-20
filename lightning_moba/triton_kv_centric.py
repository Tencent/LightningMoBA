"""KV-Centric LightningMoBA Attention — Triton Verification Implementation.

Follows the MSA (MiniMax Sparse Attention) approach:
- Grid: (num_columns, batch * num_heads_kv) — KV-head level dispatch
- Each CTA processes one KV column for ALL Q heads in its KV group
- O_partial layout: [topk, total_q, num_heads, head_dim] (split-indexed)
- LSE_partial layout: [topk, total_q, num_heads]
- A separate Triton combine kernel merges partial outputs using online softmax

Key differences from previous version:
1. Uses split_slot indexing (MSA style) instead of CSC-offset indexing
2. Proper GQA support — one CTA processes all Q heads in a KV group
3. Triton combine kernel replaces Python loop
4. Correct causal mask with varlen support

Usage:
    from lightning_moba.triton_kv_centric import lightning_moba_kv_centric_func

    out = lightning_moba_kv_centric_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        col_offsets, col_nnz, indices, chunk_size, topk, causal=True,
    )
"""

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple


# =============================================================================
# KV-Centric Attention Forward Kernel (K1), with CSC partial indexing
# =============================================================================

@triton.jit
def _kv_centric_attn_fwd_v2(
    # Q/K/V pointers
    Q, K, V,
    # CSC index pointers
    Col_offsets, Col_nnz, Indices,
    # Output partial pointers (CSC-offset indexed — simpler, no atomics)
    O_partial, LSE_partial,
    # Strides for Q: [total_q, num_heads, head_dim]
    stride_qr, stride_qh, stride_qd,
    # Strides for K: [total_k, num_heads_kv, head_dim]
    stride_kr, stride_kh, stride_kd,
    # Strides for V
    stride_vr, stride_vh, stride_vd,
    # O_partial stride: flat [nnz_total, head_dim]
    stride_op_r, stride_op_d,
    # Dimensions
    seqlen_q, seqlen_k, head_dim: tl.constexpr,
    num_heads, num_heads_kv,
    chunk_size: tl.constexpr,
    # Scaling
    softmax_scale,
    # Flags
    CAUSAL: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # GQA ratio
    H_H_K_RATIO: tl.constexpr,
):
    """Simpler version: O_partial indexed directly by CSC offset (no atomics needed).

    This is the preferred version for verification since it avoids atomic contention.
    The combine step uses CSC metadata to find which partials belong to each Q row.

    Grid: (num_columns, batch * num_heads)
    Each CTA processes one (column, batch, q_head) triple.
    """
    # --- Decode grid position ---
    col_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    bidh = bh_idx % num_heads
    bidb = bh_idx // num_heads
    bidh_kv = bidh // H_H_K_RATIO

    # --- KV block boundaries ---
    kv_start = col_idx * chunk_size
    if kv_start >= seqlen_k:
        return
    kv_end = tl.minimum(kv_start + chunk_size, seqlen_k)

    # --- Read CSC metadata for this (batch, head, column) ---
    num_columns = tl.cdiv(seqlen_k, chunk_size)
    col_meta_idx = bidb * num_heads * num_columns + bidh * num_columns + col_idx
    col_offset = tl.load(Col_offsets + col_meta_idx)
    col_count = tl.load(Col_nnz + col_meta_idx)

    if col_count == 0:
        return

    # --- Base pointers ---
    Q_base = Q + bidb * seqlen_q * stride_qr + bidh * stride_qh
    K_base = K + bidb * seqlen_k * stride_kr + bidh_kv * stride_kh
    V_base = V + bidb * seqlen_k * stride_vr + bidh_kv * stride_vh

    d_offs = tl.arange(0, BLOCK_D)
    NUM_KV_BLOCKS: tl.constexpr = chunk_size // BLOCK_N

    # --- Process Q rows in batches of BLOCK_M ---
    for q_batch_iter in range(0, 4096):
        q_batch_idx = q_batch_iter * BLOCK_M
        if q_batch_idx < col_count:
            cur_q_count = tl.minimum(BLOCK_M, col_count - q_batch_idx)

            # --- Gather Q row indices from CSC ---
            q_local_offs = tl.arange(0, BLOCK_M)
            q_idx_ptrs = Indices + col_offset + q_batch_idx + q_local_offs
            q_mask = q_local_offs < cur_q_count
            q_row_indices = tl.load(q_idx_ptrs, mask=q_mask, other=0)

            # --- Initialize accumulators ---
            acc_o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
            row_max = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
            row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

            # --- Load Q rows (gathered) ---
            q_ptrs = Q_base + q_row_indices[:, None] * stride_qr + d_offs[None, :]
            q_block = tl.load(
                q_ptrs,
                mask=q_mask[:, None] & (d_offs[None, :] < head_dim),
                other=0.0
            )

            # --- Inner loop over KV sub-blocks ---
            for n_block in range(NUM_KV_BLOCKS):
                kv_block_start = kv_start + n_block * BLOCK_N
                kv_offs = tl.arange(0, BLOCK_N)
                kv_row = kv_block_start + kv_offs
                kv_mask = kv_row < kv_end

                # Load K: [BLOCK_N, BLOCK_D]
                k_ptrs = K_base + kv_row[:, None] * stride_kr + d_offs[None, :]
                k_block = tl.load(
                    k_ptrs,
                    mask=kv_mask[:, None] & (d_offs[None, :] < head_dim),
                    other=0.0
                )

                # QK^T: [BLOCK_M, BLOCK_N]
                scores = tl.dot(q_block, tl.trans(k_block)) * softmax_scale

                # Causal mask
                if CAUSAL:
                    causal_offset = seqlen_k - seqlen_q
                    max_kv_allowed = q_row_indices[:, None] + causal_offset
                    kv_positions = kv_row[None, :]
                    causal_mask = kv_positions <= max_kv_allowed
                    scores = tl.where(
                        causal_mask & kv_mask[None, :] & q_mask[:, None],
                        scores, -float('inf')
                    )
                else:
                    scores = tl.where(
                        kv_mask[None, :] & q_mask[:, None],
                        scores, -float('inf')
                    )

                # Online softmax
                block_max = tl.max(scores, axis=1)
                new_max = tl.maximum(row_max, block_max)
                has_valid_scores = new_max > -float('inf')
                scale_prev = tl.where(
                    has_valid_scores,
                    tl.exp2((row_max - new_max) * 1.44269504),
                    0.0,
                )
                row_sum = row_sum * scale_prev
                acc_o = acc_o * scale_prev[:, None]

                scores_scaled = tl.where(
                    has_valid_scores[:, None],
                    tl.exp2((scores - new_max[:, None]) * 1.44269504),
                    0.0,
                )
                block_sum = tl.sum(scores_scaled, axis=1)
                row_sum = row_sum + block_sum
                row_max = new_max

                # Load V: [BLOCK_N, BLOCK_D]
                v_ptrs = V_base + kv_row[:, None] * stride_vr + d_offs[None, :]
                v_block = tl.load(
                    v_ptrs,
                    mask=kv_mask[:, None] & (d_offs[None, :] < head_dim),
                    other=0.0
                )

                # P @ V
                acc_o += tl.dot(scores_scaled.to(v_block.dtype), v_block)

            # --- Normalize output and compute LSE ---
            safe_sum = tl.where(row_sum > 0, row_sum, 1.0)
            acc_o = acc_o / safe_sum[:, None]
            lse = tl.where(row_sum > 0, row_max + tl.log(safe_sum), -float('inf'))

            # --- Write O_partial and LSE_partial using CSC offset directly ---
            out_base_idx = col_offset + q_batch_idx
            q_local_offs2 = tl.arange(0, BLOCK_M)
            out_row_indices = out_base_idx + q_local_offs2
            o_flat_ptrs = O_partial + out_row_indices[:, None] * stride_op_r + d_offs[None, :] * stride_op_d
            tl.store(
                o_flat_ptrs,
                acc_o,
                mask=q_mask[:, None] & (d_offs[None, :] < head_dim)
            )

            lse_flat_ptrs = LSE_partial + out_row_indices
            tl.store(lse_flat_ptrs, lse, mask=q_mask)


# =============================================================================
# Python Combine (correct, for verification)
# =============================================================================

def _combine_partial_outputs(
    o_partial: torch.Tensor,      # [total_nnz, head_dim] fp32
    lse_partial: torch.Tensor,    # [total_nnz] fp32
    col_offsets: torch.Tensor,    # [num_heads, num_columns] int64 (for batch=1)
    col_nnz: torch.Tensor,        # [num_heads, num_columns] int32
    indices: torch.Tensor,        # [total_nnz] int32  (per-head CSC indices)
    seqlen_q: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized combine using scatter-add style operations.

    For each head, iterate over columns (vectorized within each column).
    Uses online softmax merge: for each Q position, accumulate across columns.
    """
    device = o_partial.device
    out = torch.zeros(seqlen_q, num_heads, head_dim, device=device, dtype=torch.float32)
    lse_acc = torch.full((seqlen_q, num_heads), -float('inf'), device=device, dtype=torch.float32)

    num_columns = col_offsets.shape[1]

    for h in range(num_heads):
        for col in range(num_columns):
            offset = col_offsets[h, col].item()
            count = col_nnz[h, col].item()
            if count == 0:
                continue

            # Get Q indices and their partial results
            q_indices = indices[offset:offset + count].long()  # [count]
            o_rows = o_partial[offset:offset + count]          # [count, head_dim]
            lse_rows = lse_partial[offset:offset + count]      # [count]

            # Online softmax merge (vectorized over count)
            prev_lse = lse_acc[q_indices, h]  # [count]
            new_lse = lse_rows               # [count]

            # Compute merged LSE
            lse_max = torch.maximum(prev_lse, new_lse)
            # Handle -inf cases
            is_first = (prev_lse == -float('inf'))

            exp_prev = torch.exp(prev_lse - lse_max)
            exp_new = torch.exp(new_lse - lse_max)
            total = exp_prev + exp_new
            # Avoid division by zero
            total = torch.where(total > 0, total, torch.ones_like(total))

            w_prev = exp_prev / total
            w_new = exp_new / total

            # For first-time entries, just copy
            w_prev = torch.where(is_first, torch.zeros_like(w_prev), w_prev)
            w_new = torch.where(is_first, torch.ones_like(w_new), w_new)

            # Update output
            prev_out = out[q_indices, h, :]   # [count, head_dim]
            new_out = w_prev[:, None] * prev_out + w_new[:, None] * o_rows
            out[q_indices, h, :] = new_out

            # Update LSE
            new_lse_merged = torch.where(
                is_first,
                new_lse,
                lse_max + torch.log(total)
            )
            lse_acc[q_indices, h] = new_lse_merged

    return out.to(dtype), lse_acc


# =============================================================================
# Triton Combine Kernel (vectorized, correct)
# =============================================================================

@triton.jit
def _triton_combine_kernel(
    # Inputs
    O_partial,        # flat [total_nnz, head_dim]
    LSE_partial,      # flat [total_nnz]
    # Reverse index: for each Q row, which CSC entries it belongs to
    Q2col_offsets,    # [total_q * num_heads, topk] — CSC offsets for each (q, h) pair
    Q2col_valid,      # [total_q * num_heads] — number of valid columns per (q, h)
    # Output
    O_out,            # [total_q, num_heads, head_dim]
    # Strides
    stride_or, stride_oh, stride_od,
    stride_op_r, stride_op_d,
    stride_q2c_r, stride_q2c_c,
    # Dimensions
    total_q, num_heads, head_dim: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Combine partial outputs for one (q_row, head) pair using pre-built reverse index.

    Grid: (total_q * num_heads,)
    Each program handles one (q_idx, head) pair, merging up to topk partial outputs.
    """
    pid = tl.program_id(0)
    q_idx = pid // num_heads
    h_idx = pid % num_heads

    if q_idx >= total_q:
        return

    # Number of valid partials for this (q, h)
    qh_idx = q_idx * num_heads + h_idx
    n_valid = tl.load(Q2col_valid + qh_idx)
    if n_valid == 0:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    # Initialize accumulator
    acc_o = tl.zeros([BLOCK_D], dtype=tl.float32)
    row_max = tl.full([1], value=-float('inf'), dtype=tl.float32)
    row_sum = tl.zeros([1], dtype=tl.float32)

    # Iterate over valid partials (up to topk)
    for s in range(topk):
        if s < n_valid:
            # Get the CSC offset for this partial
            csc_offset = tl.load(Q2col_offsets + qh_idx * stride_q2c_r + s * stride_q2c_c)

            # Load partial O and LSE
            o_ptr = O_partial + csc_offset * stride_op_r + d_offs * stride_op_d
            partial_o = tl.load(o_ptr, mask=d_mask, other=0.0)
            partial_lse = tl.load(LSE_partial + csc_offset)

            # Skip entries where LSE is -inf (fully masked by causal)
            is_valid_lse = partial_lse > -float('inf')
            if is_valid_lse:
                # Online softmax merge
                new_max = tl.maximum(row_max, partial_lse)
                # Guard against -inf - (-inf) = NaN
                scale_prev = tl.where(row_max > -float('inf'),
                                      tl.exp2((row_max - new_max) * 1.44269504),
                                      0.0)
                scale_new = tl.exp2((partial_lse - new_max) * 1.44269504)

                row_sum = row_sum * scale_prev + scale_new
                acc_o = acc_o * scale_prev + scale_new * partial_o
                row_max = new_max

    # Normalize
    safe_sum = tl.where(row_sum > 0, row_sum, 1.0)
    acc_o = acc_o / safe_sum

    # Write output
    o_out_ptr = O_out + q_idx * stride_or + h_idx * stride_oh + d_offs * stride_od
    tl.store(o_out_ptr, acc_o, mask=d_mask)


# =============================================================================
# Helper: build Q-to-CSC-offset reverse index
# =============================================================================


def _build_q2col_reverse_index_fast(
    col_offsets: torch.Tensor,   # [num_heads, num_columns] int64
    col_nnz: torch.Tensor,       # [num_heads, num_columns] int32
    indices: torch.Tensor,       # [total_nnz] int32
    seqlen_q: int,
    num_heads: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized version of reverse index building (much faster for large inputs)."""
    device = col_offsets.device
    q2col_offsets = torch.zeros(seqlen_q * num_heads, topk, dtype=torch.int64, device=device)
    q2col_valid = torch.zeros(seqlen_q * num_heads, dtype=torch.int32, device=device)

    num_columns = col_offsets.shape[1]

    for h in range(num_heads):
        for col in range(num_columns):
            offset = col_offsets[h, col].item()
            count = col_nnz[h, col].item()
            if count == 0:
                continue

            q_indices = indices[offset:offset + count].long()  # [count]
            qh_indices = q_indices * num_heads + h              # [count]
            slots = q2col_valid[qh_indices].long()              # current slot for each
            valid_mask = slots < topk

            if valid_mask.any():
                valid_qh = qh_indices[valid_mask]
                valid_slots = slots[valid_mask]
                valid_csc_offsets = (offset + torch.arange(count, device=device)[valid_mask])

                q2col_offsets[valid_qh, valid_slots] = valid_csc_offsets
                q2col_valid[valid_qh] = (slots[valid_mask] + 1).to(torch.int32)
                # Note: this has race conditions if same Q appears multiple times
                # in same column, but CSC guarantees uniqueness within a column.
                # However, the sequential head/column loop avoids races.

    return q2col_offsets, q2col_valid


def _build_q2col_reverse_index_varlen(
    col_offsets: torch.Tensor,   # [batch, num_heads, num_columns] int64
    col_nnz: torch.Tensor,       # [batch, num_heads, num_columns] int32
    indices: torch.Tensor,       # [total_nnz] int32, sequence-local q rows
    cu_seqlens_q: torch.Tensor,  # [batch + 1] int32
    total_q: int,
    num_heads: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build reverse index for varlen/multi-request CSC offsets.

    The CUDA KV-centric kernel writes one partial output per CSC entry. The
    combine kernel wants the reverse mapping from each global (q, head) row to
    the CSC entries that contributed to it.
    """
    device = col_offsets.device
    q2col_offsets = torch.zeros(total_q * num_heads, topk, dtype=torch.int64, device=device)
    q2col_valid = torch.zeros(total_q * num_heads, dtype=torch.int32, device=device)

    batch_size = col_offsets.shape[0]
    num_columns = col_offsets.shape[2]

    for b in range(batch_size):
        q_base = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        if q_end <= q_base:
            continue

        for h in range(num_heads):
            for col in range(num_columns):
                offset = int(col_offsets[b, h, col].item())
                count = int(col_nnz[b, h, col].item())
                if count == 0:
                    continue

                q_indices = indices[offset:offset + count].long() + q_base
                qh_indices = q_indices * num_heads + h
                slots = q2col_valid[qh_indices].long()
                valid_mask = slots < topk

                if valid_mask.any():
                    valid_qh = qh_indices[valid_mask]
                    valid_slots = slots[valid_mask]
                    valid_csc_offsets = offset + torch.arange(count, device=device)[valid_mask]
                    q2col_offsets[valid_qh, valid_slots] = valid_csc_offsets
                    q2col_valid[valid_qh] = (slots[valid_mask] + 1).to(torch.int32)

    return q2col_offsets, q2col_valid


@dataclass(frozen=True)
class MobaDecodeBucket:
    """Complete fixed-shape signature for one replayable MoBA decode graph."""

    batch_size: int
    total_q: int
    total_k: int
    max_seqlen_q: int
    max_seqlen_k: int
    num_heads: int
    num_heads_kv: int
    head_dim: int
    topk: int
    chunk_size: int
    num_columns: int
    row_capacity: int
    causal: bool
    softmax_scale: float
    device: torch.device | str
    dtype: torch.dtype
    k1_mode: str = "default"

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        if min(self.batch_size, self.total_q, self.total_k, self.max_seqlen_q,
               self.max_seqlen_k, self.num_heads, self.num_heads_kv,
               self.head_dim, self.topk, self.chunk_size, self.num_columns,
               self.row_capacity) <= 0:
            raise ValueError("all MoBA decode bucket capacities must be positive")
        if self.num_heads % self.num_heads_kv:
            raise ValueError("num_heads must be divisible by num_heads_kv")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MoBA decode bucket dtype must be fp16 or bf16")

    @property
    def cache_key(self) -> tuple:
        return (
            "moba-decode-v1", self.batch_size, self.total_q, self.total_k,
            self.max_seqlen_q, self.max_seqlen_k, self.num_heads,
            self.num_heads_kv, self.head_dim, self.topk, self.chunk_size,
            self.num_columns, self.row_capacity, self.causal,
            self.softmax_scale, str(self.device), str(self.dtype), self.k1_mode,
        )


@dataclass
class KvCentricGraphWorkspace:
    """Fixed-shape buffers reused by a KV-centric CUDA-Graph bucket.

    The object owns the reverse-index and final-output storage.  CSC metadata
    and Q/K/V remain caller-owned inputs so a serving adapter can update their
    values in place before replay.  `out` is borrowed: consume or copy it
    before the next replay of this workspace.
    """

    bucket: MobaDecodeBucket
    q2col_offsets: torch.Tensor
    q2col_valid: torch.Tensor
    out: torch.Tensor

    @classmethod
    def allocate(cls, bucket: MobaDecodeBucket) -> "KvCentricGraphWorkspace":
        q2col_offsets = torch.empty(
            (bucket.total_q * bucket.num_heads, bucket.topk),
            device=bucket.device, dtype=torch.int32,
        )
        return cls(
            bucket=bucket,
            q2col_offsets=q2col_offsets,
            q2col_valid=torch.empty(
                (bucket.total_q * bucket.num_heads,), device=bucket.device, dtype=torch.int32
            ),
            out=torch.empty(
                (bucket.total_q, bucket.num_heads, bucket.head_dim),
                device=bucket.device, dtype=bucket.dtype,
            ),
        )

    def validate(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                 max_seqlen_q, max_seqlen_k, col_offsets, col_nnz, indices,
                 chunk_size, topk, causal, softmax_scale) -> None:
        b = self.bucket
        current_mode = os.environ.get("LIGHTNING_MOBA_KVCENTRIC_SM90_K1", "default")
        valid = (
            tuple(q.shape) == (b.total_q, b.num_heads, b.head_dim)
            and tuple(k.shape) == (b.total_k, b.num_heads_kv, b.head_dim)
            and tuple(v.shape) == tuple(k.shape)
            and tuple(col_offsets.shape) == (b.batch_size, b.num_heads, b.num_columns)
            and tuple(col_nnz.shape) == tuple(col_offsets.shape)
            and indices.numel() == b.row_capacity
            and cu_seqlens_q.numel() == b.batch_size + 1
            and cu_seqlens_k.numel() == b.batch_size + 1
            and max_seqlen_q == b.max_seqlen_q and max_seqlen_k == b.max_seqlen_k
            and chunk_size == b.chunk_size and topk == b.topk and causal == b.causal
            and float(softmax_scale) == b.softmax_scale and current_mode == b.k1_mode
            and q.device == b.device and k.device == b.device and v.device == b.device
            and q.dtype == b.dtype and k.dtype == b.dtype and v.dtype == b.dtype
        )
        if not valid:
            raise ValueError(
                "KV-centric graph workspace bucket mismatch; create/capture a new workspace"
            )

    @property
    def graph_bucket_key(self) -> tuple:
        return self.bucket.cache_key


def lightning_moba_cuda_kv_centric_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    col_offsets: torch.Tensor,
    col_nnz: torch.Tensor,
    indices: torch.Tensor,
    chunk_size: int,
    topk: Optional[int] = None,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    q2col_offsets: Optional[torch.Tensor] = None,
    q2col_valid: Optional[torch.Tensor] = None,
    use_cuda_combine: bool = True,
    graph_workspace: Optional[KvCentricGraphWorkspace] = None,
    *,
    backend: str = "legacy",
    q_offsets: Optional[torch.Tensor] = None,
    q_csc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Consume CSC metadata with an explicit execution backend.

    ``backend='sm90'`` accepts variable-cardinality compact CSC and Q-CSR
    produced by the relative selector's fused builder. It directly executes
    the SM90 pipeline and variable-NNZ combine; do not provide a Top-K budget.
    Metadata must come from the corresponding builder for these Q/K inputs.

    ``backend='legacy'`` preserves the Top-K wrapper, including its automatic
    streaming dispatch. The two backends never silently fall back to each other.
    """
    import lightning_moba_cuda as lightning_moba_gpu

    if backend not in ("legacy", "sm90"):
        raise ValueError("backend must be 'legacy' or 'sm90'")
    if backend == "sm90":
        if topk is not None or q2col_offsets is not None or q2col_valid is not None or graph_workspace is not None:
            raise ValueError("SM90 dynamic attention uses variable Q-CSR, not Top-K capacity or a legacy graph workspace")
        if chunk_size != 128 or causal is not True or not use_cuda_combine:
            raise ValueError("SM90 dynamic attention requires chunk_size=128, causal=True and CUDA combine")
        if q_offsets is None or q_csc is None:
            raise ValueError("SM90 dynamic attention requires q_offsets and q_csc from the fused CSC/Q-CSR builder")
        if any(t.ndim != 3 for t in (q, k, v)) or k.shape != v.shape:
            raise ValueError("expected packed rank-3 Q/K/V with matching K/V")
        if any(t.requires_grad for t in (q, k, v)):
            raise ValueError("SM90 dynamic attention is inference-only")
        tensors = (q, k, v, cu_seqlens_q, cu_seqlens_k, col_offsets, col_nnz, indices, q_offsets, q_csc)
        if any(not t.is_cuda or t.device != q.device for t in tensors):
            raise ValueError("all inputs must share one CUDA device")
        if any(t.dtype != torch.bfloat16 or t.shape[2] != 128 for t in (q, k, v)):
            raise ValueError("SM90 dynamic attention requires BF16 and head_dim=128")
        if q.shape[1] < 1 or k.shape[1] < 1 or q.shape[1] % k.shape[1]:
            raise ValueError("KV heads must divide query heads")
        for tensor, dtype, ndim in (
            (cu_seqlens_q, torch.int32, 1), (cu_seqlens_k, torch.int32, 1),
            (col_offsets, torch.int64, 3), (col_nnz, torch.int32, 3),
            (indices, torch.int32, 1), (q_offsets, torch.int64, 1), (q_csc, torch.int32, 1),
        ):
            if tensor.dtype != dtype or tensor.ndim != ndim or not tensor.is_contiguous():
                raise ValueError("invalid CSC/Q-CSR metadata dtype, rank or contiguity")
        if (col_offsets.shape != col_nnz.shape or cu_seqlens_q.numel() != cu_seqlens_k.numel()
                or col_nnz.shape[:2] != (cu_seqlens_q.numel() - 1, q.shape[1])
                or q_offsets.numel() != q.shape[0] * q.shape[1] + 1
                or q_csc.numel() != indices.numel()):
            raise ValueError("CSC/Q-CSR dimensions do not match Q and batch metadata")
        with torch.cuda.device(q.device):
            if torch.cuda.get_device_capability(q.device) != (9, 0):
                raise ValueError("backend='sm90' requires SM90; no legacy fallback is performed")
            for name in ("moba_sm90_phase4b_csc_pipe", "moba_combine_q_csr"):
                if not hasattr(lightning_moba_gpu, name):
                    raise RuntimeError(f"rebuild lightning_moba_cuda: missing {name}")
            scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
            q, k, v = (tensor.contiguous() for tensor in (q, k, v))
            partial, lse = lightning_moba_gpu.moba_sm90_phase4b_csc_pipe(
                q, k, v, col_offsets, col_nnz,
                indices, cu_seqlens_q, cu_seqlens_k, True, scale,
            )
            return lightning_moba_gpu.moba_combine_q_csr(
                partial, lse, q_offsets, q_csc, torch.empty_like(q),
            )

    if q_offsets is not None or q_csc is not None:
        raise ValueError("variable Q-CSR metadata requires backend='sm90'")
    if type(topk) is not int or topk <= 0:
        raise ValueError("the legacy backend requires an explicit positive Top-K capacity")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    total_q, num_heads, head_dim = q.shape
    from lightning_moba.streaming_kv_centric import (
        lightning_moba_streaming_kv_centric_func, streaming_chunk_edges,
    )
    stream_edges = streaming_chunk_edges()
    if (stream_edges and graph_workspace is None and max_seqlen_q > 1
            and use_cuda_combine and q2col_offsets is None and q2col_valid is None
            and not torch.cuda.is_current_stream_capturing()):
        out, _ = lightning_moba_streaming_kv_centric_func(
            q, k, v, col_offsets, col_nnz, indices, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale, causal, chunk_size,
            chunk_edges=stream_edges,
        )
        return out

    if graph_workspace is not None:
        graph_workspace.validate(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, col_offsets, col_nnz, indices,
            chunk_size, topk, causal, softmax_scale,
        )
        if not use_cuda_combine:
            raise ValueError("graph_workspace requires the CUDA combine path")
        q2col_offsets, q2col_valid = graph_workspace.q2col_offsets, graph_workspace.q2col_valid

    o_partial, lse_partial = lightning_moba_gpu.moba_varlen_fwd_kvcentric(
        q, k, v,
        col_offsets, col_nnz, indices,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale, causal, chunk_size,
    )

    if graph_workspace is not None:
        if not hasattr(lightning_moba_gpu, "moba_build_q2col_kvcentric_into"):
            raise RuntimeError("lightning_moba_cuda lacks moba_build_q2col_kvcentric_into; rebuild the extension")
        lightning_moba_gpu.moba_build_q2col_kvcentric_into(
            col_offsets, col_nnz, indices, cu_seqlens_q,
            total_q, num_heads, topk, q2col_offsets, q2col_valid,
        )
    elif q2col_offsets is None or q2col_valid is None:
        if hasattr(lightning_moba_gpu, "moba_build_q2col_kvcentric"):
            q2col_offsets, q2col_valid = lightning_moba_gpu.moba_build_q2col_kvcentric(
                col_offsets, col_nnz, indices,
                cu_seqlens_q,
                total_q, num_heads, topk,
            )
        elif col_offsets.shape[0] == 1 and int(cu_seqlens_q[0].item()) == 0:
            q2col_offsets, q2col_valid = _build_q2col_reverse_index_fast(
                col_offsets[0], col_nnz[0], indices,
                max_seqlen_q, num_heads, topk,
            )
        else:
            q2col_offsets, q2col_valid = _build_q2col_reverse_index_varlen(
                col_offsets, col_nnz, indices, cu_seqlens_q,
                total_q, num_heads, topk,
            )

    if use_cuda_combine and hasattr(lightning_moba_gpu, "moba_combine_kvcentric"):
        out = graph_workspace.out if graph_workspace is not None else torch.empty(
            total_q, num_heads, head_dim, device=q.device, dtype=q.dtype
        )
        out = lightning_moba_gpu.moba_combine_kvcentric(
            o_partial, lse_partial,
            q2col_offsets, q2col_valid,
            out,
            topk,
        )
        return out

    out = torch.zeros(total_q, num_heads, head_dim, device=q.device, dtype=q.dtype)
    block_d = triton.next_power_of_2(head_dim)
    _triton_combine_kernel[(total_q * num_heads,)](
        o_partial, lse_partial,
        q2col_offsets, q2col_valid,
        out,
        out.stride(0), out.stride(1), out.stride(2),
        head_dim, 1,
        q2col_offsets.stride(0), q2col_offsets.stride(1),
        total_q, num_heads, head_dim,
        topk, block_d,
    )
    return out


# =============================================================================
# Python entry point
# =============================================================================

def lightning_moba_kv_centric_func(
    q: torch.Tensor,              # [total_q, num_heads, head_dim]
    k: torch.Tensor,              # [total_k, num_heads_kv, head_dim]
    v: torch.Tensor,              # [total_k, num_heads_kv, head_dim]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    col_offsets: torch.Tensor,    # [batch, num_heads, num_columns] int64
    col_nnz: torch.Tensor,        # [batch, num_heads, num_columns] int32
    indices: torch.Tensor,        # [nnz_total] int32
    chunk_size: int,
    topk: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    use_triton_combine: bool = True,
    # Pre-computed reverse index (skip expensive Python build if provided)
    q2col_offsets: Optional[torch.Tensor] = None,
    q2col_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KV-centric LightningMoBA attention using Triton kernels.

    Step 1: KV-centric attention kernel (one CTA per column per head)
    Step 2: Combine kernel (merge partial outputs)

    This is the v2 design using CSC-offset indexing (no atomics).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    batch_size = cu_seqlens_q.shape[0] - 1
    total_q, num_heads, head_dim = q.shape
    num_heads_kv = k.shape[1]
    h_h_k_ratio = num_heads // num_heads_kv
    num_columns = col_offsets.shape[2]

    # For now, assume batch=1 and flat layout
    assert batch_size == 1, "KV-centric Triton kernel currently supports batch=1"
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k

    # Total nnz = indices.shape[0]
    total_nnz = indices.shape[0]

    # Allocate partial output buffers (CSC-offset indexed)
    o_partial = torch.empty(total_nnz, head_dim, device=q.device, dtype=torch.float32)
    lse_partial = torch.empty((total_nnz,), device=q.device, dtype=torch.float32)

    # --- Tile sizes ---
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    # --- Flatten col_offsets and col_nnz for batch=1 ---
    col_offsets_flat = col_offsets[0]  # [num_heads, num_columns]
    col_nnz_flat = col_nnz[0]         # [num_heads, num_columns]

    # --- Launch attention kernel ---
    # Grid: (num_columns, batch * num_heads)
    grid_attn = (num_columns, num_heads)

    _kv_centric_attn_fwd_v2[grid_attn](
        q, k, v,
        col_offsets_flat, col_nnz_flat, indices,
        o_partial, lse_partial,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2),
        # K strides
        k.stride(0), k.stride(1), k.stride(2),
        # V strides
        v.stride(0), v.stride(1), v.stride(2),
        # O_partial strides (flat [nnz, head_dim])
        head_dim, 1,
        # Dimensions
        seqlen_q, seqlen_k, BLOCK_D,
        num_heads, num_heads_kv,
        chunk_size,
        softmax_scale,
        causal,
        BLOCK_M, BLOCK_N, BLOCK_D,
        h_h_k_ratio,
    )

    # --- Combine partial outputs ---
    if use_triton_combine:
        # Build reverse index (skip if pre-computed)
        if q2col_offsets is None or q2col_valid is None:
            q2col_offsets, q2col_valid = _build_q2col_reverse_index_fast(
                col_offsets_flat, col_nnz_flat, indices,
                seqlen_q, num_heads, topk,
            )

        # Triton combine returns without writing when n_valid == 0.
        out = torch.zeros(total_q, num_heads, head_dim, device=q.device, dtype=q.dtype)

        # Launch Triton combine
        grid_combine = (total_q * num_heads,)
        _triton_combine_kernel[grid_combine](
            o_partial, lse_partial,
            q2col_offsets, q2col_valid,
            out,
            # O_out strides
            out.stride(0), out.stride(1), out.stride(2),
            # O_partial strides
            head_dim, 1,
            # Q2col strides
            q2col_offsets.stride(0), q2col_offsets.stride(1),
            # Dimensions
            total_q, num_heads, BLOCK_D,
            topk, BLOCK_D,
        )
    else:
        # Python combine (slower but simpler for debugging)
        out, _ = _combine_partial_outputs(
            o_partial, lse_partial,
            col_offsets_flat, col_nnz_flat, indices,
            seqlen_q, num_heads, head_dim,
            q.dtype,
        )

    return out


# =============================================================================
# Validation
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from lightning_moba import flash_topk_varlen_func, lightning_moba_attn_varlen_func
    from lightning_moba.lightning_moba_interface import decide_lg_block_m

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    configs = [
        # (seqlen, num_heads, num_heads_kv, head_dim, topk, chunk_size)
        (2048, 8, 2, 128, 4, 128),
        (4096, 8, 2, 128, 4, 128),
        (2048, 8, 8, 128, 4, 128),   # MHA (no GQA)
        (8192, 8, 2, 128, 8, 128),
        (2048, 32, 4, 128, 4, 128),  # large GQA ratio
    ]

    print("=" * 80)
    print("KV-Centric LightningMoBA Triton Verification")
    print("=" * 80)

    all_passed = True
    for seqlen, num_heads, num_heads_kv, head_dim, topk, chunk_size in configs:
        batch_size = 1
        total_q = batch_size * seqlen
        q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_q, num_heads_kv, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_q, num_heads_kv, head_dim, device=device, dtype=dtype)
        cu_seqlens_q = torch.arange(0, total_q + 1, seqlen, device=device, dtype=torch.int32)
        cu_seqlens_k = cu_seqlens_q.clone()

        # TopK selection
        col_offsets, col_nnz, indices_out = flash_topk_varlen_func(
            q, k, cu_seqlens_q, cu_seqlens_k, seqlen, seqlen,
            moba_topk=topk, moba_chunk_size=chunk_size, causal=True,
        )

        # Reference: original LightningMoBA kernel
        lg_block_m = decide_lg_block_m(topk, chunk_size, seqlen, True, max_seqlen_q=seqlen)
        out_ref = lightning_moba_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen, seqlen,
            col_offsets, col_nnz, indices_out, lg_block_m, chunk_size,
            dropout_p=0.0, causal=True,
        )

        # KV-centric Triton (with Triton combine)
        out_triton = lightning_moba_kv_centric_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen, seqlen,
            col_offsets, col_nnz, indices_out, chunk_size, topk,
            causal=True, use_triton_combine=True,
        )

        # Also test Python combine
        out_triton_py = lightning_moba_kv_centric_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen, seqlen,
            col_offsets, col_nnz, indices_out, chunk_size, topk,
            causal=True, use_triton_combine=False,
        )

        # Compare
        max_diff_triton = (out_ref.float() - out_triton.float()).abs().max().item()
        mean_diff_triton = (out_ref.float() - out_triton.float()).abs().mean().item()
        cos_sim_triton = torch.nn.functional.cosine_similarity(
            out_ref.float().reshape(-1, head_dim),
            out_triton.float().reshape(-1, head_dim),
        ).mean().item()

        max_diff_py = (out_ref.float() - out_triton_py.float()).abs().max().item()

        passed = max_diff_triton < 0.05 and cos_sim_triton > 0.999
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False

        print(f"\nConfig: seq={seqlen}, heads={num_heads}/{num_heads_kv}, "
              f"topk={topk}, chunk={chunk_size}, GQA_ratio={num_heads // num_heads_kv}")
        print(f"  Triton combine:  max_diff={max_diff_triton:.6f}, "
              f"mean_diff={mean_diff_triton:.6f}, cos_sim={cos_sim_triton:.6f}")
        print(f"  Python combine:  max_diff={max_diff_py:.6f}")
        print(f"  [{status}]")

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 80)
