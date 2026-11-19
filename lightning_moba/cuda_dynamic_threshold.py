"""CUDA/CuTe dynamic-threshold sparse-pattern builder.

The phase-1 CUDA path intentionally mirrors the Triton reference contract and
keeps the fixed-capacity workspace. It targets BF16, head_dim=128 and causal
prefill; the Triton implementation remains the correctness oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from lightning_moba.triton_dynamic_threshold import DynamicThresholdPattern


@dataclass
class CompactDynamicThresholdPattern:
    """Capacity-free selector output consumable by KV-centric attention."""

    col_offsets: torch.Tensor
    col_nnz: torch.Tensor
    csc_indices: torch.Tensor
    row_bitmask: torch.Tensor
    row_nnz: torch.Tensor
    row_overflow: torch.Tensor
    # Present for the max-relative selector.  Keeping it attached to the
    # pattern lets memory-cap probes reuse the exact same first QK pass.
    row_max: torch.Tensor | None = None

    @property
    def total_nnz(self) -> torch.Tensor:
        return self.row_nnz.sum()

    @property
    def total_overflow(self) -> torch.Tensor:
        return self.row_overflow.sum()


@dataclass
class FlashPrefillDynamicPattern:
    """Block-adaptive FlashPrefill pattern expanded to LightningMoBA token CSC."""

    col_offsets: torch.Tensor
    col_nnz: torch.Tensor
    csc_indices: torch.Tensor
    row_nnz: torch.Tensor
    block_mask: torch.Tensor
    threshold: torch.Tensor
    scores: torch.Tensor

    @property
    def total_nnz(self) -> torch.Tensor:
        return self.row_nnz.sum(dtype=torch.int64)


def build_compact_kvcentric_msplit_worklist(
    col_nnz: torch.Tensor, rows_per_work: int = 64
) -> torch.Tensor:
    """Build live-only records for the CUDA KV-centric M-split kernel.

    The extension's diagnostic builder pads every CSC column to the global
    maximum split count.  A causal sink column can span the full sequence, so
    that representation becomes mostly empty at long context.  This builder
    emits exactly ``ceil(col_nnz / 64)`` records per live column.
    """
    if col_nnz.ndim != 3 or col_nnz.dtype != torch.int32 or not col_nnz.is_cuda:
        raise ValueError("col_nnz must be a CUDA int32 [batch, heads, columns] tensor")
    if rows_per_work != 64:
        raise ValueError("the current KV-centric M-split kernel requires 64 rows/work")
    counts = col_nnz.reshape(-1)
    split_counts = torch.div(
        counts + rows_per_work - 1, rows_per_work, rounding_mode="floor"
    )
    column_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=torch.int64),
        split_counts,
    )
    starts = torch.cumsum(split_counts, dim=0) - split_counts
    record_starts = torch.repeat_interleave(starts, split_counts)
    split_ids = (
        torch.arange(column_ids.numel(), device=counts.device, dtype=torch.int64)
        - record_starts
    )
    q_begin = split_ids * rows_per_work
    q_count = torch.minimum(
        counts[column_ids].to(torch.int64) - q_begin,
        torch.full_like(q_begin, rows_per_work),
    )
    max_columns = col_nnz.shape[-1]
    num_heads = col_nnz.shape[-2]
    column = column_ids.remainder(max_columns)
    head = torch.div(column_ids, max_columns, rounding_mode="floor").remainder(
        num_heads
    )
    batch = torch.div(
        column_ids, max_columns * num_heads, rounding_mode="floor"
    )
    return torch.stack((batch, head, column, q_begin, q_count), dim=1).to(
        torch.int32
    ).contiguous()


def _run_kvcentric_k1(
    lightning_moba_cuda,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pattern,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    chunk_size: int,
    kvcentric_msplit: bool,
):
    if not kvcentric_msplit:
        return lightning_moba_cuda.moba_varlen_fwd_kvcentric(
            q, k, v, pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            softmax_scale, causal, chunk_size,
        )
    if not hasattr(lightning_moba_cuda, "moba_varlen_fwd_kvcentric_worklist"):
        raise RuntimeError(
            "KV-centric M-split is enabled but lightning_moba_cuda lacks the worklist ABI"
        )
    worklist = build_compact_kvcentric_msplit_worklist(pattern.col_nnz)
    return lightning_moba_cuda.moba_varlen_fwd_kvcentric_worklist(
        q, k, v, pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
        worklist, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale, causal, chunk_size,
    )


def flash_cuda_flashprefill_coarse_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    fine_block: int = 32,
    exec_block: int = 128,
    num_splits: int = 8,
) -> torch.Tensor:
    """Compute fused meanmax scores directly on LightningMoBA's coarse grid.

    The returned layout is ``[q_blocks, q_heads, kv_blocks]``.  Unlike the
    historical Triton path this does not expose the quadratic fine-grid score
    tensor; the 4x4 fine-to-coarse max reduction happens inside the CUDA CTA.
    """
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("native FlashPrefill scorer supports BF16 only")
    if q.ndim != 3 or k.ndim != 3 or q.shape[2] != 128 or k.shape[2] != 128:
        raise ValueError("native FlashPrefill scorer requires rank-3 D128 q/k")
    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k token counts must match")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    scores, _, _ = lightning_moba_dynamic_cuda.moba_flashprefill_coarse_scores(
        q.contiguous(), k.contiguous(), float(softmax_scale),
        fine_block, exec_block, num_splits,
    )
    return scores


def flash_cuda_flashprefill_dynamic_pattern(
    q: torch.Tensor,
    k: torch.Tensor,
    density: float,
    *,
    alpha: float = 0.1,
    threshold_mode: str = "abs_max",
    softmax_scale: float | None = None,
    chunk_size: int = 128,
    fine_block: int = 32,
    num_splits: int = 8,
    sink: int = 1,
    window: int = 2,
    build_csc: bool = True,
) -> FlashPrefillDynamicPattern:
    """Fused meanmax score, selectable max-relative threshold, and compact CSC.

    Routing is decided once per ``chunk_size`` Q block and expanded to token
    rows for the existing KV-centric attention ABI.  The returned CSC rows are
    sorted within each column, so no ``varlen_sort`` is required.
    """
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("native FlashPrefill pattern supports BF16 only")
    if q.ndim != 3 or k.ndim != 3 or q.shape[2] != 128 or k.shape[2] != 128:
        raise ValueError("native FlashPrefill pattern requires rank-3 D128 q/k")
    if q.shape[0] != k.shape[0] or q.shape[0] % chunk_size:
        raise ValueError("q/k lengths must match and be chunk aligned")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if threshold_mode not in {"abs_max", "auto_max", "exact_topk_budget"}:
        raise ValueError(
            "threshold_mode must be abs_max, auto_max, or exact_topk_budget"
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    args = (
        q.contiguous(), k.contiguous(), float(density), float(alpha),
        threshold_mode in {"auto_max", "exact_topk_budget"},
        float(softmax_scale), chunk_size, fine_block, num_splits,
        sink, window, build_csc,
    )
    exact_density = threshold_mode == "exact_topk_budget"
    try:
        outputs = lightning_moba_dynamic_cuda.moba_flashprefill_dynamic_pattern(
            *args, exact_density
        )
    except TypeError:
        if exact_density:
            raise RuntimeError(
                "exact_topk_budget requires rebuilding lightning_moba_dynamic_cuda"
            ) from None
        # Preserve compatibility with an extension built before the optional
        # exact-density ABI argument was added.
        outputs = lightning_moba_dynamic_cuda.moba_flashprefill_dynamic_pattern(*args)
    return FlashPrefillDynamicPattern(*outputs)


def flash_cuda_flashprefill_kv_centric_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    density: float,
    *,
    alpha: float = 0.1,
    threshold_mode: str = "abs_max",
    chunk_size: int = 128,
    fine_block: int = 32,
    num_splits: int = 8,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
    softmax_scale: float | None = None,
    kvcentric_msplit: bool = True,
) -> torch.Tensor:
    """Native FlashPrefill selector -> compact CSC -> K1 -> Q-CSR combine."""
    import lightning_moba_cuda
    import lightning_moba_dynamic_cuda

    if not causal:
        raise NotImplementedError("native FlashPrefill integration is causal-only")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    pattern = flash_cuda_flashprefill_dynamic_pattern(
        q, k, density, alpha=alpha, threshold_mode=threshold_mode,
        softmax_scale=softmax_scale,
        chunk_size=chunk_size, fine_block=fine_block,
        num_splits=num_splits, sink=sink, window=window, build_csc=True,
    )
    o_partial, lse_partial = _run_kvcentric_k1(
        lightning_moba_cuda, q, k, v, pattern, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, True, chunk_size,
        kvcentric_msplit,
    )
    q_offsets, q_csc_indices = lightning_moba_dynamic_cuda.moba_build_q_csr_from_csc(
        pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
        cu_seqlens_q, pattern.row_nnz,
    )
    return lightning_moba_dynamic_cuda.moba_combine_q_csr(
        o_partial, lse_partial, q_offsets, q_csc_indices, torch.empty_like(q)
    )


def _normalize_threshold(q: torch.Tensor, threshold: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(threshold, torch.Tensor):
        if threshold.numel() == 1:
            return threshold.to(device=q.device, dtype=torch.float32).expand(q.shape[1]).contiguous()
        if threshold.numel() == q.shape[1]:
            return threshold.to(device=q.device, dtype=torch.float32).reshape(q.shape[1]).contiguous()
        raise ValueError("threshold must be scalar or contain one value per Q head")
    return torch.full((q.shape[1],), float(threshold), device=q.device, dtype=torch.float32)


def flash_cuda_dynamic_threshold_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    threshold: Union[float, torch.Tensor],
    chunk_size: int,
    *,
    capacity: int = 256,
    causal: bool = True,
    build_csc: bool = True,
) -> DynamicThresholdPattern:
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or km.dtype != torch.bfloat16:
        raise TypeError("CUDA phase-1 supports BF16 only")
    if q.ndim != 3 or km.ndim != 3 or q.shape[2] != 128 or km.shape[2] != 128:
        raise ValueError("CUDA phase-1 requires q/km rank 3 and head_dim=128")
    thresholds = _normalize_threshold(q, threshold)

    outputs = lightning_moba_dynamic_cuda.moba_dynamic_threshold(
        q.contiguous(),
        km.contiguous(),
        thresholds,
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(),
        max_seqlen_q,
        max_seqlen_k,
        chunk_size,
        capacity,
        causal,
        build_csc,
    )
    return DynamicThresholdPattern(*outputs)


def flash_cuda_dynamic_threshold_compact_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    threshold: Union[float, torch.Tensor],
    chunk_size: int,
    *,
    causal: bool = True,
    build_csc: bool = True,
) -> CompactDynamicThresholdPattern:
    """Build exact-sized CSC without a per-row selection capacity.

    The row bitmask represents all key blocks and therefore supports every
    attainable density (the project's validation range is 2% through 30%).
    """
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or km.dtype != torch.bfloat16:
        raise TypeError("CUDA compact path supports BF16 only")
    if q.ndim != 3 or km.ndim != 3 or q.shape[2] != 128 or km.shape[2] != 128:
        raise ValueError("CUDA compact path requires q/km rank 3 and head_dim=128")
    thresholds = _normalize_threshold(q, threshold)
    outputs = lightning_moba_dynamic_cuda.moba_dynamic_threshold_compact(
        q.contiguous(),
        km.contiguous(),
        thresholds,
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(),
        max_seqlen_q,
        max_seqlen_k,
        chunk_size,
        causal,
        build_csc,
    )
    return CompactDynamicThresholdPattern(*outputs)


def flash_cuda_dynamic_relative_threshold_compact_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    alpha: float,
    chunk_size: int,
    *,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
    min_blocks_per_row: int = 0,
    build_csc: bool = True,
) -> CompactDynamicThresholdPattern:
    """Select mean-pooled K blocks by a per-Q-row max-relative threshold.

    The CUDA kernel performs two streaming score passes and never materializes
    the quadratic score matrix.  It keeps a block when
    ``exp(scale * (score - row_max)) >= alpha``; the current causal block is
    always retained.  This preserves the score space used by fused TopK while
    replacing sorting with thresholding.
    """
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or km.dtype != torch.bfloat16:
        raise TypeError("CUDA relative selector supports BF16 only")
    if q.ndim != 3 or km.ndim != 3 or q.shape[2] != 128 or km.shape[2] != 128:
        raise ValueError("CUDA relative selector requires q/km rank 3 and head_dim=128")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0 <= min_blocks_per_row <= 8:
        raise ValueError("min_blocks_per_row must be in [0, 8]")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    outputs = lightning_moba_dynamic_cuda.moba_dynamic_relative_threshold_compact(
        q.contiguous(), km.contiguous(),
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(), max_seqlen_q, max_seqlen_k, chunk_size,
        float(alpha), float(softmax_scale), sink, window, causal,
        int(min_blocks_per_row), build_csc,
    )
    return CompactDynamicThresholdPattern(*outputs)


def cap_compact_pattern_per_row(
    pattern: CompactDynamicThresholdPattern,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    chunk_size: int,
    *,
    sink: int = 1,
    window: int = 2,
    max_blocks_per_row: int,
) -> torch.Tensor:
    """Trim each Q/head row before CSC, preserving anchors and recent selected blocks.

    The score-thresholded pattern is unchanged unless a row exceeds the cap.
    Returns device-side counts of removed edges and affected rows.
    """
    import lightning_moba_dynamic_cuda

    if max_blocks_per_row < sink + window or chunk_size <= 0:
        raise ValueError("max_blocks_per_row must cover sink + window")
    if pattern.csc_indices.numel():
        raise ValueError("cap must run before CSC materialization")
    if not hasattr(lightning_moba_dynamic_cuda, "moba_cap_compact_rows"):
        raise RuntimeError("rebuild the selected LightningMoBA extension with row-cap support")
    return lightning_moba_dynamic_cuda.moba_cap_compact_rows(
        pattern.row_bitmask, pattern.row_nnz, pattern.col_nnz,
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        chunk_size, sink, window, max_blocks_per_row,
    )


def flash_cuda_dynamic_relative_multi_probe_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    alphas: torch.Tensor,
    chunk_size: int,
    *,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
) -> torch.Tensor:
    """Return aggregate NNZ for up to 16 alphas using one CUDA QK scan."""
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or km.dtype != torch.bfloat16:
        raise TypeError("CUDA relative multi-probe supports BF16 only")
    if alphas.ndim != 1 or alphas.numel() not in {1, 4, 8, 16}:
        raise ValueError("alphas must contain 1, 4, 8, or 16 values")
    if alphas.device != q.device:
        raise ValueError("alphas must be on the same CUDA device as q")
    alphas = alphas.to(dtype=torch.float32).contiguous()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return lightning_moba_dynamic_cuda.moba_dynamic_relative_threshold_multi_probe(
        q.contiguous(), km.contiguous(), alphas,
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(), max_seqlen_q, max_seqlen_k, chunk_size,
        float(softmax_scale), sink, window, causal,
    )


def flash_cuda_dynamic_relative_probe_from_row_max_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    row_max: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    alphas: Union[float, torch.Tensor],
    chunk_size: int,
    *,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
) -> torch.Tensor:
    """Count edges at one or more alphas without recomputing row maxima."""
    import lightning_moba_dynamic_cuda

    if isinstance(alphas, torch.Tensor):
        probe_alphas = alphas.to(device=q.device, dtype=torch.float32).reshape(-1).contiguous()
    else:
        probe_alphas = torch.tensor([float(alphas)], device=q.device, dtype=torch.float32)
    if probe_alphas.numel() not in {1, 4, 8, 16}:
        raise ValueError("cached row-max probes require 1, 4, 8, or 16 alphas")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return lightning_moba_dynamic_cuda.moba_dynamic_relative_threshold_probe_from_row_max(
        q.contiguous(), km.contiguous(), row_max.contiguous(), probe_alphas,
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(), max_seqlen_q, max_seqlen_k, chunk_size,
        float(softmax_scale), sink, window, causal,
    )


def flash_cuda_dynamic_relative_interpolate_device_func(
    probe_counts: torch.Tensor,
    probe_alphas: torch.Tensor,
    target_count: float,
    *,
    interpolation_safety: float = 1.01,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose a capped alpha without synchronizing probe counts to the host.

    Returns device tensors ``(selected_alpha, relative_log_threshold,
    selected_index)``.  Keeping the log threshold resident on the CUDA stream
    allows the following materialization kernel to consume it directly.
    """
    import lightning_moba_dynamic_cuda

    if probe_counts.device != probe_alphas.device:
        raise ValueError("probe_counts and probe_alphas must share a device")
    counts = probe_counts.to(dtype=torch.int64).reshape(-1).contiguous()
    alphas = probe_alphas.to(dtype=torch.float32).reshape(-1).contiguous()
    if counts.numel() != alphas.numel() or counts.numel() not in {1, 4, 8, 16}:
        raise ValueError("device interpolation requires 1, 4, 8, or 16 matched probes")
    return tuple(
        lightning_moba_dynamic_cuda.moba_dynamic_relative_threshold_interpolate_device(
            counts, alphas, float(target_count), float(interpolation_safety),
            float(softmax_scale),
        )
    )


def flash_cuda_dynamic_relative_materialize_from_row_max_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    row_max: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    alpha: float,
    chunk_size: int,
    *,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
) -> CompactDynamicThresholdPattern:
    """Materialize one final pattern from an already computed row maximum."""
    import lightning_moba_dynamic_cuda

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    outputs = lightning_moba_dynamic_cuda.moba_dynamic_relative_threshold_materialize_from_row_max(
        q.contiguous(), km.contiguous(), row_max.contiguous(),
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(), max_seqlen_q, max_seqlen_k, chunk_size,
        float(alpha), float(softmax_scale), sink, window, causal,
    )
    return CompactDynamicThresholdPattern(*outputs, row_max=row_max)


def flash_cuda_dynamic_relative_materialize_from_row_max_device_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    row_max: torch.Tensor,
    relative_log_threshold: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    chunk_size: int,
    *,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
) -> CompactDynamicThresholdPattern:
    """Materialize from a device scalar produced by cap interpolation."""
    import lightning_moba_dynamic_cuda

    threshold = relative_log_threshold.to(
        device=q.device, dtype=torch.float32
    ).reshape(-1).contiguous()
    if threshold.numel() != 1:
        raise ValueError("relative_log_threshold must contain one value")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    outputs = lightning_moba_dynamic_cuda.moba_dynamic_relative_threshold_materialize_from_row_max_device(
        q.contiguous(), km.contiguous(), row_max.contiguous(), threshold,
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(), max_seqlen_q, max_seqlen_k, chunk_size,
        float(softmax_scale), sink, window, causal,
    )
    return CompactDynamicThresholdPattern(*outputs, row_max=row_max)


def flash_cuda_dynamic_relative_refine_in_place_device_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    pattern: CompactDynamicThresholdPattern,
    probe_alphas: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    chunk_size: int,
    target_count: float,
    *,
    interpolation_safety: float = 1.01,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
) -> tuple[CompactDynamicThresholdPattern, tuple[torch.Tensor, ...]]:
    """Conditionally cap ``pattern`` in place with no device-to-host scalar.

    The CUDA extension checks the existing aggregate NNZ. Safe patterns are
    left untouched; only over-cap patterns execute the QK probe and rewrite.
    """
    import lightning_moba_dynamic_cuda

    if pattern.row_max is None:
        raise ValueError("in-place device refinement requires cached row_max")
    alphas = probe_alphas.to(
        device=q.device, dtype=torch.float32
    ).reshape(-1).contiguous()
    if alphas.numel() not in {1, 4, 8, 16}:
        raise ValueError("device refinement requires 1, 4, 8, or 16 probes")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    metadata = tuple(
        lightning_moba_dynamic_cuda.moba_dynamic_relative_threshold_refine_in_place_device(
            q.contiguous(), km.contiguous(), pattern.row_max.contiguous(),
            pattern.row_bitmask, pattern.row_nnz, pattern.row_overflow,
            pattern.col_offsets, pattern.col_nnz, alphas,
            cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
            cu_seqlens_km.contiguous(), max_seqlen_q, max_seqlen_k,
            chunk_size, float(target_count), float(interpolation_safety),
            float(softmax_scale), sink, window, causal,
        )
    )
    return pattern, metadata


def flash_cuda_dynamic_relative_head_threshold_compact_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    alphas: torch.Tensor,
    chunk_size: int,
    *,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
    build_csc: bool = True,
) -> CompactDynamicThresholdPattern:
    """Max-relative selection with one aggregate-budget control per Q head.

    ``alphas[h]`` changes the number of blocks assigned to head ``h`` while
    every Q row in that head still selects a data-dependent number of blocks.
    This is intended for an offline-calibrated alpha table; it is not per-row
    top-k and therefore retains threshold selection's sort-free hot path.
    """
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or km.dtype != torch.bfloat16:
        raise TypeError("CUDA relative selector supports BF16 only")
    if q.ndim != 3 or km.ndim != 3 or q.shape[2] != 128 or km.shape[2] != 128:
        raise ValueError("CUDA relative selector requires q/km rank 3 and head_dim=128")
    if alphas.ndim != 1 or alphas.numel() != q.shape[1]:
        raise ValueError("alphas must contain one value per local Q head")
    if alphas.device != q.device:
        raise ValueError("alphas must be on the same CUDA device as q")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    alphas = alphas.to(dtype=torch.float32).contiguous()
    if bool(torch.any((alphas <= 0) | (alphas > 1)).item()):
        raise ValueError("every alpha must be in (0, 1]")
    relative_offsets = alphas.log().div(float(softmax_scale)).contiguous()
    return flash_cuda_dynamic_relative_head_offsets_compact_varlen_func(
        q, km, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km,
        max_seqlen_q, max_seqlen_k, relative_offsets, chunk_size,
        sink=sink, window=window, causal=causal, build_csc=build_csc,
    )


def flash_cuda_dynamic_relative_head_offsets_compact_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    relative_offsets: torch.Tensor,
    chunk_size: int,
    *,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
    build_csc: bool = True,
) -> CompactDynamicThresholdPattern:
    """Hot-path form using cached ``log(alpha_h) / softmax_scale`` offsets."""
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or km.dtype != torch.bfloat16:
        raise TypeError("CUDA relative selector supports BF16 only")
    if relative_offsets.shape != (q.shape[1],):
        raise ValueError("relative_offsets must contain one value per local Q head")
    if relative_offsets.device != q.device or relative_offsets.dtype != torch.float32:
        raise ValueError("relative_offsets must be CUDA float32 on q.device")
    if not hasattr(
        lightning_moba_dynamic_cuda, "moba_dynamic_relative_head_threshold_compact"
    ):
        raise RuntimeError(
            "lightning_moba_dynamic_cuda lacks per-head relative thresholds; rebuild the extension"
        )
    outputs = lightning_moba_dynamic_cuda.moba_dynamic_relative_head_threshold_compact(
        q.contiguous(), km.contiguous(), relative_offsets.contiguous(),
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(), max_seqlen_q, max_seqlen_k, chunk_size,
        sink, window, causal, build_csc,
    )
    return CompactDynamicThresholdPattern(*outputs)


def flash_cuda_dynamic_relative_threshold_padded_varlen_func(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    alpha: float,
    chunk_size: int,
    *,
    softmax_scale: float | None = None,
    sink: int = 1,
    window: int = 2,
    causal: bool = True,
) -> CompactDynamicThresholdPattern:
    """Graph-safe relative selector with fixed-capacity CSC storage.

    The logical pattern is identical to the compact variant.  Only the
    physical ``csc_indices`` allocation differs: it is padded to the maximum
    possible number of entries so no device-to-host NNZ read is needed during
    CUDA graph capture.  Downstream kernels consume the valid prefixes encoded
    by ``col_offsets`` and ``col_nnz``.
    """
    import lightning_moba_dynamic_cuda

    if q.dtype != torch.bfloat16 or km.dtype != torch.bfloat16:
        raise TypeError("CUDA relative selector supports BF16 only")
    if q.ndim != 3 or km.ndim != 3 or q.shape[2] != 128 or km.shape[2] != 128:
        raise ValueError("CUDA relative selector requires q/km rank 3 and head_dim=128")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if not hasattr(
        lightning_moba_dynamic_cuda, "moba_dynamic_relative_threshold_padded"
    ):
        raise RuntimeError(
            "lightning_moba_dynamic_cuda lacks the padded relative selector; "
            "rebuild the extension"
        )
    outputs = lightning_moba_dynamic_cuda.moba_dynamic_relative_threshold_padded(
        q.contiguous(), km.contiguous(),
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        cu_seqlens_km.contiguous(), max_seqlen_q, max_seqlen_k, chunk_size,
        float(alpha), float(softmax_scale), sink, window, causal,
    )
    return CompactDynamicThresholdPattern(*outputs)


def flash_cuda_compact_csc_from_bitmask(
    pattern: CompactDynamicThresholdPattern,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    chunk_size: int,
) -> CompactDynamicThresholdPattern:
    """Attach exact-sized CSC storage to a selection-only compact pattern."""
    import lightning_moba_dynamic_cuda

    col_offsets, csc_indices = lightning_moba_dynamic_cuda.moba_compact_csc_from_bitmask(
        pattern.row_bitmask, pattern.row_nnz, pattern.col_nnz,
        cu_seqlens_q.contiguous(), cu_seqlens_km.contiguous(),
        max_seqlen_q, max_seqlen_k, chunk_size,
    )
    pattern.col_offsets = col_offsets
    pattern.csc_indices = csc_indices
    return pattern


def flash_cuda_compact_csc_qcsr_from_bitmask(
    pattern: CompactDynamicThresholdPattern,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    chunk_size: int,
) -> tuple[CompactDynamicThresholdPattern, torch.Tensor, torch.Tensor]:
    """Attach CSC and reverse Q-CSR with one fused row-centric scatter."""
    import lightning_moba_dynamic_cuda

    if not hasattr(
        lightning_moba_dynamic_cuda, "moba_compact_csc_qcsr_from_bitmask"
    ):
        raise RuntimeError(
            "lightning_moba_dynamic_cuda lacks fused CSC/Q-CSR; rebuild the extension"
        )
    col_offsets, csc_indices, q_offsets, q_csc_indices = (
        lightning_moba_dynamic_cuda.moba_compact_csc_qcsr_from_bitmask(
            pattern.row_bitmask, pattern.row_nnz, pattern.col_nnz,
            cu_seqlens_q.contiguous(), cu_seqlens_km.contiguous(),
            max_seqlen_q, max_seqlen_k, chunk_size,
        )
    )
    pattern.col_offsets = col_offsets
    pattern.csc_indices = csc_indices
    return pattern, q_offsets, q_csc_indices


def flash_cuda_dynamic_kv_centric_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    threshold: Union[float, torch.Tensor],
    chunk_size: int,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    kvcentric_msplit: bool = True,
) -> torch.Tensor:
    """Dynamic selector -> compact CSC -> KV-centric K1 -> variable combine."""
    import lightning_moba_cuda
    import lightning_moba_dynamic_cuda

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    pattern = flash_cuda_dynamic_threshold_compact_varlen_func(
        q, km, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km,
        max_seqlen_q, max_seqlen_k, threshold, chunk_size,
        causal=causal, build_csc=True,
    )
    from lightning_moba.streaming_kv_centric import (
        lightning_moba_streaming_kv_centric_func, streaming_chunk_edges,
    )
    stream_edges = streaming_chunk_edges()
    if stream_edges and max_seqlen_q > 1 and not torch.cuda.is_current_stream_capturing():
        out, _ = lightning_moba_streaming_kv_centric_func(
            q, k, v, pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            softmax_scale, causal, chunk_size, chunk_edges=stream_edges,
        )
        return out
    o_partial, lse_partial = _run_kvcentric_k1(
        lightning_moba_cuda, q, k, v, pattern, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, causal, chunk_size,
        kvcentric_msplit,
    )
    q_offsets, q_csc_indices = lightning_moba_dynamic_cuda.moba_build_q_csr_from_csc(
        pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
        cu_seqlens_q, pattern.row_nnz,
    )
    out = torch.empty_like(q)
    return lightning_moba_dynamic_cuda.moba_combine_q_csr(
        o_partial, lse_partial, q_offsets, q_csc_indices, out
    )
