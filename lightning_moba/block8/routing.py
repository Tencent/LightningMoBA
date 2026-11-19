"""Original scalar-alpha CUDA selector, regrouped for small-block attention."""
import math
from numbers import Real

from .selector import mean_pool_keys
from .csc import csc_pattern


def validate_alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('alpha must be one finite scalar in (0, 1], shared by all layers/heads')
    return float(alpha)


def cuda_pattern(q, k, alpha, *, sink=1, window=2, **kwargs):
    """Q/K are head-major; use the unmodified original CUDA selection rule."""
    import torch
    from lightning_moba.cuda_dynamic_threshold import flash_cuda_dynamic_relative_threshold_compact_varlen_func

    alpha = validate_alpha(alpha)
    sq, sk = q.shape[1], k.shape[1]
    km = mean_pool_keys(k, 8)
    cuq = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
    cuk = torch.tensor([0, sk], device=q.device, dtype=torch.int32)
    cum = torch.tensor([0, sk // 8], device=q.device, dtype=torch.int32)
    selected = flash_cuda_dynamic_relative_threshold_compact_varlen_func(
        q.transpose(0, 1), km.transpose(0, 1), cuq, cuk, cum, sq, sk,
        alpha, 8, sink=sink, window=window, min_blocks_per_row=0, build_csc=False,
    )
    selection = dict(keep=selected.row_bitmask.transpose(0, 1).contiguous(),
                     rowmax=selected.row_max.transpose(0, 1).contiguous() * q.shape[-1] ** -0.5)
    return csc_pattern(q, k, None, km=km, selection=selection, **kwargs)
