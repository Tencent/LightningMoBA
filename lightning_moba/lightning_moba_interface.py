from typing import Optional, Sequence, Tuple, Union
from numbers import Real
import math

import torch
import torch.nn as nn
import os

import lightning_moba_cuda as lightning_moba_gpu
from lightning_moba.triton_mean_pool import flash_topk_mean_pool
from lightning_moba._compat import supports_custom_ops

_TORCH_HAS_CUSTOM_OP = supports_custom_ops(torch.__version__)

##########################################################################################################################
# Helper functions
##########################################################################################################################

def round_multiple(x: int, m: int) -> int:
    """Round x up to the nearest multiple of m."""
    return ((x + m - 1) // m) * m

##########################################################################################################################

def decide_lg_block_m(
    top_k: int,
    chunk_size: int,
    seqlen: int,
    causal: bool = False,
    max_seqlen_q: int = None,
) -> int:
    """Heuristic to decide lg_block_m (per-CTA q-row count) for lightning_moba attn kernel.

    The original heuristic depends only on (top_k, chunk_size, seqlen, causal) via a
    sparsity bucket.  On H200 we found this leaves significant performance on the table:
    causal-sparse workloads have q-row work imbalance, large lg_block_m makes the
    heaviest q-row of a CTA dominate wave latency, and smaller lg_block_m unlocks
    20-74% kernel-time wins on prefill shapes (microbench evidence:
    bench_compare_v2/.../profile/lg_block_m_sweep_finding_h200.md).

    To stay opt-in / byte-identical with the legacy default path we:
      - keep the legacy signature (positional args unchanged); ``max_seqlen_q`` defaults
        to ``None`` which selects the legacy sparsity heuristic.
      - only switch to the new shape-aware heuristic when the env var
        ``LIGHTNING_MOBA_LG_BLOCK_M_SHAPE_AWARE=1`` is set AND ``max_seqlen_q`` is given.
    """
    # ------- legacy sparsity heuristic (unchanged) -------
    sparsity = 0.0
    budget = top_k * chunk_size
    if causal:
        density = (2*(budget * seqlen) - budget**2) / (seqlen**2)
    else:
        density = budget / seqlen

    sparsity = 1 - density

    if sparsity <= 0.5:
        lg_block_m = 128
    elif sparsity <= 0.7:
        lg_block_m = 256
    elif sparsity <= 0.8:
        lg_block_m = 512
    elif sparsity <= 0.9:
        lg_block_m = 768
    else:
        lg_block_m = 1024

    # [Optimization] Hardware-aware cap for A6000/3090/4090 to avoid Shared Memory OOM
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # sm86 (A6000, 3090) and sm89 (4090, L40) have smaller shared memory than A100 (sm80)
        if major == 8 and minor > 0:
            lg_block_m = min(lg_block_m, 512)

    # ------- opt-in shape-aware override (H200, microbench-derived) -------
    # Only fires when explicitly enabled AND we know the q dimension.
    if (
        max_seqlen_q is not None
        and os.environ.get("LIGHTNING_MOBA_LG_BLOCK_M_SHAPE_AWARE", "0") == "1"
    ):
        new_lg = _shape_aware_lg_block_m(
            top_k=top_k,
            chunk_size=chunk_size,
            seqlen_k=seqlen,
            seqlen_q=max_seqlen_q,
            causal=causal,
            legacy_value=lg_block_m,
        )
        # one-time debug print per process (for e2e verification)
        if not getattr(decide_lg_block_m, "_shape_aware_logged", False):
            try:
                import sys
                print(f"[LIGHTNING_MOBA] shape-aware lg_block_m enabled: "
                      f"q={max_seqlen_q} k={seqlen} topk={top_k} chunk={chunk_size} "
                      f"causal={causal} -> legacy={lg_block_m} new={new_lg}",
                      file=sys.stderr, flush=True)
                decide_lg_block_m._shape_aware_logged = True
            except Exception:
                pass
        lg_block_m = new_lg

    # Hopper-only experiment: larger logical M blocks improve sparse-row packing
    # for very long square prefill. The C++ API validates architecture and dtype.
    extended_lg_raw = os.environ.get("LIGHTNING_MOBA_FWD_HDIM128_CAUSAL_LG_BLOCK_M")
    try:
        extended_lg = int(extended_lg_raw) if extended_lg_raw is not None else 0
    except ValueError:
        extended_lg = 0
    if (
        1024 < extended_lg <= 4096
        and extended_lg % 64 == 0
        and causal
        and max_seqlen_q is not None
        and max_seqlen_q >= 65536
    ):
        lg_block_m = extended_lg
        if not getattr(decide_lg_block_m, "_extended_lg_logged", False):
            print(
                f"[LIGHTNING_MOBA] extended lg_block_m experiment enabled: "
                f"q={max_seqlen_q} k={seqlen} -> lg_block_m={lg_block_m}",
                flush=True,
            )
            decide_lg_block_m._extended_lg_logged = True

    return lg_block_m


def _shape_aware_lg_block_m(
    top_k: int,
    chunk_size: int,
    seqlen_k: int,
    seqlen_q: int,
    causal: bool,
    legacy_value: int,
) -> int:
    """H200 microbench-derived shape-aware heuristic for ``lg_block_m``.

    Evidence: ``profile/lg_block_m_sweep_finding_h200.md``
      - decode (q==1): lg_block_m has no effect; pick 128 (kernel lower bound).
      - chunked-prefill (q << k): smaller is better, lg_block_m=128 wins (+24~31%).
      - small full-prefill (q == k, q <= 8192): smallest wins (+74% at 8k).
      - large full-prefill (q == k, q >= 16384): mid-range wins (lg_block_m=512
        on 32k, +30.9% vs baseline 1024).

    Cutoffs are conservative; values outside the explored shape grid fall back
    to the legacy sparsity-derived value.
    """
    # decode / single q-row
    if seqlen_q <= 1:
        return 128

    # chunked-prefill: q significantly smaller than k -> per-CTA work imbalance
    # is dominated by the few "tail" rows; smallest CTA wins.
    if seqlen_k >= 4 * seqlen_q:
        return 128

    # small full-prefill (q ~ k, both small): wave count low, smallest CTA still wins.
    if seqlen_q <= 8192:
        return 128

    # large full-prefill (q ~ k, both large): too many CTAs would cause wave-tail
    # imbalance and per-CTA fixed overhead inflation; mid-range wins.
    if seqlen_q <= 32768:
        return 512

    # very large q (> 32k): we did not benchmark; fall back to legacy value to
    # avoid regression on un-explored shapes.
    return legacy_value

##########################################################################################################################

# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if _TORCH_HAS_CUSTOM_OP:
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


##########################################################################################################################
# Custom ops
##########################################################################################################################

@_torch_custom_op_wrapper("lightning_moba::_moba_fused_topk", mutates_args=(), device_types="cuda")
def _moba_fused_topk(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # MOBA sparse pattern parameters
    moba_topk: int,
    moba_chunk_size: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, km = [maybe_contiguous(x) for x in (q, km)]
    
    # The shared extension exposes the original ten-argument, five-output ABI.
    col_offsets, col_nnz, indices, topk_values, topk_indices = lightning_moba_gpu.moba_fused_topk(
        q, km, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km,
        max_seqlen_q, max_seqlen_k, moba_topk, moba_chunk_size, causal,
    )
    flat_bias = torch.zeros(indices.shape, device=indices.device, dtype=torch.float32)
    topk_bias = torch.zeros(topk_values.shape, device=topk_values.device, dtype=torch.float32)
    return col_offsets, col_nnz, indices, flat_bias, topk_values, topk_indices, topk_bias

@_torch_register_fake_wrapper("lightning_moba::_moba_fused_topk")
def _moba_fused_topk_fake(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_km: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # MOBA sparse pattern parameters
    moba_topk: int,
    moba_chunk_size: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, km = [maybe_contiguous(x) for x in (q, km)]
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape
    
    max_lg_col_num = (max_seqlen_k + moba_chunk_size - 1) // moba_chunk_size

    col_offsets = torch.empty((batch_size, num_heads, max_lg_col_num), device=q.device, dtype=torch.int64)
    col_nnz = torch.empty((batch_size, num_heads, max_lg_col_num), device=q.device, dtype=torch.int32)
    indices = torch.empty((total_q * num_heads * moba_topk), device=q.device, dtype=torch.int32)
    topk_rounded = round_multiple(moba_topk, 16)
    flat_bias = torch.empty((total_q * num_heads * moba_topk,), device=q.device, dtype=torch.float32)
    topk_values = torch.empty((total_q, num_heads, topk_rounded), device=q.device, dtype=torch.float32)
    topk_indices = torch.empty((total_q, num_heads, topk_rounded), device=q.device, dtype=torch.int32)
    topk_bias = torch.empty((total_q, num_heads, topk_rounded), device=q.device, dtype=torch.float32)

    return col_offsets, col_nnz, indices, flat_bias, topk_values, topk_indices, topk_bias

if _TORCH_HAS_CUSTOM_OP:
    _wrapped_moba_fused_topk = torch.ops.lightning_moba._moba_fused_topk
else:
    _wrapped_moba_fused_topk = _moba_fused_topk
    
##########################################################################################################################

@_torch_custom_op_wrapper("lightning_moba::_varlen_sort", mutates_args=(), device_types="cuda")
def _varlen_sort(
    col_offsets: torch.Tensor,
    col_nnz: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    col_offset_ends = col_offsets.view(-1) + col_nnz.view(-1)
    return lightning_moba_gpu.varlen_sort(
        col_offsets.view(-1), col_offset_ends, indices
    )

@_torch_register_fake_wrapper("lightning_moba::_varlen_sort")
def _varlen_sort_fake(
    col_offsets: torch.Tensor,
    col_nnz: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    # varlen_sort is out-of-place
    col_offset_ends = col_offsets.view(-1) + col_nnz.view(-1)
    return torch.empty_like(indices)

if _TORCH_HAS_CUSTOM_OP:
    _wrapped_varlen_sort = torch.ops.lightning_moba._varlen_sort
else:
    _wrapped_varlen_sort = _varlen_sort

##########################################################################################################################

@_torch_custom_op_wrapper("lightning_moba::_lightning_moba_attn_varlen_forward", mutates_args=(), device_types="cuda")
def _lightning_moba_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # MOBA sparse pattern parameters
    moba_col_offsets: torch.Tensor,
    moba_col_nnz: torch.Tensor,
    moba_row_indices: torch.Tensor,
    lg_block_m: int,
    lg_block_n: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    moba_col_offsets = maybe_contiguous(moba_col_offsets)
    moba_col_nnz = maybe_contiguous(moba_col_nnz)
    moba_row_indices = maybe_contiguous(moba_row_indices)
    
    out, softmax_lse, S_dmask, rng_state = lightning_moba_gpu.moba_varlen_fwd(
        q,
        k,
        v,
        None,
        moba_col_offsets,
        moba_col_nnz,
        moba_row_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        softcap,
        return_softmax,
        lg_block_m,
        lg_block_n,
        None,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    return out, softmax_lse, S_dmask, rng_state

@_torch_register_fake_wrapper("lightning_moba::_lightning_moba_attn_varlen_forward")
def _lightning_moba_attn_varlen_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # MOBA sparse pattern parameters
    moba_col_offsets: torch.Tensor,
    moba_col_nnz: torch.Tensor,
    moba_row_indices: torch.Tensor,
    lg_block_m: int,
    lg_block_n: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape
    
    out = torch.empty_like(q)
    softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128)
    if return_softmax:
        p = torch.empty((batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state

if _TORCH_HAS_CUSTOM_OP:
    _wrapped_lightning_moba_attn_varlen_forward = torch.ops.lightning_moba._lightning_moba_attn_varlen_forward
else:
    _wrapped_lightning_moba_attn_varlen_forward = _lightning_moba_attn_varlen_forward

##########################################################################################################################

@_torch_custom_op_wrapper("lightning_moba::_lightning_moba_attn_varlen_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _lightning_moba_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # MOBA sparse pattern parameters
    moba_col_offsets: torch.Tensor,
    moba_col_nnz: torch.Tensor,
    moba_row_indices: torch.Tensor,
    lg_block_m: int,
    lg_block_n: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = lightning_moba_gpu.moba_varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        moba_col_offsets,
        moba_col_nnz,
        moba_row_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        softcap,
        deterministic,
        lg_block_m,
        lg_block_n,
        None,
        rng_state,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return softmax_d

@_torch_register_fake_wrapper("lightning_moba::_lightning_moba_attn_varlen_backward")
def _lightning_moba_attn_varlen_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # MOBA sparse pattern parameters
    moba_col_offsets: torch.Tensor,
    moba_col_nnz: torch.Tensor,
    moba_row_indices: torch.Tensor,
    lg_block_m: int,
    lg_block_n: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    softmax_d = torch.empty((num_heads, total_q + 128 * batch_size), device=q.device, dtype=torch.float32)
    
    return softmax_d

if _TORCH_HAS_CUSTOM_OP:
    _wrapped_lightning_moba_attn_varlen_backward = torch.ops.lightning_moba._lightning_moba_attn_varlen_backward
else:
    _wrapped_lightning_moba_attn_varlen_backward = _lightning_moba_attn_varlen_backward

##########################################################################################################################

class LightningMobaAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        # MOBA sparse pattern parameters
        moba_col_offsets,
        moba_col_nnz,
        moba_row_indices,
        lg_block_m,
        lg_block_n,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_lightning_moba_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            moba_col_offsets,
            moba_col_nnz,
            moba_row_indices,
            lg_block_m,
            lg_block_n,
            dropout_p,
            softmax_scale,
            causal=causal,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state,
                moba_col_offsets, moba_col_nnz, moba_row_indices
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.lg_block_m = lg_block_m
            ctx.lg_block_n = lg_block_n

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, moba_col_offsets, moba_col_nnz, moba_row_indices = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_lightning_moba_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            moba_col_offsets,
            moba_col_nnz,
            moba_row_indices,
            ctx.lg_block_m,
            ctx.lg_block_n,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

##########################################################################################################################

def flash_topk_varlen_func(
    q,
    k,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    # MOBA sparse pattern parameters
    moba_topk,
    moba_chunk_size,
    causal=False,
    topk_pooling=None,
    return_scores=False,
    hybrid_lambda=0.0,
    block_summaries=None,
    cu_seqlens_summaries=None,
):
    """
    Computes the top-k indices for Mixture-of-Blocks Attention (MOBA).
    This function handles variable length sequences.

    Args:
        q (torch.Tensor): Query tensor of shape (total_q, num_heads, head_size).
        k (torch.Tensor): Key tensor of shape (total_k, num_heads, head_size).
        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries, shape (batch_size + 1,).
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys, shape (batch_size + 1,).
        max_seqlen_q (int): Maximum sequence length for queries.
        max_seqlen_k (int): Maximum sequence length for keys.
        moba_topk (int): The number of top-k elements to select.
        moba_chunk_size (int): The chunk size for MOBA.
        causal (bool): Whether to apply causal masking.
        hybrid_lambda (float): Compatibility parameter; only 0.0 is supported.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - col_offsets (torch.Tensor): Column offsets for the sparse matrix.
            - col_nnz (torch.Tensor): Number of non-zero elements per column block.
            - indices (torch.Tensor): The top-k indices.
    """
    if hybrid_lambda != 0.0:
        raise NotImplementedError("hybrid_lambda is not supported by the shared CUDA extension; use 0.0")
    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])

    if topk_pooling not in (None, "mean", "avg", "average"):
        raise ValueError(
            "This LightningMoBA compatibility path currently supports mean "
            f"pooling only, got {topk_pooling!r}"
        )
    if block_summaries is None:
        km, cu_seqlens_km, _ = flash_topk_mean_pool(
            k, cu_seqlens_k, max_seqlen_k, moba_chunk_size
        )
    else:
        if cu_seqlens_summaries is None:
            raise ValueError("cu_seqlens_summaries is required with block_summaries")
        km = block_summaries.contiguous()
        cu_seqlens_km = cu_seqlens_summaries.to(
            device=k.device, dtype=torch.int32
        ).contiguous()

    (
        col_offsets,
        col_nnz,
        indices,
        flat_bias,
        topk_values,
        topk_indices,
        topk_bias,
    ) = _wrapped_moba_fused_topk(
        q,
        km,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_km,
        max_seqlen_q,
        max_seqlen_k,
        moba_topk,
        moba_chunk_size,
        causal=causal,
    )

    if return_scores:
        indices = _wrapped_varlen_sort(col_offsets, col_nnz, indices)
        flat_bias = torch.zeros_like(flat_bias)
        return (
            col_offsets,
            col_nnz,
            indices,
            km.contiguous(),
            cu_seqlens_km.contiguous(),
            topk_values.contiguous(),
            topk_indices.contiguous(),
            topk_bias.contiguous(),
            flat_bias.contiguous(),
        )

    indices = _wrapped_varlen_sort(col_offsets, col_nnz, indices)
    return col_offsets, col_nnz, indices

##########################################################################################################################

def lightning_moba_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    # MOBA sparse pattern parameters
    moba_col_offsets,
    moba_col_nnz,
    moba_row_indices,
    lg_block_m=128,
    lg_block_n=64,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        moba_col_offsets: Optional[torch.Tensor]. Column offsets for MOBA sparse pattern.
            Shape: (batch_size, num_heads, max_lg_col_num), dtype: int64
        moba_col_nnz: Optional[torch.Tensor]. Non-zero counts per column for MOBA sparse pattern.
            Shape: (batch_size, num_heads, max_lg_col_num), dtype: int32
        moba_row_indices: Optional[torch.Tensor]. Row indices for MOBA sparse pattern (flattened).
            dtype: int32
        lg_block_m: int. Logical block size in M dimension (query). Default: 128
        lg_block_n: int. Logical block size in N dimension (key). Default: 64
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return LightningMobaAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        moba_col_offsets,
        moba_col_nnz,
        moba_row_indices,
        lg_block_m,
        lg_block_n,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )
    
##########################################################################################################################

def lightning_moba_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    moba_chunk_size=128,
    *,
    alpha,
    causal=True,
    softmax_scale=None,
    sink=1,
    window=2,
):
    """Relative-threshold routing followed by the SM90 KV-centric pipeline.

    Q/K/V use packed [tokens, heads, 128] BF16 layout on one SM90 device.
    Each query sequence is positionally aligned with the corresponding K suffix.
    Non-anchor blocks satisfy scaled_score >= scaled_row_max + log(alpha).
    Alpha is a required scalar in (0, 1], not a Top-K budget. Sink/window
    count 128-token blocks. There is no per-row cap or minimum-budget padding.

    This inference-only, eager entry point explicitly calls the pipelined
    CUDA kernel and variable-length Q-CSR combine, without environment-based
    dispatch or fallback. Partial-output storage scales with selected pairs.
    Use lightning_moba_topk_varlen_func for the Top-K/query-centric baseline.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be a finite scalar in (0, 1]")
    if type(moba_chunk_size) is not int or moba_chunk_size != 128:
        raise ValueError("the default SM90 pipeline requires moba_chunk_size=128")
    if causal is not True:
        raise ValueError("the default dynamic-threshold path is causal-only")
    if type(sink) is not int or type(window) is not int or sink < 0 or window < 1:
        raise ValueError("sink must be a nonnegative integer and window a positive integer")
    if any(type(length) is not int or not 0 < length <= 2147483647
           for length in (max_seqlen_q, max_seqlen_k)):
        raise ValueError("maximum sequence lengths must be positive int32 integers")
    if softmax_scale is None:
        softmax_scale = 128 ** -0.5
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, Real) or not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if any(not isinstance(t, torch.Tensor) or t.ndim != 3 for t in (q, k, v)):
        raise ValueError("expected packed rank-3 Q/K/V tensors")
    if any(t.requires_grad for t in (q, k, v)):
        raise ValueError("the default KV-centric pipeline is inference-only; use the explicit Top-K baseline for backward")
    if any(not t.is_cuda or t.device != q.device for t in (q, k, v)):
        raise ValueError("Q/K/V must share one CUDA device")
    if any(t.dtype != torch.bfloat16 or t.shape[2] != 128 for t in (q, k, v)):
        raise ValueError("the default SM90 pipeline requires BF16 and head_dim=128")
    if k.shape != v.shape or q.shape[1] < 1 or k.shape[1] < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("K/V shapes must match and KV heads must divide query heads")
    if q.shape[0] * q.shape[1] > 2147483647 or k.shape[0] > 2147483647:
        raise ValueError("packed dimensions exceed the int32 indexing range")
    for cu in (cu_seqlens_q, cu_seqlens_k):
        if (not isinstance(cu, torch.Tensor) or cu.ndim != 1 or cu.dtype != torch.int32
                or cu.device != q.device or not cu.is_contiguous()):
            raise ValueError("cu_seqlens must be contiguous CUDA int32 vectors on the Q device")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
        raise ValueError("Q/K cumulative lengths must describe the same nonempty batch")
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("dynamic pattern construction is eager-only, not CUDA-graph-safe")
        if torch.cuda.get_device_capability(q.device) != (9, 0):
            raise ValueError("the default pipelined backend requires SM90; no baseline fallback is performed")
        required = (
            "moba_dynamic_relative_threshold_compact",
            "moba_compact_csc_qcsr_from_bitmask",
            "moba_sm90_phase4b_csc_pipe",
            "moba_combine_q_csr",
        )
        missing = [name for name in required if not hasattr(lightning_moba_gpu, name)]
        if missing:
            raise RuntimeError(f"rebuild lightning_moba_cuda with the SM90 pipeline; missing: {', '.join(missing)}")
        q_boundaries = cu_seqlens_q.cpu().tolist()
        k_boundaries = cu_seqlens_k.cpu().tolist()
        if (q_boundaries[0] != 0 or k_boundaries[0] != 0
                or q_boundaries[-1] != q.shape[0] or k_boundaries[-1] != k.shape[0]):
            raise ValueError("cumulative lengths must start at zero and end at the packed token counts")
        for i in range(len(q_boundaries) - 1):
            qlen = q_boundaries[i + 1] - q_boundaries[i]
            klen = k_boundaries[i + 1] - k_boundaries[i]
            if not 0 < qlen <= klen or qlen > max_seqlen_q or klen > max_seqlen_k:
                raise ValueError("each sequence must satisfy 0 < query length <= key length and its maximum length")

        from lightning_moba.triton_mean_pool import flash_dynamic_mean_pool
        from lightning_moba.cuda_dynamic_threshold import (
            flash_cuda_dynamic_relative_threshold_compact_varlen_func,
            flash_cuda_compact_csc_qcsr_from_bitmask,
        )

        q, k, v = (t.contiguous() for t in (q, k, v))
        km, cu_seqlens_km, _ = flash_dynamic_mean_pool(
            k, cu_seqlens_k, max_seqlen_k, moba_chunk_size,
        )
        pattern = flash_cuda_dynamic_relative_threshold_compact_varlen_func(
            q, km, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km,
            max_seqlen_q, max_seqlen_k, float(alpha), moba_chunk_size,
            softmax_scale=float(softmax_scale), sink=sink, window=window,
            causal=True, min_blocks_per_row=0, build_csc=False,
        )
        pattern, q_offsets, q_csc_indices = flash_cuda_compact_csc_qcsr_from_bitmask(
            pattern, cu_seqlens_q, cu_seqlens_km,
            max_seqlen_q, max_seqlen_k, moba_chunk_size,
        )
        o_partial, lse_partial = lightning_moba_gpu.moba_sm90_phase4b_csc_pipe(
            q, k, v, pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
            cu_seqlens_q, cu_seqlens_k, True, float(softmax_scale),
        )
        return lightning_moba_gpu.moba_combine_q_csr(
            o_partial, lse_partial, q_offsets, q_csc_indices, torch.empty_like(q),
        )


def lightning_moba_topk_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    moba_chunk_size,
    moba_topk,
    causal=True,
):
    """Explicit Top-K/query-centric baseline, including its backward path."""
    col_offsets, col_nnz, indices = flash_topk_varlen_func(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        # MOBA sparse pattern parameters
        moba_topk,
        moba_chunk_size,
        causal=causal,
    )
    
    lg_block_m = decide_lg_block_m(moba_topk, moba_chunk_size, max_seqlen_k, causal, max_seqlen_q=max_seqlen_q)
    
    return lightning_moba_attn_varlen_func(
                q, k, v, 
                cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k, 
                col_offsets,
                col_nnz,
                indices,
                lg_block_m,
                moba_chunk_size,
                dropout_p=0.0, 
                causal=causal, 
    )
