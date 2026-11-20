"""Block128 prefill: pool -> relative threshold -> CSC/Q-CSR -> SM90 -> combine."""
import argparse
import math

import torch


def selector_attention(q, k, v, cu_seqlens, seq_len, alpha):
    from lightning_moba import (
        flash_dynamic_mean_pool,
        flash_cuda_dynamic_relative_threshold_compact_varlen_func,
        flash_cuda_compact_csc_qcsr_from_bitmask,
        lightning_moba_cuda_kv_centric_func,
    )

    km, cu_km, _ = flash_dynamic_mean_pool(k, cu_seqlens, seq_len, 128)
    pattern = flash_cuda_dynamic_relative_threshold_compact_varlen_func(
        q, km, cu_seqlens, cu_seqlens, cu_km, seq_len, seq_len,
        alpha, 128, sink=1, window=2, causal=True,
        min_blocks_per_row=0, build_csc=False,
    )
    pattern, q_offsets, q_csc = flash_cuda_compact_csc_qcsr_from_bitmask(
        pattern, cu_seqlens, cu_km, seq_len, seq_len, 128,
    )
    out = lightning_moba_cuda_kv_centric_func(
        q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len,
        pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
        chunk_size=128, backend="sm90", q_offsets=q_offsets, q_csc=q_csc,
    )
    return out, pattern


def reference(q, k, v, alpha):
    keys = k.transpose(0, 1)
    means = torch.stack([
        keys[:, start:start + 128].float().mean(1)
        for start in range(0, k.shape[0], 128)
    ], dim=1).to(k.dtype).float()
    ratio = q.shape[1] // k.shape[1]
    queries = q.transpose(0, 1).float()
    scores = queries @ means.repeat_interleave(ratio, 0).transpose(-1, -2) / math.sqrt(128)
    positions = torch.arange(q.shape[0], device=q.device)[:, None]
    columns = torch.arange(means.shape[1], device=q.device)[None, :]
    local = positions // 128
    forced = (columns < 1) | ((columns <= local) & (local - columns < 2))
    completed = (columns + 1) * 128 <= positions
    row_max = scores.masked_fill(~(forced | completed), -float("inf")).amax(-1, keepdim=True)
    selected = forced | (completed & (scores >= row_max + math.log(alpha)))
    tokens = torch.arange(k.shape[0], device=q.device)
    allowed = selected[:, :, tokens // 128] & (tokens[None, :] <= positions)
    logits = queries @ keys.repeat_interleave(ratio, 0).float().transpose(-1, -2) / math.sqrt(128)
    values = v.transpose(0, 1).repeat_interleave(ratio, 0).float()
    return (logits.masked_fill(~allowed, -float("inf")).softmax(-1) @ values).transpose(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--check", action="store_true", help="compare against independent FP32 attention")
    args = parser.parse_args()
    if not 0 < args.seq_len <= 2147483647 or not math.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        parser.error("seq-len must be positive and alpha finite in (0, 1]")
    if args.check and args.seq_len > 4096:
        parser.error("--check materializes dense attention; use seq-len <= 4096")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        parser.error("this example requires an SM90 GPU and the compiled LightningMoBA extension")
    torch.manual_seed(42)
    with torch.inference_mode():
        q = torch.randn(args.seq_len, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(args.seq_len, 1, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        cu = torch.tensor([0, args.seq_len], device="cuda", dtype=torch.int32)
        out, pattern = selector_attention(q, k, v, cu, args.seq_len, args.alpha)
        if args.check:
            expected = reference(q, k, v, args.alpha)
            torch.testing.assert_close(out.float(), expected, atol=0.04, rtol=0.04)
        print({"shape": tuple(out.shape), "alpha": args.alpha,
               "mean_retained_blocks": pattern.row_nnz.float().mean().item(),
               "backend": "sm90", "checked": args.check})


if __name__ == "__main__":
    main()
