"""Matched-budget block128 routing sweep; pooling and attention are not timed."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


def calibrate(count_pairs, target, tolerance=0.0005, iterations=50):
    """Bracket in log(alpha); never break ties or truncate the selected pattern."""
    if target <= 0 or not 0 < tolerance < 1 or iterations < 1:
        raise ValueError("positive target/iterations and tolerance in (0, 1) required")
    observations = {}

    def evaluate(log_alpha):
        if log_alpha not in observations:
            observations[log_alpha] = int(count_pairs(math.exp(log_alpha)))
        return observations[log_alpha]

    upper = 0.0
    minimum = evaluate(upper)
    lower = upper
    if minimum < target:
        lower = -1.0
        while evaluate(lower) < target and lower > -80.0:
            lower = max(-80.0, lower * 2)
        if evaluate(lower) >= target:
            for _ in range(iterations):
                middle = (lower + upper) / 2
                count = evaluate(middle)
                if abs(count - target) / target <= tolerance:
                    break
                if count > target:
                    lower = middle
                else:
                    upper = middle

    best = min(observations, key=lambda value: (abs(observations[value] - target), -value))
    pairs = observations[best]
    error = abs(pairs - target) / target
    return {"alpha": math.exp(best), "target_pairs": target, "selected_pairs": pairs,
            "relative_pair_error": error, "matched": error <= tolerance,
            "evaluations": len(observations),
            "bracket": [{"alpha": math.exp(value), "pairs": observations[value]}
                        for value in sorted({lower, upper})]}


def tensor_pair_count(counts):
    return int(counts.sum(dtype=torch.int64).item())


def validate_csc(offsets, counts, indices, qlen):
    flat = counts.reshape(-1).long()
    starts = offsets.reshape(-1)
    if bool((flat < 0).any()) or bool((starts < 0).any()) or bool((starts + flat > indices.numel()).any()):
        raise RuntimeError("CSC count/offset exceeds allocated storage")
    pairs = tensor_pair_count(counts)
    if pairs > 0:
        for begin in range(0, flat.numel(), 64):
            lengths = flat[begin:begin + 64]
            columns = torch.repeat_interleave(torch.arange(lengths.numel(), device=indices.device), lengths)
            prefix = lengths.cumsum(0) - lengths
            slots = torch.arange(columns.numel(), device=indices.device) - prefix[columns]
            values = indices[starts[begin + columns] + slots]
            if bool(((values < 0) | (values >= qlen)).any()):
                raise RuntimeError("CSC contains out-of-range query indices")
    return pairs


def measure(fn, metadata, expected_counts, mode, warmup, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            output = fn()
        stream.synchronize()
        if mode == "graph-native":
            del output
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = fn()
            action = graph.replay
            for _ in range(warmup):
                action()
            stream.synchronize()
        else:
            action = fn
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        end.record(stream)
        end.synchronize()
        times = []
        for _ in range(repeats):
            start.record(stream)
            result = action()
            end.record(stream)
            end.synchronize()
            times.append(start.elapsed_time(end))
            if graph is None:
                output = result
        offsets, counts, indices = metadata(output)
        if not torch.equal(counts, expected_counts):
            raise RuntimeError("routing column counts changed between validation and timed execution")
        pairs = tensor_pair_count(counts)
    torch.cuda.current_stream().wait_stream(stream)
    return {"median_ms": statistics.median(times), "min_ms": min(times),
            "max_ms": max(times), "pairs_after_timing": pairs}


def capture_input(path, layout):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not all(isinstance(payload.get(key), torch.Tensor) for key in ("q", "k")):
        raise ValueError("capture must be a tensor dictionary containing q and k")
    q, k = payload["q"], payload["k"]
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("capture Q/K must be rank-3 tensors")
    if layout == "head-major":
        q, k = q.transpose(0, 1), k.transpose(0, 1)
    if (q.dtype != torch.bfloat16 or k.dtype != q.dtype or q.shape[2] != 128 or k.shape[2] != 128
            or q.shape[0] != k.shape[0] or q.shape[1] < 1 or k.shape[1] < 1 or q.shape[1] % k.shape[1]):
        raise ValueError("capture must contain BF16 D128 full-prefill Q/K with valid GQA dimensions")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return q.contiguous(), k.contiguous(), digest.hexdigest()


def run_point(q, k, topk, args, native, pool, relative, padded, public_topk):
    length, heads, _ = q.shape
    columns = (length + 127) // 128
    capacity = length * heads * columns
    if args.mode == "graph-native" and capacity > 2147483647:
        return {"status": "unsupported", "reason": "padded CSC exceeds int32 indexing"}
    if length * heads * topk > 2147483647:
        return {"status": "unsupported", "reason": "Top-K allocation exceeds int32 indexing"}
    free, _ = torch.cuda.mem_get_info()
    estimated = (capacity * 4 if args.mode == "graph-native" else length * heads * topk * 4)
    estimated += length * heads * ((columns + 31) // 32) * 4
    estimated += length * heads * topk * 24 + 256 * 1024**2
    if 2 * estimated > free * 0.8:
        gc.collect()
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
    if 2 * estimated > free * 0.8:
        return {"status": "insufficient_memory", "estimated_workspace_bytes": 2 * estimated,
                "free_bytes": free, "reason": "preflight estimate; budgets are never truncated"}
    cu = torch.tensor([0, length], dtype=torch.int32, device=q.device)
    km, cukm, _ = pool(k, cu, length, 128)
    baseline_args = (q, km, cu, cu, cukm, length, length, topk, 128, True)
    if args.mode == "graph-native":
        baseline = lambda: native.moba_fused_topk(*baseline_args)
    else:
        baseline = lambda: public_topk(
            q, k, cu, cu, length, length, topk, 128, causal=True,
            block_summaries=km, cu_seqlens_summaries=cukm,
        )
    reference = baseline()
    target = validate_csc(*reference[:3], length)
    expected_baseline = reference[1].clone()
    del reference

    def count(alpha):
        pattern = relative(q, km, cu, cu, cukm, length, length, alpha, 128,
                           sink=args.sink, window=args.window, build_csc=False,
                           min_blocks_per_row=0)
        pairs = tensor_pair_count(pattern.col_nnz)
        if pairs != tensor_pair_count(pattern.row_nnz) or bool(pattern.row_overflow.any()):
            raise RuntimeError("selector row/column counts disagree or overflowed")
        return pairs

    calibration = calibrate(count, target, args.tolerance, args.calibration_steps)
    if not calibration["matched"]:
        return {"status": "unmatched", "calibration": calibration, "speedup": None}
    alpha = calibration["alpha"]
    if args.mode == "graph-native":
        threshold = lambda: padded(q, km, cu, cu, cukm, length, length, alpha, 128,
                                   sink=args.sink, window=args.window)
    else:
        threshold = lambda: relative(q, km, cu, cu, cukm, length, length, alpha, 128,
                                     sink=args.sink, window=args.window, min_blocks_per_row=0,
                                     build_csc=True)
    pattern = threshold()
    actual = validate_csc(pattern.col_offsets, pattern.col_nnz, pattern.csc_indices, length)
    if actual != calibration["selected_pairs"] or actual != tensor_pair_count(pattern.row_nnz) or bool(pattern.row_overflow.any()):
        raise RuntimeError("complete CSC construction disagrees with calibration")
    expected_threshold = pattern.col_nnz.clone()
    del pattern
    baseline_time = measure(baseline, lambda value: value[:3], expected_baseline,
                            args.mode, args.warmup, args.repeats)
    gc.collect()
    threshold_time = measure(threshold, lambda value: (value.col_offsets, value.col_nnz, value.csc_indices),
                             expected_threshold, args.mode, args.warmup, args.repeats)
    return {"status": "ok", "calibration": calibration,
            "mean_retained_blocks_baseline": target / (length * heads),
            "mean_retained_blocks_threshold": actual / (length * heads),
            "topk": baseline_time, "threshold": threshold_time,
            "speedup": baseline_time["median_ms"] / threshold_time["median_ms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, help="trusted local tensor dictionary with BF16 q/k")
    parser.add_argument("--layout", choices=("token-major", "head-major"), default="token-major")
    parser.add_argument("--model-label", default="unspecified")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096])
    parser.add_argument("--topks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--heads", type=int, default=4, help="query heads for synthetic input only")
    parser.add_argument("--kv-heads", type=int, default=1, help="KV heads for synthetic input only")
    parser.add_argument("--mode", choices=("graph-native", "eager-public"), default="graph-native")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=0.0005)
    parser.add_argument("--calibration-steps", type=int, default=50)
    parser.add_argument("--sink", type=int, default=1)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (any(n <= 0 or n % 128 for n in args.lengths) or any(n not in (16, 32, 64) for n in args.topks)
            or args.heads < 1 or args.kv_heads < 1 or args.heads % args.kv_heads
            or args.warmup < 1 or args.repeats < 1 or args.calibration_steps < 1
            or not 0 < args.tolerance < 1 or args.sink < 0 or args.window < 1):
        parser.error("invalid dimensions, budget, timing, calibration or anchor settings; lengths must be 128-aligned")
    if args.output.exists():
        parser.error("output exists; choose a new filename to preserve previous results")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU and the compiled LightningMoBA extension are required")

    import os
    import lightning_moba_cuda as native
    from lightning_moba import (
        flash_dynamic_mean_pool as pool,
        flash_topk_varlen_func as public_topk,
        flash_cuda_dynamic_relative_threshold_compact_varlen_func as relative,
        flash_cuda_dynamic_relative_threshold_padded_varlen_func as padded,
    )

    os.environ["LIGHTNING_MOBA_TOPK_SINK"] = str(args.sink)
    os.environ["LIGHTNING_MOBA_TOPK_WINDOW"] = str(args.window)
    torch.manual_seed(args.seed)
    source = None
    checksum = None
    if args.capture:
        source_q, source_k, checksum = capture_input(args.capture, args.layout)
        source = (source_q, source_k)
        if max(args.lengths) > source_q.shape[0]:
            parser.error("requested length exceeds captured tensor length")
    report = {
        "input_kind": "captured" if source else "synthetic_smoke_not_paper_results",
        "capture_name": args.capture.name if args.capture else None,
        "capture_sha256": checksum, "input_layout": args.layout,
        "model_label": args.model_label, "mode": args.mode,
        "scope": "scoring + selection + CSC; excludes pooling, attention and alpha calibration",
        "topk_sort_included": args.mode == "eager-public",
        "scope_note": ("native unsorted CSC builders consumed by KV-centric attention; excludes public Top-K segmented sort"
                       if args.mode == "graph-native" else "public Top-K includes segmented sort; threshold emits KV-consumable unsorted compact CSC; not Table 4 CUDA Graph timing"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "warmup": args.warmup, "repeats": args.repeats,
        "pair_tolerance": args.tolerance, "sink": args.sink, "window": args.window,
        "seed": args.seed, "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination, torch.inference_mode():
        try:
            for length in args.lengths:
                if source:
                    q, k = (tensor[:length].to("cuda").contiguous() for tensor in source)
                else:
                    q = torch.randn(length, args.heads, 128, device="cuda", dtype=torch.bfloat16)
                    k = torch.randn(length, args.kv_heads, 128, device="cuda", dtype=q.dtype)
                for topk in args.topks:
                    row = {"length": length, "reference_topk": topk,
                           "q_heads": q.shape[1], "kv_heads": k.shape[1], "head_dim": 128,
                           "captured_prefix_slice": bool(source and length != source[0].shape[0])}
                    try:
                        row.update(run_point(q, k, topk, args, native, pool, relative, padded, public_topk))
                    except (RuntimeError, ValueError) as error:
                        row.update(status="error", error=str(error), speedup=None)
                    report["rows"].append(row)
                    print(json.dumps(row, allow_nan=False), flush=True)
                    gc.collect()
                del q, k
        finally:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
    if any(row["status"] != "ok" for row in report["rows"]):
        raise SystemExit("Some points failed, were unmatched or lacked memory; see the report. No timing-mode fallback was used.")


if __name__ == "__main__":
    main()
