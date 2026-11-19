import os
import unittest
from unittest.mock import patch

import torch

from lightning_moba.streaming_kv_centric import (
    build_streaming_q_csr,
    lightning_moba_streaming_kv_centric_func,
    streaming_chunk_edges,
)


def make_pattern(q_lengths, k_lengths, heads, block_size=128, sparse=False, reverse=False):
    counts, rows = [], []
    columns = (max(k_lengths) + block_size - 1) // block_size
    for qlen, klen in zip(q_lengths, k_lengths):
        for head in range(heads):
            for col in range(columns):
                selected = [
                    row for row in range(qlen)
                    if col * block_size < klen
                    and (not sparse or col == 0 or (row + head + col) % 3 == 0)
                ]
                if reverse:
                    selected.reverse()
                counts.append(len(selected))
                rows.extend(selected)
    counts = torch.tensor(counts, dtype=torch.int32).reshape(len(q_lengths), heads, columns)
    flat = counts.reshape(-1).to(torch.int64)
    offsets = (flat.cumsum(0) - flat).reshape_as(counts)
    return offsets, counts, torch.tensor(rows, dtype=torch.int32)


def reference(q, k, v, offsets, counts, indices, q_lengths, k_lengths, causal):
    heads, dim = q.shape[1:]
    output = torch.zeros_like(q)
    global_lse = torch.full((heads, q.shape[0]), float("inf"), device=q.device)
    qb = kb = 0
    for batch, (qlen, klen) in enumerate(zip(q_lengths, k_lengths)):
        for head in range(heads):
            mask = torch.zeros((qlen, klen), dtype=torch.bool, device=q.device)
            for col in range(counts.shape[2]):
                begin = int(offsets[batch, head, col])
                end = begin + int(counts[batch, head, col])
                local_rows = indices[begin:end].to(device=q.device, dtype=torch.int64)
                mask[local_rows, col * 128:min((col + 1) * 128, klen)] = True
            if causal:
                mask &= torch.arange(klen, device=q.device)[None, :] <= (
                    torch.arange(qlen, device=q.device)[:, None] + klen - qlen
                )
            kv_head = head // (heads // k.shape[1])
            scores = q[qb:qb + qlen, head].float() @ k[kb:kb + klen, kv_head].float().T
            scores *= dim ** -0.5
            scores = scores.masked_fill(~mask, -float("inf"))
            valid = mask.any(1)
            probabilities = torch.softmax(scores, dim=-1).nan_to_num()
            output[qb:qb + qlen, head] = (probabilities @ v[kb:kb + klen, kv_head].float()).to(q.dtype)
            global_lse[head, qb:qb + qlen] = torch.where(
                valid, torch.logsumexp(scores, dim=-1), float("inf")
            )
        qb += qlen
        kb += klen
    return output, global_lse


class StreamingMetadataTest(unittest.TestCase):
    def test_packed_unsorted_csc_and_unused_tail(self):
        offsets, counts, indices = make_pattern([3, 5], [129, 257], 2, sparse=True, reverse=True)
        cuq = torch.tensor([0, 3, 8], dtype=torch.int32)
        padded = torch.cat((indices, torch.full((7,), 999, dtype=torch.int32)))
        q_offsets, q_csc = build_streaming_q_csr(offsets, counts, padded, cuq, 8, 2)
        self.assertEqual(q_csc.numel(), indices.numel())
        self.assertEqual(int(q_offsets[-1]), indices.numel())
        self.assertTrue(torch.equal(q_csc.sort().values, torch.arange(indices.numel(), dtype=torch.int32)))
        for row in range(16):
            edges = q_csc[q_offsets[row]:q_offsets[row + 1]]
            self.assertTrue(bool((edges[1:] > edges[:-1]).all()))
        flat = counts.reshape(-1).long()
        columns = torch.repeat_interleave(torch.arange(flat.numel()), flat)
        batch = columns // (2 * counts.shape[-1])
        head = (columns // counts.shape[-1]) % 2
        qh = (indices.long() + cuq[batch]) * 2 + head
        grouped = torch.repeat_interleave(torch.arange(16), q_offsets[1:] - q_offsets[:-1])
        self.assertTrue(torch.equal(qh[q_csc.long()], grouped))

    def test_empty_csc(self):
        offsets = torch.zeros((1, 2, 1), dtype=torch.int64)
        counts = torch.zeros_like(offsets, dtype=torch.int32)
        q_offsets, q_csc = build_streaming_q_csr(
            offsets, counts, torch.empty(0, dtype=torch.int32), torch.tensor([0, 3]), 3, 2,
        )
        self.assertEqual(q_offsets.tolist(), [0] * 7)
        self.assertEqual(q_csc.numel(), 0)

    def test_capacity_switch(self):
        for value in (0, 1, 2000000, 2147483647):
            with patch.dict(os.environ, {"LIGHTNING_MOBA_STREAM_CHUNK_EDGES": str(value)}):
                self.assertEqual(streaming_chunk_edges(), value)
        for value in (-1, 2147483648):
            with self.assertRaises(ValueError):
                streaming_chunk_edges(value)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
class StrictRowCapCudaTest(unittest.TestCase):
    def test_per_query_head_cap_and_csc_counts(self):
        from lightning_moba.cuda_dynamic_threshold import (
            CompactDynamicThresholdPattern, cap_compact_pattern_per_row,
            flash_cuda_compact_csc_from_bitmask,
        )
        from lightning_moba.streaming_kv_centric import build_streaming_q_csr

        columns, tokens, heads = 258, 4, 2
        device = "cuda"
        masks = torch.zeros((tokens, heads, (columns + 31) // 32), device=device, dtype=torch.int32)
        for row in range(tokens):
            for head in range(heads):
                selected = range(columns) if head == 0 else (0, 1, columns - 2, columns - 1)
                for col in selected:
                    bit = 1 << (col % 32)
                    masks[row, head, col // 32] |= bit if bit < 2**31 else bit - 2**32
        counts = torch.tensor([[columns, 4]] * tokens, device=device, dtype=torch.int32)
        col_counts = torch.stack((
            torch.full((columns,), tokens, device=device, dtype=torch.int32),
            torch.tensor([tokens, tokens] + [0] * (columns - 4) + [tokens, tokens],
                         device=device, dtype=torch.int32),
        )).unsqueeze(0)
        pattern = CompactDynamicThresholdPattern(
            col_offsets=torch.empty(0, device=device, dtype=torch.int64),
            col_nnz=col_counts, csc_indices=torch.empty(0, device=device, dtype=torch.int32),
            row_bitmask=masks, row_nnz=counts,
            row_overflow=torch.zeros_like(counts),
        )
        cuq = torch.tensor([0, tokens], device=device, dtype=torch.int32)
        cuk = torch.tensor([0, columns * 128], device=device, dtype=torch.int32)
        stats = cap_compact_pattern_per_row(
            pattern, cuq, cuk, 128, sink=1, window=2, max_blocks_per_row=256,
        )
        self.assertEqual(pattern.row_nnz.tolist(), [[256, 4]] * tokens)
        self.assertEqual(stats.tolist(), [8, 4])
        for row in range(tokens):
            word = pattern.row_bitmask[row, 0].to(torch.int64).bitwise_and(0xffffffff)
            bits = [c for c in range(columns) if int(word[c // 32]) & (1 << (c % 32))]
            self.assertEqual(bits, [0] + list(range(3, columns)))
        self.assertEqual(int(pattern.col_nnz.sum()), int(pattern.row_nnz.sum()))
        self.assertEqual(pattern.col_nnz[0, 0, :4].tolist(), [4, 0, 0, 4])
        cukm = torch.tensor([0, columns], device=device, dtype=torch.int32)
        result = flash_cuda_compact_csc_from_bitmask(pattern, cuq, cukm, tokens, columns * 128, 128)
        self.assertEqual(result.csc_indices.numel(), int(pattern.row_nnz.sum()))
        q_offsets, q_csc = build_streaming_q_csr(
            result.col_offsets, result.col_nnz, result.csc_indices, cuq, tokens, heads,
        )
        self.assertEqual(int(q_offsets[-1]), int(pattern.row_nnz.sum()))
        self.assertEqual(q_csc.numel(), result.csc_indices.numel())

        pattern.csc_indices = torch.empty(0, device=device, dtype=torch.int32)
        stats_128 = cap_compact_pattern_per_row(
            pattern, cuq, cuk, 128, sink=1, window=2, max_blocks_per_row=128,
        )
        self.assertEqual(stats_128.tolist(), [512, 4])
        self.assertEqual(pattern.row_nnz.tolist(), [[128, 4]] * tokens)
        for row in range(tokens):
            word = pattern.row_bitmask[row, 0].to(torch.int64).bitwise_and(0xffffffff)
            bits = [c for c in range(columns) if int(word[c // 32]) & (1 << (c % 32))]
            self.assertEqual(bits, [0] + list(range(131, columns)))
        result = flash_cuda_compact_csc_from_bitmask(pattern, cuq, cukm, tokens, columns * 128, 128)
        self.assertEqual(result.csc_indices.numel(), int(pattern.row_nnz.sum()))
        self.assertEqual(int(pattern.col_nnz.sum()), int(pattern.row_nnz.sum()))

    def test_anchor_budget_and_small_cap(self):
        from lightning_moba.cuda_dynamic_threshold import (
            CompactDynamicThresholdPattern, cap_compact_pattern_per_row,
        )

        device = "cuda"
        mask = torch.tensor([[[0b11111]]], device=device, dtype=torch.int32)
        rows = torch.tensor([[5]], device=device, dtype=torch.int32)
        cols = torch.ones((1, 1, 5), device=device, dtype=torch.int32)
        pattern = CompactDynamicThresholdPattern(
            torch.empty(0, device=device, dtype=torch.int64), cols,
            torch.empty(0, device=device, dtype=torch.int32), mask, rows,
            torch.zeros_like(rows),
        )
        cuq = torch.tensor([0, 1], device=device, dtype=torch.int32)
        cuk = torch.tensor([0, 640], device=device, dtype=torch.int32)
        with self.assertRaises(ValueError):
            cap_compact_pattern_per_row(
                pattern, cuq, cuk, 128, sink=1, window=2, max_blocks_per_row=2,
            )
        stats = cap_compact_pattern_per_row(
            pattern, cuq, cuk, 128, sink=1, window=2, max_blocks_per_row=3,
        )
        self.assertEqual(pattern.row_nnz.tolist(), [[3]])
        self.assertEqual(pattern.col_nnz.flatten().tolist(), [1, 0, 0, 1, 1])
        self.assertEqual(stats.tolist(), [2, 1])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
class StreamingCudaTest(unittest.TestCase):
    def test_output_lse_and_ranged_launch(self):
        import lightning_moba_cuda
        self.assertTrue(hasattr(lightning_moba_cuda, "moba_varlen_fwd_kvcentric_stream"))
        torch.manual_seed(1234)
        for dim, dtype, causal, sparse in (
            (128, torch.bfloat16, True, False),
            (128, torch.bfloat16, True, True),
            (64, torch.float16, False, True),
        ):
            qlens, klens, heads = [65, 129], [129, 257], 4
            q = torch.randn(sum(qlens), heads, dim, device="cuda", dtype=dtype)
            k = torch.randn(sum(klens), 1, dim, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            offsets, counts, indices = make_pattern(qlens, klens, heads, sparse=sparse, reverse=True)
            cuq = torch.tensor([0, 65, 194], device="cuda", dtype=torch.int32)
            cuk = torch.tensor([0, 129, 386], device="cuda", dtype=torch.int32)
            expected, expected_lse = reference(q, k, v, offsets, counts, indices, qlens, klens, causal)
            args = (q, k, v, offsets.cuda(), counts.cuda(), indices.cuda(), cuq, cuk, 129, 257)
            for edges in (65, 257, indices.numel() + 1):
                result, lse = lightning_moba_streaming_kv_centric_func(
                    *args, causal=causal, chunk_edges=edges, restrict_grid=True,
                )
                full_grid, full_lse = lightning_moba_streaming_kv_centric_func(
                    *args, causal=causal, chunk_edges=edges, restrict_grid=False,
                )
                torch.testing.assert_close(result, expected, atol=0.03, rtol=0.03)
                torch.testing.assert_close(lse, expected_lse, atol=0.03, rtol=0.03)
                self.assertTrue(torch.equal(result, full_grid))
                self.assertTrue(torch.equal(lse, full_lse))

    def test_empty_rows_and_zero_queries(self):
        for tokens in (0, 3):
            q = torch.randn(tokens, 2, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(129, 1, 128, device="cuda", dtype=q.dtype)
            offsets = torch.zeros((1, 2, 2), device="cuda", dtype=torch.int64)
            counts = torch.zeros_like(offsets, dtype=torch.int32)
            out, lse = lightning_moba_streaming_kv_centric_func(
                q, k, k, offsets, counts, torch.empty(0, device="cuda", dtype=torch.int32),
                torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
                torch.tensor([0, 129], device="cuda", dtype=torch.int32), tokens, 129, chunk_edges=1,
            )
            self.assertTrue(torch.equal(out, torch.zeros_like(q)))
            self.assertTrue(bool(torch.isposinf(lse).all()))

    def test_legacy_combine_parity_and_memory(self):
        import lightning_moba_cuda
        torch.manual_seed(1234)
        q = torch.randn(2048, 2, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2048, 1, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        offsets, counts, indices = make_pattern([2048], [2048], 2)
        offsets, counts, indices = offsets.cuda(), counts.cuda(), indices.cuda()
        cu = torch.tensor([0, 2048], device="cuda", dtype=torch.int32)
        q_offsets, q_csc = build_streaming_q_csr(offsets, counts, indices, cu, 2048, 2)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        partial, local_lse = lightning_moba_cuda.moba_varlen_fwd_kvcentric(
            q, k, v, offsets, counts, indices, cu, cu, 2048, 2048, 128 ** -0.5, True, 128,
        )
        expected = lightning_moba_cuda.moba_combine_q_csr(partial, local_lse, q_offsets, q_csc, torch.empty_like(q))
        torch.cuda.synchronize()
        full_peak = torch.cuda.max_memory_allocated() - baseline
        del partial, local_lse
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out, _ = lightning_moba_streaming_kv_centric_func(
            q, k, v, offsets, counts, indices, cu, cu, 2048, 2048, chunk_edges=2048,
            q_offsets=q_offsets, q_csc=q_csc,
        )
        torch.cuda.synchronize()
        stream_peak = torch.cuda.max_memory_allocated() - baseline
        torch.testing.assert_close(out, expected, atol=0.03, rtol=0.03)
        self.assertLess(stream_peak, full_peak)
        print(f"MEMORY_CHECK full_extra_bytes={full_peak} stream_extra_bytes={stream_peak}", flush=True)

    def test_selector_entrypoints_and_stream_toggle(self):
        from lightning_moba import flash_topk_varlen_func
        from lightning_moba.cuda_dynamic_threshold import flash_cuda_dynamic_kv_centric_func
        from lightning_moba.triton_kv_centric import lightning_moba_cuda_kv_centric_func

        torch.manual_seed(1234)
        tokens = 640
        q = torch.randn(tokens, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(tokens, 1, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        offsets, counts, indices = flash_topk_varlen_func(q, k, cu, cu, tokens, tokens, 3, 128, causal=True)
        km = k.reshape(5, 128, 1, 128).float().mean(1).to(k.dtype)
        cukm = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
        for fn, args in (
            (lightning_moba_cuda_kv_centric_func,
             (q, k, v, cu, cu, tokens, tokens, offsets, counts, indices, 128, 3)),
            (flash_cuda_dynamic_kv_centric_func,
             (q, k, v, km, cu, cu, cukm, tokens, tokens, 0.05, 128)),
        ):
            with patch.dict(os.environ, {"LIGHTNING_MOBA_STREAM_CHUNK_EDGES": "0"}):
                expected = fn(*args)
            with patch.dict(os.environ, {"LIGHTNING_MOBA_STREAM_CHUNK_EDGES": "257"}):
                result = fn(*args)
            torch.testing.assert_close(result, expected, atol=0.03, rtol=0.03)

    @unittest.skipUnless(os.environ.get("LIGHTNING_MOBA_TEST_BACKEND") == "1", "optional vLLM backend test")
    def test_vllm_backend_prefill_and_topk_padding(self):
        from types import SimpleNamespace
        from dynamic_gqa_backend import DynamicGqaFlashAttentionImpl

        torch.manual_seed(1234)
        settings = {
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_TOPK": "3",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_PROFILE": "1",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_DIAGNOSTICS": "0",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_RECORD": "0",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_DISABLE_MEMORY_GUARD": "1",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_EMPTY_CACHE": "0",
        }
        for selector, tokens in (("topk", 640), ("cuda_relative", 640), ("cuda_flashprefill", 641)):
            settings["LIGHTNING_MOBA_DYNAMIC_BACKEND_SELECTOR"] = selector
            with patch.dict(os.environ, settings):
                impl = DynamicGqaFlashAttentionImpl(
                    num_heads=4, head_size=128, scale=128 ** -0.5, num_kv_heads=1,
                    alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                )
            q = torch.randn(tokens, 4, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(tokens, 1, 128, device="cuda", dtype=q.dtype)
            v = torch.randn_like(k)
            metadata = SimpleNamespace(
                num_actual_tokens=tokens,
                query_start_loc=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
            )
            expected = torch.empty_like(q)
            result = torch.empty_like(q)
            with patch.dict(os.environ, {"LIGHTNING_MOBA_STREAM_CHUNK_EDGES": "0"}):
                impl._run_dynamic(q, k, v, expected, metadata)
            with patch.dict(os.environ, {"LIGHTNING_MOBA_STREAM_CHUNK_EDGES": "257"}):
                with patch.object(impl, "_write_record") as write_record:
                    impl._run_dynamic(q, k, v, result, metadata)
            torch.testing.assert_close(result, expected, atol=0.03, rtol=0.03)
            self.assertTrue(write_record.call_args.args[0]["streaming"])
            self.assertEqual(write_record.call_args.args[0]["stream_chunk_edges"], 257)

    @unittest.skipUnless(os.environ.get("LIGHTNING_MOBA_TEST_BACKEND") == "1", "optional vLLM backend test")
    def test_vllm_relative_strict_row_cap(self):
        from types import SimpleNamespace
        from dynamic_gqa_backend import DynamicGqaFlashAttentionImpl

        tokens = 769
        settings = {
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_SELECTOR": "cuda_relative",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_FP_ALPHA": "0.000001",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_TOPK": "32",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_MAX_BLOCKS_PER_ROW": "3",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_DISABLE_MEMORY_GUARD": "1",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_RECORD": "0",
            "LIGHTNING_MOBA_DYNAMIC_BACKEND_DIAGNOSTICS": "0",
        }
        with patch.dict(os.environ, settings):
            impl = DynamicGqaFlashAttentionImpl(
                num_heads=4, head_size=128, scale=128 ** -0.5, num_kv_heads=1,
                alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
            )
        q = torch.randn(tokens, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(tokens, 1, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        metadata = SimpleNamespace(
            num_actual_tokens=tokens,
            query_start_loc=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
        )
        result = torch.empty_like(q)
        with patch.dict(os.environ, {"LIGHTNING_MOBA_STREAM_CHUNK_EDGES": "257"}):
            with patch.object(impl, "_write_record") as record:
                impl._run_dynamic(q, k, v, result, metadata)
        metrics = record.call_args.args[0]
        self.assertEqual(metrics["max_blocks_per_row"], 3)
        self.assertEqual(metrics["row_cap_max_observed"], 3)
        self.assertGreater(metrics["row_cap_removed_edges"], 0)
        self.assertGreater(metrics["row_cap_affected_rows"], 0)
        self.assertTrue(torch.isfinite(result).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
