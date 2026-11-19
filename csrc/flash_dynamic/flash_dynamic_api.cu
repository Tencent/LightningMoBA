#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/numeric_types.h>
#include <torch/extension.h>

#include "src/flash_dynamic.h"
#include "src/flash_dynamic_kernel.h"
#include "src/flashprefill_score_kernel.h"
#include "src/flashprefill_pattern_kernel.h"

namespace flash_dynamic {

__global__ void cap_compact_rows_kernel(
    uint32_t* bitmask, int* row_nnz, int* col_nnz, const int* cu_q,
    const int* cu_k, int batch_size, int heads, int columns, int words,
    int chunk_size, int sink, int window, int max_blocks, int total_q,
    int* cap_stats) {
    const int64_t meta = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (meta >= static_cast<int64_t>(total_q) * heads) return;
    int count = row_nnz[meta];
    if (count <= max_blocks) return;
    const int global_q = meta / heads;
    const int head = meta % heads;
    int batch = 0;
    while (batch + 1 < batch_size && global_q >= cu_q[batch + 1]) ++batch;
    const int local_q = global_q - cu_q[batch];
    const int causal_q = local_q + (cu_k[batch + 1] - cu_k[batch])
        - (cu_q[batch + 1] - cu_q[batch]);
    const int local_block = causal_q / chunk_size;
    const int64_t mask_base = meta * words;
    const int64_t col_base = (static_cast<int64_t>(batch) * heads + head) * columns;
    int forced_count = 0;
    for (int col = 0; col < columns; ++col) {
        const bool selected = (bitmask[mask_base + (col >> 5)] >> (col & 31)) & 1u;
        if (selected && (col < sink ||
            (col <= local_block && local_block - col < window))) ++forced_count;
    }
    // The host rejects configurations with sink + window > max_blocks.
    int available = max_blocks - forced_count;
    int kept = 0;
    for (int col = columns - 1; col >= 0; --col) {
        const uint32_t bit = 1u << (col & 31);
        uint32_t& word = bitmask[mask_base + (col >> 5)];
        if (!(word & bit)) continue;
        const bool forced = col < sink ||
            (col <= local_block && local_block - col < window);
        if (forced) continue;
        if (kept++ < available) continue;
        word &= ~bit;
        atomicSub(col_nnz + col_base + col, 1);
        --count;
    }
    const int removed = row_nnz[meta] - count;
    row_nnz[meta] = count;
    if (removed) {
        atomicAdd(cap_stats, removed);
        atomicAdd(cap_stats + 1, 1);
    }
}

at::Tensor moba_cap_compact_rows(
    const at::Tensor& row_bitmask, const at::Tensor& row_nnz,
    const at::Tensor& col_nnz, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, int chunk_size, int sink,
    int window, int max_blocks) {
    at::cuda::CUDAGuard device_guard{row_bitmask.device()};
    TORCH_CHECK(row_bitmask.is_cuda() && row_nnz.is_cuda() && col_nnz.is_cuda() &&
        cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cap metadata must be on CUDA");
    TORCH_CHECK(row_bitmask.device() == row_nnz.device() &&
        row_bitmask.device() == col_nnz.device() &&
        row_bitmask.device() == cu_seqlens_q.device() &&
        row_bitmask.device() == cu_seqlens_k.device(), "cap metadata device mismatch");
    TORCH_CHECK(row_bitmask.is_contiguous() && row_nnz.is_contiguous() &&
        col_nnz.is_contiguous() && cu_seqlens_q.is_contiguous() &&
        cu_seqlens_k.is_contiguous(), "cap metadata must be contiguous");
    TORCH_CHECK(row_bitmask.scalar_type() == at::kInt &&
        row_nnz.scalar_type() == at::kInt && col_nnz.scalar_type() == at::kInt &&
        cu_seqlens_q.scalar_type() == at::kInt &&
        cu_seqlens_k.scalar_type() == at::kInt, "cap metadata must be int32");
    TORCH_CHECK(row_bitmask.dim() == 3 && row_nnz.dim() == 2 &&
        col_nnz.dim() == 3 && cu_seqlens_q.dim() == 1 &&
        cu_seqlens_k.sizes() == cu_seqlens_q.sizes(), "invalid cap metadata ranks");
    TORCH_CHECK(row_bitmask.size(0) == row_nnz.size(0) &&
        row_bitmask.size(1) == row_nnz.size(1) &&
        row_nnz.size(1) == col_nnz.size(1) &&
        col_nnz.size(0) == cu_seqlens_q.numel() - 1 &&
        row_bitmask.size(2) == (col_nnz.size(2) + 31) / 32,
        "invalid cap metadata shapes");
    TORCH_CHECK(chunk_size > 0 && sink >= 0 && window >= 0 &&
        max_blocks > 0 && max_blocks >= sink + window,
        "max_blocks must cover mandatory sink and window blocks");
    TORCH_CHECK(row_nnz.numel() <= std::numeric_limits<int>::max(),
        "cap row count exceeds int32 capacity");
    auto stats = at::zeros({2}, row_nnz.options());
    const auto rows = row_nnz.numel();
    if (rows) {
        cap_compact_rows_kernel<<<(rows + 127) / 128, 128, 0,
            at::cuda::getCurrentCUDAStream().stream()>>>(
            reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>()),
            row_nnz.data_ptr<int>(), col_nnz.data_ptr<int>(),
            cu_seqlens_q.data_ptr<int>(), cu_seqlens_k.data_ptr<int>(),
            col_nnz.size(0), row_nnz.size(1), col_nnz.size(2),
            row_bitmask.size(2), chunk_size, sink, window, max_blocks,
            row_nnz.size(0), stats.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return stats;
}

std::vector<at::Tensor> moba_flashprefill_coarse_scores(
    const at::Tensor& q,
    const at::Tensor& k,
    double softmax_scale,
    int fine_block,
    int exec_block,
    int num_splits) {
    at::cuda::CUDAGuard device_guard{q.device()};
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q and k must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16,
                "native FlashPrefill scorer supports BF16 only");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "q and k must be rank 3");
    TORCH_CHECK(q.size(0) == k.size(0), "q and k token counts must match");
    TORCH_CHECK(q.size(2) == 128 && k.size(2) == 128,
                "native FlashPrefill scorer supports head_dim=128 only");
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1,
                "q and k last dimension must be contiguous");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "KV heads must divide Q heads");
    TORCH_CHECK(fine_block == 32 && exec_block == 128,
                "phase-1 native FlashPrefill scorer requires fine=32, exec=128");
    TORCH_CHECK(q.size(0) % exec_block == 0,
                "phase-1 native FlashPrefill scorer requires 128-aligned sequence length");
    TORCH_CHECK(num_splits >= 1 && num_splits <= 32,
                "native FlashPrefill scorer num_splits must be in [1, 32]");

    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = k.size(1);
    const int fine_blocks = total_q / fine_block;
    const int coarse_blocks = total_q / exec_block;
    auto pooled = k.reshape({fine_blocks, fine_block, kv_heads, 128});
    // Pool in FP32 then cast once, matching Triton's BF16 pooled output while
    // keeping this first implementation concise.  These reductions are O(SD),
    // tiny beside the O(S^2D/block) score calculation.
    auto k_mean = pooled.to(at::kFloat).mean(1).to(at::kBFloat16).contiguous();
    auto k_max = std::get<0>(pooled.max(1)).contiguous();
    auto scores = at::full(
        {coarse_blocks, q_heads, coarse_blocks},
        -std::numeric_limits<float>::infinity(), q.options().dtype(at::kFloat));

    FlashPrefillScoreParams params{};
    params.q_ptr = q.data_ptr();
    params.k_mean_ptr = k_mean.data_ptr();
    params.k_max_ptr = k_max.data_ptr();
    params.coarse_score_ptr = scores.data_ptr<float>();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.k_row_stride = k_mean.stride(0);
    params.k_head_stride = k_mean.stride(1);
    params.total_q = total_q;
    params.q_heads = q_heads;
    params.kv_heads = kv_heads;
    params.head_ratio = q_heads / kv_heads;
    params.fine_blocks = fine_blocks;
    params.coarse_blocks = coarse_blocks;
    params.softmax_scale_log2 = static_cast<float>(softmax_scale) * 1.4426950408889634f;
    run_flashprefill_coarse_score<cutlass::bfloat16_t>(
        params, num_splits, at::cuda::getCurrentCUDAStream().stream());
    return {scores, k_mean, k_max};
}

std::vector<at::Tensor> moba_flashprefill_dynamic_pattern(
    const at::Tensor& q,
    const at::Tensor& k,
    double density,
    double alpha,
    bool auto_density,
    double softmax_scale,
    int chunk_size,
    int fine_block,
    int num_splits,
    int sink,
    int window,
    bool build_csc,
    bool exact_density) {
    at::cuda::CUDAGuard device_guard{q.device()};
    TORCH_CHECK(density > 0.0 && density <= 1.0, "density must be in (0, 1]");
    TORCH_CHECK(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
    TORCH_CHECK(chunk_size == 128 && fine_block == 32,
                "native FlashPrefill pattern requires chunk=128 and fine=32");
    TORCH_CHECK(q.size(0) % chunk_size == 0,
                "phase-1 native FlashPrefill pattern requires 128-aligned tokens");
    TORCH_CHECK(sink >= 0 && window >= 0, "sink and window must be non-negative");

    auto score_outputs = moba_flashprefill_coarse_scores(
        q, k, softmax_scale, fine_block, chunk_size, num_splits);
    auto scores = score_outputs[0];
    const int tokens = q.size(0);
    const int heads = q.size(1);
    const int blocks = tokens / chunk_size;
    auto float_opts = q.options().dtype(at::kFloat);
    auto int_opts = q.options().dtype(at::kInt);
    auto long_opts = q.options().dtype(at::kLong);
    auto byte_opts = q.options().dtype(at::kByte);
    auto row_max = at::empty({blocks, heads}, float_opts);
    auto histogram = auto_density
        ? at::zeros({kFlashPrefillHistogramBins}, int_opts)
        : at::empty({0}, int_opts);
    auto exact_meta = exact_density
        ? at::zeros({4}, int_opts)
        : at::empty({0}, int_opts);
    const float threshold_offset = static_cast<float>(std::log2(alpha));
    auto threshold = at::full({1}, threshold_offset, float_opts);
    auto block_mask = at::zeros({blocks, heads, blocks}, byte_opts);
    auto block_row_nnz = at::empty({blocks, heads}, int_opts);
    auto row_nnz = at::empty({tokens, heads}, int_opts);
    auto col_nnz = at::zeros({1, heads, blocks}, int_opts);
    auto col_offsets = at::empty({1, heads, blocks}, long_opts);

    FlashPrefillPatternParams params{};
    params.scores = scores.data_ptr<float>();
    params.row_max = row_max.data_ptr<float>();
    params.threshold = threshold.data_ptr<float>();
    params.histogram = auto_density ? histogram.data_ptr<int>() : nullptr;
    params.exact_meta = exact_density ? exact_meta.data_ptr<int>() : nullptr;
    params.block_mask = block_mask.data_ptr<uint8_t>();
    params.block_row_nnz = block_row_nnz.data_ptr<int>();
    params.row_nnz = row_nnz.data_ptr<int>();
    params.col_offsets = col_offsets.data_ptr<int64_t>();
    params.col_nnz = col_nnz.data_ptr<int>();
    params.blocks = blocks;
    params.heads = heads;
    params.tokens = tokens;
    params.chunk_size = chunk_size;
    params.sink = sink;
    params.window = window;
    params.density = density;
    params.auto_density = auto_density;
    params.exact_density = exact_density;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_flashprefill_pattern_select(params, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (!build_csc) {
        auto csc_rows = at::empty({0}, int_opts);
        return {col_offsets, col_nnz, csc_rows, row_nnz, block_mask,
                threshold, scores};
    }
    const int64_t total_nnz = col_nnz.sum().item<int64_t>();
    auto csc_rows = at::empty({total_nnz}, int_opts);
    params.csc_rows = csc_rows.data_ptr<int>();
    run_flashprefill_pattern_csc(params, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {col_offsets, col_nnz, csc_rows, row_nnz, block_mask,
            threshold, scores};
}

void run_dynamic_threshold_bf16_hdim128(
    Dynamic_threshold_params& params,
    bool build_csc,
    cudaStream_t stream) {
    run_dynamic_threshold<cutlass::bfloat16_t>(params, build_csc, stream);
}

void run_dynamic_threshold_compact_select_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    run_dynamic_threshold_compact_select<cutlass::bfloat16_t>(params, stream);
}

void run_dynamic_relative_threshold_compact_select_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    run_dynamic_relative_threshold_compact_select<cutlass::bfloat16_t>(params, stream);
}

void run_dynamic_relative_threshold_compact_select_from_row_max_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    run_dynamic_relative_threshold_compact_select_from_row_max<cutlass::bfloat16_t>(
        params, stream);
}

void run_dynamic_relative_threshold_probe_from_row_max_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    run_dynamic_relative_threshold_probe_from_row_max<cutlass::bfloat16_t>(
        params, stream);
}

void run_dynamic_relative_threshold_multi_probe_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    run_dynamic_relative_threshold_multi_probe<cutlass::bfloat16_t>(params, stream);
}

void run_dynamic_threshold_compact_csc(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    run_dynamic_threshold_compact_csc_impl(params, stream);
}

namespace {

void check_dynamic_inputs(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& thresholds,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int chunk_size,
    bool causal) {
    TORCH_CHECK(q.is_cuda() && km.is_cuda() && thresholds.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && km.scalar_type() == at::kBFloat16,
                "CUDA dynamic selector supports BF16 only");
    TORCH_CHECK(thresholds.scalar_type() == at::kFloat && thresholds.is_contiguous(),
                "thresholds must be contiguous float32");
    TORCH_CHECK(q.dim() == 3 && km.dim() == 3, "q and km must be rank 3");
    TORCH_CHECK(q.size(2) == 128 && km.size(2) == 128, "CUDA dynamic selector supports head_dim=128 only");
    TORCH_CHECK(q.stride(2) == 1 && km.stride(2) == 1, "q and km last dimension must be contiguous");
    TORCH_CHECK(q.size(1) % km.size(1) == 0, "KV heads must divide Q heads");
    TORCH_CHECK(thresholds.numel() == q.size(1), "one threshold per Q head is required");
    TORCH_CHECK(causal, "CUDA dynamic selector supports causal=True only");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda() && cu_seqlens_km.is_cuda(),
                "cu_seqlens tensors must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt &&
                cu_seqlens_k.scalar_type() == at::kInt &&
                cu_seqlens_km.scalar_type() == at::kInt,
                "cu_seqlens tensors must be int32");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous() && cu_seqlens_km.is_contiguous(),
                "cu_seqlens tensors must be contiguous");
}

}  // namespace

std::vector<at::Tensor> moba_dynamic_threshold(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& thresholds,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    int capacity,
    bool causal,
    bool build_csc) {
    at::cuda::CUDAGuard device_guard{q.device()};
    check_dynamic_inputs(q, km, thresholds, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km, chunk_size, causal);
    TORCH_CHECK(capacity > 0, "capacity must be positive");

    const int batch = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    TORCH_CHECK(batch > 0, "batch must be positive");
    TORCH_CHECK(cu_seqlens_k.numel() == batch + 1 && cu_seqlens_km.numel() == batch + 1,
                "cu_seqlens lengths must agree");

    auto int_opts = q.options().dtype(at::kInt);
    auto long_opts = q.options().dtype(at::kLong);
    auto row_indices = at::empty({total_q, q_heads, capacity}, int_opts);
    auto row_nnz = at::empty({total_q, q_heads}, int_opts);
    auto row_overflow = at::empty({total_q, q_heads}, int_opts);
    auto col_offsets = at::empty({batch, q_heads, max_km}, long_opts);
    auto col_nnz = at::zeros({batch, q_heads, max_km}, int_opts);
    auto col_cursor = build_csc ? at::zeros_like(col_nnz) : at::empty({0}, int_opts);
    auto csc_indices = build_csc
        ? at::empty({static_cast<int64_t>(total_q) * q_heads * capacity}, int_opts)
        : at::empty({0}, int_opts);

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.threshold_ptr = thresholds.data_ptr<float>();
    params.row_indices_ptr = row_indices.data_ptr<int>();
    params.row_bitmask_ptr = nullptr;
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.row_overflow_ptr = row_overflow.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.col_cursor_ptr = build_csc ? col_cursor.data_ptr<int>() : nullptr;
    params.csc_indices_ptr = build_csc ? csc_indices.data_ptr<int>() : nullptr;
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = total_q;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.capacity = capacity;
    params.bitmask_words = 0;
    params.sink = 1;
    params.window = 2;

    run_dynamic_threshold_bf16_hdim128(
        params, build_csc, at::cuda::getCurrentCUDAStream().stream());
    return {col_offsets, col_nnz, csc_indices, row_indices, row_nnz, row_overflow};
}

std::vector<at::Tensor> moba_dynamic_threshold_compact(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& thresholds,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    bool causal,
    bool build_csc) {
    at::cuda::CUDAGuard device_guard{q.device()};
    check_dynamic_inputs(q, km, thresholds, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km, chunk_size, causal);

    const int batch = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    const int bitmask_words = (max_km + 31) / 32;
    TORCH_CHECK(batch > 0, "batch must be positive");
    TORCH_CHECK(cu_seqlens_k.numel() == batch + 1 && cu_seqlens_km.numel() == batch + 1,
                "cu_seqlens lengths must agree");

    auto int_opts = q.options().dtype(at::kInt);
    auto long_opts = q.options().dtype(at::kLong);
    auto row_bitmask = at::zeros({total_q, q_heads, bitmask_words}, int_opts);
    auto row_nnz = at::empty({total_q, q_heads}, int_opts);
    auto row_overflow = at::zeros({total_q, q_heads}, int_opts);
    auto col_offsets = at::empty({batch, q_heads, max_km}, long_opts);
    auto col_nnz = at::zeros({batch, q_heads, max_km}, int_opts);

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.threshold_ptr = thresholds.data_ptr<float>();
    params.row_indices_ptr = nullptr;
    params.row_bitmask_ptr = reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>());
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.row_overflow_ptr = row_overflow.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = total_q;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.capacity = 0;
    params.bitmask_words = bitmask_words;
    params.sink = 1;
    params.window = 2;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_dynamic_threshold_compact_select_bf16_hdim128(params, stream);

    if (!build_csc) {
        auto csc_indices = at::empty({0}, int_opts);
        return {col_offsets, col_nnz, csc_indices, row_bitmask, row_nnz,
                row_overflow};
    }

    // Exact-sized output requires observing the data-dependent NNZ once.  A
    // later CUDA-graph path will use a preallocated density bucket instead.
    const int64_t total_nnz = col_nnz.sum().item<int64_t>();
    auto col_cursor = at::zeros_like(col_nnz);
    auto csc_indices = at::empty({total_nnz}, int_opts);
    params.col_cursor_ptr = col_cursor.data_ptr<int>();
    params.csc_indices_ptr = csc_indices.data_ptr<int>();
    run_dynamic_threshold_compact_csc(params, stream);

    return {col_offsets, col_nnz, csc_indices, row_bitmask, row_nnz,
            row_overflow};
}

at::Tensor moba_dynamic_relative_threshold_probe_from_row_max(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& row_max,
    const at::Tensor& alphas,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    double softmax_scale,
    int sink,
    int window,
    bool causal) {
    at::cuda::CUDAGuard device_guard{q.device()};
    TORCH_CHECK(row_max.is_cuda() && row_max.device() == q.device(),
                "row_max must be on the same CUDA device as q");
    TORCH_CHECK(row_max.scalar_type() == at::kFloat && row_max.is_contiguous(),
                "row_max must be contiguous float32");
    TORCH_CHECK(row_max.dim() == 2 && row_max.size(0) == q.size(0) &&
                row_max.size(1) == q.size(1), "row_max shape mismatch");
    TORCH_CHECK(alphas.is_cuda() && alphas.device() == q.device(),
                "alphas must be on the same CUDA device as q");
    TORCH_CHECK(alphas.scalar_type() == at::kFloat && alphas.is_contiguous(),
                "alphas must be contiguous float32");
    TORCH_CHECK(alphas.dim() == 1 &&
                (alphas.numel() == 1 || alphas.numel() == 4 ||
                 alphas.numel() == 8 || alphas.numel() == 16),
                "alphas must contain 1, 4, 8, or 16 values");
    TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");
    TORCH_CHECK(sink >= 0 && window >= 1, "invalid forced-anchor configuration");
    auto validation_thresholds = at::zeros(
        {q.size(1)}, q.options().dtype(at::kFloat));
    check_dynamic_inputs(q, km, validation_thresholds, cu_seqlens_q,
                         cu_seqlens_k, cu_seqlens_km, chunk_size, causal);

    const int batch = cu_seqlens_q.numel() - 1;
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    auto offsets = alphas.log().div(softmax_scale).contiguous();
    const int64_t ctas = static_cast<int64_t>(
        (max_seqlen_q + 63) / 64) * batch * q_heads;
    auto block_counts = at::zeros(
        {ctas, alphas.numel()}, q.options().dtype(at::kLong));

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.row_max_ptr = row_max.data_ptr<float>();
    params.probe_threshold_ptr = offsets.data_ptr<float>();
    params.probe_nnz_ptr = block_counts.data_ptr<int64_t>();
    params.num_probes = alphas.numel();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = q.size(0);
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.sink = sink;
    params.window = window;
    run_dynamic_relative_threshold_probe_from_row_max_bf16_hdim128(
        params, at::cuda::getCurrentCUDAStream().stream());
    return block_counts.sum(0, false, at::kLong);
}

namespace {

__global__ void interpolate_cap_alpha_kernel(
    const int64_t* counts,
    const float* alphas,
    int num_probes,
    double target_count,
    float interpolation_safety,
    float inv_softmax_scale,
    float* selected_alpha,
    float* selected_relative_log_threshold,
    int* selected_index) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int safe_index = -1;
    for (int index = 0; index < num_probes; ++index) {
        if (static_cast<double>(counts[index]) <= target_count) {
            safe_index = index;
            break;
        }
    }

    // If the configured probe bracket did not reach a safe endpoint, alpha=1
    // is the minimum non-forced pattern and is safer than accepting the last
    // known-unsafe grid point.  Normal calibrated brackets take the branch
    // below and preserve the existing log-space interpolation semantics.
    float alpha = safe_index >= 0 ? alphas[safe_index] : 1.0f;
    if (safe_index > 0) {
        const double low_count = static_cast<double>(counts[safe_index - 1]);
        const double high_count = static_cast<double>(counts[safe_index]);
        if (low_count > target_count && low_count > high_count) {
            double fraction =
                (low_count - target_count) / (low_count - high_count);
            fraction = fmin(1.0, fmax(0.0, fraction));
            const float low_alpha = alphas[safe_index - 1];
            const float predicted_alpha = expf(
                logf(low_alpha) + static_cast<float>(fraction) *
                (logf(alpha) - logf(low_alpha)));
            alpha = fminf(alpha, predicted_alpha * interpolation_safety);
        }
    }
    alpha = fminf(1.0f, fmaxf(alpha, 1.0e-8f));
    selected_alpha[0] = alpha;
    selected_relative_log_threshold[0] = logf(alpha) * inv_softmax_scale;
    selected_index[0] = safe_index;
}

__global__ void set_cap_refine_flag_kernel(
    const int64_t* candidate_count,
    double target_count,
    int* refine_flag) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        refine_flag[0] =
            static_cast<double>(candidate_count[0]) > target_count ? 1 : 0;
    }
}

__global__ void conditional_clear_pattern_kernel(
    const int* refine_flag,
    uint32_t* row_bitmask,
    int64_t row_bitmask_count,
    int* row_overflow,
    int64_t row_count,
    int* col_nnz,
    int64_t col_count) {
    if (refine_flag[0] == 0) return;
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < row_bitmask_count) row_bitmask[index] = 0;
    if (index < row_count) row_overflow[index] = 0;
    if (index < col_count) col_nnz[index] = 0;
}

std::vector<at::Tensor> materialize_from_row_max_impl(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& row_max,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    double alpha,
    const at::Tensor& device_relative_log_threshold,
    double softmax_scale,
    int sink,
    int window,
    bool causal) {
    at::cuda::CUDAGuard device_guard{q.device()};
    const bool use_device_threshold = device_relative_log_threshold.defined();
    if (use_device_threshold) {
        TORCH_CHECK(
            device_relative_log_threshold.is_cuda() &&
                device_relative_log_threshold.device() == q.device(),
            "device relative log threshold must be on q's CUDA device");
        TORCH_CHECK(
            device_relative_log_threshold.scalar_type() == at::kFloat &&
                device_relative_log_threshold.is_contiguous() &&
                device_relative_log_threshold.numel() == 1,
            "device relative log threshold must be one contiguous float32 value");
    } else {
        TORCH_CHECK(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
    }
    TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");
    TORCH_CHECK(row_max.is_cuda() && row_max.device() == q.device() &&
                row_max.scalar_type() == at::kFloat && row_max.is_contiguous(),
                "row_max must be contiguous float32 on q's CUDA device");
    TORCH_CHECK(row_max.dim() == 2 && row_max.size(0) == q.size(0) &&
                row_max.size(1) == q.size(1), "row_max shape mismatch");
    auto thresholds = at::zeros({q.size(1)}, q.options().dtype(at::kFloat));
    check_dynamic_inputs(q, km, thresholds, cu_seqlens_q, cu_seqlens_k,
                         cu_seqlens_km, chunk_size, causal);

    const int batch = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    const int bitmask_words = (max_km + 31) / 32;
    auto int_opts = q.options().dtype(at::kInt);
    auto long_opts = q.options().dtype(at::kLong);
    auto row_bitmask = at::zeros({total_q, q_heads, bitmask_words}, int_opts);
    auto row_nnz = at::empty({total_q, q_heads}, int_opts);
    auto row_overflow = at::zeros({total_q, q_heads}, int_opts);
    auto col_offsets = at::empty({batch, q_heads, max_km}, long_opts);
    auto col_nnz = at::zeros({batch, q_heads, max_km}, int_opts);

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.threshold_ptr = thresholds.data_ptr<float>();
    params.row_max_ptr = row_max.data_ptr<float>();
    params.relative_log_threshold =
        use_device_threshold
        ? 0.0f
        : static_cast<float>(std::log(alpha) / softmax_scale);
    params.relative_log_threshold_ptr = use_device_threshold
        ? device_relative_log_threshold.data_ptr<float>()
        : nullptr;
    params.use_row_relative_threshold = true;
    params.row_bitmask_ptr =
        reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>());
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.row_overflow_ptr = row_overflow.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = total_q;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.bitmask_words = bitmask_words;
    params.sink = sink;
    params.window = window;
    run_dynamic_relative_threshold_compact_select_from_row_max_bf16_hdim128(
        params, at::cuda::getCurrentCUDAStream().stream());
    auto csc_indices = at::empty({0}, int_opts);
    return {col_offsets, col_nnz, csc_indices, row_bitmask, row_nnz, row_overflow};
}

}  // namespace

std::vector<at::Tensor> moba_dynamic_relative_threshold_interpolate_device(
    const at::Tensor& probe_counts,
    const at::Tensor& probe_alphas,
    double target_count,
    double interpolation_safety,
    double softmax_scale) {
    TORCH_CHECK(probe_counts.is_cuda() && probe_alphas.is_cuda(),
                "probe counts and alphas must be CUDA tensors");
    TORCH_CHECK(probe_counts.device() == probe_alphas.device(),
                "probe counts and alphas must share a CUDA device");
    TORCH_CHECK(probe_counts.scalar_type() == at::kLong &&
                probe_counts.is_contiguous(),
                "probe counts must be contiguous int64");
    TORCH_CHECK(probe_alphas.scalar_type() == at::kFloat &&
                probe_alphas.is_contiguous(),
                "probe alphas must be contiguous float32");
    TORCH_CHECK(probe_counts.dim() == 1 && probe_alphas.dim() == 1 &&
                probe_counts.numel() == probe_alphas.numel(),
                "probe counts and alphas must be equal-length vectors");
    TORCH_CHECK(probe_counts.numel() == 1 || probe_counts.numel() == 4 ||
                probe_counts.numel() == 8 || probe_counts.numel() == 16,
                "device interpolation supports 1, 4, 8, or 16 probes");
    TORCH_CHECK(target_count > 0.0, "target count must be positive");
    TORCH_CHECK(interpolation_safety >= 1.0 && interpolation_safety <= 1.2,
                "interpolation safety must be in [1.0, 1.2]");
    TORCH_CHECK(softmax_scale > 0.0, "softmax scale must be positive");

    at::cuda::CUDAGuard device_guard{probe_counts.device()};
    auto float_opts = probe_alphas.options().dtype(at::kFloat);
    auto int_opts = probe_alphas.options().dtype(at::kInt);
    auto selected_alpha = at::empty({1}, float_opts);
    auto relative_log_threshold = at::empty({1}, float_opts);
    auto selected_index = at::empty({1}, int_opts);
    interpolate_cap_alpha_kernel<<<
        1, 1, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        probe_counts.data_ptr<int64_t>(), probe_alphas.data_ptr<float>(),
        static_cast<int>(probe_counts.numel()), target_count,
        static_cast<float>(interpolation_safety),
        static_cast<float>(1.0 / softmax_scale),
        selected_alpha.data_ptr<float>(),
        relative_log_threshold.data_ptr<float>(),
        selected_index.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {selected_alpha, relative_log_threshold, selected_index};
}

std::vector<at::Tensor>
moba_dynamic_relative_threshold_refine_in_place_device(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& row_max,
    const at::Tensor& row_bitmask,
    const at::Tensor& row_nnz,
    const at::Tensor& row_overflow,
    const at::Tensor& col_offsets,
    const at::Tensor& col_nnz,
    const at::Tensor& probe_alphas,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    double target_count,
    double interpolation_safety,
    double softmax_scale,
    int sink,
    int window,
    bool causal) {
    at::cuda::CUDAGuard device_guard{q.device()};
    TORCH_CHECK(row_max.is_cuda() && row_max.device() == q.device() &&
                row_max.scalar_type() == at::kFloat && row_max.is_contiguous(),
                "row_max must be contiguous float32 on q's CUDA device");
    TORCH_CHECK(row_bitmask.is_cuda() && row_bitmask.device() == q.device() &&
                row_bitmask.scalar_type() == at::kInt && row_bitmask.is_contiguous(),
                "row_bitmask must be contiguous int32 on q's CUDA device");
    TORCH_CHECK(row_nnz.is_cuda() && row_nnz.device() == q.device() &&
                row_nnz.scalar_type() == at::kInt && row_nnz.is_contiguous(),
                "row_nnz must be contiguous int32 on q's CUDA device");
    TORCH_CHECK(row_overflow.is_cuda() && row_overflow.device() == q.device() &&
                row_overflow.scalar_type() == at::kInt && row_overflow.is_contiguous(),
                "row_overflow must be contiguous int32 on q's CUDA device");
    TORCH_CHECK(col_offsets.is_cuda() && col_offsets.device() == q.device() &&
                col_offsets.scalar_type() == at::kLong && col_offsets.is_contiguous(),
                "col_offsets must be contiguous int64 on q's CUDA device");
    TORCH_CHECK(col_nnz.is_cuda() && col_nnz.device() == q.device() &&
                col_nnz.scalar_type() == at::kInt && col_nnz.is_contiguous(),
                "col_nnz must be contiguous int32 on q's CUDA device");
    TORCH_CHECK(probe_alphas.is_cuda() && probe_alphas.device() == q.device() &&
                probe_alphas.scalar_type() == at::kFloat &&
                probe_alphas.is_contiguous(),
                "probe_alphas must be contiguous float32 on q's CUDA device");
    TORCH_CHECK(probe_alphas.dim() == 1 &&
                (probe_alphas.numel() == 1 || probe_alphas.numel() == 4 ||
                 probe_alphas.numel() == 8 || probe_alphas.numel() == 16),
                "probe_alphas must contain 1, 4, 8, or 16 values");
    TORCH_CHECK(target_count > 0.0, "target count must be positive");
    TORCH_CHECK(interpolation_safety >= 1.0 && interpolation_safety <= 1.2,
                "interpolation safety must be in [1.0, 1.2]");
    TORCH_CHECK(softmax_scale > 0.0, "softmax scale must be positive");

    auto validation_thresholds = at::zeros(
        {q.size(1)}, q.options().dtype(at::kFloat));
    check_dynamic_inputs(q, km, validation_thresholds, cu_seqlens_q,
                         cu_seqlens_k, cu_seqlens_km, chunk_size, causal);

    const int batch = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    const int bitmask_words = (max_km + 31) / 32;
    TORCH_CHECK(row_max.numel() == static_cast<int64_t>(total_q) * q_heads,
                "row_max shape mismatch");
    TORCH_CHECK(row_nnz.numel() == static_cast<int64_t>(total_q) * q_heads &&
                row_overflow.numel() == row_nnz.numel(),
                "row metadata shape mismatch");
    TORCH_CHECK(row_bitmask.numel() ==
                    static_cast<int64_t>(total_q) * q_heads * bitmask_words,
                "row_bitmask shape mismatch");
    TORCH_CHECK(col_nnz.numel() ==
                    static_cast<int64_t>(batch) * q_heads * max_km &&
                col_offsets.numel() == col_nnz.numel(),
                "column metadata shape mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto candidate_count = row_nnz.sum(at::kLong).reshape({1});
    auto refine_flag = at::empty({1}, q.options().dtype(at::kInt));
    set_cap_refine_flag_kernel<<<1, 1, 0, stream>>>(
        candidate_count.data_ptr<int64_t>(), target_count,
        refine_flag.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto probe_offsets = probe_alphas.log().div(softmax_scale).contiguous();
    const int64_t ctas = static_cast<int64_t>(
        (max_seqlen_q + 63) / 64) * batch * q_heads;
    auto block_counts = at::zeros(
        {ctas, probe_alphas.numel()}, q.options().dtype(at::kLong));

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.row_max_ptr = row_max.data_ptr<float>();
    params.probe_threshold_ptr = probe_offsets.data_ptr<float>();
    params.probe_nnz_ptr = block_counts.data_ptr<int64_t>();
    params.num_probes = probe_alphas.numel();
    params.execute_flag_ptr = refine_flag.data_ptr<int>();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = total_q;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.sink = sink;
    params.window = window;
    run_dynamic_relative_threshold_probe_from_row_max_bf16_hdim128(
        params, stream);
    auto probe_counts = block_counts.sum(0, false, at::kLong).contiguous();

    auto selected_alpha = at::empty({1}, q.options().dtype(at::kFloat));
    auto relative_log_threshold = at::empty(
        {1}, q.options().dtype(at::kFloat));
    auto selected_index = at::empty({1}, q.options().dtype(at::kInt));
    interpolate_cap_alpha_kernel<<<1, 1, 0, stream>>>(
        probe_counts.data_ptr<int64_t>(), probe_alphas.data_ptr<float>(),
        static_cast<int>(probe_alphas.numel()), target_count,
        static_cast<float>(interpolation_safety),
        static_cast<float>(1.0 / softmax_scale),
        selected_alpha.data_ptr<float>(),
        relative_log_threshold.data_ptr<float>(),
        selected_index.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t clear_count = std::max(
        row_bitmask.numel(), std::max(row_nnz.numel(), col_nnz.numel()));
    constexpr int kClearThreads = 256;
    conditional_clear_pattern_kernel<<<
        (clear_count + kClearThreads - 1) / kClearThreads,
        kClearThreads, 0, stream>>>(
        refine_flag.data_ptr<int>(),
        reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>()),
        row_bitmask.numel(), row_overflow.data_ptr<int>(), row_nnz.numel(),
        col_nnz.data_ptr<int>(), col_nnz.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    params.threshold_ptr = validation_thresholds.data_ptr<float>();
    params.probe_threshold_ptr = nullptr;
    params.probe_nnz_ptr = nullptr;
    params.num_probes = 0;
    params.relative_log_threshold_ptr = relative_log_threshold.data_ptr<float>();
    params.use_row_relative_threshold = true;
    params.row_bitmask_ptr =
        reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>());
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.row_overflow_ptr = row_overflow.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.bitmask_words = bitmask_words;
    run_dynamic_relative_threshold_compact_select_from_row_max_bf16_hdim128(
        params, stream);

    return {candidate_count, selected_alpha, relative_log_threshold,
            selected_index, refine_flag, probe_counts};
}

std::vector<at::Tensor> moba_dynamic_relative_threshold_materialize_from_row_max(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& row_max,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    double alpha,
    double softmax_scale,
    int sink,
    int window,
    bool causal) {
    return materialize_from_row_max_impl(
        q, km, row_max, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km,
        max_seqlen_q, max_seqlen_k, chunk_size, alpha, at::Tensor(),
        softmax_scale, sink, window, causal);
}

std::vector<at::Tensor>
moba_dynamic_relative_threshold_materialize_from_row_max_device(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& row_max,
    const at::Tensor& device_relative_log_threshold,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    double softmax_scale,
    int sink,
    int window,
    bool causal) {
    return materialize_from_row_max_impl(
        q, km, row_max, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km,
        max_seqlen_q, max_seqlen_k, chunk_size, 1.0,
        device_relative_log_threshold, softmax_scale, sink, window, causal);
}

std::vector<at::Tensor> moba_dynamic_relative_threshold_compact(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    double alpha,
    double softmax_scale,
    int sink,
    int window,
    bool causal,
    int min_blocks_per_row,
    bool build_csc) {
    at::cuda::CUDAGuard device_guard{q.device()};
    TORCH_CHECK(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
    TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");
    TORCH_CHECK(sink >= 0 && window >= 1, "invalid forced-anchor configuration");
    TORCH_CHECK(min_blocks_per_row >= 0 && min_blocks_per_row <= 8,
                "min_blocks_per_row must be in [0, 8]");
    auto thresholds = at::zeros({q.size(1)}, q.options().dtype(at::kFloat));
    check_dynamic_inputs(q, km, thresholds, cu_seqlens_q, cu_seqlens_k,
                         cu_seqlens_km, chunk_size, causal);

    const int batch = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    const int bitmask_words = (max_km + 31) / 32;
    auto int_opts = q.options().dtype(at::kInt);
    auto long_opts = q.options().dtype(at::kLong);
    auto row_bitmask = at::zeros({total_q, q_heads, bitmask_words}, int_opts);
    auto row_nnz = at::empty({total_q, q_heads}, int_opts);
    auto row_overflow = at::zeros({total_q, q_heads}, int_opts);
    auto row_max = at::empty({total_q, q_heads}, q.options().dtype(at::kFloat));
    auto col_offsets = at::empty({batch, q_heads, max_km}, long_opts);
    auto col_nnz = at::zeros({batch, q_heads, max_km}, int_opts);

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.threshold_ptr = thresholds.data_ptr<float>();
    params.row_max_ptr = row_max.data_ptr<float>();
    // GEMM produces unscaled logits.  Comparing exp(scale*(s-max)) >= alpha
    // is equivalent to s >= max + log(alpha)/scale.
    params.relative_log_threshold = static_cast<float>(std::log(alpha) / softmax_scale);
    params.use_row_relative_threshold = true;
    params.row_indices_ptr = nullptr;
    params.row_bitmask_ptr = reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>());
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.row_overflow_ptr = row_overflow.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = total_q;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.capacity = 0;
    params.bitmask_words = bitmask_words;
    params.sink = sink;
    params.window = window;
    params.min_blocks_per_row = min_blocks_per_row;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_dynamic_relative_threshold_compact_select_bf16_hdim128(params, stream);
    if (!build_csc) {
        auto csc_indices = at::empty({0}, int_opts);
        return {col_offsets, col_nnz, csc_indices, row_bitmask, row_nnz,
                row_overflow, row_max};
    }
    const int64_t total_nnz = col_nnz.sum().item<int64_t>();
    auto col_cursor = at::zeros_like(col_nnz);
    auto csc_indices = at::empty({total_nnz}, int_opts);
    params.col_cursor_ptr = col_cursor.data_ptr<int>();
    params.csc_indices_ptr = csc_indices.data_ptr<int>();
    run_dynamic_threshold_compact_csc(params, stream);
    return {col_offsets, col_nnz, csc_indices, row_bitmask, row_nnz,
            row_overflow, row_max};
}

at::Tensor moba_dynamic_relative_threshold_multi_probe(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& alphas,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    double softmax_scale,
    int sink,
    int window,
    bool causal) {
    at::cuda::CUDAGuard device_guard{q.device()};
    TORCH_CHECK(alphas.is_cuda() && alphas.device() == q.device(),
                "alphas must be on the same CUDA device as q");
    TORCH_CHECK(alphas.scalar_type() == at::kFloat && alphas.is_contiguous(),
                "alphas must be contiguous float32");
    TORCH_CHECK(alphas.dim() == 1 &&
                (alphas.numel() == 1 || alphas.numel() == 4 ||
                 alphas.numel() == 8 || alphas.numel() == 16),
                "alphas must contain 1, 4, 8, or 16 values");
    TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");
    TORCH_CHECK(sink >= 0 && window >= 1, "invalid forced-anchor configuration");
    auto validation_thresholds = at::zeros(
        {q.size(1)}, q.options().dtype(at::kFloat));
    check_dynamic_inputs(q, km, validation_thresholds, cu_seqlens_q, cu_seqlens_k,
                         cu_seqlens_km, chunk_size, causal);

    const int batch = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    auto row_max = at::empty({total_q, q_heads}, q.options().dtype(at::kFloat));
    auto offsets = alphas.log().div(softmax_scale).contiguous();
    // Dynamic_threshold_traits uses a 64-row Q tile.
    const int64_t ctas = static_cast<int64_t>(
        (max_seqlen_q + 63) / 64) * batch * q_heads;
    auto block_counts = at::zeros(
        {ctas, alphas.numel()}, q.options().dtype(at::kLong));

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.row_max_ptr = row_max.data_ptr<float>();
    params.probe_threshold_ptr = offsets.data_ptr<float>();
    params.probe_nnz_ptr = block_counts.data_ptr<int64_t>();
    params.num_probes = alphas.numel();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = total_q;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.sink = sink;
    params.window = window;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_dynamic_relative_threshold_multi_probe_bf16_hdim128(params, stream);
    return block_counts.sum(0, false, at::kLong);
}

// Per-local-Q-head variant of the max-relative selector.  relative_offsets
// contains log(alpha_h) / softmax_scale, so the hot kernel only performs one
// coalesced float load per query row and does not evaluate log().  Giving each
// head its own alpha lets the caller equalise aggregate head budgets while Q
// rows remain genuinely dynamic.
std::vector<at::Tensor> moba_dynamic_relative_head_threshold_compact(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& relative_offsets,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    int sink,
    int window,
    bool causal,
    bool build_csc) {
    at::cuda::CUDAGuard device_guard{q.device()};
    TORCH_CHECK(relative_offsets.is_cuda() && relative_offsets.device() == q.device(),
                "relative_offsets must be on the same CUDA device as q");
    TORCH_CHECK(relative_offsets.scalar_type() == at::kFloat && relative_offsets.is_contiguous(),
                "relative_offsets must be contiguous float32");
    TORCH_CHECK(relative_offsets.dim() == 1 && relative_offsets.numel() == q.size(1),
                "relative_offsets must contain one value per local Q head");
    TORCH_CHECK(sink >= 0 && window >= 1, "invalid forced-anchor configuration");
    check_dynamic_inputs(q, km, relative_offsets, cu_seqlens_q, cu_seqlens_k,
                         cu_seqlens_km, chunk_size, causal);

    const int batch = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    const int bitmask_words = (max_km + 31) / 32;
    auto int_opts = q.options().dtype(at::kInt);
    auto long_opts = q.options().dtype(at::kLong);
    auto row_bitmask = at::zeros({total_q, q_heads, bitmask_words}, int_opts);
    auto row_nnz = at::empty({total_q, q_heads}, int_opts);
    auto row_overflow = at::zeros({total_q, q_heads}, int_opts);
    auto row_max = at::empty({total_q, q_heads}, q.options().dtype(at::kFloat));
    auto col_offsets = at::empty({batch, q_heads, max_km}, long_opts);
    auto col_nnz = at::zeros({batch, q_heads, max_km}, int_opts);

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.threshold_ptr = relative_offsets.data_ptr<float>();
    params.row_max_ptr = row_max.data_ptr<float>();
    params.relative_log_threshold = 0.0f;
    params.use_row_relative_threshold = true;
    params.use_head_relative_threshold = true;
    params.row_indices_ptr = nullptr;
    params.row_bitmask_ptr = reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>());
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.row_overflow_ptr = row_overflow.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = total_q;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.capacity = 0;
    params.bitmask_words = bitmask_words;
    params.sink = sink;
    params.window = window;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_dynamic_relative_threshold_compact_select_bf16_hdim128(params, stream);
    if (!build_csc) {
        auto csc_indices = at::empty({0}, int_opts);
        return {col_offsets, col_nnz, csc_indices, row_bitmask, row_nnz, row_overflow};
    }
    const int64_t total_nnz = col_nnz.sum().item<int64_t>();
    auto col_cursor = at::zeros_like(col_nnz);
    auto csc_indices = at::empty({total_nnz}, int_opts);
    params.col_cursor_ptr = col_cursor.data_ptr<int>();
    params.csc_indices_ptr = csc_indices.data_ptr<int>();
    run_dynamic_threshold_compact_csc(params, stream);
    return {col_offsets, col_nnz, csc_indices, row_bitmask, row_nnz, row_overflow};
}

// CUDA-graph-safe relative selector.  The compact API above sizes its CSC
// with col_nnz.sum().item(), which synchronizes the host.  Decode has one Q
// row, so reserving the exact mathematical upper bound is small even at 128K.
// Consumers use col_offsets/col_nnz and ignore the unspecified padded tail.
std::vector<at::Tensor> moba_dynamic_relative_threshold_padded(
    const at::Tensor& q,
    const at::Tensor& km,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size,
    double alpha,
    double softmax_scale,
    int sink,
    int window,
    bool causal) {
    at::cuda::CUDAGuard device_guard{q.device()};
    TORCH_CHECK(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
    TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");
    TORCH_CHECK(sink >= 0 && window >= 1, "invalid forced-anchor configuration");
    auto thresholds = at::zeros({q.size(1)}, q.options().dtype(at::kFloat));
    check_dynamic_inputs(q, km, thresholds, cu_seqlens_q, cu_seqlens_k,
                         cu_seqlens_km, chunk_size, causal);

    const int batch = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int q_heads = q.size(1);
    const int kv_heads = km.size(1);
    const int max_km = (max_seqlen_k + chunk_size - 1) / chunk_size;
    const int bitmask_words = (max_km + 31) / 32;
    auto int_opts = q.options().dtype(at::kInt);
    auto long_opts = q.options().dtype(at::kLong);
    auto row_bitmask = at::zeros({total_q, q_heads, bitmask_words}, int_opts);
    auto row_nnz = at::empty({total_q, q_heads}, int_opts);
    auto row_overflow = at::zeros({total_q, q_heads}, int_opts);
    auto row_max = at::empty({total_q, q_heads}, q.options().dtype(at::kFloat));
    auto col_offsets = at::empty({batch, q_heads, max_km}, long_opts);
    auto col_nnz = at::zeros({batch, q_heads, max_km}, int_opts);
    auto col_cursor = at::zeros_like(col_nnz);
    const int64_t padded_capacity =
        static_cast<int64_t>(total_q) * q_heads * max_km;
    auto csc_indices = at::empty({padded_capacity}, int_opts);

    Dynamic_threshold_params params{};
    params.q_ptr = q.data_ptr();
    params.km_ptr = km.data_ptr();
    params.threshold_ptr = thresholds.data_ptr<float>();
    params.row_max_ptr = row_max.data_ptr<float>();
    params.relative_log_threshold =
        static_cast<float>(std::log(alpha) / softmax_scale);
    params.use_row_relative_threshold = true;
    params.row_indices_ptr = nullptr;
    params.row_bitmask_ptr =
        reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>());
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.row_overflow_ptr = row_overflow.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.col_cursor_ptr = col_cursor.data_ptr<int>();
    params.csc_indices_ptr = csc_indices.data_ptr<int>();
    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);
    params.km_row_stride = km.stride(0);
    params.km_head_stride = km.stride(1);
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.h_k = kv_heads;
    params.h_h_k_ratio = q_heads / kv_heads;
    params.d = 128;
    params.total_q = total_q;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.capacity = 0;
    params.bitmask_words = bitmask_words;
    params.sink = sink;
    params.window = window;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_dynamic_relative_threshold_compact_select_bf16_hdim128(params, stream);
    run_dynamic_threshold_compact_csc(params, stream);
    return {col_offsets, col_nnz, csc_indices, row_bitmask, row_nnz, row_overflow};
}

std::vector<at::Tensor> moba_compact_csc_from_bitmask(
    const at::Tensor& row_bitmask,
    const at::Tensor& row_nnz,
    const at::Tensor& col_nnz,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size) {
    at::cuda::CUDAGuard device_guard{row_bitmask.device()};
    TORCH_CHECK(row_bitmask.is_cuda() && row_nnz.is_cuda() && col_nnz.is_cuda(),
                "compact CSC inputs must be CUDA tensors");
    TORCH_CHECK(row_bitmask.scalar_type() == at::kInt &&
                row_nnz.scalar_type() == at::kInt && col_nnz.scalar_type() == at::kInt,
                "compact CSC inputs must be int32");
    TORCH_CHECK(row_bitmask.dim() == 3 && row_nnz.dim() == 2 && col_nnz.dim() == 3,
                "invalid compact CSC input ranks");
    const int batch = col_nnz.size(0);
    const int q_heads = col_nnz.size(1);
    const int max_km = col_nnz.size(2);
    TORCH_CHECK(row_nnz.size(0) == row_bitmask.size(0) &&
                row_nnz.size(1) == q_heads, "row metadata shape mismatch");
    auto long_opts = row_nnz.options().dtype(at::kLong);
    auto col_offsets = at::empty({batch, q_heads, max_km}, long_opts);
    const int64_t total_nnz = col_nnz.sum().item<int64_t>();
    auto col_cursor = at::zeros_like(col_nnz);
    auto csc_indices = at::empty({total_nnz}, row_nnz.options());
    Dynamic_threshold_params params{};
    params.row_bitmask_ptr = reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>());
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.col_cursor_ptr = col_cursor.data_ptr<int>();
    params.csc_indices_ptr = csc_indices.data_ptr<int>();
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.total_q = row_nnz.size(0);
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.bitmask_words = row_bitmask.size(2);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_dynamic_threshold_compact_csc(params, stream);
    return {col_offsets, csc_indices};
}

// Build CSC and its reverse Q-CSR in the same row-centric scatter.  The old
// path scanned the completed CSC a second time and used another atomic scatter
// to recover this exact mapping.  Here each source row already owns a stable
// row-local slot, while atomicAdd returns the final CSC position for free.
std::vector<at::Tensor> moba_compact_csc_qcsr_from_bitmask(
    const at::Tensor& row_bitmask,
    const at::Tensor& row_nnz,
    const at::Tensor& col_nnz,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_km,
    int max_seqlen_q,
    int max_seqlen_k,
    int chunk_size) {
    at::cuda::CUDAGuard device_guard{row_bitmask.device()};
    TORCH_CHECK(row_bitmask.is_cuda() && row_nnz.is_cuda() && col_nnz.is_cuda(),
                "compact CSC/QCSR inputs must be CUDA tensors");
    TORCH_CHECK(row_bitmask.scalar_type() == at::kInt &&
                row_nnz.scalar_type() == at::kInt && col_nnz.scalar_type() == at::kInt,
                "compact CSC/QCSR inputs must be int32");
    TORCH_CHECK(row_bitmask.dim() == 3 && row_nnz.dim() == 2 && col_nnz.dim() == 3,
                "invalid compact CSC/QCSR input ranks");
    const int batch = col_nnz.size(0);
    const int q_heads = col_nnz.size(1);
    const int max_km = col_nnz.size(2);
    const int64_t qh = row_nnz.numel();
    TORCH_CHECK(row_nnz.size(0) == row_bitmask.size(0) &&
                row_nnz.size(1) == q_heads, "row metadata shape mismatch");

    auto long_opts = row_nnz.options().dtype(at::kLong);
    auto col_offsets = at::empty({batch, q_heads, max_km}, long_opts);
    auto counts64 = row_nnz.reshape({qh}).to(at::kLong);
    auto prefix = at::cumsum(counts64, 0);
    auto q_offsets = at::empty({qh + 1}, long_opts);
    q_offsets.narrow(0, 0, 1).zero_();
    q_offsets.narrow(0, 1, qh).copy_(prefix);
    const int64_t total_nnz = col_nnz.sum().item<int64_t>();
    TORCH_CHECK(total_nnz <= std::numeric_limits<int>::max(),
                "fused Q-CSR int32 positions require total_nnz <= INT_MAX");
    auto col_cursor = at::zeros_like(col_nnz);
    auto csc_indices = at::empty({total_nnz}, row_nnz.options());
    auto q_csc_indices = at::empty({total_nnz}, row_nnz.options());

    Dynamic_threshold_params params{};
    params.row_bitmask_ptr = reinterpret_cast<uint32_t*>(row_bitmask.data_ptr<int>());
    params.row_nnz_ptr = row_nnz.data_ptr<int>();
    params.col_offsets_ptr = col_offsets.data_ptr<int64_t>();
    params.col_nnz_ptr = col_nnz.data_ptr<int>();
    params.col_cursor_ptr = col_cursor.data_ptr<int>();
    params.csc_indices_ptr = csc_indices.data_ptr<int>();
    params.q_offsets_ptr = q_offsets.data_ptr<int64_t>();
    params.q_csc_indices_ptr = q_csc_indices.data_ptr<int>();
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_km = cu_seqlens_km.data_ptr<int>();
    params.b = batch;
    params.h = q_heads;
    params.total_q = row_nnz.size(0);
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_km = max_km;
    params.moba_chunk_size = chunk_size;
    params.max_lg_col_num = max_km;
    params.bitmask_words = row_bitmask.size(2);
    run_dynamic_threshold_compact_csc(
        params, at::cuda::getCurrentCUDAStream().stream());
    return {col_offsets, csc_indices, q_offsets, q_csc_indices};
}

}  // namespace flash_dynamic

void init_flash_dynamic(pybind11::module& m) {
    m.def(
        "moba_cap_compact_rows",
        &flash_dynamic::moba_cap_compact_rows,
        "Enforce a per-query/head block limit before CSC materialization, preserving anchors");
    m.def(
        "moba_flashprefill_coarse_scores",
        &flash_dynamic::moba_flashprefill_coarse_scores,
        "Native fused FlashPrefill meanmax coarse scores");
    m.def(
        "moba_flashprefill_dynamic_pattern",
        &flash_dynamic::moba_flashprefill_dynamic_pattern,
        "Native FlashPrefill scorer, auto threshold, and compact CSC");
    m.def(
        "moba_dynamic_threshold",
        &flash_dynamic::moba_dynamic_threshold,
        "LightningMoBA CUDA dynamic-threshold selector and CSC builder");
    m.def(
        "moba_dynamic_threshold_compact",
        &flash_dynamic::moba_dynamic_threshold_compact,
        "LightningMoBA CUDA capacity-free dynamic-threshold selector and compact CSC builder");
    m.def(
        "moba_dynamic_relative_threshold_compact",
        &flash_dynamic::moba_dynamic_relative_threshold_compact,
        "LightningMoBA CUDA per-row max-relative threshold selector and compact CSC builder");
    m.def(
        "moba_dynamic_relative_threshold_multi_probe",
        &flash_dynamic::moba_dynamic_relative_threshold_multi_probe,
        "LightningMoBA CUDA multi-alpha aggregate NNZ probe with one QK scan");
    m.def(
        "moba_dynamic_relative_threshold_probe_from_row_max",
        &flash_dynamic::moba_dynamic_relative_threshold_probe_from_row_max,
        "Count selected edges for one or more alphas using a cached row maximum");
    m.def(
        "moba_dynamic_relative_threshold_interpolate_device",
        &flash_dynamic::moba_dynamic_relative_threshold_interpolate_device,
        "Interpolate a capped relative alpha entirely on the CUDA stream");
    m.def(
        "moba_dynamic_relative_threshold_refine_in_place_device",
        &flash_dynamic::moba_dynamic_relative_threshold_refine_in_place_device,
        "Conditionally refine a compact pattern in place without host synchronization");
    m.def(
        "moba_dynamic_relative_threshold_materialize_from_row_max",
        &flash_dynamic::moba_dynamic_relative_threshold_materialize_from_row_max,
        "Materialize the compact pattern using a cached row maximum");
    m.def(
        "moba_dynamic_relative_threshold_materialize_from_row_max_device",
        &flash_dynamic::moba_dynamic_relative_threshold_materialize_from_row_max_device,
        "Materialize from a device-resident relative log threshold");
    m.def(
        "moba_dynamic_relative_head_threshold_compact",
        &flash_dynamic::moba_dynamic_relative_head_threshold_compact,
        "LightningMoBA CUDA per-head, per-row max-relative selector and compact CSC builder");
    m.def(
        "moba_dynamic_relative_threshold_padded",
        &flash_dynamic::moba_dynamic_relative_threshold_padded,
        "Graph-safe max-relative selector with padded CSC storage");
    m.def(
        "moba_compact_csc_from_bitmask",
        &flash_dynamic::moba_compact_csc_from_bitmask,
        "Build exact CSC metadata from a compact selector bitmask");
    m.def(
        "moba_compact_csc_qcsr_from_bitmask",
        &flash_dynamic::moba_compact_csc_qcsr_from_bitmask,
        "Build compact CSC and reverse Q-CSR in one row-centric scatter");
}
