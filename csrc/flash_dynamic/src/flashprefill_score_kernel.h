#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "../../flash_topk/src/kernel_traits.h"
#include "../../flash_topk/src/namespace_config.h"
#include "../../flash_topk/src/utils.h"

namespace flash_dynamic {

using namespace cute;

// Phase 1 of the native FlashPrefill selector.  It preserves the historical
// fine=32 / exec=128 meanmax score, but fuses fine->coarse reduction and never
// materializes the [4092, 4092, H] fine score tensor.
template <typename Element_>
struct FlashPrefillScoreTraits
    : Fused_topk_kernel_traits<128, 128, 64, 4, 16, true, Element_> {
    using Base = Fused_topk_kernel_traits<128, 128, 64, 4, 16, true, Element_>;
    static_assert(Base::kBlockM == 128 && Base::kBlockN == 64,
                  "FlashPrefill scorer code and CuTe tile must agree");
    static constexpr int kFineScores = 4 * Base::kBlockN;
    static constexpr int kScoreSmemSize =
        Base::kSmemKSize +
        std::max(Base::kSmemQSize, Base::kSmemTopKReduceSize) +
        kFineScores * sizeof(float);
};

struct FlashPrefillScoreParams {
    const void* q_ptr;
    const void* k_mean_ptr;
    const void* k_max_ptr;
    float* coarse_score_ptr;
    int64_t q_row_stride;
    int64_t q_head_stride;
    int64_t k_row_stride;
    int64_t k_head_stride;
    int total_q;
    int q_heads;
    int kv_heads;
    int head_ratio;
    int fine_blocks;
    int coarse_blocks;
    float softmax_scale_log2;
};

__device__ __forceinline__ float fp_warp_max(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
    }
    return value;
}

__device__ __forceinline__ float fp_warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

__device__ __forceinline__ float fp_logaddexp2(float lhs, float rhs) {
    const float hi = fmaxf(lhs, rhs);
    const float lo = fminf(lhs, rhs);
    return hi + log2f(1.0f + exp2f(fmaxf(lo - hi, -60.0f)));
}

template <typename Traits>
__global__ void flashprefill_coarse_score_kernel(FlashPrefillScoreParams params) {
    using Element = typename Traits::Element;
    using ElementAccum = typename Traits::ElementAccum;
    constexpr int kBlockM = Traits::kBlockM;  // one 128-token execution block
    constexpr int kBlockN = Traits::kBlockN;  // 64 fine pooled K vectors
    constexpr int kHeadDim = Traits::kHeadDim;

    const int q_block = blockIdx.x;
    const int q_head = blockIdx.y;
    const int split = blockIdx.z;
    const int tidx = threadIdx.x;
    const int lane = tidx & 31;
    const int warp = tidx >> 5;
    if (q_block >= params.coarse_blocks || q_head >= params.q_heads) return;
    const int max_fine_k = min(params.fine_blocks, (q_block + 1) * 4);
    const int k_tiles = cute::ceil_div(max_fine_k, kBlockN);
    // Split CTAs own disjoint coarse-K tiles, so no cross-CTA reduction or
    // atomics are needed.  Return before loading Q for empty early-row splits.
    if (split >= k_tiles) return;

    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(const_cast<void*>(params.q_ptr))),
        make_shape(params.total_q, params.q_heads, kHeadDim),
        make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(
        mQ(_, q_head, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(q_block, 0));

    const int kv_head = q_head / params.head_ratio;
    Tensor mKMean = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(const_cast<void*>(params.k_mean_ptr))),
        make_shape(params.fine_blocks, params.kv_heads, kHeadDim),
        make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor mKMax = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(const_cast<void*>(params.k_max_ptr))),
        make_shape(params.fine_blocks, params.kv_heads, kHeadDim),
        make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gKMean = local_tile(
        mKMean(_, kv_head, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_coord(_, 0));
    Tensor gKMax = local_tile(
        mKMax(_, kv_head, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_coord(_, 0));

    extern __shared__ char smem[];
    Tensor sK = make_tensor(
        make_smem_ptr(reinterpret_cast<Element*>(smem)),
        typename Traits::SmemLayoutKV{});
    char* q_smem = smem + sizeof(Element) * size(sK);
    Tensor sQ = make_tensor(
        make_smem_ptr(reinterpret_cast<Element*>(q_smem)),
        typename Traits::SmemLayoutQ{});
    Tensor sReduce = make_tensor(
        make_smem_ptr(reinterpret_cast<ElementAccum*>(q_smem)),
        typename Traits::SmemLayoutTopKReduce{});
    float* fine_scores = reinterpret_cast<float*>(
        q_smem + std::max(Traits::kSmemQSize, Traits::kSmemTopKReduceSize));

    typename Traits::GmemTiledCopyQKV gmem_copy;
    auto gmem_thr = gmem_copy.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr.partition_S(gQ);
    Tensor tQsQ = gmem_thr.partition_D(sQ);
    Tensor tKMean = gmem_thr.partition_S(gKMean);
    Tensor tKMax = gmem_thr.partition_S(gKMax);
    Tensor tKsK = gmem_thr.partition_D(sK);

    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    auto copy_q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto copy_k = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_q = copy_q.get_thread_slice(tidx);
    auto thr_copy_k = copy_k.get_thread_slice(tidx);
    Tensor tSsQ = thr_copy_q.partition_S(sQ);
    Tensor tSsK = thr_copy_k.partition_S(sK);
    auto copy_r = make_tiled_copy_C(typename Traits::SmemCopyAtomR{}, tiled_mma);
    auto thr_copy_r = copy_r.get_thread_slice(tidx);
    Tensor taccOsR = thr_copy_r.partition_D(sReduce);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr.partition_S(cQ);
    Tensor tKcK = gmem_thr.partition_S(cK);
    Tensor q_pred = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor k_pred = make_tensor<bool>(make_shape(size<2>(tKsK)));
#pragma unroll
    for (int i = 0; i < size(q_pred); ++i) q_pred(i) = true;
#pragma unroll
    for (int i = 0; i < size(k_pred); ++i) k_pred(i) = true;

    const int q_remaining = params.total_q - q_block * kBlockM;
    FLASH_TOPK_NAMESPACE::copy<false, true>(
        gmem_copy, tQgQ, tQsQ, tQcQ, q_pred, q_remaining);
    cute::cp_async_fence();
    FLASH_TOPK_NAMESPACE::cp_async_wait<0>();
    __syncthreads();
    Tensor q_reg_view = thr_copy_q.retile_D(tSrQ);
    cute::copy(copy_q, tSsQ, q_reg_view);
    __syncthreads();

    // Only fine K blocks ending before at least one query in this execution
    // block can have a score. Sink/local blocks are forced later.
    for (int k_tile = split; k_tile < k_tiles; k_tile += gridDim.z) {
        const int k_base = k_tile * kBlockN;
        const int k_remaining = params.fine_blocks - k_base;

        FLASH_TOPK_NAMESPACE::copy<false, true>(
            gmem_copy, tKMean(_, _, _, k_tile), tKsK, tKcK, k_pred,
            k_remaining);
        cute::cp_async_fence();
        FLASH_TOPK_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor acc = partition_fragment_C(
            tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc);
        FLASH_TOPK_NAMESPACE::gemm<true>(
            acc, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
            copy_q, copy_k, thr_copy_q, thr_copy_k);
        Tensor acc_view = thr_copy_r.retile_S(acc);
        cute::copy(copy_r, acc_view, taccOsR);
        __syncthreads();

        if (warp < 4) {
            const int q_local = warp * 32 + lane;
            const int q_token = q_block * 128 + q_local;
#pragma unroll
            for (int j = 0; j < kBlockN; ++j) {
                const int fine_k = k_base + j;
                const bool valid = q_token < params.total_q &&
                    fine_k < params.fine_blocks && q_token >= (fine_k + 1) * 32 - 1;
                float value = valid
                    ? sReduce(q_local, j) * params.softmax_scale_log2
                    : -INFINITY;
                float maximum = fp_warp_max(value);
                maximum = __shfl_sync(0xffffffffu, maximum, 0);
                float weight = valid ? exp2f(value - maximum) : 0.0f;
                float total = fp_warp_sum(weight);
                if (lane == 0) {
                    fine_scores[warp * kBlockN + j] =
                        total > 0.0f ? maximum + log2f(total) : -INFINITY;
                }
            }
        }
        __syncthreads();

        FLASH_TOPK_NAMESPACE::copy<false, true>(
            gmem_copy, tKMax(_, _, _, k_tile), tKsK, tKcK, k_pred,
            k_remaining);
        cute::cp_async_fence();
        FLASH_TOPK_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        clear(acc);
        FLASH_TOPK_NAMESPACE::gemm<true>(
            acc, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
            copy_q, copy_k, thr_copy_q, thr_copy_k);
        Tensor acc_view2 = thr_copy_r.retile_S(acc);
        cute::copy(copy_r, acc_view2, taccOsR);
        __syncthreads();

        if (warp < 4) {
            const int q_local = warp * 32 + lane;
            const int q_token = q_block * 128 + q_local;
#pragma unroll
            for (int j = 0; j < kBlockN; ++j) {
                const int fine_k = k_base + j;
                const bool valid = q_token < params.total_q &&
                    fine_k < params.fine_blocks && q_token >= (fine_k + 1) * 32 - 1;
                float value = valid
                    ? sReduce(q_local, j) * params.softmax_scale_log2
                    : -INFINITY;
                float maximum = fp_warp_max(value);
                maximum = __shfl_sync(0xffffffffu, maximum, 0);
                float weight = valid ? exp2f(value - maximum) : 0.0f;
                float total = fp_warp_sum(weight);
                if (lane == 0) {
                    const int slot = warp * kBlockN + j;
                    const float max_score = total > 0.0f
                        ? maximum + log2f(total) : -INFINITY;
                    const float mean_score = fine_scores[slot];
                    fine_scores[slot] = isfinite(mean_score) || isfinite(max_score)
                        ? fp_logaddexp2(mean_score, max_score)
                        : -INFINITY;
                }
            }
        }
        __syncthreads();

        // 64 fine K blocks map to 16 execution K blocks.  Sixteen threads each
        // reduce 4 query sub-blocks x 4 key sub-blocks.
        if (tidx < 16) {
            const int coarse_k = k_tile * 16 + tidx;
            if (coarse_k < params.coarse_blocks && coarse_k <= q_block) {
                float score = -INFINITY;
#pragma unroll
                for (int qf = 0; qf < 4; ++qf) {
#pragma unroll
                    for (int kf = 0; kf < 4; ++kf) {
                        score = fmaxf(score,
                            fine_scores[qf * kBlockN + tidx * 4 + kf]);
                    }
                }
                const int64_t out =
                    (static_cast<int64_t>(q_block) * params.q_heads + q_head) *
                    params.coarse_blocks + coarse_k;
                params.coarse_score_ptr[out] = score;
            }
        }
        __syncthreads();
    }
}

template <typename Element>
void run_flashprefill_coarse_score(
    FlashPrefillScoreParams& params, int num_splits, cudaStream_t stream) {
    using Traits = FlashPrefillScoreTraits<Element>;
    constexpr int smem_size = Traits::kScoreSmemSize;
    auto kernel = &flashprefill_coarse_score_kernel<Traits>;
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    dim3 grid(params.coarse_blocks, params.q_heads, num_splits);
    kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace flash_dynamic
