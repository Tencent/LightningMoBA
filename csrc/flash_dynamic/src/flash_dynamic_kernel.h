#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "flash_dynamic.h"
#include "../../flash_topk/src/block_info.h"
#include "../../flash_topk/src/kernel_traits.h"
#include "../../flash_topk/src/namespace_config.h"
#include "../../flash_topk/src/utils.h"

namespace flash_dynamic {

using namespace cute;

template <typename Element_>
struct Dynamic_threshold_traits
    : Fused_topk_kernel_traits<128, 64, 32, 4, 16, true, Element_> {
    using Base = Fused_topk_kernel_traits<128, 64, 32, 4, 16, true, Element_>;
    // Q is copied to registers before the scan, so its shared-memory region is
    // reused by the score tile. No top-k shared arrays are allocated.
    static constexpr int kDynamicSmemSize =
        Base::kSmemKSize + std::max(Base::kSmemQSize, Base::kSmemTopKReduceSize);
};

template <typename Kernel_traits, bool Is_even_MN, bool Use_bitmask = false,
          bool Find_row_max = false, bool Multi_probe = false,
          bool Enforce_min_blocks = false,
          int Multi_probe_count = 16, bool Warp_aggregate_counts = false>
inline __device__ void compute_dynamic_threshold_1rowblock(
    const Dynamic_threshold_params& params,
    int bidb,
    int bidh,
    int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const int tidx = threadIdx.x;
    const int chunk_size = params.moba_chunk_size;
    const FLASH_TOPK_NAMESPACE::TopKBlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    int chunk_block_max = binfo.actual_seqlen_km;
    chunk_block_max = min(
        chunk_block_max,
        cute::ceil_div(
            (m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q,
            chunk_size));
    int n_block = cute::ceil_div(chunk_block_max, kBlockN) - 1;
    if (n_block < 0) return;

    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.sum_s_q * params.q_row_stride),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(
        mQ(_, bidh, _),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, 0));

    Tensor mKM = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.km_ptr) + binfo.sum_s_km * params.km_row_stride),
        make_shape(binfo.actual_seqlen_km, params.h_k, params.d),
        make_stride(params.km_row_stride, params.km_head_stride, _1{}));
    Tensor gKM = local_tile(
        mKM(_, bidh / params.h_h_k_ratio, _),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_coord(_, 0));

    extern __shared__ char smem_[];
    Tensor sKM = make_tensor(
        make_smem_ptr(reinterpret_cast<Element*>(smem_)),
        typename Kernel_traits::SmemLayoutKV{});
    char* smem_q = smem_ + sizeof(Element) * size(sKM);
    Tensor sQ = make_tensor(
        make_smem_ptr(reinterpret_cast<Element*>(smem_q)),
        typename Kernel_traits::SmemLayoutQ{});
    Tensor sReduce = make_tensor(
        make_smem_ptr(reinterpret_cast<ElementAccum*>(smem_q)),
        typename Kernel_traits::SmemLayoutTopKReduce{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
    Tensor tKMgKM = gmem_thr_copy.partition_S(gKM);
    Tensor tKMsKM = gmem_thr_copy.partition_D(sKM);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sKM);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sKM);
    auto smem_tiled_copy_R = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomR{}, tiled_mma);
    auto smem_thr_copy_R = smem_tiled_copy_R.get_thread_slice(tidx);
    Tensor taccOsR = smem_thr_copy_R.partition_D(sReduce);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKM = make_identity_tensor(make_shape(size<0>(sKM), size<1>(sKM)));
    Tensor tQcQ = gmem_thr_copy.partition_S(cQ);
    Tensor tKMcKM = gmem_thr_copy.partition_S(cKM);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKMpKM = make_tensor<bool>(make_shape(size<2>(tKMsKM)));
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) tQpQ(k) = true;
#pragma unroll
    for (int k = 0; k < size(tKMpKM); ++k) tKMpKM(k) = true;

    FLASH_TOPK_NAMESPACE::copy<false, true>(
        gmem_tiled_copy, tQgQ, tQsQ, tQcQ, tQpQ,
        binfo.actual_seqlen_q - m_block * kBlockM);
    cute::cp_async_fence();
    FLASH_TOPK_NAMESPACE::cp_async_wait<0>();
    __syncthreads();

    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();

    FLASH_TOPK_NAMESPACE::copy<false, true>(
        gmem_tiled_copy, tKMgKM(_, _, _, n_block), tKMsKM, tKMcKM, tKMpKM,
        binfo.actual_seqlen_km - n_block * kBlockN);
    cute::cp_async_fence();

    const int num_q_rows = Is_even_MN
        ? kBlockM
        : min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);
    const bool owns_row = tidx < num_q_rows;
    const int row_idx = m_block * kBlockM + tidx;
    const int global_row = binfo.sum_s_q + row_idx;
    int selected_count = 0;
    constexpr int kMaxMinBlocks = Enforce_min_blocks ? 8 : 1;
    float rejected_top_scores[kMaxMinBlocks];
    int rejected_top_indices[kMaxMinBlocks];
    int rejected_count = 0;
#pragma unroll
    for (int slot = 0; slot < kMaxMinBlocks; ++slot) {
        rejected_top_scores[slot] = -INFINITY;
        rejected_top_indices[slot] = -1;
    }
    constexpr int kMaxProbes = Multi_probe ? Multi_probe_count : 1;
    int probe_counts[kMaxProbes];
#pragma unroll
    for (int p = 0; p < kMaxProbes; ++p) probe_counts[p] = 0;
    int overflow = 0;
    float effective_threshold = -INFINITY;
    float row_max = -INFINITY;
    if (owns_row && !Find_row_max && !Multi_probe) {
        const int64_t meta_idx = static_cast<int64_t>(global_row) * params.h + bidh;
        if (params.use_row_relative_threshold) {
            const float relative_log_threshold =
                params.relative_log_threshold_ptr != nullptr
                ? params.relative_log_threshold_ptr[0]
                : params.relative_log_threshold;
            effective_threshold = params.row_max_ptr[meta_idx]
                + (params.use_head_relative_threshold
                    ? params.threshold_ptr[bidh]
                    : relative_log_threshold);
        } else {
        const Element* q_row = reinterpret_cast<const Element*>(params.q_ptr)
            + global_row * params.q_row_stride + bidh * params.q_head_stride;
        float sum_sq = 0.0f;
#pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            float value = static_cast<float>(q_row[d]);
            sum_sq = fmaf(value, value, sum_sq);
        }
        effective_threshold = params.threshold_ptr[bidh] * sqrtf(sum_sq / kHeadDim);
        }
    }

    const int64_t row_base = (static_cast<int64_t>(global_row) * params.h + bidh) * params.capacity;
    const int64_t bitmask_base =
        (static_cast<int64_t>(global_row) * params.h + bidh) * params.bitmask_words;
    const int64_t col_base = (static_cast<int64_t>(bidb) * params.h + bidh) * params.max_lg_col_num;

    for (; n_block >= 0; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        FLASH_TOPK_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        FLASH_TOPK_NAMESPACE::gemm<true>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
            smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K);
        if (params.coarse_anchor_block_size > 0) __syncthreads();

        if (n_block > 0) {
            FLASH_TOPK_NAMESPACE::copy<true, true>(
                gmem_tiled_copy, tKMgKM(_, _, _, n_block - 1),
                tKMsKM, tKMcKM, tKMpKM);
            cute::cp_async_fence();
        }

        Tensor taccOrR = smem_thr_copy_R.retile_S(acc_s);
        cute::copy(smem_tiled_copy_R, taccOrR, taccOsR);
        __syncthreads();

        if (owns_row) {
            const int actual_k = min(kBlockN, binfo.actual_seqlen_km - n_block * kBlockN);
#pragma unroll
            for (int j = 0; j < kBlockN; ++j) {
                if (j >= actual_k) break;
                const int km_idx = n_block * kBlockN + j;
                // Q and K have the same length during prefill, but decode uses
                // bottom-right causal alignment (Sq=1, Sk=the full history).
                // Anchor the forced local window at the absolute causal Q
                // position.  Using row_idx alone makes every decode token look
                // like block 0 and drops the current/incomplete KV block.
                const int causal_row =
                    row_idx + binfo.actual_seqlen_k - binfo.actual_seqlen_q;
                const int local_block = causal_row / chunk_size;
                const bool completed = (km_idx + 1) * chunk_size <=
                    row_idx + binfo.actual_seqlen_k - binfo.actual_seqlen_q;
                const int anchor_ratio = params.coarse_anchor_block_size > 0
                    ? params.coarse_anchor_block_size / chunk_size : 1;
                const int anchor_q = local_block / anchor_ratio;
                const int anchor_k = km_idx / anchor_ratio;
                const bool forced = params.coarse_anchor_block_size > 0
                    ? (causal_row >= 0 && km_idx <= local_block &&
                       (anchor_k < params.sink ||
                        (anchor_k <= anchor_q && anchor_q - anchor_k < params.window)))
                    : (km_idx < params.sink ||
                       (km_idx <= local_block && local_block - km_idx < params.window));
                const bool eligible = forced || completed;
                if constexpr (Find_row_max) {
                    if (eligible) row_max = fmaxf(row_max, sReduce(tidx, j));
                } else if constexpr (Multi_probe) {
                    if (eligible) {
                        const int64_t meta_idx =
                            static_cast<int64_t>(global_row) * params.h + bidh;
                        const float base = params.row_max_ptr[meta_idx];
#pragma unroll
                        for (int p = 0; p < kMaxProbes; ++p) {
                            const bool keep = forced ||
                                (completed && sReduce(tidx, j) >=
                                    base + params.probe_threshold_ptr[p]);
                            probe_counts[p] += static_cast<int>(keep);
                        }
                    }
                } else {
                const bool keep = forced ||
                    (completed && sReduce(tidx, j) >= effective_threshold);
                if constexpr (Use_bitmask && Warp_aggregate_counts) {
                    // The 32 lanes of a warp process 32 Q rows for the same
                    // KV column.  Count their selected edges with one atomic
                    // instead of up to 32 contending atomics.  The mask is
                    // evaluated by every active row lane, including rejects.
                    const unsigned active = __activemask();
                    const unsigned votes = __ballot_sync(active, keep);
                    if ((tidx & 31) == 0 && votes != 0) {
                        atomicAdd(params.col_nnz_ptr + col_base + km_idx,
                                  __popc(votes));
                    }
                }
                if (keep) {
                    if constexpr (Use_bitmask) {
                        params.row_bitmask_ptr[bitmask_base + (km_idx >> 5)] |=
                            uint32_t(1) << (km_idx & 31);
                        if constexpr (!Warp_aggregate_counts) {
                            atomicAdd(params.col_nnz_ptr + col_base + km_idx, 1);
                        }
                        ++selected_count;
                    } else if (selected_count < params.capacity) {
                        params.row_indices_ptr[row_base + selected_count] = km_idx;
                        atomicAdd(params.col_nnz_ptr + col_base + km_idx, 1);
                        ++selected_count;
                    } else {
                        ++overflow;
                    }
                } else if constexpr (Enforce_min_blocks) {
                    // Keep a tiny register-resident TopK among rejected
                    // completed blocks. This path is opt-in and preserves
                    // exact score order for the row-level floor.
                    // Once this row already satisfies the floor, later
                    // rejected candidates can never be inserted into the
                    // final mask.  Stop maintaining the fallback TopK for
                    // those rows; this is especially useful for long rows,
                    // where threshold-selected blocks quickly reach the
                    // minimum but many KV blocks remain to be scanned.
                    if (completed &&
                        selected_count < params.min_blocks_per_row) {
                        const float score = sReduce(tidx, j);
                        const int candidate_idx = km_idx;
                        const auto better_than = [&](int lhs, int rhs) {
                            return rejected_top_scores[lhs] > rejected_top_scores[rhs] ||
                                (rejected_top_scores[lhs] == rejected_top_scores[rhs] &&
                                 rejected_top_indices[lhs] < rejected_top_indices[rhs]);
                        };
                        if (rejected_count < kMaxMinBlocks) {
                            int insert = rejected_count++;
                            rejected_top_scores[insert] = score;
                            rejected_top_indices[insert] = candidate_idx;
                            while (insert > 0) {
                                const int parent = (insert - 1) / 2;
                                if (!better_than(insert, parent)) break;
                                const float parent_score = rejected_top_scores[parent];
                                const int parent_idx = rejected_top_indices[parent];
                                rejected_top_scores[parent] = rejected_top_scores[insert];
                                rejected_top_indices[parent] = rejected_top_indices[insert];
                                rejected_top_scores[insert] = parent_score;
                                rejected_top_indices[insert] = parent_idx;
                                insert = parent;
                            }
                        } else {
                            // Maintain a max-heap (best candidate at the
                            // root).  Its minimum must be one of the four
                            // leaves, so reject most candidates after four
                            // comparisons instead of scanning/shifting all
                            // eight sorted slots for every rejected block.
                            int worst = 4;
#pragma unroll
                            for (int slot = 5; slot < kMaxMinBlocks; ++slot) {
                                if (better_than(worst, slot)) worst = slot;
                            }
                            const bool better = score > rejected_top_scores[worst] ||
                                (score == rejected_top_scores[worst] &&
                                 candidate_idx < rejected_top_indices[worst]);
                            if (better) {
                                int insert = worst;
                                while (insert > 0) {
                                    const int parent = (insert - 1) / 2;
                                    const bool candidate_better_than_parent =
                                        score > rejected_top_scores[parent] ||
                                        (score == rejected_top_scores[parent] &&
                                         candidate_idx < rejected_top_indices[parent]);
                                    if (!candidate_better_than_parent) break;
                                    rejected_top_scores[insert] = rejected_top_scores[parent];
                                    rejected_top_indices[insert] = rejected_top_indices[parent];
                                    insert = parent;
                                }
                                rejected_top_scores[insert] = score;
                                rejected_top_indices[insert] = candidate_idx;
                            }
                        }
                    }
                }
                }
            }
        }
    }

    if constexpr (Enforce_min_blocks) {
        if (owns_row && !Find_row_max && !Multi_probe &&
            selected_count < params.min_blocks_per_row) {
            const int needed = params.min_blocks_per_row - selected_count;
            // Restore descending order only for rows that actually need the
            // floor.  The streaming max-heap keeps exact top-k membership;
            // this tiny row-local sort makes materialization deterministic.
#pragma unroll
            for (int slot = 1; slot < kMaxMinBlocks; ++slot) {
                const float score = rejected_top_scores[slot];
                const int col = rejected_top_indices[slot];
                int insert = slot;
                while (insert > 0 &&
                    (score > rejected_top_scores[insert - 1] ||
                     (score == rejected_top_scores[insert - 1] &&
                      col < rejected_top_indices[insert - 1]))) {
                    rejected_top_scores[insert] = rejected_top_scores[insert - 1];
                    rejected_top_indices[insert] = rejected_top_indices[insert - 1];
                    --insert;
                }
                rejected_top_scores[insert] = score;
                rejected_top_indices[insert] = col;
            }
#pragma unroll
            for (int slot = 0; slot < kMaxMinBlocks; ++slot) {
                if (slot < needed && rejected_top_indices[slot] >= 0) {
                    const int col = rejected_top_indices[slot];
                    params.row_bitmask_ptr[bitmask_base + (col >> 5)] |=
                        uint32_t(1) << (col & 31);
                    atomicAdd(params.col_nnz_ptr + col_base + col, 1);
                    ++selected_count;
                }
            }
        }
    }

    if constexpr (Multi_probe) {
        // Reuse the Q/reduction shared-memory region after the GEMM scan.
        // Global atomics from every Q row serialize catastrophically at 128K;
        // aggregate one vector per CTA and reduce those small vectors later.
        int* block_probe_counts = reinterpret_cast<int*>(smem_q);
        if (tidx < kMaxProbes) block_probe_counts[tidx] = 0;
        __syncthreads();
        if (owns_row) {
#pragma unroll
            for (int p = 0; p < kMaxProbes; ++p) {
                if (p < params.num_probes) {
                    atomicAdd(block_probe_counts + p, probe_counts[p]);
                }
            }
        }
        __syncthreads();
        if (tidx < kMaxProbes) {
            const int64_t cta =
                (static_cast<int64_t>(blockIdx.x) * params.b + blockIdx.y)
                * params.h + blockIdx.z;
            params.probe_nnz_ptr[cta * params.num_probes + tidx] =
                block_probe_counts[tidx];
        }
    } else if (owns_row) {
        const int64_t meta_idx = static_cast<int64_t>(global_row) * params.h + bidh;
        if constexpr (Find_row_max) {
            params.row_max_ptr[meta_idx] = row_max;
        } else {
            params.row_nnz_ptr[meta_idx] = selected_count;
            params.row_overflow_ptr[meta_idx] = overflow;
        }
    }
}

template <typename Kernel_traits, bool Is_even_MN, bool Use_bitmask = false,
          bool Find_row_max = false, bool Multi_probe = false,
          bool Enforce_min_blocks = false,
          int Multi_probe_count = 16, bool Warp_aggregate_counts = false>
__global__ void dynamic_threshold_kernel(Dynamic_threshold_params params) {
    if (params.execute_flag_ptr != nullptr && params.execute_flag_ptr[0] == 0) {
        return;
    }
    compute_dynamic_threshold_1rowblock<Kernel_traits, Is_even_MN, Use_bitmask,
                                        Find_row_max, Multi_probe, Enforce_min_blocks,
                                        Multi_probe_count, Warp_aggregate_counts>(
        params, blockIdx.y, blockIdx.z, blockIdx.x);
}

__global__ void dynamic_csc_offset_kernel(Dynamic_threshold_params params) {
    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;
    if (threadIdx.x != 0) return;
    const int q_start = params.cu_seqlens_q[bidb];
    const int q_len = params.cu_seqlens_q[bidb + 1] - q_start;
    const int km_len = params.cu_seqlens_km[bidb + 1] - params.cu_seqlens_km[bidb];
    const int64_t col_base = (static_cast<int64_t>(bidb) * params.h + bidh) * params.max_lg_col_num;
    int64_t offset = (static_cast<int64_t>(q_start) * params.h + static_cast<int64_t>(q_len) * bidh)
        * params.capacity;
    for (int col = 0; col < km_len; ++col) {
        params.col_offsets_ptr[col_base + col] = offset;
        offset += params.col_nnz_ptr[col_base + col];
    }
}

__global__ void dynamic_csc_scatter_kernel(Dynamic_threshold_params params) {
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    const int q_start = params.cu_seqlens_q[bidb];
    const int q_len = params.cu_seqlens_q[bidb + 1] - q_start;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= q_len) return;
    const int global_row = q_start + row;
    const int64_t meta_idx = static_cast<int64_t>(global_row) * params.h + bidh;
    const int64_t row_base = meta_idx * params.capacity;
    const int64_t col_base = (static_cast<int64_t>(bidb) * params.h + bidh) * params.max_lg_col_num;
    const int nnz = params.row_nnz_ptr[meta_idx];
    for (int slot = 0; slot < nnz; ++slot) {
        const int col = params.row_indices_ptr[row_base + slot];
        const int position = atomicAdd(params.col_cursor_ptr + col_base + col, 1);
        params.csc_indices_ptr[params.col_offsets_ptr[col_base + col] + position] = row;
    }
}

// Compact CSC uses one global exclusive prefix instead of reserving capacity
// elements for every (Q, head) row.  The column metadata is small (for the
// target 128K/chunk128 shape: 32 * 1024 entries), so this serial pass is not a
// long-context bottleneck and keeps the first implementation easy to audit.
__global__ void dynamic_compact_csc_offset_kernel(Dynamic_threshold_params params) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int64_t offset = 0;
    const int64_t num_cols =
        static_cast<int64_t>(params.b) * params.h * params.max_lg_col_num;
    for (int64_t idx = 0; idx < num_cols; ++idx) {
        params.col_offsets_ptr[idx] = offset;
        offset += params.col_nnz_ptr[idx];
    }
}

__global__ void dynamic_compact_csc_scatter_kernel(Dynamic_threshold_params params) {
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    const int q_start = params.cu_seqlens_q[bidb];
    const int q_len = params.cu_seqlens_q[bidb + 1] - q_start;
    const int km_len = params.cu_seqlens_km[bidb + 1] - params.cu_seqlens_km[bidb];
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= q_len) return;
    const int global_row = q_start + row;
    const int64_t meta_idx = static_cast<int64_t>(global_row) * params.h + bidh;
    const int64_t bitmask_base =
        meta_idx * params.bitmask_words;
    const int64_t col_base =
        (static_cast<int64_t>(bidb) * params.h + bidh) * params.max_lg_col_num;
    const int words = (km_len + 31) / 32;
    int row_slot = 0;
    for (int word_idx = 0; word_idx < words; ++word_idx) {
        uint32_t bits = params.row_bitmask_ptr[bitmask_base + word_idx];
        while (bits != 0) {
            const int bit = __ffs(bits) - 1;
            const int col = word_idx * 32 + bit;
            if (col < km_len) {
                const int position = atomicAdd(params.col_cursor_ptr + col_base + col, 1);
                const int64_t csc_position =
                    params.col_offsets_ptr[col_base + col] + position;
                params.csc_indices_ptr[csc_position] = row;
                if (params.q_csc_indices_ptr != nullptr) {
                    params.q_csc_indices_ptr[
                        params.q_offsets_ptr[meta_idx] + row_slot] =
                            static_cast<int>(csc_position);
                }
                ++row_slot;
            }
            bits &= bits - 1;
        }
    }
}

// One warp per Q row: each lane owns one 32-column bitmask word.  This
// parallelizes the long-context word scan and the column atomics while the
// warp prefix preserves the exact row-local Q-CSR slot order.
__global__ void dynamic_compact_csc_scatter_warp_kernel(Dynamic_threshold_params params) {
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    const int q_start = params.cu_seqlens_q[bidb];
    const int q_len = params.cu_seqlens_q[bidb + 1] - q_start;
    const int km_len = params.cu_seqlens_km[bidb + 1] - params.cu_seqlens_km[bidb];
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= q_len) return;
    const int global_row = q_start + row;
    const int64_t meta_idx = static_cast<int64_t>(global_row) * params.h + bidh;
    const int64_t col_base =
        (static_cast<int64_t>(bidb) * params.h + bidh) * params.max_lg_col_num;
    const int words = (km_len + 31) / 32;
    const uint32_t bits = lane < words
        ? params.row_bitmask_ptr[meta_idx * params.bitmask_words + lane] : 0;
    const int count = __popc(bits);
    int prefix = count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int prior = __shfl_up_sync(0xffffffffu, prefix, offset);
        if (lane >= offset) prefix += prior;
    }
    int row_slot = prefix - count;
    uint32_t remaining = bits;
    while (remaining != 0) {
        const int bit = __ffs(remaining) - 1;
        const int col = lane * 32 + bit;
        if (col < km_len) {
            const int position = atomicAdd(params.col_cursor_ptr + col_base + col, 1);
            const int64_t csc_position = params.col_offsets_ptr[col_base + col] + position;
            params.csc_indices_ptr[csc_position] = row;
            if (params.q_csc_indices_ptr != nullptr) {
                params.q_csc_indices_ptr[params.q_offsets_ptr[meta_idx] + row_slot] =
                    static_cast<int>(csc_position);
            }
            ++row_slot;
        }
        remaining &= remaining - 1;
    }
}

// Four lanes cooperate on one row.  Compared with one lane/row this reduces
// the 128K bitmask scan from 32 serial words to eight; compared with one
// warp/row it keeps 32 rows per CTA to avoid excessive tiny CTAs.
__global__ void dynamic_compact_csc_scatter_lanes4_kernel(Dynamic_threshold_params params) {
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    const int q_start = params.cu_seqlens_q[bidb];
    const int q_len = params.cu_seqlens_q[bidb + 1] - q_start;
    const int km_len = params.cu_seqlens_km[bidb + 1] - params.cu_seqlens_km[bidb];
    const int lane = threadIdx.x & 3;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 2);
    if (row >= q_len) return;
    const int global_row = q_start + row;
    const int64_t meta_idx = static_cast<int64_t>(global_row) * params.h + bidh;
    const int64_t col_base =
        (static_cast<int64_t>(bidb) * params.h + bidh) * params.max_lg_col_num;
    const int words = (km_len + 31) / 32;
    const unsigned mask = __activemask();
    int base_slot = 0;
    for (int group = 0; group < words; group += 4) {
        const int word_idx = group + lane;
        const uint32_t bits = word_idx < words
            ? params.row_bitmask_ptr[meta_idx * params.bitmask_words + word_idx] : 0;
        const int count = __popc(bits);
        int prefix = count;
#pragma unroll
        for (int offset = 1; offset < 4; offset <<= 1) {
            const int prior = __shfl_up_sync(mask, prefix, offset, 4);
            if (lane >= offset) prefix += prior;
        }
        int row_slot = base_slot + prefix - count;
        base_slot += __shfl_sync(mask, prefix, 3, 4);
        uint32_t remaining = bits;
        while (remaining != 0) {
            const int bit = __ffs(remaining) - 1;
            const int col = word_idx * 32 + bit;
            if (col < km_len) {
                const int position = atomicAdd(params.col_cursor_ptr + col_base + col, 1);
                const int64_t csc_position = params.col_offsets_ptr[col_base + col] + position;
                params.csc_indices_ptr[csc_position] = row;
                if (params.q_csc_indices_ptr != nullptr) {
                    params.q_csc_indices_ptr[params.q_offsets_ptr[meta_idx] + row_slot] =
                        static_cast<int>(csc_position);
                }
                ++row_slot;
            }
            remaining &= remaining - 1;
        }
    }
}

template <typename Element, bool Use_bitmask, bool Find_row_max = false,
          bool Multi_probe = false, bool Enforce_min_blocks = false,
          int Multi_probe_count = 16, bool Warp_aggregate_counts = false>
void launch_dynamic_selection(Dynamic_threshold_params& params, cudaStream_t stream) {
    using Traits = Dynamic_threshold_traits<Element>;
    constexpr int smem_size = Traits::kDynamicSmemSize;
    const int num_m_blocks = cute::ceil_div(params.seqlen_q, Traits::kBlockM);
    dim3 grid(num_m_blocks, params.b, params.h);
    const bool even_mn = params.seqlen_q % Traits::kBlockM == 0 &&
        params.seqlen_km % Traits::kBlockN == 0;
    if (even_mn) {
        auto kernel = &dynamic_threshold_kernel<Traits, true, Use_bitmask, Find_row_max, Multi_probe, Enforce_min_blocks, Multi_probe_count, Warp_aggregate_counts>;
        if (smem_size >= 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);
    } else {
        auto kernel = &dynamic_threshold_kernel<Traits, false, Use_bitmask, Find_row_max, Multi_probe, Enforce_min_blocks, Multi_probe_count, Warp_aggregate_counts>;
        if (smem_size >= 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Element>
void run_dynamic_threshold(
    Dynamic_threshold_params& params,
    bool build_csc,
    cudaStream_t stream) {
    launch_dynamic_selection<Element, false>(params, stream);

    if (build_csc) {
        dynamic_csc_offset_kernel<<<dim3(params.b, params.h), 1, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        dim3 scatter_grid(cute::ceil_div(params.seqlen_q, 128), params.b, params.h);
        dynamic_csc_scatter_kernel<<<scatter_grid, 128, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <typename Element>
void run_dynamic_threshold_compact_select(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    launch_dynamic_selection<Element, true>(params, stream);
}

template <typename Element>
void run_dynamic_relative_threshold_compact_select(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    launch_dynamic_selection<Element, true, true>(params, stream);
    const char* aggregate_env = std::getenv("LIGHTNING_MOBA_SELECTOR_WARP_COUNT");
    const bool aggregate_counts = aggregate_env != nullptr &&
        std::strcmp(aggregate_env, "1") == 0;
    if (params.min_blocks_per_row > 0) {
        if (aggregate_counts) {
            launch_dynamic_selection<Element, true, false, false, true, 16, true>(params, stream);
        } else {
            launch_dynamic_selection<Element, true, false, false, true>(params, stream);
        }
    } else {
        if (aggregate_counts) {
            launch_dynamic_selection<Element, true, false, false, false, 16, true>(params, stream);
        } else {
            launch_dynamic_selection<Element, true, false>(params, stream);
        }
    }
}

template <typename Element>
void run_dynamic_relative_threshold_compact_select_from_row_max(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    // The caller owns row_max_ptr and guarantees that it was produced from
    // the same q/km tensors and varlen metadata.  This is the materialization
    // half of the normal two-pass relative selector.
    launch_dynamic_selection<Element, true, false>(params, stream);
}

template <typename Element>
void run_dynamic_relative_threshold_probe_from_row_max(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    // Count-only cap probes reuse an already computed row maximum.  No
    // bitmask, per-column counters, or CSC metadata are touched here.
    switch (params.num_probes) {
        case 1:
            launch_dynamic_selection<Element, false, false, true, false, 1>(params, stream);
            break;
        case 4:
            launch_dynamic_selection<Element, false, false, true, false, 4>(params, stream);
            break;
        case 8:
            launch_dynamic_selection<Element, false, false, true, false, 8>(params, stream);
            break;
        case 16:
            launch_dynamic_selection<Element, false, false, true, false, 16>(params, stream);
            break;
        default:
            throw std::runtime_error("row-max probe count must be 1, 4, 8, or 16");
    }
}

template <typename Element>
void run_dynamic_relative_threshold_multi_probe(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    // Exactly the same row maxima and score comparisons as the production
    // selector, but all scalar offsets share the second QK traversal.  Keep
    // probe count compile-time-specialized so 4-way search does not pay 16-way
    // register and instruction cost.
    launch_dynamic_selection<Element, false, true, false>(params, stream);
    switch (params.num_probes) {
        case 1:
            launch_dynamic_selection<Element, false, false, true, false, 1>(params, stream);
            break;
        case 4:
            launch_dynamic_selection<Element, false, false, true, false, 4>(params, stream);
            break;
        case 8:
            launch_dynamic_selection<Element, false, false, true, false, 8>(params, stream);
            break;
        case 16:
            launch_dynamic_selection<Element, false, false, true, false, 16>(params, stream);
            break;
        default:
            throw std::runtime_error("multi-probe count must be 1, 4, 8, or 16");
    }
}

inline void run_dynamic_threshold_compact_csc_impl(
    Dynamic_threshold_params& params,
    cudaStream_t stream) {
    dynamic_compact_csc_offset_kernel<<<1, 1, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const char* warp_env = std::getenv("LIGHTNING_MOBA_CSC_WARP");
    const bool use_warp = warp_env != nullptr && std::strcmp(warp_env, "1") == 0 &&
        params.bitmask_words <= 32 && params.seqlen_q >= 32768;
    const bool use_lanes4 = warp_env != nullptr && std::strcmp(warp_env, "4") == 0 &&
        params.bitmask_words <= 32 && params.seqlen_q >= 32768;
    if (use_lanes4) {
        dim3 scatter_grid(cute::ceil_div(params.seqlen_q, 32), params.b, params.h);
        dynamic_compact_csc_scatter_lanes4_kernel<<<scatter_grid, 128, 0, stream>>>(params);
    } else if (use_warp) {
        dim3 scatter_grid(cute::ceil_div(params.seqlen_q, 4), params.b, params.h);
        dynamic_compact_csc_scatter_warp_kernel<<<scatter_grid, 128, 0, stream>>>(params);
    } else {
        const char* block_env = std::getenv("LIGHTNING_MOBA_CSC_BLOCK");
        const int scatter_threads = block_env != nullptr && std::strcmp(block_env, "256") == 0
            ? 256 : block_env != nullptr && std::strcmp(block_env, "64") == 0 ? 64 : 128;
        dim3 scatter_grid(cute::ceil_div(params.seqlen_q, scatter_threads), params.b, params.h);
        dynamic_compact_csc_scatter_kernel<<<scatter_grid, scatter_threads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace flash_dynamic
