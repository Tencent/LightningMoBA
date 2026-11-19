/******************************************************************************
 * KV-Centric Forward Kernel for LightningMoBA
 * 
 * Each CTA processes one KV column (block), computing attention for all Q rows
 * that selected this column. Outputs O_partial + LSE_partial per CSC entry.
 *
 * Grid: (num_columns, batch, num_heads)  
 * Each CTA: fixed KV block, iterates over Q rows from CSC indices
 *
 * Copyright (c) 2025-2026, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <type_traits>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"

namespace LIGHTNING_MOBA_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_K, typename Params,
         bool ForceCscFp32=false>
inline __device__ void compute_moba_attn_1colblock(
    const Params &params,
    const int bidb,
    const int bidh,
    const int n_col,
    const int q_range_begin = 0,
    const int q_range_count = -1,
    const int64_t partial_base = 0
) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const int chunk_size = params.n_lg_block_dim;

    // --- Block info (varlen) ---
    // KV-centric is only exposed through the varlen API.  Use cumulative
    // sequence offsets here; otherwise multi-request packed Q/K/V reads use
    // max-seqlen batch strides and corrupt every request after bidb=0.
    const BlockInfo</*Varlen=*/true> binfo(params, bidb);
    const int seqlen_q = binfo.actual_seqlen_q;
    const int seqlen_k = binfo.actual_seqlen_k;

    if (n_col * chunk_size >= seqlen_k) return;

    // --- CSC metadata ---
    const index_t col_idx = bidb * params.h * params.max_lg_col_num 
                          + bidh * params.max_lg_col_num + n_col;
    const index_t col_offset = reinterpret_cast<index_t*>(params.col_offsets_ptr)[col_idx];
    const int col_count = reinterpret_cast<int*>(params.col_nnz_ptr)[col_idx];
    if (col_count == 0) return;
    const int q_begin_nonnegative = q_range_begin < 0 ? 0 : q_range_begin;
    const int q_begin = q_begin_nonnegative > col_count ? col_count : q_begin_nonnegative;
    const int q_end_requested = q_range_count < 0 ? col_count : q_begin + q_range_count;
    const int q_end_capped = q_end_requested > col_count ? col_count : q_end_requested;
    const int q_end = q_end_capped < q_begin ? q_begin : q_end_capped;
    if (q_begin >= q_end) return;

    int* col_indices = reinterpret_cast<int*>(params.indices_ptr) + col_offset;

    // --- Pointers ---
    Element* k_head_ptr = reinterpret_cast<Element*>(params.k_ptr) 
        + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Element* v_head_ptr = reinterpret_cast<Element*>(params.v_ptr)
        + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Element* q_head_ptr = reinterpret_cast<Element*>(params.q_ptr)
        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + bidh * params.q_head_stride;

    ElementAccum* lse_partial_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr);

    // --- Shared memory layout ---
    // Layout: sQ | sK | sV | optional resident sK1/sV1 | sMobaIdx
    using SmemLayoutKVUse = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutVtUse = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzleUse = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

    static constexpr int kSmemQElems = cosize_v<typename Kernel_traits::SmemLayoutQ>;
    static constexpr int kSmemKVComputeElems = cosize_v<SmemLayoutKVUse>;
    Element* smem_elements = reinterpret_cast<Element*>(smem_);
    Tensor sQ = make_tensor(make_smem_ptr(smem_elements),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(smem_elements + kSmemQElems), SmemLayoutKVUse{});
    Tensor sV = make_tensor(make_smem_ptr(smem_elements + kSmemQElems + kSmemKVComputeElems), SmemLayoutKVUse{});
    Tensor sK1 = make_tensor(make_smem_ptr(smem_elements + kSmemQElems + 2 * kSmemKVComputeElems), SmemLayoutKVUse{});
    Tensor sV1 = make_tensor(make_smem_ptr(smem_elements + kSmemQElems + 3 * kSmemKVComputeElems), SmemLayoutKVUse{});
    constexpr int kSmemKVSlotCount = Kernel_traits::KvResidentKV ? 4 : 2;
    char* after_kv_smem = smem_ + sizeof(Element) * (kSmemQElems + kSmemKVSlotCount * kSmemKVComputeElems);
    Tensor sVt = make_tensor(sV.data(), SmemLayoutVtUse{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), SmemLayoutVtNoSwizzleUse{});
    Tensor sVt1 = make_tensor(sV1.data(), SmemLayoutVtUse{});
    Tensor sVt1NoSwizzle = make_tensor(sV1.data().get(), SmemLayoutVtNoSwizzleUse{});
    int* sMobaIdx_ptr = reinterpret_cast<int*>(after_kv_smem);
    // Use a simple array view (max kBlockM indices needed at a time)
    auto sMobaIdx = make_tensor(make_smem_ptr(sMobaIdx_ptr), make_shape(kBlockM));

    // --- Copy atoms ---
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tKsK1 = gmem_thr_copy_QKV.partition_D(sK1);
    Tensor tVsV1 = gmem_thr_copy_QKV.partition_D(sV1);

    // K/V global tensors for this column
    const int kv_start = n_col * chunk_size;
    Tensor mK = make_tensor(make_gmem_ptr(k_head_ptr),
                            make_shape(seqlen_k, params.d),
                            make_stride(params.k_row_stride, _1{}));
    Tensor gK = local_tile(mK, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);

    Tensor mV = make_tensor(make_gmem_ptr(v_head_ptr),
                            make_shape(seqlen_k, params.d),
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV = local_tile(mV, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);

    // --- MMA setup ---
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tSrK1 = thr_mma.partition_fragment_B(sK1);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor tOrVt1 = thr_mma.partition_fragment_B(sVt1NoSwizzle);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tSsK1 = smem_thr_copy_K.partition_S(sK1);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tOsVt1 = smem_thr_copy_V.partition_S(sVt1);

    // --- Predicates ---
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // --- Mask ---
    const float alibi_slope = 0.0f;
    LIGHTNING_MOBA_NAMESPACE::Mask<Is_causal, false, false> mask(
        seqlen_k, seqlen_q, seqlen_k, Is_causal ? 0 : seqlen_k, alibi_slope);

    // --- O_partial epilogue: use sQ's memory as staging buffer (reused after Q load) ---
    // sOtmp is fp32 [kBlockM, kHeadDim] — overlaps with sQ since they're not used simultaneously
    // Actually sQ IS needed during the inner loop. Use separate staging after inner loop.
    // We'll use sMobaIdx area + extra space for O staging, or just do direct scatter writes.
    // For correctness-first approach: use o_tmp_store pattern with sQ as staging (after inner loop done).

    // =========================================================================
    // Main loop: iterate over Q rows in batches of kBlockM
    // =========================================================================
    const int n_block_start = kv_start / kBlockN;
    const int n_block_end = min((kv_start + chunk_size + kBlockN - 1) / kBlockN,
                                (seqlen_k + kBlockN - 1) / kBlockN);
    // Resident K/V reuse requires two complete tiles and a tile-aligned
    // packed-sequence base. Tail tiles use predicated cp.async loads.
    const bool tma_batch_offset_aligned =
        binfo.sum_s_k < 0 || (binfo.sum_s_k % kBlockN == 0);
    const bool use_resident_kv = Kernel_traits::KvResidentKV &&
        tma_batch_offset_aligned &&
        (n_block_end - n_block_start == 2) &&
        ((n_block_start + 2) * kBlockN <= seqlen_k);

    for (int q_batch_start = q_begin; q_batch_start < q_end; q_batch_start += kBlockM) {
        const int cur_q_count = min(kBlockM, q_end - q_batch_start);
        
        // --- Load Q row indices to smem ---
        load_row_indices_to_smem<kBlockM>(col_indices + q_batch_start, cur_q_count, sMobaIdx);

        // --- Create gather tensor for Q and load to sQ ---
        Tensor gQ = make_gather_tensor_Q<Element, kBlockM, kHeadDim>(
            params, q_head_ptr, cur_q_count, sMobaIdx_ptr);
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);

        int n_block = n_block_start;
        if constexpr (Kernel_traits::KvResidentKV) {
            if (use_resident_kv) {
                if (q_batch_start == q_begin) {
                    {
                        LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block_start), tKsK, tKVcKV, tKVpKV);
                        LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block_start + 1), tKsK1, tKVcKV, tKVpKV);
                        LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block_start), tVsV, tKVcKV, tKVpKV);
                        LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block_start + 1), tVsV1, tKVcKV, tKVpKV);
                        cute::cp_async_fence();
                    }
                }
            } else {
                {
                    // The legacy cp.async path used to mark every K tile as
                    // fully valid.  For packed varlen input that reads past
                    // the allocation whenever the final request ends on a
                    // partial 64-row tile.  Keep the zero-overhead full-tile
                    // path and predicate only the tail tile.
                    if ((n_block + 1) * kBlockN <= seqlen_k) {
                        LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV);
                    } else {
                        LIGHTNING_MOBA_NAMESPACE::copy<false, Is_even_K, true>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                            max(0, seqlen_k - n_block * kBlockN));
                    }
                }
            }
        } else {
            {
                if ((n_block + 1) * kBlockN <= seqlen_k) {
                    LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV);
                } else {
                    LIGHTNING_MOBA_NAMESPACE::copy<false, Is_even_K, true>(
                        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                        max(0, seqlen_k - n_block * kBlockN));
                }
            }
        }

        // Load Q (gathered) into sQ
        LIGHTNING_MOBA_NAMESPACE::copy<false, Is_even_K>(
            gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, cur_q_count);
        cute::cp_async_fence();


        // --- Initialize accumulator ---
        Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        clear(acc_o);

        // Softmax state: need one entry per physical row owned by this thread.
        // acc_s has layout (MMA=size<0>, MMA_M=size<1>, MMA_N=size<2>)
        // For SM80 16x8x16: size<0>=4 (2 rows × 2 cols in the fragment)
        // Each (i_row, mi) pair is a unique row: i_row = i / 2 (0 or 1), giving 2*MMA_M rows
        // Use 2*size<1>(acc_o) to cover all rows
        constexpr int kNumRows = 2 * size<1>(acc_o);  // = 2 * MMA_M
        Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<kNumRows>>{});
        Tensor scores_sum = make_tensor<ElementAccum>(Shape<Int<kNumRows>>{});
        #pragma unroll
        for (int i = 0; i < kNumRows; ++i) {
            scores_max(i) = -INFINITY;
            scores_sum(i) = 0.0f;
        }

        // --- Inner loop over KV sub-blocks ---
        for (; n_block < n_block_end; ++n_block) {
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            if (!(Kernel_traits::KvResidentKV && use_resident_kv)) {
                LIGHTNING_MOBA_NAMESPACE::cp_async_wait<0>();
                __syncthreads();
            }

            if (!(Kernel_traits::KvResidentKV && use_resident_kv)) {
                {
                    if ((n_block + 1) * kBlockN <= seqlen_k) {
                        LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
                    } else {
                        LIGHTNING_MOBA_NAMESPACE::copy<false, Is_even_K, true>(
                            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
                            max(0, seqlen_k - n_block * kBlockN));
                    }
                }
                cute::cp_async_fence();
            }

            // QK^T GEMM
            if constexpr (Kernel_traits::KvResidentKV) {
                if (use_resident_kv && n_block == n_block_start + 1) {
                    LIGHTNING_MOBA_NAMESPACE::gemm<false>(
                        acc_s, tSrQ, tSrK1, tSsQ, tSsK1, tiled_mma,
                        smem_tiled_copy_Q, smem_tiled_copy_K,
                        smem_thr_copy_Q, smem_thr_copy_K);
                } else {
                    LIGHTNING_MOBA_NAMESPACE::gemm<false>(
                        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                        smem_tiled_copy_Q, smem_tiled_copy_K,
                        smem_thr_copy_Q, smem_thr_copy_K);
                }
            } else {
                LIGHTNING_MOBA_NAMESPACE::gemm<false>(
                    acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                    smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K);
            }

            if constexpr (Kernel_traits::KvEarlyKPrefetch) {
                if (!(Kernel_traits::KvResidentKV && use_resident_kv) && n_block < n_block_end - 1) {
                    {
                        if ((n_block + 2) * kBlockN <= seqlen_k) {
                            LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1), tKsK, tKVcKV, tKVpKV);
                        } else {
                            LIGHTNING_MOBA_NAMESPACE::copy<false, Is_even_K, true>(
                                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1), tKsK, tKVcKV, tKVpKV,
                                max(0, seqlen_k - (n_block + 1) * kBlockN));
                        }
                    }
                    cute::cp_async_fence();
                }
            }

            // Apply mask (causal + padding)
            mask.template apply_moba_mask<Is_causal, false>(
                acc_s, n_block * kBlockN,
                (tidx / 32) * 16 + (tidx % 32) / 4,
                kNWarps * 16,
                sMobaIdx, cur_q_count, 0);

            // Wait for V. In the late-V-wait experiment, keep V in flight
            // through softmax and wait only immediately before P*V.
            if constexpr (!Kernel_traits::KvLateVWait) {
                if constexpr (Kernel_traits::KvEarlyKPrefetch) {
                    if (!(Kernel_traits::KvResidentKV && use_resident_kv) && n_block < n_block_end - 1) {
                        LIGHTNING_MOBA_NAMESPACE::cp_async_wait<1>();
                    } else {
                        LIGHTNING_MOBA_NAMESPACE::cp_async_wait<0>();
                    }
                } else {
                    LIGHTNING_MOBA_NAMESPACE::cp_async_wait<0>();
                }
                __syncthreads();
            }

            // Prefetch next K
            if (!Kernel_traits::KvEarlyKPrefetch && !(Kernel_traits::KvResidentKV && use_resident_kv) && n_block < n_block_end - 1) {
                {
                    if ((n_block + 2) * kBlockN <= seqlen_k) {
                        LIGHTNING_MOBA_NAMESPACE::copy<true, Is_even_K>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1), tKsK, tKVcKV, tKVpKV);
                    } else {
                        LIGHTNING_MOBA_NAMESPACE::copy<false, Is_even_K, true>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1), tKsK, tKVcKV, tKVpKV,
                            max(0, seqlen_k - (n_block + 1) * kBlockN));
                    }
                }
                cute::cp_async_fence();
            }

            // Online softmax + rescale O
            // acc_s layout: (MMA=4, MMA_M, MMA_N) where MMA dim has 2 row-groups × 2 col-groups
            // Row index for (i, mi): row_idx = (i / 2) + mi * 2  (i/2 gives 0 or 1)
            // This matches convert_layout_acc_rowcol used by mask/softmax utilities
            #pragma unroll
            for (int mi = 0; mi < size<1>(acc_s); ++mi) {
                #pragma unroll
                for (int ii = 0; ii < 2; ++ii) {  // two row groups per mi
                    const int row_state_idx = mi * 2 + ii;
                    // Find max across all columns for this row
                    ElementAccum row_max_new = scores_max(row_state_idx);
                    #pragma unroll
                    for (int ni = 0; ni < size<2>(acc_s); ++ni) {
                        #pragma unroll
                        for (int ci = 0; ci < 2; ++ci) {  // two col groups
                            row_max_new = max(row_max_new, acc_s(make_coord(ii * 2 + ci, mi, ni)));
                        }
                    }
                    MaxOp<float> max_op;
                    row_max_new = Allreduce<4>::run(row_max_new, max_op);
                    // Rescale previous accumulator
                    const bool row_has_valid_scores = row_max_new != -INFINITY;
                    ElementAccum scale = row_has_valid_scores
                        ? exp2f((scores_max(row_state_idx) - row_max_new) * params.scale_softmax_log2)
                        : 0.0f;
                    scores_max(row_state_idx) = row_max_new;
                    scores_sum(row_state_idx) *= scale;
                    // Scale acc_o for this row
                    #pragma unroll
                    for (int di = 0; di < size<2>(acc_o); ++di) {
                        #pragma unroll
                        for (int ci = 0; ci < 2; ++ci) {
                            acc_o(make_coord(ii * 2 + ci, mi, di)) *= scale;
                        }
                    }
                    // Compute exp and accumulate sum
                    ElementAccum scores_sum_cur = 0.0f;
                    #pragma unroll
                    for (int ni = 0; ni < size<2>(acc_s); ++ni) {
                        #pragma unroll
                        for (int ci = 0; ci < 2; ++ci) {
                            acc_s(make_coord(ii * 2 + ci, mi, ni)) = row_has_valid_scores
                                ? exp2f((acc_s(make_coord(ii * 2 + ci, mi, ni)) - row_max_new) * params.scale_softmax_log2)
                                : 0.0f;
                            scores_sum_cur += acc_s(make_coord(ii * 2 + ci, mi, ni));
                        }
                    }
                    SumOp<float> sum_op;
                    scores_sum(row_state_idx) += Allreduce<4>::run(scores_sum_cur, sum_op);
                }
            }

            if constexpr (Kernel_traits::KvLateVWait) {
                if constexpr (Kernel_traits::KvEarlyKPrefetch) {
                    if (!(Kernel_traits::KvResidentKV && use_resident_kv) && n_block < n_block_end - 1) {
                        LIGHTNING_MOBA_NAMESPACE::cp_async_wait<1>();
                    } else {
                        LIGHTNING_MOBA_NAMESPACE::cp_async_wait<0>();
                    }
                } else {
                    LIGHTNING_MOBA_NAMESPACE::cp_async_wait<0>();
                }
                __syncthreads();
            }

            // P * V GEMM
            Tensor rP = LIGHTNING_MOBA_NAMESPACE::convert_type<Element>(acc_s);
            Tensor tOrP = make_tensor(rP.data(), 
                LIGHTNING_MOBA_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            if constexpr (Kernel_traits::KvResidentKV) {
                if (use_resident_kv && n_block == n_block_start + 1) {
                    LIGHTNING_MOBA_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt1, tOsVt1, tiled_mma,
                        smem_tiled_copy_V, smem_thr_copy_V);
                } else {
                    LIGHTNING_MOBA_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma,
                        smem_tiled_copy_V, smem_thr_copy_V);
                }
            } else {
                LIGHTNING_MOBA_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma,
                    smem_tiled_copy_V, smem_thr_copy_V);
            }
        }

        // =================================================================
        // Epilogue: write O_partial and LSE_partial
        // Use o_tmp_store pattern: acc_o → sQ (as fp32 staging) → gmem
        // =================================================================
        __syncthreads();

        // Normalize acc_o: each row group has its own sum
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_o); ++mi) {
            #pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                const int row_state_idx = mi * 2 + ii;
                ElementAccum inv_sum = (scores_sum(row_state_idx) > 0.0f && scores_sum(row_state_idx) == scores_sum(row_state_idx))
                    ? (1.0f / scores_sum(row_state_idx))
                    : 0.0f;
                #pragma unroll
                for (int di = 0; di < size<2>(acc_o); ++di) {
                    #pragma unroll
                    for (int ci = 0; ci < 2; ++ci) {
                        ElementAccum o_val = acc_o(make_coord(ii * 2 + ci, mi, di)) * inv_sum;
                        acc_o(make_coord(ii * 2 + ci, mi, di)) = (o_val == o_val) ? o_val : 0.0f;
                    }
                }
            }
        }

        // Write O_partial using scatter pattern (each thread writes its acc elements)
        // taccOcO gives (row, col) for each element in acc_o
        auto caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        auto taccOcO = thr_mma.partition_C(caccO);

        if constexpr (ForceCscFp32) {
            ElementAccum* o_partial_ptr = reinterpret_cast<ElementAccum*>(params.o_tmp_ptr);
            #pragma unroll
            for (int mi = 0; mi < size<1>(acc_o); ++mi) {
                #pragma unroll
                for (int di = 0; di < size<2>(acc_o); ++di) {
                    #pragma unroll
                    for (int i = 0; i < size<0>(acc_o); ++i) {
                        const int row = get<0>(taccOcO(i, mi, di));
                        const int col = get<1>(taccOcO(i, mi, di));
                        if (row < cur_q_count && col < params.d) {
                            const index_t csc_offset = col_offset + q_batch_start + row;
                            
                            o_partial_ptr[csc_offset * params.d + col] = acc_o(make_coord(i, mi, di));
                        }
                    }
                }
            }
        } else {
            if (params.kv_use_slot_major) {
                ElementAccum* o_partial_ptr = reinterpret_cast<ElementAccum*>(params.o_tmp_ptr);
                #pragma unroll
                for (int mi = 0; mi < size<1>(acc_o); ++mi) {
                    #pragma unroll
                    for (int di = 0; di < size<2>(acc_o); ++di) {
                        #pragma unroll
                        for (int i = 0; i < size<0>(acc_o); ++i) {
                            const int row = get<0>(taccOcO(i, mi, di));
                            const int col = get<1>(taccOcO(i, mi, di));
                            if (row < cur_q_count && col < params.d) {
                                const index_t csc_offset = col_offset + q_batch_start + row;
                                const int slot = params.kv_csc_to_slot_ptr[csc_offset];
                                if (slot >= 0) {
                                    const int q_global = params.cu_seqlens_q[bidb] + sMobaIdx_ptr[row];
                                    const index_t out_idx =
                                        slot * params.kv_slot_o_slot_stride +
                                        q_global * params.kv_slot_o_row_stride +
                                        bidh * params.kv_slot_o_head_stride +
                                        col;
                                    o_partial_ptr[out_idx] = acc_o(make_coord(i, mi, di));
                                }
                            }
                        }
                    }
                }
            } else if (params.kv_partial_dtype == 0) {
                const int output_dim = params.kv_partial_v_dim > 0 ? params.kv_partial_v_dim : params.d;
                ElementAccum* o_partial_ptr = reinterpret_cast<ElementAccum*>(params.o_tmp_ptr);
                #pragma unroll
                for (int mi = 0; mi < size<1>(acc_o); ++mi) {
                    #pragma unroll
                    for (int di = 0; di < size<2>(acc_o); ++di) {
                        #pragma unroll
                        for (int i = 0; i < size<0>(acc_o); ++i) {
                            const int row = get<0>(taccOcO(i, mi, di));
                            const int col = get<1>(taccOcO(i, mi, di));
                            if (row < cur_q_count && col < output_dim) {
                                const index_t out_row = col_offset + q_batch_start + row;
                                o_partial_ptr[(out_row - partial_base) * output_dim + col] = acc_o(make_coord(i, mi, di));
                            }
                        }
                    }
                }
            } else {
                Element* o_partial_ptr = reinterpret_cast<Element*>(params.o_tmp_ptr);
                #pragma unroll
                for (int mi = 0; mi < size<1>(acc_o); ++mi) {
                    #pragma unroll
                    for (int di = 0; di < size<2>(acc_o); ++di) {
                        #pragma unroll
                        for (int i = 0; i < size<0>(acc_o); ++i) {
                            const int row = get<0>(taccOcO(i, mi, di));
                            const int col = get<1>(taccOcO(i, mi, di));
                            if (row < cur_q_count && col < params.d) {
                                const index_t out_row = col_offset + q_batch_start + row;
                                o_partial_ptr[out_row * params.d + col] = Element(acc_o(make_coord(i, mi, di)));
                            }
                        }
                    }
                }
            }
        }

        // Write LSE_partial: need to write one per physical row.
        // Use taccOcO to find which elements correspond to col=0 and extract the row.
        // For each (ii, mi) row group, compute LSE and write via the thread that owns it.
        // The thread that holds element (ii*2, mi, 0) in acc_o holds row get<0>(taccOcO(ii*2, mi, 0))
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_o); ++mi) {
            #pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                const int row = get<0>(taccOcO(ii * 2, mi, 0));
                const int col = get<1>(taccOcO(ii * 2, mi, 0));
                if (col == 0 && row < cur_q_count) {
                    const int row_state_idx = mi * 2 + ii;
                    // LSE = raw_max * softmax_scale + log(sum)
                    // softmax_scale = scale_softmax_log2 / log2(e) = scale_softmax_log2 / 1.44269504
                    ElementAccum lse_val = (scores_sum(row_state_idx) > 0.0f && scores_sum(row_state_idx) == scores_sum(row_state_idx))
                        ? scores_max(row_state_idx) * (params.scale_softmax_log2 / 1.44269504f)
                            + __logf(scores_sum(row_state_idx))
                        : -INFINITY;
                    const index_t lse_idx = col_offset + q_batch_start + row;
                    if constexpr (ForceCscFp32) {
                        
                        lse_partial_ptr[lse_idx] = lse_val;
                    } else {
                        if (params.kv_use_slot_major) {
                            const int slot = params.kv_csc_to_slot_ptr[lse_idx];
                            if (slot >= 0) {
                                const int q_global = params.cu_seqlens_q[bidb] + sMobaIdx_ptr[row];
                                const index_t slot_lse_idx =
                                    slot * params.kv_slot_lse_slot_stride +
                                    q_global * params.kv_slot_lse_row_stride +
                                    bidh * params.kv_slot_lse_head_stride;
                                lse_partial_ptr[slot_lse_idx] = lse_val;
                            }
                        } else {
                            lse_partial_ptr[lse_idx - partial_base] = lse_val;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }  // end q_batch loop
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_K, typename Params>
inline __device__ void compute_moba_attn_kv_centric(const Params &params) {
    const int n_col = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;

    compute_moba_attn_1colblock<Kernel_traits, Is_causal, Is_even_K>(
        params, bidb, bidh, n_col);
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_K, typename Params>
inline __device__ void compute_moba_attn_kv_centric_csc_fp32(const Params &params) {
    const int n_col = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;

    compute_moba_attn_1colblock<Kernel_traits, Is_causal, Is_even_K, Params, true>(
        params, bidb, bidh, n_col);
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_K, typename Params>
inline __device__ void compute_moba_attn_kv_centric_worklist(
    const Params &params, const int64_t partial_base = 0, const int64_t partial_end = 0) {
    const int work_idx = blockIdx.x;
    if (work_idx >= params.kv_work_count) return;
    const int* work = params.kv_worklist_ptr + work_idx * 5;
    const int bidb = work[0];
    const int bidh = work[1];
    const int n_col = work[2];
    int q_begin = work[3];
    int q_count = work[4];
    if (q_count <= 0 || bidb < 0 || bidb >= params.b ||
        bidh < 0 || bidh >= params.h || n_col < 0 || n_col >= params.max_lg_col_num) {
        return;
    }
    if (partial_end > partial_base) {
        const int64_t col = (static_cast<int64_t>(bidb) * params.h + bidh)
            * params.max_lg_col_num + n_col;
        const int64_t offset = reinterpret_cast<const int64_t*>(params.col_offsets_ptr)[col];
        const int64_t begin = offset + q_begin;
        const int64_t end = begin + q_count;
        if (begin >= partial_end || end <= partial_base) return;
        const int64_t clipped_begin = begin > partial_base ? begin : partial_base;
        const int64_t clipped_end = end < partial_end ? end : partial_end;
        q_begin += static_cast<int>(clipped_begin - begin);
        q_count = static_cast<int>(clipped_end - clipped_begin);
    }

    compute_moba_attn_1colblock<Kernel_traits, Is_causal, Is_even_K>(
        params, bidb, bidh, n_col, q_begin, q_count, partial_base);
}

} // namespace LIGHTNING_MOBA_NAMESPACE
