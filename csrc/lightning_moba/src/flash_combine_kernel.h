/******************************************************************************
 * Combine Kernel for KV-Centric LightningMoBA
 *
 * Merges partial outputs from KV-centric attention kernel using online softmax.
 * Each thread block handles a tile of Q positions, merging their topK partial
 * outputs (O_partial + LSE_partial) into the final output.
 *
 * Grid: (ceil(total_q / TILE_M), num_heads, batch)
 * Block: (NUM_THREADS)
 *
 * Input layout:
 *   O_partial: [total_nnz, head_dim]  fp32 — indexed by CSC offset
 *   LSE_partial: [total_nnz]          fp32 — indexed by CSC offset
 *   Q2col_offsets: [total_q, num_heads, topk]  int64 — reverse index
 *   Q2col_valid: [total_q, num_heads]  int32 — number of valid partials
 *
 * Output:
 *   O_out: [total_q, num_heads, head_dim]  bf16/fp16
 *   LSE_out: [total_q, num_heads]  fp32 (optional)
 *
 * Copyright (c) 2025-2026, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

namespace LIGHTNING_MOBA_NAMESPACE {

// =============================================================================
// Combine kernel parameters
// =============================================================================

struct Lightning_moba_combine_params {
    // Input partial results (from KV-centric kernel)
    const void* __restrict__ o_partial_ptr;    // [total_nnz, head_dim], fp32/bf16/fp16
    const float* __restrict__ lse_partial_ptr; // [total_nnz]
    
    // Reverse index: Q -> CSC offsets
    const int* __restrict__ q2col_offsets_ptr;      // [total_q, num_heads, topk]
    const int* __restrict__ q2col_valid_ptr;        // [total_q, num_heads]
    
    // Output
    void* __restrict__ o_out_ptr;        // [total_q, num_heads, head_dim] bf16/fp16
    float* __restrict__ lse_out_ptr;     // [total_q, num_heads] fp32 (optional, can be null)
    
    // Strides for O_partial
    int o_partial_row_stride;   // = head_dim
    
    // Strides for Q2col_offsets
    int q2col_q_stride;         // = num_heads * topk
    int q2col_h_stride;         // = topk
    
    // Strides for output
    int o_out_row_stride;       // stride between Q positions
    int o_out_head_stride;      // stride between heads
    
    // Dimensions
    int total_q;
    int num_heads;
    int head_dim;
    int topk;
    int partial_dtype;  // 0 = fp32, 1 = bf16, 2 = fp16
};

struct Lightning_moba_slot_combine_params {
    const float* __restrict__ o_partial_ptr;   // [topk, total_q, num_heads, head_dim]
    const float* __restrict__ lse_partial_ptr; // [topk, total_q, num_heads]
    const int* __restrict__ split_count_ptr;   // [total_q * num_heads]
    void* __restrict__ o_out_ptr;              // [total_q, num_heads, head_dim]
    float* __restrict__ lse_out_ptr;

    int64_t o_partial_slot_stride;
    int64_t o_partial_row_stride;
    int64_t o_partial_head_stride;
    int64_t lse_partial_slot_stride;
    int64_t lse_partial_row_stride;
    int64_t lse_partial_head_stride;
    int64_t o_out_row_stride;
    int64_t o_out_head_stride;
    int total_q;
    int num_heads;
    int head_dim;
    int topk;
};

struct Lightning_moba_q2col_params {
    const int64_t* __restrict__ col_offsets_ptr;  // [batch, num_heads, num_columns]
    const int* __restrict__ col_nnz_ptr;          // [batch, num_heads, num_columns]
    const int* __restrict__ indices_ptr;          // [total_nnz], sequence-local q row
    const int* __restrict__ cu_seqlens_q_ptr;     // [batch + 1]
    int* __restrict__ q2col_offsets_ptr;          // [total_q * num_heads, topk]
    int* __restrict__ q2col_valid_ptr;            // [total_q * num_heads]

    int batch_size;
    int num_heads;
    int num_columns;
    int total_q;
    int topk;

    int col_offsets_batch_stride;
    int col_offsets_head_stride;
    int col_nnz_batch_stride;
    int col_nnz_head_stride;
};

struct Lightning_moba_slot_meta_params {
    const int64_t* __restrict__ col_offsets_ptr;  // [batch, num_heads, num_columns]
    const int* __restrict__ col_nnz_ptr;          // [batch, num_heads, num_columns]
    const int* __restrict__ indices_ptr;          // [total_nnz], sequence-local q row
    const int* __restrict__ cu_seqlens_q_ptr;     // [batch + 1]
    int* __restrict__ csc_to_slot_ptr;            // [total_nnz]
    int* __restrict__ split_count_ptr;            // [total_q * num_heads]

    int batch_size;
    int num_heads;
    int num_columns;
    int total_q;
    int total_nnz;
    int topk;

    int col_offsets_batch_stride;
    int col_offsets_head_stride;
    int col_nnz_batch_stride;
    int col_nnz_head_stride;
};

#ifdef __CUDACC__

__device__ inline float lightning_moba_partial_to_float(float x) {
    return x;
}

__device__ inline float lightning_moba_partial_to_float(__half x) {
    return __half2float(x);
}

__device__ inline float lightning_moba_partial_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__global__ void lightning_moba_build_q2col_kernel(const Lightning_moba_q2col_params params) {
    // Grid: (num_columns, num_heads, batch)
    const int col = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;

    if (b >= params.batch_size || h >= params.num_heads || col >= params.num_columns) {
        return;
    }

    const int meta_idx_offsets =
        b * params.col_offsets_batch_stride + h * params.col_offsets_head_stride + col;
    const int meta_idx_nnz =
        b * params.col_nnz_batch_stride + h * params.col_nnz_head_stride + col;
    const int64_t offset = params.col_offsets_ptr[meta_idx_offsets];
    const int count = params.col_nnz_ptr[meta_idx_nnz];
    const int q_base = params.cu_seqlens_q_ptr[b];

    for (int i = tid; i < count; i += blockDim.x) {
        const int q_local = params.indices_ptr[offset + i];
        const int q_global = q_base + q_local;
        if (q_global < 0 || q_global >= params.total_q) {
            continue;
        }
        const int qh_idx = q_global * params.num_heads + h;
        const int slot = atomicAdd(params.q2col_valid_ptr + qh_idx, 1);
        if (slot < params.topk) {
            params.q2col_offsets_ptr[static_cast<int64_t>(qh_idx) * params.topk + slot] =
                static_cast<int>(offset + i);
        } else {
            // Keep combine bounded even if malformed input produces too many partials.
            atomicMin(params.q2col_valid_ptr + qh_idx, params.topk);
        }
    }
}

__global__ void lightning_moba_build_slot_meta_kernel(const Lightning_moba_slot_meta_params params) {
    const int col = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;

    if (b >= params.batch_size || h >= params.num_heads || col >= params.num_columns) {
        return;
    }

    const int meta_idx_offsets =
        b * params.col_offsets_batch_stride + h * params.col_offsets_head_stride + col;
    const int meta_idx_nnz =
        b * params.col_nnz_batch_stride + h * params.col_nnz_head_stride + col;
    const int64_t offset = params.col_offsets_ptr[meta_idx_offsets];
    const int count = params.col_nnz_ptr[meta_idx_nnz];
    const int q_base = params.cu_seqlens_q_ptr[b];

    for (int i = tid; i < count; i += blockDim.x) {
        const int64_t csc_offset64 = offset + i;
        if (csc_offset64 < 0 || csc_offset64 >= params.total_nnz) {
            continue;
        }
        const int csc_offset = static_cast<int>(csc_offset64);
        const int q_local = params.indices_ptr[csc_offset];
        const int q_global = q_base + q_local;
        if (q_global < 0 || q_global >= params.total_q) {
            params.csc_to_slot_ptr[csc_offset] = -1;
            continue;
        }

        const int qh_idx = q_global * params.num_heads + h;
        const int slot = atomicAdd(params.split_count_ptr + qh_idx, 1);
        if (slot < params.topk) {
            params.csc_to_slot_ptr[csc_offset] = slot;
        } else {
            params.csc_to_slot_ptr[csc_offset] = -1;
            atomicMin(params.split_count_ptr + qh_idx, params.topk);
        }
    }
}

// =============================================================================
// Combine kernel: one thread per (q, head, d_chunk) — simple version
// =============================================================================

// =============================================================================
// Optimized combine: warp-level, vectorized loads
// =============================================================================

template<
    int HEAD_DIM,
    int TOPK_MAX,
    typename PartialType,
    typename OutputType,
    bool PREFETCH_PARTIALS = false,
    bool SHARED_WEIGHTS = false,
    bool PARALLEL_WEIGHTS = false>
__global__ void lightning_moba_combine_kernel_v2(const Lightning_moba_combine_params params) {
    // Grid: (total_q, num_heads, 1)
    // Block: (HEAD_DIM) — one thread per head-dim element
    // Each thread block handles exactly one (q, h) pair
    
    const int q_idx = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int d_idx = threadIdx.x;
    
    if (q_idx >= params.total_q) return;
    if (d_idx >= params.head_dim) return;

    const PartialType* o_partial_ptr = reinterpret_cast<const PartialType*>(params.o_partial_ptr);
    
    const int qh_idx = q_idx * params.num_heads + h_idx;
    const int n_valid = min(params.q2col_valid_ptr[qh_idx], params.topk);
    
    if (n_valid == 0) {
        const int out_offset = q_idx * params.o_out_row_stride + h_idx * params.o_out_head_stride + d_idx;
        reinterpret_cast<OutputType*>(params.o_out_ptr)[out_offset] = OutputType(0.0f);
        if (params.lse_out_ptr && d_idx == 0) {
            params.lse_out_ptr[qh_idx] = -INFINITY;
        }
        return;
    }

    __shared__ int s_csc_offsets[TOPK_MAX];
    __shared__ float s_partial_lse[TOPK_MAX];
    __shared__ float s_scale_prev[TOPK_MAX];
    __shared__ float s_scale_new[TOPK_MAX];
    __shared__ float s_row_max;
    __shared__ float s_row_sum;

    const int* q2col_ptr = params.q2col_offsets_ptr +
        q_idx * params.q2col_q_stride + h_idx * params.q2col_h_stride;

    for (int s = d_idx; s < TOPK_MAX; s += blockDim.x) {
        if (s < n_valid) {
            const int csc_offset = q2col_ptr[s];
            s_csc_offsets[s] = csc_offset;
            s_partial_lse[s] = params.lse_partial_ptr[csc_offset];
        } else {
            s_csc_offsets[s] = 0;
            s_partial_lse[s] = -INFINITY;
        }
    }
    __syncthreads();

    if (n_valid == 1) {
        const int out_offset = q_idx * params.o_out_row_stride + h_idx * params.o_out_head_stride + d_idx;
        const float partial_lse = s_partial_lse[0];
        if (!isfinite(partial_lse)) {
            reinterpret_cast<OutputType*>(params.o_out_ptr)[out_offset] = OutputType(0.0f);
            if (params.lse_out_ptr && d_idx == 0) {
                params.lse_out_ptr[qh_idx] = -INFINITY;
            }
            return;
        }
        const int csc_offset = s_csc_offsets[0];
        const float partial_o = lightning_moba_partial_to_float(o_partial_ptr[
            static_cast<int64_t>(csc_offset) * params.o_partial_row_stride + d_idx]);
        reinterpret_cast<OutputType*>(params.o_out_ptr)[out_offset] = OutputType(partial_o);
        if (params.lse_out_ptr && d_idx == 0) {
            params.lse_out_ptr[qh_idx] = partial_lse;
        }
        return;
    }
    
    // Each thread loads one d-element from each partial and merges
    float acc_o = 0.0f;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    
    if constexpr (PARALLEL_WEIGHTS) {
        // The first half-warp computes the mathematically equivalent
        // log-sum-exp weights in parallel.  Unlike SHARED_WEIGHTS, this avoids
        // a 16-step thread-0 online recurrence.  The reduction order changes,
        // so this specialization is accuracy-gated rather than bitwise-exact.
        if (d_idx < TOPK_MAX) {
            constexpr unsigned kTopkMask = (1u << TOPK_MAX) - 1u;
            const float partial_lse = s_partial_lse[d_idx];
            float lane_max = isfinite(partial_lse) ? partial_lse : -INFINITY;
            #pragma unroll
            for (int offset = TOPK_MAX / 2; offset > 0; offset >>= 1) {
                lane_max = fmaxf(
                    lane_max,
                    __shfl_xor_sync(kTopkMask, lane_max, offset, TOPK_MAX));
            }

            const float lane_weight =
                isfinite(partial_lse) && isfinite(lane_max)
                ? exp2f((partial_lse - lane_max) * 1.44269504f)
                : 0.0f;
            float weight_sum = lane_weight;
            #pragma unroll
            for (int offset = TOPK_MAX / 2; offset > 0; offset >>= 1) {
                weight_sum += __shfl_xor_sync(
                    kTopkMask, weight_sum, offset, TOPK_MAX);
            }
            s_scale_new[d_idx] = lane_weight;
            if (d_idx == 0) {
                s_row_max = lane_max;
                s_row_sum = weight_sum;
            }
        }
        __syncthreads();

        float prefetched_o[TOPK_MAX];
        #pragma unroll
        for (int s = 0; s < TOPK_MAX; ++s) {
            const int csc_offset = s_csc_offsets[s];
            prefetched_o[s] = (s < n_valid && isfinite(s_partial_lse[s]))
                ? lightning_moba_partial_to_float(o_partial_ptr[
                    static_cast<int64_t>(csc_offset) *
                        params.o_partial_row_stride + d_idx])
                : 0.0f;
        }
        #pragma unroll
        for (int s = 0; s < TOPK_MAX; ++s) {
            acc_o = fmaf(s_scale_new[s], prefetched_o[s], acc_o);
        }
        row_max = s_row_max;
        row_sum = s_row_sum;
    } else if constexpr (SHARED_WEIGHTS) {
        // The merge weights depend only on the LSE values, not on d_idx.
        // Compute the exact online recurrence once per CTA, while retaining
        // the original per-dimension accumulation order below.
        if (d_idx == 0) {
            #pragma unroll
            for (int s = 0; s < TOPK_MAX; ++s) {
                if (s < n_valid && isfinite(s_partial_lse[s])) {
                    const float partial_lse = s_partial_lse[s];
                    const float new_max = fmaxf(row_max, partial_lse);
                    const float scale_prev = isfinite(row_max)
                        ? exp2f((row_max - new_max) * 1.44269504f) : 0.0f;
                    const float scale_new =
                        exp2f((partial_lse - new_max) * 1.44269504f);
                    s_scale_prev[s] = scale_prev;
                    s_scale_new[s] = scale_new;
                    row_sum = row_sum * scale_prev + scale_new;
                    row_max = new_max;
                } else {
                    s_scale_prev[s] = 1.0f;
                    s_scale_new[s] = 0.0f;
                }
            }
            s_row_max = row_max;
            s_row_sum = row_sum;
        }
        __syncthreads();

        float prefetched_o[TOPK_MAX];
        #pragma unroll
        for (int s = 0; s < TOPK_MAX; ++s) {
            const int csc_offset = s_csc_offsets[s];
            prefetched_o[s] = (s < n_valid && isfinite(s_partial_lse[s]))
                ? lightning_moba_partial_to_float(o_partial_ptr[
                    static_cast<int64_t>(csc_offset) *
                        params.o_partial_row_stride + d_idx])
                : 0.0f;
        }
        #pragma unroll
        for (int s = 0; s < TOPK_MAX; ++s) {
            acc_o = acc_o * s_scale_prev[s] +
                s_scale_new[s] * prefetched_o[s];
        }
        row_max = s_row_max;
        row_sum = s_row_sum;
    } else if constexpr (PREFETCH_PARTIALS) {
        // The CSC rows selected by TopK are far apart and have almost no L2
        // reuse. Issue all independent row loads before consuming them so the
        // warp can expose memory-level parallelism instead of serializing a
        // load-use dependency for every split.
        float prefetched_o[TOPK_MAX];
        #pragma unroll
        for (int s = 0; s < TOPK_MAX; ++s) {
            const int csc_offset = s_csc_offsets[s];
            prefetched_o[s] = (s < n_valid && isfinite(s_partial_lse[s]))
                ? lightning_moba_partial_to_float(o_partial_ptr[
                    static_cast<int64_t>(csc_offset) * params.o_partial_row_stride + d_idx])
                : 0.0f;
        }
        #pragma unroll
        for (int s = 0; s < TOPK_MAX; ++s) {
            if (s < n_valid) {
                const float partial_lse = s_partial_lse[s];
                if (!isfinite(partial_lse)) {
                    continue;
                }
                const float new_max = fmaxf(row_max, partial_lse);
                const float scale_prev = isfinite(row_max)
                    ? exp2f((row_max - new_max) * 1.44269504f) : 0.0f;
                const float scale_new =
                    exp2f((partial_lse - new_max) * 1.44269504f);
                row_sum = row_sum * scale_prev + scale_new;
                acc_o = acc_o * scale_prev + scale_new * prefetched_o[s];
                row_max = new_max;
            }
        }
    } else {
        #pragma unroll
        for (int s = 0; s < TOPK_MAX; ++s) {
            if (s < n_valid) {
                const int csc_offset = s_csc_offsets[s];
                const float partial_lse = s_partial_lse[s];
                const float partial_o = lightning_moba_partial_to_float(o_partial_ptr[
                    static_cast<int64_t>(csc_offset) * params.o_partial_row_stride + d_idx]);
                if (!isfinite(partial_lse)) {
                    continue;
                }
                const float new_max = fmaxf(row_max, partial_lse);
                const float scale_prev = isfinite(row_max)
                    ? exp2f((row_max - new_max) * 1.44269504f) : 0.0f;
                const float scale_new =
                    exp2f((partial_lse - new_max) * 1.44269504f);
                row_sum = row_sum * scale_prev + scale_new;
                acc_o = acc_o * scale_prev + scale_new * partial_o;
                row_max = new_max;
            }
        }
    }
    
    // Normalize
    const float safe_sum = (row_sum > 0.0f) ? row_sum : 1.0f;
    acc_o = acc_o / safe_sum;
    
    // Write output
    const int out_offset = q_idx * params.o_out_row_stride + h_idx * params.o_out_head_stride + d_idx;
    reinterpret_cast<OutputType*>(params.o_out_ptr)[out_offset] = OutputType(acc_o);
    
    // Write LSE
    if (params.lse_out_ptr && d_idx == 0) {
        params.lse_out_ptr[qh_idx] = row_max + logf(safe_sum);
    }
}

// =============================================================================
// Launch wrapper
// =============================================================================

template<typename PartialType, typename OutputType>
void run_lightning_moba_combine(
    const Lightning_moba_combine_params &params,
    cudaStream_t stream
) {
    // Use v2 kernel: one block per (q, h), HEAD_DIM threads per block.  The
    // PARALLEL_WEIGHTS specialization is intentionally reserved for topk=16:
    // its warp-shuffle LSE reduction cannot represent topk=64.  The generic
    // specialization below keeps LSEs in shared memory and performs an online
    // softmax merge, so it is valid for both 32 and 64 partials.
    dim3 grid(params.total_q, params.num_heads, 1);
    dim3 block(params.head_dim, 1, 1);  // HEAD_DIM threads

    if (params.topk <= 32) {
        if (params.head_dim <= 128) {
            lightning_moba_combine_kernel_v2<128, 32, PartialType, OutputType>
                <<<grid, block, 0, stream>>>(params);
        } else {
            lightning_moba_combine_kernel_v2<256, 32, PartialType, OutputType>
                <<<grid, block, 0, stream>>>(params);
        }
    } else {
        if (params.head_dim <= 128) {
            lightning_moba_combine_kernel_v2<128, 64, PartialType, OutputType>
                <<<grid, block, 0, stream>>>(params);
        } else {
            lightning_moba_combine_kernel_v2<256, 64, PartialType, OutputType>
                <<<grid, block, 0, stream>>>(params);
        }
    }
}

template<typename PartialType, typename OutputType>
void run_lightning_moba_combine_topk16_prefetch(
    const Lightning_moba_combine_params &params,
    cudaStream_t stream
) {
    dim3 grid(params.total_q, params.num_heads, 1);
    dim3 block(params.head_dim, 1, 1);
    lightning_moba_combine_kernel_v2<
        128, 16, PartialType, OutputType, true>
        <<<grid, block, 0, stream>>>(params);
}

template<typename PartialType, typename OutputType>
void run_lightning_moba_combine_topk16_parallel_weights(
    const Lightning_moba_combine_params &params,
    cudaStream_t stream
) {
    dim3 grid(params.total_q, params.num_heads, 1);
    dim3 block(params.head_dim, 1, 1);
    lightning_moba_combine_kernel_v2<
        128, 16, PartialType, OutputType, true, false, true>
        <<<grid, block, 0, stream>>>(params);
}

template<int TILE_M, int HEAD_DIM, int TOPK_MAX, typename OutputType>
__global__ void lightning_moba_combine_slot_tiled_kernel(const Lightning_moba_slot_combine_params params) {
    // MSA-style combine scheduling: one CTA covers a tile of flattened
    // (q, head) rows and the full D=128 vector. This keeps the split-major
    // partial layout but avoids launching one CTA per single (q, head) row.
    const int qh_tile_start = blockIdx.x * TILE_M;
    const int tid = threadIdx.x;
    const int total_qh = params.total_q * params.num_heads;

    for (int linear = tid; linear < TILE_M * HEAD_DIM; linear += blockDim.x) {
        const int row_local = linear / HEAD_DIM;
        const int d_idx = linear - row_local * HEAD_DIM;
        const int qh_idx = qh_tile_start + row_local;
        if (qh_idx >= total_qh || d_idx >= params.head_dim) {
            continue;
        }

        const int q_idx = qh_idx / params.num_heads;
        const int h_idx = qh_idx - q_idx * params.num_heads;
        const int n_valid = min(params.split_count_ptr[qh_idx], params.topk);
        const int64_t out_offset =
            q_idx * params.o_out_row_stride + h_idx * params.o_out_head_stride + d_idx;

        if (n_valid == 0) {
            reinterpret_cast<OutputType*>(params.o_out_ptr)[out_offset] = OutputType(0.0f);
            if (params.lse_out_ptr && d_idx == 0) {
                params.lse_out_ptr[qh_idx] = -INFINITY;
            }
            continue;
        }

        if (n_valid == 1) {
            const int64_t lse_idx =
                q_idx * params.lse_partial_row_stride +
                h_idx * params.lse_partial_head_stride;
            const float partial_lse = params.lse_partial_ptr[lse_idx];
            if (!isfinite(partial_lse)) {
                reinterpret_cast<OutputType*>(params.o_out_ptr)[out_offset] = OutputType(0.0f);
                if (params.lse_out_ptr && d_idx == 0) {
                    params.lse_out_ptr[qh_idx] = -INFINITY;
                }
                continue;
            }

            const int64_t o_idx =
                q_idx * params.o_partial_row_stride +
                h_idx * params.o_partial_head_stride +
                d_idx;
            reinterpret_cast<OutputType*>(params.o_out_ptr)[out_offset] =
                OutputType(params.o_partial_ptr[o_idx]);
            if (params.lse_out_ptr && d_idx == 0) {
                params.lse_out_ptr[qh_idx] = partial_lse;
            }
            continue;
        }

        float acc_o = 0.0f;
        float row_max = -INFINITY;
        float row_sum = 0.0f;

        #pragma unroll
        for (int s = 0; s < TOPK_MAX; ++s) {
            if (s < n_valid) {
                const int64_t lse_idx =
                    s * params.lse_partial_slot_stride +
                    q_idx * params.lse_partial_row_stride +
                    h_idx * params.lse_partial_head_stride;
                const float partial_lse = params.lse_partial_ptr[lse_idx];
                if (!isfinite(partial_lse)) {
                    continue;
                }

                const int64_t o_idx =
                    s * params.o_partial_slot_stride +
                    q_idx * params.o_partial_row_stride +
                    h_idx * params.o_partial_head_stride +
                    d_idx;
                const float partial_o = params.o_partial_ptr[o_idx];

                const float new_max = fmaxf(row_max, partial_lse);
                const float scale_prev = isfinite(row_max) ? exp2f((row_max - new_max) * 1.44269504f) : 0.0f;
                const float scale_new = exp2f((partial_lse - new_max) * 1.44269504f);

                row_sum = row_sum * scale_prev + scale_new;
                acc_o = acc_o * scale_prev + scale_new * partial_o;
                row_max = new_max;
            }
        }

        const float safe_sum = (row_sum > 0.0f) ? row_sum : 1.0f;
        acc_o = acc_o / safe_sum;

        reinterpret_cast<OutputType*>(params.o_out_ptr)[out_offset] = OutputType(acc_o);
        if (params.lse_out_ptr && d_idx == 0) {
            params.lse_out_ptr[qh_idx] = row_max + logf(safe_sum);
        }
    }
}

template<typename OutputType>
void run_lightning_moba_combine_slot(
    const Lightning_moba_slot_combine_params &params,
    cudaStream_t stream
) {
    constexpr int TOPK_MAX = 32;
    constexpr int TILE_M = 64;

    dim3 grid((params.total_q * params.num_heads + TILE_M - 1) / TILE_M, 1, 1);
    dim3 block(256, 1, 1);

    if (params.head_dim <= 128) {
        lightning_moba_combine_slot_tiled_kernel<TILE_M, 128, TOPK_MAX, OutputType>
            <<<grid, block, 0, stream>>>(params);
    } else if (params.head_dim <= 256) {
        lightning_moba_combine_slot_tiled_kernel<TILE_M, 256, TOPK_MAX, OutputType>
            <<<grid, block, 0, stream>>>(params);
    } else {
        lightning_moba_combine_slot_tiled_kernel<TILE_M, 128, TOPK_MAX, OutputType>
            <<<grid, block, 0, stream>>>(params);
    }
}

#endif  // __CUDACC__

void run_lightning_moba_combine_dispatch(
    const Lightning_moba_combine_params &params,
    bool is_bf16,
    cudaStream_t stream
);

void run_lightning_moba_combine_slot_dispatch(
    const Lightning_moba_slot_combine_params &params,
    bool is_bf16,
    cudaStream_t stream
);

void run_lightning_moba_build_q2col_dispatch(
    const Lightning_moba_q2col_params &params,
    cudaStream_t stream
);

void run_lightning_moba_build_slot_meta_dispatch(
    const Lightning_moba_slot_meta_params &params,
    cudaStream_t stream
);

}  // namespace LIGHTNING_MOBA_NAMESPACE
