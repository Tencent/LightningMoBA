#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

namespace flash_dynamic {

// The auto_max policy searches an offset in [-60, 0].  16384 bins give a
// log2 resolution of 0.00366, comparable with the historical 14-step binary
// search, while reading the score matrix only once.
constexpr int kFlashPrefillHistogramBins = 16384;
constexpr float kFlashPrefillOffsetFloor = -60.0f;

struct FlashPrefillPatternParams {
    const float* scores;       // [Q blocks, heads, K blocks]
    float* row_max;            // [Q blocks, heads]
    float* threshold;          // scalar log2 offset
    int* histogram;
    // exact_meta = {selected_bin, tie_budget, tie_counter, forced_count}.
    // It is used only by the exact-density comparison mode.
    int* exact_meta;
    uint8_t* block_mask;       // [Q blocks, heads, K blocks]
    int* block_row_nnz;        // [Q blocks, heads]
    int* row_nnz;              // [tokens, heads]
    int64_t* col_offsets;      // [1, heads, K blocks]
    int* col_nnz;              // token rows per CSC column
    int* csc_rows;
    int blocks;
    int heads;
    int tokens;
    int chunk_size;
    int sink;
    int window;
    // Keep the target in FP64: exact-budget mode rounds density * legal and
    // must recover the requested integer cardinality without FP32 drift.
    double density;
    bool auto_density;
    bool exact_density;
};

__device__ __forceinline__ float fp_pattern_warp_max(float value) {
#pragma unroll
    for (int delta = 16; delta; delta >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
    }
    return value;
}

__global__ void flashprefill_row_max_kernel(FlashPrefillPatternParams p) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    float maximum = -INFINITY;
    const int64_t base = (static_cast<int64_t>(row) * p.heads + head) * p.blocks;
    for (int col = tid; col <= row; col += blockDim.x) {
        maximum = fmaxf(maximum, p.scores[base + col]);
    }
    maximum = fp_pattern_warp_max(maximum);
    __shared__ float warp_maxima[8];
    if ((tid & 31) == 0) warp_maxima[tid >> 5] = maximum;
    __syncthreads();
    if (tid < 32) {
        float value = tid < (blockDim.x >> 5) ? warp_maxima[tid] : -INFINITY;
        value = fp_pattern_warp_max(value);
        if (tid == 0) p.row_max[static_cast<int64_t>(row) * p.heads + head] = value;
    }
}

__global__ void flashprefill_histogram_kernel(FlashPrefillPatternParams p) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t base = (static_cast<int64_t>(row) * p.heads + head) * p.blocks;
    const float maximum = p.row_max[static_cast<int64_t>(row) * p.heads + head];
    constexpr float scale =
        float(kFlashPrefillHistogramBins - 1) / -kFlashPrefillOffsetFloor;
    for (int col = tid; col <= row; col += blockDim.x) {
        const float score = p.scores[base + col];
        const int distance = row - col;
        const bool forced = col < p.sink ||
            (distance >= 0 && distance < p.window);
        // The reference BSA implementation overwrites anchor scores with
        // +1e9.  Count them before checking the computed score so anchors are
        // unconditional and remain part of the requested total budget.
        if (p.exact_density && forced) {
            atomicAdd(p.exact_meta + 3, 1);
            continue;
        }
        if (isfinite(score)) {
            const float delta = fmaxf(kFlashPrefillOffsetFloor, fminf(0.0f, score - maximum));
            const int bin = min(kFlashPrefillHistogramBins - 1,
                                max(0, __float2int_rd((delta - kFlashPrefillOffsetFloor) * scale)));
            atomicAdd(p.histogram + bin, 1);
        }
    }
}

__global__ void flashprefill_resolve_threshold_kernel(FlashPrefillPatternParams p) {
    if (blockIdx.x || threadIdx.x) return;
    const int64_t legal = static_cast<int64_t>(p.heads) * p.blocks * (p.blocks + 1) / 2;
    int64_t wanted = __double2ll_rn(p.density * double(legal));
    wanted = wanted < 1 ? 1 : (wanted > legal ? legal : wanted);
    if (p.exact_density) {
        wanted -= int64_t(p.exact_meta[3]);
        wanted = wanted < 0 ? 0 : wanted;
    }
    int64_t cumulative = 0;
    int64_t cumulative_above = 0;
    int selected_bin = kFlashPrefillHistogramBins - 1;
    for (int bin = kFlashPrefillHistogramBins - 1; bin >= 0; --bin) {
        cumulative += p.histogram[bin];
        if (cumulative >= wanted) {
            selected_bin = bin;
            cumulative_above = cumulative - p.histogram[bin];
            break;
        }
    }
    // Select the complete histogram bin.  This can exceed the target by at
    // most one quantisation bin and never drops equal-score entries randomly.
    p.threshold[0] = kFlashPrefillOffsetFloor +
        float(selected_bin) * (-kFlashPrefillOffsetFloor) /
        float(kFlashPrefillHistogramBins - 1);
    if (p.exact_density) {
        p.exact_meta[0] = selected_bin;
        const int64_t tie_budget = wanted - cumulative_above;
        p.exact_meta[1] = int(tie_budget < 0 ? 0 : tie_budget);
        p.exact_meta[2] = 0;
    }
}

__global__ void flashprefill_select_kernel(FlashPrefillPatternParams p) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t base = (static_cast<int64_t>(row) * p.heads + head) * p.blocks;
    const float maximum = p.row_max[static_cast<int64_t>(row) * p.heads + head];
    const float cutoff = maximum + p.threshold[0];
    int local_count = 0;
    for (int col = tid; col <= row; col += blockDim.x) {
        const float score = p.scores[base + col];
        const int distance = row - col;
        const bool forced = col < p.sink || (distance >= 0 && distance < p.window);
        bool keep = forced;
        if (!forced && isfinite(score)) {
            if (p.exact_density) {
                constexpr float scale =
                    float(kFlashPrefillHistogramBins - 1) /
                    -kFlashPrefillOffsetFloor;
                const float delta = fmaxf(
                    kFlashPrefillOffsetFloor, fminf(0.0f, score - maximum));
                const int bin = min(
                    kFlashPrefillHistogramBins - 1,
                    max(0, __float2int_rd(
                        (delta - kFlashPrefillOffsetFloor) * scale)));
                if (bin > p.exact_meta[0]) {
                    keep = true;
                } else if (bin == p.exact_meta[0]) {
                    const int ticket = atomicAdd(p.exact_meta + 2, 1);
                    keep = ticket < p.exact_meta[1];
                }
            } else {
                keep = score >= cutoff;
            }
        }
        p.block_mask[base + col] = static_cast<uint8_t>(keep);
        if (keep) {
            ++local_count;
            atomicAdd(p.col_nnz + static_cast<int64_t>(head) * p.blocks + col,
                      min(p.chunk_size, p.tokens - row * p.chunk_size));
        }
    }
    // The upper triangle is zero-initialised and is never consumed.
    __shared__ int warp_counts[8];
#pragma unroll
    for (int delta = 16; delta; delta >>= 1) {
        local_count += __shfl_down_sync(0xffffffffu, local_count, delta);
    }
    if ((tid & 31) == 0) warp_counts[tid >> 5] = local_count;
    __syncthreads();
    if (tid < 32) {
        int value = tid < (blockDim.x >> 5) ? warp_counts[tid] : 0;
#pragma unroll
        for (int delta = 16; delta; delta >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, delta);
        }
        if (tid == 0) {
            p.block_row_nnz[static_cast<int64_t>(row) * p.heads + head] = value;
        }
    }
}

__global__ void flashprefill_expand_row_nnz_kernel(FlashPrefillPatternParams p) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    if (token >= p.tokens) return;
    const int block = token / p.chunk_size;
    p.row_nnz[static_cast<int64_t>(token) * p.heads + head] =
        p.block_row_nnz[static_cast<int64_t>(block) * p.heads + head];
}

__global__ void flashprefill_csc_offset_kernel(FlashPrefillPatternParams p) {
    if (blockIdx.x || threadIdx.x) return;
    int64_t offset = 0;
    const int64_t columns = static_cast<int64_t>(p.heads) * p.blocks;
    for (int64_t col = 0; col < columns; ++col) {
        p.col_offsets[col] = offset;
        offset += p.col_nnz[col];
    }
}

// One CTA owns a (head, K block) CSC column.  Scanning Q blocks in ascending
// order emits sorted token rows without atomics or a separate sort pass.
__global__ void flashprefill_csc_scatter_kernel(FlashPrefillPatternParams p) {
    const int col = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    int64_t output = p.col_offsets[static_cast<int64_t>(head) * p.blocks + col];
    for (int row = col; row < p.blocks; ++row) {
        const int64_t mask_idx =
            (static_cast<int64_t>(row) * p.heads + head) * p.blocks + col;
        if (p.block_mask[mask_idx]) {
            const int q_start = row * p.chunk_size;
            const int q_len = min(p.chunk_size, p.tokens - q_start);
            for (int token = tid; token < q_len; token += blockDim.x) {
                p.csc_rows[output + token] = q_start + token;
            }
            output += q_len;
        }
    }
}

inline void run_flashprefill_pattern_select(
    FlashPrefillPatternParams& p, cudaStream_t stream) {
    dim3 row_grid(p.blocks, p.heads);
    flashprefill_row_max_kernel<<<row_grid, 256, 0, stream>>>(p);
    if (p.auto_density) {
        flashprefill_histogram_kernel<<<row_grid, 256, 0, stream>>>(p);
        flashprefill_resolve_threshold_kernel<<<1, 1, 0, stream>>>(p);
    }
    flashprefill_select_kernel<<<row_grid, 256, 0, stream>>>(p);
    dim3 token_grid((p.tokens + 255) / 256, p.heads);
    flashprefill_expand_row_nnz_kernel<<<token_grid, 256, 0, stream>>>(p);
}

inline void run_flashprefill_pattern_csc(
    FlashPrefillPatternParams& p, cudaStream_t stream) {
    flashprefill_csc_offset_kernel<<<1, 1, 0, stream>>>(p);
    flashprefill_csc_scatter_kernel<<<dim3(p.blocks, p.heads), 128, 0, stream>>>(p);
}

}  // namespace flash_dynamic
