#pragma once

#include <cstdint>
#include <torch/extension.h>

namespace flash_dynamic {

struct Dynamic_threshold_params {
    using index_t = int64_t;

    void* q_ptr;
    void* km_ptr;
    const float* threshold_ptr;
    // Optional scalar max-relative threshold probes.  The multi-probe path
    // scans QK once and accumulates only aggregate NNZ counts; it never
    // materializes a mask, so the normal selector remains the source of truth.
    const float* probe_threshold_ptr;
    int64_t* probe_nnz_ptr;
    int num_probes;
    float* row_max_ptr;
    float relative_log_threshold;
    // Optional device-resident scalar log(alpha) / softmax_scale.  This lets
    // the cap interpolation and the materializer remain on the same CUDA
    // stream without synchronizing a scalar through the host.
    const float* relative_log_threshold_ptr;
    // Optional device flag used by the fused cap-refinement path.  A zero
    // value turns a normally expensive QK launch into a uniform early return.
    const int* execute_flag_ptr;
    bool use_row_relative_threshold;
    // When true, threshold_ptr stores one additive row-max-relative logit
    // offset per local Q head.  The scalar flag remains the legacy ABI.
    bool use_head_relative_threshold;
    int* row_indices_ptr;
    uint32_t* row_bitmask_ptr;
    int* row_nnz_ptr;
    int* row_overflow_ptr;
    int64_t* col_offsets_ptr;
    int* col_nnz_ptr;
    int* col_cursor_ptr;
    int* csc_indices_ptr;
    const int64_t* q_offsets_ptr;
    int* q_csc_indices_ptr;

    index_t q_row_stride;
    index_t q_head_stride;
    index_t km_row_stride;
    index_t km_head_stride;

    int* cu_seqlens_q;
    int* cu_seqlens_k;
    int* cu_seqlens_km;

    int b;
    int h;
    int h_k;
    int h_h_k_ratio;
    int d;
    int total_q;
    int seqlen_q;
    int seqlen_k;
    int seqlen_km;
    int moba_chunk_size;
    int max_lg_col_num;
    int capacity;
    int bitmask_words;
    int sink;
    int window;
    // Optional row-level quality floor for the max-relative selector.  When
    // thresholding keeps fewer blocks, add the highest-scoring rejected
    // completed blocks.  Forced sink/window blocks count toward the floor.
    int min_blocks_per_row;
    int coarse_anchor_block_size = 0;
};

void run_dynamic_threshold_bf16_hdim128(
    Dynamic_threshold_params& params,
    bool build_csc,
    cudaStream_t stream);

void run_dynamic_threshold_compact_select_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream);

void run_dynamic_relative_threshold_compact_select_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream);

void run_dynamic_relative_threshold_compact_select_from_row_max_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream);

void run_dynamic_relative_threshold_probe_from_row_max_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream);

void run_dynamic_relative_threshold_multi_probe_bf16_hdim128(
    Dynamic_threshold_params& params,
    cudaStream_t stream);

void run_dynamic_threshold_compact_csc(
    Dynamic_threshold_params& params,
    cudaStream_t stream);

}  // namespace flash_dynamic
