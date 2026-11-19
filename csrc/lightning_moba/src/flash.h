/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <vector>

#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

namespace LIGHTNING_MOBA_NAMESPACE {
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t q_row_stride;
    index_t q_head_stride;
    index_t k_batch_stride;
    index_t k_row_stride;
    index_t k_head_stride;
    index_t v_batch_stride;
    index_t v_row_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Lightning_moba_fwd_params : public Qkv_params {
    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix (for debugging).
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions. total_k is the packed K/V row count; it differs from
    // b * seqlen_k for variable-length batches.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, total_q, total_k, rotary_dim;
    int seqlen_q_rounded, seqlen_k_rounded, d_rounded;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // Varlen sequence lengths.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ seqused_k;
    int * __restrict__ leftpad_k;
    bool is_seqlens_k_cumulative;
    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;
    index_t knew_batch_stride, vnew_batch_stride;
    index_t knew_row_stride, vnew_row_stride;
    index_t knew_head_stride, vnew_head_stride;

    // Paged KV cache
    int * __restrict__ cache_batch_idx;
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // Rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;
    bool is_rotary_interleaved;

    // Alibi slopes.
    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    // Dropout.
    float p_dropout;
    uint8_t p_dropout_in_uint8_t;
    float rp_dropout;
    float scale_softmax_rp_dropout;
    uint64_t * rng_state;
    at::PhiloxCudaState philox_args;

    // Softcapping
    float softcap;

    // For split-KV version.
    int num_splits;

    // Boolean flags.
    bool is_causal;
    int window_size_left, window_size_right;
    bool is_bf16;
    bool unpadded_lse;
    bool seqlenq_ngroups_swapped;

    // MOBA sparse attention parameters.
    bool use_moba_sparse;
    void *__restrict__ o_tmp_ptr;
    // KV-centric only: 0 = fp32 O_partial, 1 = bf16 O_partial, 2 = fp16 O_partial.
    int kv_partial_dtype;
    int kv_partial_v_dim;
    bool kv_causal_unsorted;
    bool kv_use_slot_major;
    const int *__restrict__ kv_csc_to_slot_ptr;
    // Preserve the launch-parameter layout after removing diagnostic metadata.
    const void *__restrict__ kv_reserved_metadata[3];
    index_t kv_slot_o_slot_stride;
    index_t kv_slot_o_row_stride;
    index_t kv_slot_o_head_stride;
    index_t kv_slot_lse_slot_stride;
    index_t kv_slot_lse_row_stride;
    index_t kv_slot_lse_head_stride;
    index_t o_tmp_batch_stride;
    index_t o_tmp_row_stride;
    index_t o_tmp_head_stride;
    void *__restrict__ col_offsets_ptr; // [batch_size, num_heads, max_lg_col_num]
    void *__restrict__ indices_ptr; // [total_selected_rows]
    void *__restrict__ col_nnz_ptr; // [batch_size, num_heads, max_lg_col_num]
    const int *__restrict__ kv_worklist_ptr; // [kv_work_count, 5]: batch, head, col, q_begin, q_count
    int kv_work_count;
    index_t col_offsets_batch_stride;
    index_t col_offsets_head_stride;
    int m_lg_block_dim;
    int n_lg_block_dim;
    int lg_row_factor;
    int lg_col_factor;
    int max_lg_col_num;
    int max_kblockM;
};

struct Lightning_moba_bwd_params : public Lightning_moba_fwd_params {
    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ.
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dq_row_stride;
    index_t dq_head_stride;
    index_t dk_batch_stride;
    index_t dk_row_stride;
    index_t dk_head_stride;
    index_t dv_batch_stride;
    index_t dv_row_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;
    index_t dq_accum_split_stride;

    // For deterministic backward.
    bool deterministic;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_moba_fwd_(Lightning_moba_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim, bool Is_causal> void run_moba_bwd_(Lightning_moba_bwd_params &params, cudaStream_t stream);

// KV-Centric forward kernel
template<typename T, bool Is_causal> void run_moba_fwd_kvcentric_hdim128(Lightning_moba_fwd_params &params, cudaStream_t stream);

}  // namespace LIGHTNING_MOBA_NAMESPACE
