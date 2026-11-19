/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
#include "philox_unpack.cuh"  // For at::cuda::philox::unpack

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "lightning_moba_static_switch.h"
#include "flash_combine_kernel.h"
#include "flash_kvcentric_worklist.h"
#include "flash_sm90_phase4b_csc_pipe.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace LIGHTNING_MOBA_NAMESPACE {

namespace {

int get_hdim128_causal_lg_block_m_cap() {
    static const int cap = [] {
        const char* value = std::getenv("LIGHTNING_MOBA_FWD_HDIM128_CAUSAL_LG_BLOCK_M");
        if (value == nullptr) { return 1024; }
        const int parsed = std::atoi(value);
        if (parsed <= 1024 || parsed > 4096 || parsed % 64 != 0) { return 1024; }
        return parsed <= 2048 ? 2048 : 4096;
    }();
    return cap;
}

bool use_kvcentric_input_partial_dtype(at::ScalarType q_dtype) {
    const char* value = std::getenv("LIGHTNING_MOBA_KVCENTRIC_PARTIAL_DTYPE");
    if (value == nullptr) { return false; }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "fp32") == 0 ||
        std::strcmp(value, "float") == 0 || std::strcmp(value, "float32") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "input") == 0) {
        return true;
    }
    if (std::strcmp(value, "bf16") == 0) {
        return q_dtype == torch::kBFloat16;
    }
    if (std::strcmp(value, "fp16") == 0 || std::strcmp(value, "half") == 0) {
        return q_dtype == torch::kFloat16;
    }
    return false;
}

// Phase 4-B production bridge. Keep the new pipeline behind an exact opt-in
// until the complete 64K/128K TopK=16 validation matrix is accepted.
bool use_kvcentric_sm90_phase4b_csc_pipe_float2_overlap_chunk128() {
    const char* value = std::getenv("LIGHTNING_MOBA_KVCENTRIC_SM90_K1");
    return value != nullptr &&
        std::strcmp(value, "phase4b_csc_pipe_float2_overlap_chunk128") == 0;
}

int kvcentric_partial_dtype_code(at::ScalarType dtype) {
    if (dtype == torch::kBFloat16) { return 1; }
    if (dtype == torch::kFloat16) { return 2; }
    return 0;
}

}  // namespace

void set_params_moba_fprop(Lightning_moba_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      at::Tensor out_tmp,
                      // MOBA sparse pattern parameters
                      const at::Tensor moba_col_offsets,
                      const at::Tensor moba_col_nnz,
                      const at::Tensor moba_row_indices,
                      const int m_lg_block_dim,
                      const int n_lg_block_dim,
                      const int max_kblockM,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);
    params.o_tmp_ptr = out_tmp.data_ptr();
    params.o_tmp_row_stride = out_tmp.stride(-3);
    params.o_tmp_head_stride = out_tmp.stride(-2);
    // K/V are packed across variable-length requests, so this is not in
    // general equal to b * max_seqlen_k.
    params.total_q = q.size(0);
    params.total_k = k.size(0);
    params.max_kblockM = max_kblockM;
    params.use_moba_sparse = true;  // Always enabled in MOBA variant

    // Set MOBA sparse pattern parameters
    params.col_offsets_ptr = moba_col_offsets.data_ptr();
    params.col_nnz_ptr = moba_col_nnz.data_ptr();
    params.indices_ptr = moba_row_indices.data_ptr();
    params.max_lg_col_num = moba_col_offsets.size(2);
    params.m_lg_block_dim = m_lg_block_dim;
    params.n_lg_block_dim = n_lg_block_dim;
    
    // Calculate strides for col_offsets tensor (batch_size x num_heads x max_lg_col_num)
    params.col_offsets_batch_stride = moba_col_offsets.stride(0);
    params.col_offsets_head_stride = moba_col_offsets.stride(1);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        params.o_tmp_batch_stride = out_tmp.stride(0);
        if (seqlenq_ngroups_swapped) {
            params.q_batch_stride *= seqlen_q;
            params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef LIGHTNINGMOBA_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This LightningMoBA build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef LIGHTNINGMOBA_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This LightningMoBA build does not support dropout.");
    #endif

    params.is_causal = is_causal;
    params.window_size_left = seqlen_k;
    params.window_size_right = is_causal ? 0 : seqlen_k;

    params.is_seqlens_k_cumulative = true;

    #ifdef LIGHTNINGMOBA_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This LightningMoBA build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_moba_dgrad(Lightning_moba_bwd_params &params,
                            // sizes
                            const size_t b,
                            const size_t seqlen_q,
                            const size_t seqlen_k,
                            const size_t seqlen_q_rounded,
                            const size_t seqlen_k_rounded,
                            const size_t h,
                            const size_t h_k,
                            const size_t d,
                            const size_t d_rounded,
                            // device pointers
                            const at::Tensor q,
                            const at::Tensor k,
                            const at::Tensor v,
                            const at::Tensor out,
                            const at::Tensor dout,
                            at::Tensor dq,
                            at::Tensor dk,
                            at::Tensor dv,
                            // MOBA sparse pattern parameters
                            const at::Tensor moba_col_offsets,
                            const at::Tensor moba_col_nnz,
                            const at::Tensor moba_row_indices,
                            const int m_lg_block_dim,
                            const int n_lg_block_dim,
                            void *cu_seqlens_q_d,
                            void *cu_seqlens_k_d,
                            void *dq_accum_d,
                            void *dk_accum_d,
                            void *dv_accum_d,
                            void *softmax_lse_d,
                            void *dsoftmax_sum_d,
                            float p_dropout,
                            float softmax_scale,
                            bool is_causal,
                            const float softcap,
                            bool deterministic,
                            const bool unpadded_lse) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = nullptr;

    // P = softmax(QK^T)
    params.p_ptr = nullptr;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef LIGHTNINGMOBA_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This LightningMoBA build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef LIGHTNINGMOBA_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This LightningMoBA build does not support dropout.");
    #endif

    params.is_causal = is_causal;
    params.window_size_left = seqlen_k;
    params.window_size_right = is_causal ? 0 : seqlen_k;

    params.is_seqlens_k_cumulative = true;

    #ifdef LIGHTNINGMOBA_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This LightningMoBA build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = false;

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    params.use_moba_sparse = true;  // Always enabled in MOBA variant

    // Set MOBA sparse pattern parameters
    params.col_offsets_ptr = moba_col_offsets.data_ptr();
    params.col_nnz_ptr = moba_col_nnz.data_ptr();
    params.indices_ptr = moba_row_indices.data_ptr();
    params.max_lg_col_num = moba_col_offsets.size(2);
    params.m_lg_block_dim = m_lg_block_dim;
    params.n_lg_block_dim = n_lg_block_dim;
    
    // Calculate strides for col_offsets tensor (batch_size x num_heads x max_lg_col_num)
    params.col_offsets_batch_stride = moba_col_offsets.stride(0);
    params.col_offsets_head_stride = moba_col_offsets.stride(1);

    if (cu_seqlens_q_d == nullptr) {
    params.do_batch_stride = dout.stride(0);
    params.dq_batch_stride = dq.stride(0);
    params.dk_batch_stride = dk.stride(0);
    params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

void run_moba_fwd(Lightning_moba_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_moba_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}

void set_params_alibi(Lightning_moba_fwd_params &params, std::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef LIGHTNINGMOBA_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This LightningMoBA build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

std::vector<at::Tensor>
moba_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &moba_col_offsets, // batch_size x num_heads x max_lg_col_num
               const at::Tensor &moba_col_nnz, // batch_size x num_heads x max_lg_col_num, number of wanted rows in each column
               const at::Tensor &moba_row_indices, // total_selected_rows
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<const at::Tensor> &leftpad_k_, // batch_size
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               const float softcap,
               const bool return_softmax,
               const int m_lg_block_dim,  // Size of logical blocks in M dimension
               const int n_lg_block_dim,  // Size of logical blocks in N dimension
               std::optional<at::Generator> gen_) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "LightningMoBA only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "LightningMoBA only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);
    
    // Validate MOBA sparse pattern tensors
    TORCH_CHECK(moba_col_offsets.dtype() == torch::kInt64, "moba_col_offsets must have dtype int64 (index_t)");
    TORCH_CHECK(moba_col_nnz.dtype() == torch::kInt32, "moba_col_nnz must have dtype int32");
    TORCH_CHECK(moba_row_indices.dtype() == torch::kInt32, "moba_row_indices must have dtype int32");
    
    CHECK_DEVICE(moba_col_offsets);
    CHECK_DEVICE(moba_col_nnz);
    CHECK_DEVICE(moba_row_indices);
    
    CHECK_CONTIGUOUS(moba_col_offsets);
    CHECK_CONTIGUOUS(moba_col_nnz);
    CHECK_CONTIGUOUS(moba_row_indices);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = k.size(1);
    const int max_lg_col_num = moba_col_offsets.size(2);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    const int page_block_size = 1;

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    const int seqlenq_ngroups_swapped = false;
    const int ngroups = num_heads / num_heads_k;

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "LightningMoBA forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    const int extended_lg_cap = get_hdim128_causal_lg_block_m_cap();
    TORCH_CHECK(m_lg_block_dim >= 128 && m_lg_block_dim <= extended_lg_cap,
                "m_lg_block_dim must be at least 128 and at most ", extended_lg_cap,
                ". Set LIGHTNING_MOBA_FWD_HDIM128_CAUSAL_LG_BLOCK_M to a multiple of 64 "
                "between 1088 and 4096 "
                "to enable the experimental Hopper path.");
    if (m_lg_block_dim > 1024) {
        TORCH_CHECK(cc_major >= 9, "m_lg_block_dim > 1024 is only supported on Hopper or newer GPUs");
        TORCH_CHECK(head_size == 128 && q_dtype == torch::kBFloat16 && is_causal && p_dropout == 0.0f,
                    "m_lg_block_dim > 1024 requires causal bf16 attention with head_dim=128 and dropout=0");
    }
    TORCH_CHECK(m_lg_block_dim % 64 == 0, "m_lg_block_dim must be a multiple of 64");
    TORCH_CHECK(n_lg_block_dim % 64 == 0, "n_lg_block_dim must be a multiple of 64");

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    const int total_k = k.size(0);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    CHECK_SHAPE(moba_col_offsets, batch_size, num_heads, max_lg_col_num);
    CHECK_SHAPE(moba_col_nnz, batch_size, num_heads, max_lg_col_num);
    // moba_row_indices shape will be checked later as it depends on total selected rows
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size);
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    auto opts = q.options();
    const int max_kblockM = 128; //过大的ph block 会容易浪费 mma，过小容易造成flush
    const int acc_length = total_q + batch_size * max_kblockM;
    at::Tensor out_tmp;
    out_tmp = torch::zeros({acc_length, num_heads, head_size}, opts.dtype(at::kFloat));
    // additional space for row_sum and row_max so as not to mask during the update (except for the final write back)

    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    Lightning_moba_fwd_params params;
    set_params_moba_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     out_tmp,
                     moba_col_offsets,
                     moba_col_nnz,
                     moba_row_indices,
                     m_lg_block_dim,
                     n_lg_block_dim,
                     max_kblockM,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     softcap,
                     seqlenq_ngroups_swapped,
                     /*unpadded_lse*/true);
    params.total_q = total_q;

    params.page_block_size = page_block_size;
    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_moba_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return {out, softmax_lse, p, rng_state};
}

// =============================================================================
// KV-Centric Forward Pass
// =============================================================================
// Returns: [O_partial (fp32), LSE_partial (fp32)]
// O_partial: [total_nnz, head_dim] — partial attention output per CSC entry
// LSE_partial: [total_nnz] — log-sum-exp per CSC entry
// These need to be combined by a separate combine kernel or the Triton combine.

std::vector<at::Tensor>
moba_varlen_fwd_kvcentric(
    at::Tensor &q,                         // total_q x num_heads x head_size
    const at::Tensor &k,                   // total_k x num_heads_k x head_size
    const at::Tensor &v,                   // total_k x num_heads_k x head_size
    const at::Tensor &moba_col_offsets,    // batch_size x num_heads x max_lg_col_num
    const at::Tensor &moba_col_nnz,        // batch_size x num_heads x max_lg_col_num
    const at::Tensor &moba_row_indices,    // total_selected_rows
    const at::Tensor &cu_seqlens_q,        // b+1
    const at::Tensor &cu_seqlens_k,        // b+1
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool is_causal,
    int n_lg_block_dim                     // = chunk_size
) {
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(cc_major >= 8, "LightningMoBA only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "LightningMoBA only supports fp16 and bf16");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype);

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);
    CHECK_DEVICE(moba_col_offsets); CHECK_DEVICE(moba_col_nnz); CHECK_DEVICE(moba_row_indices);

    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
    TORCH_CHECK(moba_col_offsets.dtype() == torch::kInt64, "moba_col_offsets must have dtype int64");
    TORCH_CHECK(moba_col_nnz.dtype() == torch::kInt32, "moba_col_nnz must have dtype int32");
    TORCH_CHECK(moba_row_indices.dtype() == torch::kInt32, "moba_row_indices must have dtype int32");

    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "v must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    CHECK_CONTIGUOUS(moba_col_offsets);
    CHECK_CONTIGUOUS(moba_col_nnz);
    CHECK_CONTIGUOUS(moba_row_indices);

    const auto sizes = q.sizes();
    const int total_q = sizes[0];
    const int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = k.size(1);
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int max_lg_col_num = moba_col_offsets.size(2);
    const int total_nnz = moba_row_indices.size(0);
    const int total_k = k.size(0);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 128, "KV-centric forward currently supports head_size <= 128");
    TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of KV heads must divide number of Q heads");
    TORCH_CHECK(n_lg_block_dim % 64 == 0, "n_lg_block_dim must be a multiple of 64");

    const bool use_phase4b_csc_pipe_float2_overlap =
        use_kvcentric_sm90_phase4b_csc_pipe_float2_overlap_chunk128();
    if (use_phase4b_csc_pipe_float2_overlap) {
        TORCH_CHECK(cc_major >= 9 && q_dtype == torch::kBFloat16 && head_size == 128,
                    "WGMMA CSC strict modes require SM90+ BF16 head_dim=128");
        TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                    "WGMMA CSC strict modes require contiguous packed Q/K/V");
        TORCH_CHECK(n_lg_block_dim == 128,
                    "W2SR attribution CSC WGMMA requires fixed chunk128 K columns");
        TORCH_CHECK(is_causal,
                    "Phase 4-B CSC overlap mode currently requires is_causal=true");
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    CHECK_SHAPE(moba_col_offsets, batch_size, num_heads, max_lg_col_num);
    CHECK_SHAPE(moba_col_nnz, batch_size, num_heads, max_lg_col_num);

    // Return before the legacy partial-buffer allocation below. Phase 4-B
    // allocates and fully writes its own CSC-contiguous outputs; allocating a
    // second 34/69 GiB output at 64K/128K would make this bridge unusable.
    if (use_phase4b_csc_pipe_float2_overlap) {
        auto [o, lse] = moba_sm90_phase4b_csc_pipe(
            q, k, v, moba_col_offsets, moba_col_nnz, moba_row_indices,
            cu_seqlens_q, cu_seqlens_k, is_causal, softmax_scale);
        return {o, lse};
    }

    // K1 writes every CSC entry in both buffers, including -inf LSE for
    // fully masked rows, so avoid an extra full-buffer memset here.
    auto opts = q.options();
    const at::ScalarType q_scalar_type = q.scalar_type();
    const bool use_input_partial = use_kvcentric_input_partial_dtype(q_scalar_type);
    const at::ScalarType partial_dtype = use_input_partial ? q_scalar_type : at::kFloat;
    at::Tensor o_partial = torch::empty({total_nnz, head_size}, opts.dtype(partial_dtype));
    at::Tensor lse_partial = torch::empty({total_nnz}, opts.dtype(at::kFloat));

    // Setup params (reuse existing structure)
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // set_params_moba_fprop expects a 3D out_tmp for stride computation only.
    // KV-centric overrides o_tmp_ptr below and indexes O_partial as [total_nnz, D],
    // so do not allocate [total_nnz, num_heads, D] here.
    at::Tensor out_dummy = torch::empty_like(q);  // same shape as q for out
    at::Tensor out_tmp_3d = torch::empty({1, 1, head_size}, opts.dtype(at::kFloat));

    // Dummy softmax_lse with shape [num_heads, total_q] for set_params
    at::Tensor softmax_lse_dummy = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    Lightning_moba_fwd_params params;
    set_params_moba_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out_dummy,
                     out_tmp_3d,  // 3D tensor for stride computation
                     moba_col_offsets,
                     moba_col_nnz,
                     moba_row_indices,
                     max_seqlen_q,  // m_lg_block_dim (unused in kv-centric but needed)
                     n_lg_block_dim,
                     128,  // max_kblockM
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     nullptr,  // seqused_k
                     nullptr,  // p_ptr
                     softmax_lse_dummy.data_ptr(),
                     0.0f,  // p_dropout
                     softmax_scale,
                     is_causal,
                     0.0f,  // softcap
                     false,  // seqlenq_ngroups_swapped
                     true);  // unpadded_lse
    params.total_q = total_q;

    // Override: kv-centric kernel uses o_tmp_ptr for O_partial and softmax_lseaccum_ptr for LSE
    // O_partial is [total_nnz, head_dim] — kernel indexes as o_partial[csc_offset * d + d_col]
    params.o_tmp_ptr = o_partial.data_ptr();
    params.kv_partial_dtype = kvcentric_partial_dtype_code(partial_dtype);
    params.softmax_lseaccum_ptr = lse_partial.data_ptr();

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        // Launch KV-centric kernel
        FP16_SWITCH(!params.is_bf16, [&] {
            BOOL_SWITCH(is_causal, Is_causal, [&] {
                run_moba_fwd_kvcentric_hdim128<elem_type, Is_causal>(params, stream);
            });
        });
    }

    return {o_partial, lse_partial};
}

std::vector<at::Tensor>
moba_varlen_fwd_kvcentric_worklist(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &moba_col_offsets,
    const at::Tensor &moba_col_nnz,
    const at::Tensor &moba_row_indices,
    const at::Tensor &kv_worklist,
    const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool is_causal,
    int n_lg_block_dim
) {
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(cc_major >= 8, "LightningMoBA only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "LightningMoBA only supports fp16 and bf16");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype);

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);
    CHECK_DEVICE(moba_col_offsets); CHECK_DEVICE(moba_col_nnz); CHECK_DEVICE(moba_row_indices);
    CHECK_DEVICE(kv_worklist);

    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
    TORCH_CHECK(moba_col_offsets.dtype() == torch::kInt64, "moba_col_offsets must have dtype int64");
    TORCH_CHECK(moba_col_nnz.dtype() == torch::kInt32, "moba_col_nnz must have dtype int32");
    TORCH_CHECK(moba_row_indices.dtype() == torch::kInt32, "moba_row_indices must have dtype int32");
    TORCH_CHECK(kv_worklist.dtype() == torch::kInt32, "kv_worklist must have dtype int32");

    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "v must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    CHECK_CONTIGUOUS(moba_col_offsets);
    CHECK_CONTIGUOUS(moba_col_nnz);
    CHECK_CONTIGUOUS(moba_row_indices);
    CHECK_CONTIGUOUS(kv_worklist);

    const auto sizes = q.sizes();
    const int total_q = sizes[0];
    const int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = k.size(1);
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int max_lg_col_num = moba_col_offsets.size(2);
    const int total_nnz = moba_row_indices.size(0);
    const int total_k = k.size(0);
    const int work_count = kv_worklist.size(0);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 128, "KV-centric forward currently supports head_size <= 128");
    TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of KV heads must divide number of Q heads");
    TORCH_CHECK(n_lg_block_dim % 64 == 0, "n_lg_block_dim must be a multiple of 64");
    TORCH_CHECK(kv_worklist.dim() == 2 && kv_worklist.size(1) == 5,
                "kv_worklist must have shape [work_count, 5]");
    TORCH_CHECK(work_count >= 0, "kv_worklist work_count must be non-negative");

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    CHECK_SHAPE(moba_col_offsets, batch_size, num_heads, max_lg_col_num);
    CHECK_SHAPE(moba_col_nnz, batch_size, num_heads, max_lg_col_num);

    auto opts = q.options();
    const at::ScalarType q_scalar_type = q.scalar_type();
    const bool use_input_partial = use_kvcentric_input_partial_dtype(q_scalar_type);
    const at::ScalarType partial_dtype = use_input_partial ? q_scalar_type : at::kFloat;
    at::Tensor o_partial = torch::empty({total_nnz, head_size}, opts.dtype(partial_dtype));
    at::Tensor lse_partial = torch::empty({total_nnz}, opts.dtype(at::kFloat));

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    at::Tensor out_dummy = torch::empty_like(q);
    at::Tensor out_tmp_3d = torch::empty({1, 1, head_size}, opts.dtype(at::kFloat));
    at::Tensor softmax_lse_dummy = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    Lightning_moba_fwd_params params;
    set_params_moba_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out_dummy,
                     out_tmp_3d,
                     moba_col_offsets,
                     moba_col_nnz,
                     moba_row_indices,
                     max_seqlen_q,
                     n_lg_block_dim,
                     128,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     nullptr,
                     nullptr,
                     softmax_lse_dummy.data_ptr(),
                     0.0f,
                     softmax_scale,
                     is_causal,
                     0.0f,
                     false,
                     true);
    params.total_q = total_q;
    params.o_tmp_ptr = o_partial.data_ptr();
    params.kv_partial_dtype = kvcentric_partial_dtype_code(partial_dtype);
    params.softmax_lseaccum_ptr = lse_partial.data_ptr();
    params.kv_worklist_ptr = reinterpret_cast<const int*>(kv_worklist.data_ptr());
    params.kv_work_count = work_count;

    if (max_seqlen_k > 0 && work_count > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        FP16_SWITCH(!params.is_bf16, [&] {
            BOOL_SWITCH(is_causal, Is_causal, [&] {
                run_moba_fwd_kvcentric_hdim128<elem_type, Is_causal>(params, stream);
            });
        });
    }

    return {o_partial, lse_partial};
}

std::vector<at::Tensor>
moba_varlen_fwd_kvcentric_slot(
    at::Tensor &q,                         // total_q x num_heads x head_size
    const at::Tensor &k,                   // total_k x num_heads_k x head_size
    const at::Tensor &v,                   // total_k x num_heads_k x head_size
    const at::Tensor &moba_col_offsets,    // batch_size x num_heads x max_lg_col_num
    const at::Tensor &moba_col_nnz,        // batch_size x num_heads x max_lg_col_num
    const at::Tensor &moba_row_indices,    // total_selected_rows
    const at::Tensor &csc_to_slot,         // total_selected_rows
    const at::Tensor &cu_seqlens_q,        // b+1
    const at::Tensor &cu_seqlens_k,        // b+1
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool is_causal,
    int n_lg_block_dim,
    int topk
) {
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(cc_major >= 8, "LightningMoBA only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "LightningMoBA only supports fp16 and bf16");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype);

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(moba_col_offsets); CHECK_DEVICE(moba_col_nnz); CHECK_DEVICE(moba_row_indices);
    CHECK_DEVICE(csc_to_slot);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);
    CHECK_CONTIGUOUS(moba_col_offsets);
    CHECK_CONTIGUOUS(moba_col_nnz);
    CHECK_CONTIGUOUS(moba_row_indices);
    CHECK_CONTIGUOUS(csc_to_slot);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    TORCH_CHECK(moba_col_offsets.dtype() == torch::kInt64, "moba_col_offsets must be int64");
    TORCH_CHECK(moba_col_nnz.dtype() == torch::kInt32, "moba_col_nnz must be int32");
    TORCH_CHECK(moba_row_indices.dtype() == torch::kInt32, "moba_row_indices must be int32");
    TORCH_CHECK(csc_to_slot.dtype() == torch::kInt32, "csc_to_slot must be int32");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");

    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "v must have contiguous last dimension");

    const int total_q = q.size(0);
    const int num_heads = q.size(1);
    const int head_size = q.size(2);
    const int num_heads_k = k.size(1);
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int max_lg_col_num = moba_col_offsets.size(2);
    const int total_nnz = moba_row_indices.size(0);

    TORCH_CHECK(head_size == 128, "KV-centric slot path currently supports head_dim=128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads must be divisible by num_heads_k");
    TORCH_CHECK(n_lg_block_dim % 64 == 0, "n_lg_block_dim must be a multiple of 64");
    TORCH_CHECK(topk > 0 && topk <= 32, "KV-centric slot path supports 1 <= topk <= 32");
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    CHECK_SHAPE(moba_col_offsets, batch_size, num_heads, max_lg_col_num);
    CHECK_SHAPE(moba_col_nnz, batch_size, num_heads, max_lg_col_num);
    CHECK_SHAPE(csc_to_slot, total_nnz);

    auto opts = q.options();
    at::Tensor o_partial = torch::empty(
        {topk, total_q, num_heads, head_size},
        opts.dtype(at::kFloat));
    at::Tensor lse_partial = torch::empty(
        {topk, total_q, num_heads},
        opts.dtype(at::kFloat));

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    at::Tensor out_dummy = torch::empty_like(q);
    at::Tensor out_tmp_3d = torch::empty({1, 1, head_size}, opts.dtype(at::kFloat));
    at::Tensor softmax_lse_dummy = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    Lightning_moba_fwd_params params;
    set_params_moba_fprop(params,
                 batch_size,
                 max_seqlen_q, max_seqlen_k,
                 seqlen_q_rounded, seqlen_k_rounded,
                 num_heads, num_heads_k,
                 head_size, head_size_rounded,
                 q, k, v, out_dummy, out_tmp_3d,
                 moba_col_offsets, moba_col_nnz, moba_row_indices,
                 max_seqlen_q, n_lg_block_dim, 128,
                 cu_seqlens_q.data_ptr(), cu_seqlens_k.data_ptr(),
                 nullptr, nullptr, softmax_lse_dummy.data_ptr(),
                 0.0f, softmax_scale, is_causal, 0.0f,
                 false, true);
    params.total_q = total_q;

    params.o_tmp_ptr = o_partial.data_ptr();
    params.kv_partial_dtype = 0;
    params.kv_use_slot_major = true;
    params.kv_csc_to_slot_ptr = reinterpret_cast<const int*>(csc_to_slot.data_ptr());
    params.kv_slot_o_slot_stride = o_partial.stride(0);
    params.kv_slot_o_row_stride = o_partial.stride(1);
    params.kv_slot_o_head_stride = o_partial.stride(2);
    params.kv_slot_lse_slot_stride = lse_partial.stride(0);
    params.kv_slot_lse_row_stride = lse_partial.stride(1);
    params.kv_slot_lse_head_stride = lse_partial.stride(2);
    params.softmax_lseaccum_ptr = lse_partial.data_ptr();

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        FP16_SWITCH(!params.is_bf16, [&] {
            BOOL_SWITCH(is_causal, Is_causal, [&] {
                run_moba_fwd_kvcentric_hdim128<elem_type, Is_causal>(params, stream);
            });
        });
    }

    return {o_partial, lse_partial};
}

at::Tensor
moba_combine_kvcentric(
    const at::Tensor &o_partial,        // total_nnz x head_size, fp32
    const at::Tensor &lse_partial,      // total_nnz, fp32
    const at::Tensor &q2col_offsets,    // total_q * num_heads x topk, int32
    const at::Tensor &q2col_valid,      // total_q * num_heads, int32
    at::Tensor &out,                    // total_q x num_heads x head_size, fp16/bf16
    int topk
) {
    at::cuda::CUDAGuard device_guard{out.device()};

    CHECK_DEVICE(o_partial); CHECK_DEVICE(lse_partial);
    CHECK_DEVICE(q2col_offsets); CHECK_DEVICE(q2col_valid); CHECK_DEVICE(out);
    CHECK_CONTIGUOUS(o_partial);
    CHECK_CONTIGUOUS(lse_partial);
    CHECK_CONTIGUOUS(q2col_offsets);
    CHECK_CONTIGUOUS(q2col_valid);

    TORCH_CHECK(o_partial.dtype() == torch::kFloat32 ||
                o_partial.dtype() == torch::kBFloat16 ||
                o_partial.dtype() == torch::kFloat16,
                "o_partial must be fp32, bf16, or fp16");
    TORCH_CHECK(lse_partial.dtype() == torch::kFloat32, "lse_partial must be fp32");
    TORCH_CHECK(q2col_offsets.dtype() == torch::kInt32, "q2col_offsets must be int32");
    TORCH_CHECK(q2col_valid.dtype() == torch::kInt32, "q2col_valid must be int32");
    TORCH_CHECK(out.dtype() == torch::kFloat16 || out.dtype() == torch::kBFloat16,
                "out must be fp16 or bf16");
    TORCH_CHECK(out.stride(-1) == 1, "out must have contiguous last dimension");

    TORCH_CHECK(o_partial.dim() == 2, "o_partial must have shape [total_nnz, head_dim]");
    TORCH_CHECK(lse_partial.dim() == 1, "lse_partial must have shape [total_nnz]");
    TORCH_CHECK(q2col_offsets.dim() == 2, "q2col_offsets must have shape [total_q * num_heads, topk]");
    TORCH_CHECK(q2col_valid.dim() == 1, "q2col_valid must have shape [total_q * num_heads]");
    TORCH_CHECK(out.dim() == 3, "out must have shape [total_q, num_heads, head_dim]");

    const int total_nnz = o_partial.size(0);
    const int head_size = o_partial.size(1);
    const int total_q = out.size(0);
    const int num_heads = out.size(1);
    const int out_head_size = out.size(2);
    const int qh = total_q * num_heads;

    TORCH_CHECK(head_size == out_head_size, "o_partial/out head_dim mismatch");
    TORCH_CHECK(lse_partial.size(0) == total_nnz, "lse_partial total_nnz mismatch");
    TORCH_CHECK(q2col_offsets.size(0) == qh, "q2col_offsets first dim must equal total_q * num_heads");
    TORCH_CHECK(q2col_offsets.size(1) >= topk, "q2col_offsets second dim must be >= topk");
    TORCH_CHECK(q2col_offsets.stride(1) == 1, "q2col_offsets must be contiguous along topk dimension");
    TORCH_CHECK(q2col_valid.size(0) == qh, "q2col_valid dim must equal total_q * num_heads");
    TORCH_CHECK(topk > 0 && topk <= 64, "CUDA CSC combine supports 1 <= topk <= 64");
    TORCH_CHECK(head_size <= 256, "CUDA combine supports head_dim <= 256");

    Lightning_moba_combine_params params = {};
    params.o_partial_ptr = o_partial.data_ptr();
    params.lse_partial_ptr = reinterpret_cast<const float*>(lse_partial.data_ptr());
    params.q2col_offsets_ptr = reinterpret_cast<const int*>(q2col_offsets.data_ptr());
    params.q2col_valid_ptr = reinterpret_cast<const int*>(q2col_valid.data_ptr());
    params.o_out_ptr = out.data_ptr();
    params.lse_out_ptr = nullptr;
    params.o_partial_row_stride = o_partial.stride(0);
    params.q2col_q_stride = q2col_offsets.stride(0) * num_heads;
    params.q2col_h_stride = q2col_offsets.stride(0);
    params.o_out_row_stride = out.stride(0);
    params.o_out_head_stride = out.stride(1);
    params.total_q = total_q;
    params.num_heads = num_heads;
    params.head_dim = head_size;
    params.topk = topk;
    params.partial_dtype = kvcentric_partial_dtype_code(o_partial.scalar_type());

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_lightning_moba_combine_dispatch(params, out.dtype() == torch::kBFloat16, stream);

    return out;
}

at::Tensor
moba_combine_slot_kvcentric(
    const at::Tensor &o_partial,        // topk x total_q x num_heads x head_size, fp32
    const at::Tensor &lse_partial,      // topk x total_q x num_heads, fp32
    const at::Tensor &split_count,      // total_q * num_heads, int32
    at::Tensor &out,                    // total_q x num_heads x head_size, fp16/bf16
    int topk
) {
    at::cuda::CUDAGuard device_guard{out.device()};

    CHECK_DEVICE(o_partial); CHECK_DEVICE(lse_partial);
    CHECK_DEVICE(split_count); CHECK_DEVICE(out);
    CHECK_CONTIGUOUS(o_partial);
    CHECK_CONTIGUOUS(lse_partial);
    CHECK_CONTIGUOUS(split_count);

    TORCH_CHECK(o_partial.dtype() == torch::kFloat32, "slot o_partial must be fp32");
    TORCH_CHECK(lse_partial.dtype() == torch::kFloat32, "slot lse_partial must be fp32");
    TORCH_CHECK(split_count.dtype() == torch::kInt32, "split_count must be int32");
    TORCH_CHECK(out.dtype() == torch::kFloat16 || out.dtype() == torch::kBFloat16,
                "out must be fp16 or bf16");
    TORCH_CHECK(out.stride(-1) == 1, "out must have contiguous last dimension");

    TORCH_CHECK(o_partial.dim() == 4, "o_partial must have shape [topk, total_q, num_heads, head_dim]");
    TORCH_CHECK(lse_partial.dim() == 3, "lse_partial must have shape [topk, total_q, num_heads]");
    TORCH_CHECK(split_count.dim() == 1, "split_count must have shape [total_q * num_heads]");
    TORCH_CHECK(out.dim() == 3, "out must have shape [total_q, num_heads, head_dim]");

    const int partial_topk = o_partial.size(0);
    const int total_q = out.size(0);
    const int num_heads = out.size(1);
    const int head_size = out.size(2);
    TORCH_CHECK(partial_topk >= topk, "o_partial topk dim must be >= topk");
    CHECK_SHAPE(o_partial, partial_topk, total_q, num_heads, head_size);
    CHECK_SHAPE(lse_partial, partial_topk, total_q, num_heads);
    CHECK_SHAPE(split_count, total_q * num_heads);
    TORCH_CHECK(topk > 0 && topk <= 32, "slot combine supports 1 <= topk <= 32");
    TORCH_CHECK(head_size <= 256, "slot combine supports head_dim <= 256");

    Lightning_moba_slot_combine_params params = {};
    params.o_partial_ptr = reinterpret_cast<const float*>(o_partial.data_ptr());
    params.lse_partial_ptr = reinterpret_cast<const float*>(lse_partial.data_ptr());
    params.split_count_ptr = reinterpret_cast<const int*>(split_count.data_ptr());
    params.o_out_ptr = out.data_ptr();
    params.lse_out_ptr = nullptr;
    params.o_partial_slot_stride = o_partial.stride(0);
    params.o_partial_row_stride = o_partial.stride(1);
    params.o_partial_head_stride = o_partial.stride(2);
    params.lse_partial_slot_stride = lse_partial.stride(0);
    params.lse_partial_row_stride = lse_partial.stride(1);
    params.lse_partial_head_stride = lse_partial.stride(2);
    params.o_out_row_stride = out.stride(0);
    params.o_out_head_stride = out.stride(1);
    params.total_q = total_q;
    params.num_heads = num_heads;
    params.head_dim = head_size;
    params.topk = topk;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_lightning_moba_combine_slot_dispatch(params, out.dtype() == torch::kBFloat16, stream);
    return out;
}

void
moba_build_q2col_kvcentric_into(
    const at::Tensor &moba_col_offsets,    // batch_size x num_heads x max_lg_col_num
    const at::Tensor &moba_col_nnz,        // batch_size x num_heads x max_lg_col_num
    const at::Tensor &moba_row_indices,    // total_selected_rows, sequence-local q rows
    const at::Tensor &cu_seqlens_q,        // batch_size + 1
    int total_q,
    int num_heads,
    int topk,
    at::Tensor &q2col_offsets,             // [total_q * num_heads, topk] int32
    at::Tensor &q2col_valid                // [total_q * num_heads] int32
) {
    CHECK_DEVICE(moba_col_offsets); CHECK_DEVICE(moba_col_nnz);
    CHECK_DEVICE(moba_row_indices); CHECK_DEVICE(cu_seqlens_q);
    CHECK_CONTIGUOUS(moba_col_offsets);
    CHECK_CONTIGUOUS(moba_col_nnz);
    CHECK_CONTIGUOUS(moba_row_indices);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_DEVICE(q2col_offsets); CHECK_DEVICE(q2col_valid);
    CHECK_CONTIGUOUS(q2col_offsets); CHECK_CONTIGUOUS(q2col_valid);

    TORCH_CHECK(moba_col_offsets.dtype() == torch::kInt64, "moba_col_offsets must be int64");
    TORCH_CHECK(moba_col_nnz.dtype() == torch::kInt32, "moba_col_nnz must be int32");
    TORCH_CHECK(moba_row_indices.dtype() == torch::kInt32, "moba_row_indices must be int32");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(q2col_offsets.dtype() == torch::kInt32, "q2col_offsets must be int32");
    TORCH_CHECK(q2col_valid.dtype() == torch::kInt32, "q2col_valid must be int32");
    TORCH_CHECK(moba_col_offsets.dim() == 3, "moba_col_offsets must have shape [batch, heads, columns]");
    TORCH_CHECK(moba_col_nnz.dim() == 3, "moba_col_nnz must have shape [batch, heads, columns]");
    TORCH_CHECK(moba_row_indices.dim() == 1, "moba_row_indices must be 1D");
    TORCH_CHECK(cu_seqlens_q.dim() == 1, "cu_seqlens_q must be 1D");

    const int batch_size = moba_col_offsets.size(0);
    const int heads_from_offsets = moba_col_offsets.size(1);
    const int num_columns = moba_col_offsets.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads == heads_from_offsets, "num_heads mismatch");
    CHECK_SHAPE(moba_col_nnz, batch_size, num_heads, num_columns);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    TORCH_CHECK(total_q >= 0, "total_q must be non-negative");
    TORCH_CHECK(topk > 0 && topk <= 64, "CUDA q2col builder supports 1 <= topk <= 64");
    TORCH_CHECK(q2col_offsets.device() == moba_col_offsets.device() &&
                    q2col_valid.device() == moba_col_offsets.device(),
                "q2col output buffers must be on the CSC device");
    TORCH_CHECK(q2col_offsets.sizes() == at::IntArrayRef({static_cast<int64_t>(total_q) * num_heads, topk}),
                "q2col_offsets has incorrect shape");
    TORCH_CHECK(q2col_valid.sizes() == at::IntArrayRef({static_cast<int64_t>(total_q) * num_heads}),
                "q2col_valid has incorrect shape");

    at::cuda::CUDAGuard device_guard{moba_col_offsets.device()};
    // The builder atomically appends slots, so callers do not need to remember
    // a separate clear operation before each graph replay.
    q2col_valid.zero_();

    Lightning_moba_q2col_params params = {};
    params.col_offsets_ptr = reinterpret_cast<const int64_t*>(moba_col_offsets.data_ptr());
    params.col_nnz_ptr = reinterpret_cast<const int*>(moba_col_nnz.data_ptr());
    params.indices_ptr = reinterpret_cast<const int*>(moba_row_indices.data_ptr());
    params.cu_seqlens_q_ptr = reinterpret_cast<const int*>(cu_seqlens_q.data_ptr());
    params.q2col_offsets_ptr = reinterpret_cast<int*>(q2col_offsets.data_ptr());
    params.q2col_valid_ptr = reinterpret_cast<int*>(q2col_valid.data_ptr());
    params.batch_size = batch_size;
    params.num_heads = num_heads;
    params.num_columns = num_columns;
    params.total_q = total_q;
    params.topk = topk;
    params.col_offsets_batch_stride = moba_col_offsets.stride(0);
    params.col_offsets_head_stride = moba_col_offsets.stride(1);
    params.col_nnz_batch_stride = moba_col_nnz.stride(0);
    params.col_nnz_head_stride = moba_col_nnz.stride(1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_lightning_moba_build_q2col_dispatch(params, stream);
}

std::vector<at::Tensor>
moba_build_q2col_kvcentric(
    const at::Tensor &moba_col_offsets,
    const at::Tensor &moba_col_nnz,
    const at::Tensor &moba_row_indices,
    const at::Tensor &cu_seqlens_q,
    int total_q,
    int num_heads,
    int topk
) {
    auto q2col_offsets = torch::empty(
        {static_cast<int64_t>(total_q) * num_heads, topk},
        moba_col_nnz.options().dtype(torch::kInt32));
    // The shared *_into implementation performs the required clear.
    auto q2col_valid = torch::empty(
        {static_cast<int64_t>(total_q) * num_heads},
        moba_col_nnz.options().dtype(torch::kInt32));
    moba_build_q2col_kvcentric_into(
        moba_col_offsets, moba_col_nnz, moba_row_indices, cu_seqlens_q,
        total_q, num_heads, topk, q2col_offsets, q2col_valid);
    return {q2col_offsets, q2col_valid};
}

std::vector<at::Tensor>
moba_build_slot_meta_kvcentric(
    const at::Tensor &moba_col_offsets,    // batch_size x num_heads x max_lg_col_num
    const at::Tensor &moba_col_nnz,        // batch_size x num_heads x max_lg_col_num
    const at::Tensor &moba_row_indices,    // total_selected_rows, sequence-local q rows
    const at::Tensor &cu_seqlens_q,        // batch_size + 1
    int total_q,
    int num_heads,
    int topk
) {
    CHECK_DEVICE(moba_col_offsets); CHECK_DEVICE(moba_col_nnz);
    CHECK_DEVICE(moba_row_indices); CHECK_DEVICE(cu_seqlens_q);
    CHECK_CONTIGUOUS(moba_col_offsets);
    CHECK_CONTIGUOUS(moba_col_nnz);
    CHECK_CONTIGUOUS(moba_row_indices);
    CHECK_CONTIGUOUS(cu_seqlens_q);

    TORCH_CHECK(moba_col_offsets.dtype() == torch::kInt64, "moba_col_offsets must be int64");
    TORCH_CHECK(moba_col_nnz.dtype() == torch::kInt32, "moba_col_nnz must be int32");
    TORCH_CHECK(moba_row_indices.dtype() == torch::kInt32, "moba_row_indices must be int32");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(moba_col_offsets.dim() == 3, "moba_col_offsets must have shape [batch, heads, columns]");
    TORCH_CHECK(moba_col_nnz.dim() == 3, "moba_col_nnz must have shape [batch, heads, columns]");
    TORCH_CHECK(moba_row_indices.dim() == 1, "moba_row_indices must be 1D");
    TORCH_CHECK(cu_seqlens_q.dim() == 1, "cu_seqlens_q must be 1D");

    const int batch_size = moba_col_offsets.size(0);
    const int heads_from_offsets = moba_col_offsets.size(1);
    const int num_columns = moba_col_offsets.size(2);
    const int total_nnz = moba_row_indices.size(0);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads == heads_from_offsets, "num_heads mismatch");
    CHECK_SHAPE(moba_col_nnz, batch_size, num_heads, num_columns);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    TORCH_CHECK(total_q >= 0, "total_q must be non-negative");
    TORCH_CHECK(total_nnz >= 0, "total_nnz must be non-negative");
    TORCH_CHECK(topk > 0 && topk <= 32, "CUDA slot metadata builder supports 1 <= topk <= 32");
    TORCH_CHECK(total_nnz <= std::numeric_limits<int>::max(),
                "slot metadata builder currently requires total_nnz to fit int32");

    at::cuda::CUDAGuard device_guard{moba_col_offsets.device()};
    auto csc_to_slot = torch::empty(
        {total_nnz},
        moba_col_nnz.options().dtype(torch::kInt32));
    auto split_count = torch::zeros(
        {static_cast<int64_t>(total_q) * num_heads},
        moba_col_nnz.options().dtype(torch::kInt32));

    Lightning_moba_slot_meta_params params = {};
    params.col_offsets_ptr = reinterpret_cast<const int64_t*>(moba_col_offsets.data_ptr());
    params.col_nnz_ptr = reinterpret_cast<const int*>(moba_col_nnz.data_ptr());
    params.indices_ptr = reinterpret_cast<const int*>(moba_row_indices.data_ptr());
    params.cu_seqlens_q_ptr = reinterpret_cast<const int*>(cu_seqlens_q.data_ptr());
    params.csc_to_slot_ptr = reinterpret_cast<int*>(csc_to_slot.data_ptr());
    params.split_count_ptr = reinterpret_cast<int*>(split_count.data_ptr());
    params.batch_size = batch_size;
    params.num_heads = num_heads;
    params.num_columns = num_columns;
    params.total_q = total_q;
    params.total_nnz = total_nnz;
    params.topk = topk;
    params.col_offsets_batch_stride = moba_col_offsets.stride(0);
    params.col_offsets_head_stride = moba_col_offsets.stride(1);
    params.col_nnz_batch_stride = moba_col_nnz.stride(0);
    params.col_nnz_head_stride = moba_col_nnz.stride(1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_lightning_moba_build_slot_meta_dispatch(params, stream);
    return {csc_to_slot, split_count};
}

void run_moba_bwd(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_moba_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}

std::vector<at::Tensor>
moba_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,    // h x total_q, softmax logsumexp
               std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &moba_col_offsets, // batch_size x num_heads x max_lg_col_num
               const at::Tensor &moba_col_nnz, // batch_size x num_heads x max_lg_col_num, number of wanted rows in each column
               const at::Tensor &moba_row_indices, // total_selected_rows
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               const float softcap,
               const bool deterministic,
               const int m_lg_block_dim,  // Size of logical blocks in M dimension
               const int n_lg_block_dim,  // Size of logical blocks in N dimension
               std::optional<at::Generator> gen_,
               std::optional<at::Tensor> &rng_state) {

    #ifdef LIGHTNINGMOBA_DISABLE_BACKWARD
        TORCH_CHECK(false, "This LightningMoBA build does not support backward.");
    #endif

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "LightningMoBA only supports Ampere GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "LightningMoBA only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
    // Validate MOBA sparse pattern tensors
    TORCH_CHECK(moba_col_offsets.dtype() == torch::kInt64, "moba_col_offsets must have dtype int64 (index_t)");
    TORCH_CHECK(moba_col_nnz.dtype() == torch::kInt32, "moba_col_nnz must have dtype int32");
    TORCH_CHECK(moba_row_indices.dtype() == torch::kInt32, "moba_row_indices must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    CHECK_DEVICE(moba_col_offsets);
    CHECK_DEVICE(moba_col_nnz);
    CHECK_DEVICE(moba_row_indices);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    CHECK_CONTIGUOUS(moba_col_offsets);
    CHECK_CONTIGUOUS(moba_col_nnz);
    CHECK_CONTIGUOUS(moba_row_indices);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    const int max_lg_col_num = moba_col_offsets.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "LightningMoBA backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(m_lg_block_dim % 64 == 0, "m_lg_block_dim must be a multiple of 64");
    TORCH_CHECK(n_lg_block_dim % 64 == 0, "n_lg_block_dim must be a multiple of 64");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    CHECK_SHAPE(moba_col_offsets, batch_size, num_heads, max_lg_col_num);
    CHECK_SHAPE(moba_col_nnz, batch_size, num_heads, max_lg_col_num);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    // bool loop = max_seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto opts = q.options();
    auto softmax_d = torch::empty({num_heads, total_q + 128 * batch_size}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    if (loop) {
        if (!deterministic) {
            dq_accum = torch::empty({total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if( zero_tensors ) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    Lightning_moba_bwd_params params;

    set_params_moba_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     moba_col_offsets,
                     moba_col_nnz,
                     moba_row_indices,
                     m_lg_block_dim,
                     n_lg_block_dim,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     loop ? dq_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/true);
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
    params.total_q = total_q;

    auto launch = &run_moba_bwd;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    }

    return { dq, dk, dv, softmax_d };
}
} // namespace LIGHTNING_MOBA_NAMESPACE

// Forward declare init function from flash_topk_api
void init_lightning_moba(pybind11::module &m) {
    m.def("moba_varlen_fwd",
          &LIGHTNING_MOBA_NAMESPACE::moba_varlen_fwd,
          "Forward pass of moba (variable length)");
    m.def("moba_varlen_fwd_kvcentric",
          &LIGHTNING_MOBA_NAMESPACE::moba_varlen_fwd_kvcentric,
          "KV-centric forward pass of moba (variable length)");
    m.def("moba_varlen_fwd_kvcentric_worklist",
          &LIGHTNING_MOBA_NAMESPACE::moba_varlen_fwd_kvcentric_worklist,
          "KV-centric forward pass with flat worklist M-split");
    m.def("moba_varlen_fwd_kvcentric_slot",
          &LIGHTNING_MOBA_NAMESPACE::moba_varlen_fwd_kvcentric_slot,
          "KV-centric forward pass with slot-major partial output");
    m.def("moba_sm90_phase4b_csc_pipe",
          &LIGHTNING_MOBA_NAMESPACE::moba_sm90_phase4b_csc_pipe,
          "SM90 KV-centric CSC producer/consumer kernel");
    m.def("moba_build_kvcentric_worklist",
          &LIGHTNING_MOBA_NAMESPACE::moba_build_kvcentric_worklist,
          "Build KV-centric [batch, head, column, q_begin, q_count] CUDA worklist from CSC counts");
    m.def("moba_build_kvcentric_msplit_worklist",
          &LIGHTNING_MOBA_NAMESPACE::moba_build_kvcentric_msplit_worklist,
          "Build M-split KV-centric CUDA worklist with 64-row q_begin/q_count records");
    m.def("moba_combine_kvcentric",
          &LIGHTNING_MOBA_NAMESPACE::moba_combine_kvcentric,
          "CUDA combine for KV-centric forward partial outputs");
    m.def("moba_combine_slot_kvcentric",
          &LIGHTNING_MOBA_NAMESPACE::moba_combine_slot_kvcentric,
          "CUDA combine for slot-major KV-centric partial outputs");
    m.def("moba_build_q2col_kvcentric",
          &LIGHTNING_MOBA_NAMESPACE::moba_build_q2col_kvcentric,
          "CUDA reverse-index builder for KV-centric combine");
    m.def("moba_build_q2col_kvcentric_into",
          &LIGHTNING_MOBA_NAMESPACE::moba_build_q2col_kvcentric_into,
          "CUDA reverse-index builder into caller-owned KV-centric buffers");
    m.def("moba_build_slot_meta_kvcentric",
          &LIGHTNING_MOBA_NAMESPACE::moba_build_slot_meta_kvcentric,
          "CUDA slot metadata builder for slot-major KV-centric partials");
    m.def("moba_varlen_bwd",
          &LIGHTNING_MOBA_NAMESPACE::moba_varlen_bwd,
          "Backward pass of moba (variable length)");
}
