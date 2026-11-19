#include "flash_combine_kernel.h"
#include <c10/cuda/CUDAException.h>
#include <cstdlib>
#include <cstring>

namespace LIGHTNING_MOBA_NAMESPACE {

void run_lightning_moba_combine_dispatch(
    const Lightning_moba_combine_params &params,
    bool is_bf16,
    cudaStream_t stream
) {
    const char* k1_env = std::getenv("LIGHTNING_MOBA_KVCENTRIC_SM90_K1");
    const bool phase4b_pipeline_selected =
        k1_env != nullptr &&
        std::strcmp(k1_env, "phase4b_csc_pipe_float2_overlap_chunk128") == 0;
    const bool use_topk16_prefetch =
        phase4b_pipeline_selected &&
        params.topk == 16 && params.head_dim == 128;
    const bool use_topk16_parallel_weights =
        phase4b_pipeline_selected &&
        params.topk == 16 && params.head_dim == 128;
    if (params.partial_dtype == 1) {
        if (is_bf16) {
            if (use_topk16_prefetch) {
                run_lightning_moba_combine_topk16_prefetch<__nv_bfloat16, __nv_bfloat16>(
                    params, stream);
            } else {
                run_lightning_moba_combine<__nv_bfloat16, __nv_bfloat16>(params, stream);
            }
        } else {
            if (use_topk16_prefetch) {
                run_lightning_moba_combine_topk16_prefetch<__nv_bfloat16, __half>(
                    params, stream);
            } else {
                run_lightning_moba_combine<__nv_bfloat16, __half>(params, stream);
            }
        }
    } else if (params.partial_dtype == 2) {
        if (is_bf16) {
            if (use_topk16_prefetch) {
                run_lightning_moba_combine_topk16_prefetch<__half, __nv_bfloat16>(
                    params, stream);
            } else {
                run_lightning_moba_combine<__half, __nv_bfloat16>(params, stream);
            }
        } else {
            if (use_topk16_prefetch) {
                run_lightning_moba_combine_topk16_prefetch<__half, __half>(
                    params, stream);
            } else {
                run_lightning_moba_combine<__half, __half>(params, stream);
            }
        }
    } else {
        if (is_bf16) {
            if (use_topk16_parallel_weights) {
                run_lightning_moba_combine_topk16_parallel_weights<float, __nv_bfloat16>(
                    params, stream);
            } else if (use_topk16_prefetch) {
                run_lightning_moba_combine_topk16_prefetch<float, __nv_bfloat16>(
                    params, stream);
            } else {
                run_lightning_moba_combine<float, __nv_bfloat16>(params, stream);
            }
        } else {
            if (use_topk16_prefetch) {
                run_lightning_moba_combine_topk16_prefetch<float, __half>(
                    params, stream);
            } else {
                run_lightning_moba_combine<float, __half>(params, stream);
            }
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_lightning_moba_combine_slot_dispatch(
    const Lightning_moba_slot_combine_params &params,
    bool is_bf16,
    cudaStream_t stream
) {
    if (is_bf16) {
        run_lightning_moba_combine_slot<__nv_bfloat16>(params, stream);
    } else {
        run_lightning_moba_combine_slot<__half>(params, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_lightning_moba_build_q2col_dispatch(
    const Lightning_moba_q2col_params &params,
    cudaStream_t stream
) {
    dim3 grid(params.num_columns, params.num_heads, params.batch_size);
    dim3 block(128, 1, 1);
    lightning_moba_build_q2col_kernel<<<grid, block, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_lightning_moba_build_slot_meta_dispatch(
    const Lightning_moba_slot_meta_params &params,
    cudaStream_t stream
) {
    dim3 grid(params.num_columns, params.num_heads, params.batch_size);
    dim3 block(128, 1, 1);
    lightning_moba_build_slot_meta_kernel<<<grid, block, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace LIGHTNING_MOBA_NAMESPACE
