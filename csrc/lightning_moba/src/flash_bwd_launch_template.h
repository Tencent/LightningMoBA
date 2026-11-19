/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_bwd_launch_template.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "lightning_moba_static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"

namespace LIGHTNING_MOBA_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: LightningMoBA requires building with sm version sm80-sm90, but was built for < 8.0!");

#define DEFINE_LIGHTNING_MOBA_BACKWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Lightning_moba_bwd_params params)

DEFINE_LIGHTNING_MOBA_BACKWARD_KERNEL(lightning_moba_bwd_dq_dk_dv_loop_seqk_parallel_kernel, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        LIGHTNING_MOBA_NAMESPACE::compute_moba_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void lightning_moba_bwd_dot_do_o_kernel(const Lightning_moba_bwd_params params) {
    LIGHTNING_MOBA_NAMESPACE::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void lightning_moba_bwd_convert_dq_kernel(const Lightning_moba_bwd_params params, const int nsplits) {
    LIGHTNING_MOBA_NAMESPACE::convert_dQ<Kernel_traits>(params, nsplits);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_lightning_moba_bwd_seqk_parallel(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    if (params.deterministic) {
        int num_sm = get_num_sm(get_current_device());
        gridDimx = (num_sm + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.b, params.h);

    if (!params.deterministic) {
        lightning_moba_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        lightning_moba_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    auto kernel = &lightning_moba_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Has_alibi && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !Has_alibi, Is_softcap>;
                    if (smem_size_dq_dk_dv >= 48 * 1024)  {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                    }
                    kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });

    auto kernel_dq = &lightning_moba_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_lightning_moba_bwd(Lightning_moba_bwd_params &params, cudaStream_t stream) {
#ifndef LIGHTNINGMOBA_DISABLE_BACKWARD
    run_lightning_moba_bwd_seqk_parallel<Kernel_traits, Is_dropout, Is_causal>(params, stream);
#endif
}

template<typename T, bool Is_causal>
void run_moba_bwd_hdim32(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
            if constexpr(!Is_dropout) {  // We can afford more registers to keep V in registers
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {  // 96 KB
            run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_bwd_hdim64(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 144 * 1024) {
            if (params.n_lg_block_dim % 128 == 0) {
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            if (params.n_lg_block_dim % 128 == 0) {
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_bwd_hdim96(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 116 * 1024) {
            if constexpr(!Is_dropout) {  // 92KB
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {  // 116 KB
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_bwd_hdim128(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 144 * 1024) {
            if (params.n_lg_block_dim % 128 == 0) {
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 2, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 128, 64, 8, 4, 2, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>, Is_dropout, Is_causal>(params, stream);
        }

    });
}

template<typename T, bool Is_causal>
void run_moba_bwd_hdim192(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 136 * 1024) {
            run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_bwd_hdim256(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 176 * 1024) {  // H100
            run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_dropout, Is_causal>(params, stream); // true to avoid random output, fix later
        } else if (max_smem_per_block >= 144 * 1024) {
            run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            if constexpr (!Is_dropout) {
                run_lightning_moba_bwd<Lightning_moba_bwd_kernel_traits<Headdim, 64, 32, 8, 4, 1, 2, true, true, T>, false, Is_causal>(params, stream);
            }
        }
    });
}

} // namespace LIGHTNING_MOBA_NAMESPACE
