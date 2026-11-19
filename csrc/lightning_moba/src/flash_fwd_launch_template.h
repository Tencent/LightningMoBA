/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string_view>
#include <type_traits>

#include "lightning_moba_static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"
#include "flash_fwd_kvcentric_kernel.h"

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

#define DEFINE_LIGHTNING_MOBA_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Lightning_moba_fwd_params params)

DEFINE_LIGHTNING_MOBA_FORWARD_KERNEL(lightning_moba_fwd_kernel, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        LIGHTNING_MOBA_NAMESPACE::compute_moba_attn<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_lightning_moba_fwd(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_lg_m_block = (params.seqlen_q + params.m_lg_block_dim - 1) / params.m_lg_block_dim;
    dim3 grid(num_lg_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                auto kernel = &lightning_moba_fwd_kernel<Kernel_traits, Is_dropout /*&& !Is_softcap*/, Is_causal, /*Has_alibi*/false, IsEvenMNConst && IsEvenKConst /*&& !Has_alibi*/ && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst /*&& !Has_alibi*/, /*Is_softcap*/ false, ReturnSoftmaxConst && Is_dropout /*&& !Is_softcap*/>;
                if (smem_size >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

enum class Hdim128CausalTileOverride {
    Default,
    Tile64x64,
    Tile128x32,
    Tile128x64,
};

inline Hdim128CausalTileOverride get_hdim128_causal_tile_override() {
    const char* env = std::getenv("LIGHTNING_MOBA_FWD_HDIM128_CAUSAL_TILE");
    if (env == nullptr) { return Hdim128CausalTileOverride::Default; }
    const std::string_view value(env);
    if (value == "64x64") { return Hdim128CausalTileOverride::Tile64x64; }
    if (value == "128x32") { return Hdim128CausalTileOverride::Tile128x32; }
    if (value == "128x64") { return Hdim128CausalTileOverride::Tile128x64; }
    return Hdim128CausalTileOverride::Default;
}

enum class KvcentricTileOverride {
    Default,
    Tile64x64,
    Tile64x128,
    Tile128x32,
    Tile128x64,
};

inline KvcentricTileOverride get_kvcentric_tile_override() {
    const char* env = std::getenv("LIGHTNING_MOBA_KVCENTRIC_TILE");
    if (env == nullptr) { return KvcentricTileOverride::Default; }
    const std::string_view value(env);
    if (value == "64x64") { return KvcentricTileOverride::Tile64x64; }
    if (value == "64x128") { return KvcentricTileOverride::Tile64x128; }
    if (value == "128x32") { return KvcentricTileOverride::Tile128x32; }
    if (value == "128x64") { return KvcentricTileOverride::Tile128x64; }
    return KvcentricTileOverride::Default;
}

template<typename T, bool Is_causal>
inline bool should_use_moba_hdim128_sm90_entry(const Lightning_moba_fwd_params &params, int cc_major) {
    if constexpr (!Is_causal || !std::is_same_v<T, cutlass::bfloat16_t>) {
        return false;
    } else {
        return cc_major >= 9 && params.p_dropout == 0.0f && params.m_lg_block_dim > 1024;
    }
}

template<typename T, bool Is_dropout, bool Is_causal>
inline bool try_run_moba_fwd_hdim128_sm90_entry(Lightning_moba_fwd_params &params, cudaStream_t stream, int cc_major) {
    if constexpr (Is_dropout || !Is_causal || !std::is_same_v<T, cutlass::bfloat16_t>) {
        return false;
    } else {
        if (!should_use_moba_hdim128_sm90_entry<T, Is_causal>(params, cc_major)) {
            return false;
        }
        // Stage 2 skeleton only: keep Hopper-specific dispatch isolated here first.
        // The actual SM90-native kernel will replace this fallback path incrementally.
        if (params.m_lg_block_dim > 2048) {
            run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<128, 64, 64, 4, 4096, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<128, 64, 64, 4, 2048, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        return true;
    }
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim32(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 128, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim64(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
            // Using block size (64 x 256) is 27% slower for seqlen=2k
            // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
            if (params.n_lg_block_dim % 128 == 0) {
                run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 128, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim96(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 128, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // These two are always slower
    });
}

// [MODIFIED] Optimized for A6000 (sm86) to avoid Shared Memory OOM
template<typename T, bool Is_causal>
void run_moba_fwd_hdim128(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0; // A6000, 3090, 4090
    const auto tile_override = get_hdim128_causal_tile_override();

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            if (is_sm8x) {
                // [A6000 Fix] Use 512 cap
                if constexpr(!Is_causal) {
                    run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 128, 32, 4, 512, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 512, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                if (try_run_moba_fwd_hdim128_sm90_entry<T, Is_dropout, Is_causal>(params, stream, cc_major)) {
                    return;
                }
                if constexpr (Is_causal) {
                    switch (tile_override) {
                    case Hdim128CausalTileOverride::Tile128x32:
                        run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 128, 32, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
                        break;
                    case Hdim128CausalTileOverride::Tile128x64:
                        run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 128, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
                        break;
                    case Hdim128CausalTileOverride::Tile64x64:
                    case Hdim128CausalTileOverride::Default:
                    default:
                        run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
                        break;
                    }
                } else {
                    // [Default A100/H100 path]
                    run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            }
        } else {
            if (is_sm8x) {
                // [A6000 Fix] Use 512 cap for dropout case too
                run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 128, 32, 4, 512, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                // [Default A100]
                run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 128, 32, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim192(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream); // warps = 8 cause problem for moba, check later
        } else {
            run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim256(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For A100, we want to run with 128 x 64 (128KB smem).
        // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream); // exceed smem limit for A100
        } else {
            run_lightning_moba_fwd<Lightning_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // 64 KB
        // 96 KB
    });
}

// =============================================================================
// KV-Centric Forward Kernel Launch
// =============================================================================

DEFINE_LIGHTNING_MOBA_FORWARD_KERNEL(lightning_moba_fwd_kvcentric_kernel, bool Is_causal, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        LIGHTNING_MOBA_NAMESPACE::compute_moba_attn_kv_centric<Kernel_traits, Is_causal, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_LIGHTNING_MOBA_FORWARD_KERNEL(lightning_moba_fwd_kvcentric_csc_fp32_kernel, bool Is_causal, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        LIGHTNING_MOBA_NAMESPACE::compute_moba_attn_kv_centric_csc_fp32<Kernel_traits, Is_causal, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_LIGHTNING_MOBA_FORWARD_KERNEL(lightning_moba_fwd_kvcentric_worklist_kernel, bool Is_causal, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        LIGHTNING_MOBA_NAMESPACE::compute_moba_attn_kv_centric_worklist<Kernel_traits, Is_causal, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_causal>
void run_lightning_moba_fwd_kvcentric(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool use_worklist = params.kv_worklist_ptr != nullptr && params.kv_work_count > 0;
    const bool use_csc_fp32_fastpath = !use_worklist && !params.kv_use_slot_major && params.kv_partial_dtype == 0;

    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        cudaStreamCaptureStatus capture_status{};
        C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        const bool is_capturing = capture_status != cudaStreamCaptureStatusNone;
        if (use_csc_fp32_fastpath) {
            auto kernel = &lightning_moba_fwd_kvcentric_csc_fp32_kernel<Kernel_traits, Is_causal, IsEvenKConst>;
            if (smem_size >= 48 * 1024 && !is_capturing) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            const int num_columns = params.max_lg_col_num;
            dim3 grid(num_columns, params.b, params.h);
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        } else if (use_worklist) {
            auto kernel = &lightning_moba_fwd_kvcentric_worklist_kernel<Kernel_traits, Is_causal, IsEvenKConst>;
            if (smem_size >= 48 * 1024 && !is_capturing) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            dim3 grid(params.kv_work_count, 1, 1);
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        } else {
            auto kernel = &lightning_moba_fwd_kvcentric_kernel<Kernel_traits, Is_causal, IsEvenKConst>;
            if (smem_size >= 48 * 1024 && !is_capturing) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            // Grid: (num_columns, batch, num_heads)
            const int num_columns = params.max_lg_col_num;
            dim3 grid(num_columns, params.b, params.h);
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_kvcentric_hdim128(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    // kMxLgBlockM is unused for kv-centric but needed for the shared traits template.
    switch (get_kvcentric_tile_override()) {
    case KvcentricTileOverride::Tile64x128:
        run_lightning_moba_fwd_kvcentric<Lightning_moba_fwd_kernel_traits<128, 64, 128, 4, 1024, false, false, T>, Is_causal>(params, stream);
        break;
    case KvcentricTileOverride::Tile128x32:
        run_lightning_moba_fwd_kvcentric<Lightning_moba_fwd_kernel_traits<128, 128, 32, 4, 1024, false, false, T>, Is_causal>(params, stream);
        break;
    case KvcentricTileOverride::Tile128x64:
        run_lightning_moba_fwd_kvcentric<Lightning_moba_fwd_kernel_traits<128, 128, 64, 4, 1024, false, false, T>, Is_causal>(params, stream);
        break;
    case KvcentricTileOverride::Tile64x64:
    case KvcentricTileOverride::Default:
    default:
        run_lightning_moba_fwd_kvcentric<Lightning_moba_fwd_kernel_traits<128, 64, 64, 4, 1024, false, false, T>, Is_causal>(params, stream);
        break;
    }
}

}  // namespace LIGHTNING_MOBA_NAMESPACE
