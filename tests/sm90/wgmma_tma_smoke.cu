/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Derived from CUTLASS examples/cute/tutorial/hopper/wgmma_tma_sm90.cu.
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

using namespace cute;

namespace lightning_moba_sm90_smoke {

constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 128;
constexpr int kBlockK = 64;
constexpr int kStages = 2;
constexpr int kPvN = 128;
constexpr int kPvK = 64;
constexpr bool kStage1PvBatchAcrossKBlocks = true;
constexpr bool kStage1PvPreloadAllABeforeBatch = true;

void check_cuda(cudaError_t error, char const* operation) {
    if (error != cudaSuccess) {
        std::cerr << "[FAIL] " << operation << ": " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct Sm90SharedStorage {
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> a;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> b;
    uint64_t tma_barrier[kStages];
};

template <
    class ProblemShape,
    class CtaTiler,
    class ElementA,
    class SmemLayoutA,
    class TmaA,
    class ElementB,
    class SmemLayoutB,
    class TmaB,
    class ElementC,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(size(TiledMma{}))::value)
void wgmma_tma_kernel(
    ProblemShape shape_mnk,
    CtaTiler cta_tiler,
    ElementA const* a,
    CUTLASS_GRID_CONSTANT TmaA const tma_a,
    ElementB const* b,
    CUTLASS_GRID_CONSTANT TmaB const tma_b,
    ElementC* c,
    CStride stride_c,
    TiledMma mma) {
    auto [m, n, k] = shape_mnk;
    Tensor tensor_a = tma_a.get_tma_tensor(make_shape(m, k));
    Tensor tensor_b = tma_b.get_tma_tensor(make_shape(n, k));
    Tensor tensor_c = make_tensor(make_gmem_ptr(c), make_shape(m, n), stride_c);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor global_a = local_tile(tensor_a, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor global_b = local_tile(tensor_b, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor global_c = local_tile(tensor_c, cta_tiler, cta_coord, Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    using SharedStorage = Sm90SharedStorage<ElementA, ElementB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor shared_a = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
    Tensor shared_b = make_tensor(make_smem_ptr(storage.b.begin()), SmemLayoutB{});

    auto [tma_global_a, tma_shared_a] = tma_partition(
        tma_a,
        Int<0>{},
        Layout<_1>{},
        group_modes<0, 2>(shared_a),
        group_modes<0, 2>(global_a));
    auto [tma_global_b, tma_shared_b] = tma_partition(
        tma_b,
        Int<0>{},
        Layout<_1>{},
        group_modes<0, 2>(shared_b),
        group_modes<0, 2>(global_b));

    constexpr int kTransactionBytes =
        sizeof(make_tensor_like(tensor<0>(tma_shared_a))) +
        sizeof(make_tensor_like(tensor<0>(tma_shared_b)));
    using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int elected_lane = cute::elect_one_sync();
    if (warp_idx == 0 && elected_lane) {
        for (int stage = 0; stage < kStages; ++stage) {
            ProducerBarrier::init(&storage.tma_barrier[stage], 1);
        }
    }
    cluster_sync();

    // The attention-shaped smoke has exactly two K tiles, one per pipeline stage.
    if (warp_idx == 0 && elected_lane) {
        for (int stage = 0; stage < kStages; ++stage) {
            ProducerBarrier::arrive_and_expect_tx(
                &storage.tma_barrier[stage], kTransactionBytes);
            copy(
                tma_a.with(storage.tma_barrier[stage]),
                tma_global_a(_, stage),
                tma_shared_a(_, stage));
            copy(
                tma_b.with(storage.tma_barrier[stage]),
                tma_global_b(_, stage),
                tma_shared_b(_, stage));
        }
    }

    auto thread_mma = mma.get_thread_slice(threadIdx.x);
    Tensor mma_shared_a = thread_mma.partition_A(shared_a);
    Tensor mma_shared_b = thread_mma.partition_B(shared_b);
    Tensor mma_global_c = thread_mma.partition_C(global_c);
    Tensor coord_c = make_identity_tensor(take<0, 2>(cta_tiler));
    Tensor thread_coord_c = thread_mma.partition_C(coord_c);
    Tensor fragment_a = thread_mma.make_fragment_A(mma_shared_a);
    Tensor fragment_b = thread_mma.make_fragment_B(mma_shared_b);
    Tensor accum = thread_mma.make_fragment_C(mma_global_c);
    clear(accum);

    auto read_state = cutlass::PipelineState<kStages>();
    for (int tile = 0; tile < kStages; ++tile) {
        int stage = read_state.index();
        ProducerBarrier::wait(&storage.tma_barrier[stage], read_state.phase());

        warpgroup_fence_operand(accum);
        warpgroup_arrive();
        gemm(mma, fragment_a(_, _, _, stage), fragment_b(_, _, _, stage), accum);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(accum);
        ++read_state;
    }

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
        auto coord = thread_coord_c(i);
        c[get<0>(coord) + get<1>(coord) * kM] = accum(i);
    }
}

template <class ElementA, class ElementB, class ElementC>
void launch_qk_smoke(ElementA const* a, ElementB const* b, ElementC* c) {
    auto problem_shape = make_shape(Int<kM>{}, Int<kN>{}, Int<kK>{});
    auto cta_tiler = make_shape(Int<kM>{}, Int<kN>{}, Int<kBlockK>{});
    auto stride_a = make_stride(Int<kK>{}, Int<1>{});
    auto stride_b = make_stride(Int<kK>{}, Int<1>{});
    auto stride_c = make_stride(Int<1>{}, Int<kM>{});

    auto smem_a = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<ElementA>{},
        make_shape(Int<kM>{}, Int<kBlockK>{}, Int<kStages>{}));
    auto smem_b = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<ElementB>{},
        make_shape(Int<kN>{}, Int<kBlockK>{}, Int<kStages>{}));
    auto mma = make_tiled_mma(
        SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{});

    Tensor tensor_a = make_tensor(a, make_shape(kM, kK), stride_a);
    Tensor tensor_b = make_tensor(b, make_shape(kN, kK), stride_b);
    auto tma_a = make_tma_atom(
        SM90_TMA_LOAD{}, tensor_a, smem_a(_, _, 0), make_shape(Int<kM>{}, Int<kBlockK>{}));
    auto tma_b = make_tma_atom(
        SM90_TMA_LOAD{}, tensor_b, smem_b(_, _, 0), make_shape(Int<kN>{}, Int<kBlockK>{}));

    using SharedStorage = Sm90SharedStorage<ElementA, ElementB, decltype(smem_a), decltype(smem_b)>;
    int smem_bytes = sizeof(SharedStorage);
    auto kernel = &wgmma_tma_kernel<
        decltype(problem_shape),
        decltype(cta_tiler),
        ElementA,
        decltype(smem_a),
        decltype(tma_a),
        ElementB,
        decltype(smem_b),
        decltype(tma_b),
        ElementC,
        decltype(stride_c),
        decltype(mma)>;

    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
        "cudaFuncSetAttribute");

    cutlass::ClusterLaunchParams launch_params{
        dim3(1, 1, 1), dim3(size(mma)), dim3(1, 1, 1), smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        launch_params,
        reinterpret_cast<void const*>(kernel),
        problem_shape,
        cta_tiler,
        a,
        tma_a,
        b,
        tma_b,
        c,
        stride_c,
        mma);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[FAIL] cluster launch returned status " << int(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

template <
    class ProblemShape,
    class CtaTiler,
    class ElementA,
    class SmemLayoutA,
    class TmaA,
    class ElementB,
    class SmemLayoutB,
    class TmaB,
    class ElementC,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(decltype(size(TiledMma{}))::value)
void wgmma_pv_rs_kernel(
    ProblemShape shape_mnk,
    CtaTiler cta_tiler,
    ElementA const* a,
    CUTLASS_GRID_CONSTANT TmaA const tma_a,
    ElementB const* b,
    CUTLASS_GRID_CONSTANT TmaB const tma_b,
    ElementC* c,
    CStride stride_c,
    TiledMma mma) {
    auto [m, n, k] = shape_mnk;
    Tensor tensor_a = tma_a.get_tma_tensor(make_shape(m, k));
    Tensor tensor_b = tma_b.get_tma_tensor(make_shape(n, k));
    Tensor tensor_c = make_tensor(make_gmem_ptr(c), make_shape(m, n), stride_c);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor global_a = local_tile(tensor_a, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor global_b = local_tile(tensor_b, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor global_c = local_tile(tensor_c, cta_tiler, cta_coord, Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    using SharedStorage = Sm90SharedStorage<ElementA, ElementB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor shared_a = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
    Tensor shared_b = make_tensor(make_smem_ptr(storage.b.begin()), SmemLayoutB{});

    auto [tma_global_a, tma_shared_a] = tma_partition(
        tma_a,
        Int<0>{},
        Layout<_1>{},
        group_modes<0, 2>(shared_a),
        group_modes<0, 2>(global_a));
    auto [tma_global_b, tma_shared_b] = tma_partition(
        tma_b,
        Int<0>{},
        Layout<_1>{},
        group_modes<0, 2>(shared_b),
        group_modes<0, 2>(global_b));

    constexpr int kTransactionBytes =
        sizeof(make_tensor_like(tensor<0>(tma_shared_a))) +
        sizeof(make_tensor_like(tensor<0>(tma_shared_b)));
    using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int elected_lane = cute::elect_one_sync();
    if (warp_idx == 0 && elected_lane) {
        ProducerBarrier::init(&storage.tma_barrier[0], 1);
    }
    cluster_sync();
    if (warp_idx == 0 && elected_lane) {
        ProducerBarrier::arrive_and_expect_tx(
            &storage.tma_barrier[0], kTransactionBytes);
        copy(tma_a.with(storage.tma_barrier[0]), tma_global_a(_, 0), tma_shared_a(_, 0));
        copy(tma_b.with(storage.tma_barrier[0]), tma_global_b(_, 0), tma_shared_b(_, 0));
    }

    auto thread_mma = mma.get_thread_slice(threadIdx.x);
    Tensor shared_a_pi = as_position_independent_swizzle_tensor(shared_a);
    Tensor mma_shared_b = thread_mma.partition_B(shared_b);
    Tensor mma_global_c = thread_mma.partition_C(global_c);
    Tensor coord_c = make_identity_tensor(take<0, 2>(cta_tiler));
    Tensor thread_coord_c = thread_mma.partition_C(coord_c);
    Tensor fragment_a = thread_mma.partition_fragment_A(shared_a_pi(_, _, Int<0>{}));
    Tensor fragment_b = thread_mma.make_fragment_B(mma_shared_b);
    Tensor accum = thread_mma.make_fragment_C(mma_global_c);
    clear(accum);
    CUTE_STATIC_ASSERT_V(size<2>(fragment_a) == Int<4>{});
    CUTE_STATIC_ASSERT_V(size<2>(fragment_b) == Int<4>{});

    auto smem_tiled_copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementA>{}, mma);
    auto smem_thread_copy_a = smem_tiled_copy_a.get_thread_slice(threadIdx.x);
    Tensor fragment_a_copy_view = smem_thread_copy_a.retile_D(fragment_a);
    Tensor shared_a_copy_view = smem_thread_copy_a.partition_S(shared_a_pi);

    ProducerBarrier::wait(&storage.tma_barrier[0], 0);
    warpgroup_fence_operand(accum);
    mma.accumulate_ = GMMA::ScaleOut::Zero;
    if constexpr (kStage1PvBatchAcrossKBlocks) {
        if constexpr (kStage1PvPreloadAllABeforeBatch) {
            // Stage 1 step 2: preload all shared->register A fragments, then issue
            // the full PV K-loop as one WGMMA batch.
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(fragment_a); ++k_block) {
                copy(
                    smem_tiled_copy_a,
                    shared_a_copy_view(_, _, k_block, 0),
                    fragment_a_copy_view(_, _, k_block));
            }
            warpgroup_arrive();
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(fragment_a); ++k_block) {
                gemm(mma, fragment_a(_, _, k_block), fragment_b(_, _, k_block, 0), accum);
                if (k_block == 0) {
                    mma.accumulate_ = GMMA::ScaleOut::One;
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();
        } else {
            // Stage 1 step 1: issue the full PV K-loop as one WGMMA batch, then wait once.
            warpgroup_arrive();
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(fragment_a); ++k_block) {
                copy(
                    smem_tiled_copy_a,
                    shared_a_copy_view(_, _, k_block, 0),
                    fragment_a_copy_view(_, _, k_block));
                gemm(mma, fragment_a(_, _, k_block), fragment_b(_, _, k_block, 0), accum);
                if (k_block == 0) {
                    mma.accumulate_ = GMMA::ScaleOut::One;
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(fragment_a); ++k_block) {
            copy(
                smem_tiled_copy_a,
                shared_a_copy_view(_, _, k_block, 0),
                fragment_a_copy_view(_, _, k_block));
            warpgroup_arrive();
            gemm(mma, fragment_a(_, _, k_block), fragment_b(_, _, k_block, 0), accum);
            if (k_block == 0) {
                mma.accumulate_ = GMMA::ScaleOut::One;
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();
        }
    }
    warpgroup_fence_operand(accum);

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
        auto coord = thread_coord_c(i);
        c[get<0>(coord) + get<1>(coord) * kM] = accum(i);
    }
}

template <class ElementA, class ElementB, class ElementC>
void launch_pv_rs_smoke(ElementA const* p, ElementB const* v, ElementC* output) {
    auto problem_shape = make_shape(Int<kM>{}, Int<kPvN>{}, Int<kPvK>{});
    auto cta_tiler = make_shape(Int<kM>{}, Int<kPvN>{}, Int<kPvK>{});
    auto stride_p = make_stride(Int<kPvK>{}, Int<1>{});
    // Logical B is [N, K], backed by row-major V[K, N].
    auto stride_v = make_stride(Int<1>{}, Int<kPvN>{});
    auto stride_output = make_stride(Int<1>{}, Int<kM>{});

    auto smem_p = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<ElementA>{},
        make_shape(Int<kM>{}, Int<kPvK>{}, Int<1>{}));
    auto smem_v = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<ElementB>{},
        make_shape(Int<kPvN>{}, Int<kPvK>{}, Int<1>{}));
    auto mma = make_tiled_mma(
        SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{});

    Tensor tensor_p = make_tensor(p, make_shape(kM, kPvK), stride_p);
    Tensor tensor_v = make_tensor(v, make_shape(kPvN, kPvK), stride_v);
    auto tma_p = make_tma_atom(
        SM90_TMA_LOAD{}, tensor_p, smem_p(_, _, 0), make_shape(Int<kM>{}, Int<kPvK>{}));
    auto tma_v = make_tma_atom(
        SM90_TMA_LOAD{}, tensor_v, smem_v(_, _, 0), make_shape(Int<kPvN>{}, Int<kPvK>{}));

    using SharedStorage = Sm90SharedStorage<ElementA, ElementB, decltype(smem_p), decltype(smem_v)>;
    int smem_bytes = sizeof(SharedStorage);
    auto kernel = &wgmma_pv_rs_kernel<
        decltype(problem_shape),
        decltype(cta_tiler),
        ElementA,
        decltype(smem_p),
        decltype(tma_p),
        ElementB,
        decltype(smem_v),
        decltype(tma_v),
        ElementC,
        decltype(stride_output),
        decltype(mma)>;

    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
        "cudaFuncSetAttribute(PV RS)");

    cutlass::ClusterLaunchParams launch_params{
        dim3(1, 1, 1), dim3(size(mma)), dim3(1, 1, 1), smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        launch_params,
        reinterpret_cast<void const*>(kernel),
        problem_shape,
        cta_tiler,
        p,
        tma_p,
        v,
        tma_v,
        output,
        stride_output,
        mma);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[FAIL] PV RS cluster launch returned status " << int(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace lightning_moba_sm90_smoke

using namespace lightning_moba_sm90_smoke;

int main() {
    cudaDeviceProp properties{};
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    std::cout << "[GPU] " << properties.name << " capability=" << properties.major << "."
              << properties.minor << "\n";
    if (properties.major != 9) {
        std::cerr << "[FAIL] This smoke test requires an SM90 Hopper GPU\n";
        return EXIT_FAILURE;
    }

    using Input = cutlass::bfloat16_t;
    using Output = float;
    thrust::host_vector<Input> host_a(kM * kK);
    thrust::host_vector<Input> host_b(kN * kK);
    thrust::host_vector<Output> host_c(kM * kN, 0.0f);

    for (int m = 0; m < kM; ++m) {
        for (int k = 0; k < kK; ++k) {
            host_a[m * kK + k] = Input(float(((m * 3 + k * 5) % 17) - 8) / 16.0f);
        }
    }
    for (int n = 0; n < kN; ++n) {
        for (int k = 0; k < kK; ++k) {
            host_b[n * kK + k] = Input(float(((n * 7 + k * 2) % 13) - 6) / 16.0f);
        }
    }

    thrust::device_vector<Input> device_a = host_a;
    thrust::device_vector<Input> device_b = host_b;
    thrust::device_vector<Output> device_c = host_c;

    launch_qk_smoke(
        device_a.data().get(), device_b.data().get(), device_c.data().get());
    check_cuda(cudaDeviceSynchronize(), "initial kernel synchronization");
    host_c = device_c;

    float max_abs_error = 0.0f;
    float max_reference = 0.0f;
    for (int n = 0; n < kN; ++n) {
        for (int m = 0; m < kM; ++m) {
            float reference = 0.0f;
            for (int k = 0; k < kK; ++k) {
                reference += float(host_a[m * kK + k]) * float(host_b[n * kK + k]);
            }
            float result = host_c[m + n * kM];
            max_abs_error = std::max(max_abs_error, std::abs(result - reference));
            max_reference = std::max(max_reference, std::abs(reference));
        }
    }
    float relative_error = max_abs_error / std::max(max_reference, 1.0e-12f);
    std::cout << "[CHECK QK] max_abs_error=" << max_abs_error
              << " relative_to_max=" << relative_error << "\n";
    if (!std::isfinite(max_abs_error) || max_abs_error > 1.0e-2f) {
        std::cerr << "[FAIL] WGMMA QK result does not match the host reference\n";
        return EXIT_FAILURE;
    }

    thrust::host_vector<Input> host_p(kM * kPvK);
    thrust::host_vector<Input> host_v(kPvK * kPvN);
    thrust::host_vector<Output> host_output(kM * kPvN, 0.0f);
    for (int m = 0; m < kM; ++m) {
        for (int k = 0; k < kPvK; ++k) {
            host_p[m * kPvK + k] = Input(float(((m * 5 + k * 3) % 11) - 5) / 16.0f);
        }
    }
    for (int k = 0; k < kPvK; ++k) {
        for (int n = 0; n < kPvN; ++n) {
            host_v[k * kPvN + n] = Input(float(((k * 7 + n * 2) % 15) - 7) / 16.0f);
        }
    }

    thrust::device_vector<Input> device_p = host_p;
    thrust::device_vector<Input> device_v = host_v;
    thrust::device_vector<Output> device_output = host_output;
    launch_pv_rs_smoke(
        device_p.data().get(), device_v.data().get(), device_output.data().get());
    check_cuda(cudaDeviceSynchronize(), "PV RS kernel synchronization");
    host_output = device_output;

    float pv_max_abs_error = 0.0f;
    float pv_max_reference = 0.0f;
    for (int n = 0; n < kPvN; ++n) {
        for (int m = 0; m < kM; ++m) {
            float reference = 0.0f;
            for (int k = 0; k < kPvK; ++k) {
                reference += float(host_p[m * kPvK + k]) * float(host_v[k * kPvN + n]);
            }
            float result = host_output[m + n * kM];
            pv_max_abs_error = std::max(pv_max_abs_error, std::abs(result - reference));
            pv_max_reference = std::max(pv_max_reference, std::abs(reference));
        }
    }
    float pv_relative_error =
        pv_max_abs_error / std::max(pv_max_reference, 1.0e-12f);
    std::cout << "[CHECK PV_RS] max_abs_error=" << pv_max_abs_error
              << " relative_to_max=" << pv_relative_error << "\n";
    if (!std::isfinite(pv_max_abs_error) || pv_max_abs_error > 1.0e-2f) {
        std::cerr << "[FAIL] WGMMA PV RS result does not match the host reference\n";
        return EXIT_FAILURE;
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    for (int i = 0; i < 10; ++i) {
        launch_qk_smoke(device_a.data().get(), device_b.data().get(), device_c.data().get());
    }
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    constexpr int kIterations = 100;
    for (int i = 0; i < kIterations; ++i) {
        launch_qk_smoke(device_a.data().get(), device_b.data().get(), device_c.data().get());
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float qk_elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&qk_elapsed_ms, start, stop), "cudaEventElapsedTime(QK)");

    for (int i = 0; i < 10; ++i) {
        launch_pv_rs_smoke(
            device_p.data().get(), device_v.data().get(), device_output.data().get());
    }
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start PV)");
    for (int i = 0; i < kIterations; ++i) {
        launch_pv_rs_smoke(
            device_p.data().get(), device_v.data().get(), device_output.data().get());
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop PV)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop PV)");
    float pv_elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&pv_elapsed_ms, start, stop), "cudaEventElapsedTime(PV)");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::cout << "[TIMING QK] avg_us=" << qk_elapsed_ms * 1000.0f / kIterations << "\n";
    std::cout << "[TIMING PV_RS] avg_us=" << pv_elapsed_ms * 1000.0f / kIterations << "\n";
    std::cout << "[PASS] BF16 TMA + WGMMA QK/PV_RS smoke tests passed\n";
    return EXIT_SUCCESS;
}
