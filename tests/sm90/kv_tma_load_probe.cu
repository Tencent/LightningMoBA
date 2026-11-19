/***************************************************************************************************
 * LightningMoBA SM90 KV TMA load probe.
 *
 * This is a production-layout bring-up probe for Stage D. It does not replace
 * the attention kernel yet. It verifies that contiguous K/V tiles shaped like
 * the KV-centric K1 path can be loaded by TMA + mbarrier and read back from
 * swizzled shared memory correctly.
 **************************************************************************************************/

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

namespace lightning_moba_sm90_kv_tma_probe {

constexpr int kBlockN = 64;
constexpr int kHeadDim = 128;
constexpr int kStages = 2;
constexpr int kRows = kBlockN * kStages;
constexpr int kHeadsKv = 8;
constexpr int kHeadIdx = 3;
constexpr int kGmemCols = kHeadsKv * kHeadDim;

void check_cuda(cudaError_t error, char const* operation) {
    if (error != cudaSuccess) {
        std::cerr << "[FAIL] " << operation << ": " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

template <class Element, class SmemLayoutKV>
struct KvTmaSharedStorage {
    alignas(128) cute::ArrayEngine<Element, cosize_v<SmemLayoutKV>> k;
    alignas(128) cute::ArrayEngine<Element, cosize_v<SmemLayoutKV>> v;
    uint64_t tma_barrier[kStages];
};

template <class Element, class SmemLayoutKV, class TmaK, class TmaV>
__global__ void kv_tma_load_kernel(
    Element const* k_ptr,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    Element const* v_ptr,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    Element* k_out,
    Element* v_out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    Tensor gK_tensor = tma_k.get_tma_tensor(make_shape(Int<kRows>{}, Int<kGmemCols>{}));
    Tensor gV_tensor = tma_v.get_tma_tensor(make_shape(Int<kRows>{}, Int<kGmemCols>{}));

    Tensor gK = local_tile(gK_tensor, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, Int<kHeadIdx>{}));
    Tensor gV = local_tile(gV_tensor, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, Int<kHeadIdx>{}));

    extern __shared__ char smem_raw[];
    using SharedStorage = KvTmaSharedStorage<Element, SmemLayoutKV>;
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_raw);
    Tensor sK = make_tensor(make_smem_ptr(storage.k.begin()), SmemLayoutKV{});
    Tensor sV = make_tensor(make_smem_ptr(storage.v.begin()), SmemLayoutKV{});

    auto [tKgK, tKsK] = tma_partition(
        tma_k,
        Int<0>{},
        Layout<_1>{},
        group_modes<0, 2>(sK),
        group_modes<0, 2>(gK));
    auto [tVgV, tVsV] = tma_partition(
        tma_v,
        Int<0>{},
        Layout<_1>{},
        group_modes<0, 2>(sV),
        group_modes<0, 2>(gV));

    constexpr int kTransactionBytes =
        2 * kBlockN * kHeadDim * static_cast<int>(sizeof(Element));
    using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;

    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int elected = cute::elect_one_sync();
    if (warp_idx == 0 && elected) {
        for (int stage = 0; stage < kStages; ++stage) {
            ProducerBarrier::init(&storage.tma_barrier[stage], 1);
        }
    }
    cluster_sync();

    if (warp_idx == 0 && elected) {
        for (int stage = 0; stage < kStages; ++stage) {
            ProducerBarrier::arrive_and_expect_tx(&storage.tma_barrier[stage], kTransactionBytes);
            copy(tma_k.with(storage.tma_barrier[stage]), tKgK(_, stage), tKsK(_, stage));
            copy(tma_v.with(storage.tma_barrier[stage]), tVgV(_, stage), tVsV(_, stage));
        }
    }

    auto read_state = cutlass::PipelineState<kStages>();
    for (int stage_iter = 0; stage_iter < kStages; ++stage_iter) {
        const int stage = read_state.index();
        ProducerBarrier::wait(&storage.tma_barrier[stage], read_state.phase());
        __syncthreads();

        for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
            const int row = idx / kHeadDim;
            const int col = idx - row * kHeadDim;
            const int global_row = stage * kBlockN + row;
            k_out[global_row * kHeadDim + col] = sK(row, col, stage);
            v_out[global_row * kHeadDim + col] = sV(row, col, stage);
        }
        __syncthreads();
        ++read_state;
    }
#else
    (void)k_ptr; (void)tma_k; (void)v_ptr; (void)tma_v; (void)k_out; (void)v_out;
#endif
}

template <class Element>
void launch_probe(Element const* k_ptr, Element const* v_ptr, Element* k_out, Element* v_out) {
    using SmemLayoutKVStage = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<Element>{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

    auto gmem_layout = make_layout(
        make_shape(Int<kRows>{}, Int<kGmemCols>{}),
        make_stride(Int<kGmemCols>{}, Int<1>{}));
    Tensor k_tensor = make_tensor(k_ptr, gmem_layout);
    Tensor v_tensor = make_tensor(v_ptr, gmem_layout);
    auto smem_layout = SmemLayoutKVStage{};
    auto tma_k = make_tma_atom(
        SM90_TMA_LOAD{}, k_tensor, smem_layout(_, _, 0),
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
    auto tma_v = make_tma_atom(
        SM90_TMA_LOAD{}, v_tensor, smem_layout(_, _, 0),
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));

    using Kernel = decltype(&kv_tma_load_kernel<Element, SmemLayoutKVStage, decltype(tma_k), decltype(tma_v)>);
    Kernel kernel = &kv_tma_load_kernel<Element, SmemLayoutKVStage, decltype(tma_k), decltype(tma_v)>;
    using SharedStorage = KvTmaSharedStorage<Element, SmemLayoutKVStage>;
    const int smem_bytes = sizeof(SharedStorage);
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
               "cudaFuncSetAttribute");

    cutlass::ClusterLaunchParams launch_params{
        dim3(1, 1, 1), dim3(128, 1, 1), dim3(1, 1, 1), smem_bytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        launch_params,
        reinterpret_cast<void const*>(kernel),
        k_ptr,
        tma_k,
        v_ptr,
        tma_v,
        k_out,
        v_out);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[FAIL] cluster launch returned status " << int(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

void run_probe() {
    using Element = cutlass::bfloat16_t;
    constexpr int kGmemElems = kRows * kGmemCols;
    constexpr int kOutElems = kRows * kHeadDim;

    thrust::host_vector<Element> h_k(kGmemElems), h_v(kGmemElems);
    for (int i = 0; i < kGmemElems; ++i) {
        h_k[i] = Element(float((i % 251) - 125) / 32.0f);
        h_v[i] = Element(float((i % 193) - 96) / 16.0f);
    }

    thrust::device_vector<Element> d_k = h_k;
    thrust::device_vector<Element> d_v = h_v;
    thrust::device_vector<Element> d_k_out(kOutElems);
    thrust::device_vector<Element> d_v_out(kOutElems);

    launch_probe(thrust::raw_pointer_cast(d_k.data()),
                 thrust::raw_pointer_cast(d_v.data()),
                 thrust::raw_pointer_cast(d_k_out.data()),
                 thrust::raw_pointer_cast(d_v_out.data()));
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    thrust::host_vector<Element> h_k_out = d_k_out;
    thrust::host_vector<Element> h_v_out = d_v_out;
    float max_k_err = 0.0f;
    float max_v_err = 0.0f;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kHeadDim; ++col) {
            const int src_idx = row * kGmemCols + kHeadIdx * kHeadDim + col;
            const int out_idx = row * kHeadDim + col;
            max_k_err = std::max(max_k_err, std::abs(float(h_k[src_idx]) - float(h_k_out[out_idx])));
            max_v_err = std::max(max_v_err, std::abs(float(h_v[src_idx]) - float(h_v_out[out_idx])));
        }
    }
    std::cout << "[CHECK K] max_abs_error=" << max_k_err << "\n";
    std::cout << "[CHECK V] max_abs_error=" << max_v_err << "\n";
    if (max_k_err != 0.0f || max_v_err != 0.0f) {
        std::cerr << "[FAIL] KV TMA load probe mismatch\n";
        std::exit(EXIT_FAILURE);
    }
    std::cout << "[PASS] KV TMA load probe passed\n";
}

}  // namespace lightning_moba_sm90_kv_tma_probe

int main() {
    int device = 0;
    lightning_moba_sm90_kv_tma_probe::check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    lightning_moba_sm90_kv_tma_probe::check_cuda(
        cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::cout << "[GPU] " << prop.name << " capability=" << prop.major << "." << prop.minor << "\n";
    if (prop.major < 9) {
        std::cerr << "[SKIP] SM90+ GPU required\n";
        return 0;
    }
    lightning_moba_sm90_kv_tma_probe::run_probe();
    return 0;
}
