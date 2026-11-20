/******************************************************************************
 * Production SM90 KV-centric CSC attention pipeline.
 ******************************************************************************/

#include "sm90_kv_centric_pipeline.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <climits>

#include "flash_sm90_traits.h"
#include "index_gather_scatter.h"
#include "utils.h"

namespace LIGHTNING_MOBA_NAMESPACE {
using namespace cute;
namespace {

constexpr int kThreads = 160;
constexpr int kConsumerThreads = 128;
constexpr int kProducerBegin = 128;
constexpr int kHeadDim = 128;
constexpr int kTileRows = 64;
constexpr int kPartialSpillDims = 32;
constexpr int kWorkRecordInts = 5;
using Element = cutlass::bfloat16_t;
using Traits = Sm90WgmmaQk64Traits<Element>;
using QkMma = typename Traits::TiledMma;
using PvMma = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}));
using PvMma64 = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}));
using SmemLayoutP = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<64>, Int<64>>{}));
using SmemLayoutV = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<128>, Int<64>>{}));

template <int Stages>
using Pipeline = cutlass::PipelineAsync<Stages>;

template <bool Enabled>
struct PartialPvScratch {};

template <>
struct alignas(128) PartialPvScratch<true> {
    alignas(128) float pv0[kTileRows * kPartialSpillDims];
};

template <int Stages, bool PartialSpill>
struct alignas(128) SharedStorage : PartialPvScratch<PartialSpill> {
    typename Pipeline<Stages>::SharedStorage pipeline;
    alignas(128) cute::ArrayEngine<Element, cosize_v<typename Traits::SmemLayoutQ>> q[Stages];
    alignas(128) int q_positions[Stages][kTileRows];
    alignas(128) int valid_cols[Stages][kTileRows][2];
    alignas(128) uint8_t kv_tma_overlay[
        sizeof(cute::ArrayEngine<Element, cosize_v<typename Traits::SmemLayoutK>>)];
    alignas(128) cute::ArrayEngine<Element, cosize_v<typename Traits::SmemLayoutK>> k1;
    alignas(128) cute::ArrayEngine<Element, cosize_v<SmemLayoutV>> v0;
    alignas(128) cute::ArrayEngine<Element, cosize_v<SmemLayoutV>> v1;
    alignas(64) cute::TmaDescriptor tma_descriptor_k;
    alignas(64) cute::TmaDescriptor tma_descriptor_v;
    alignas(128) uint64_t tma_barrier[2];
    alignas(128) float online_max[kTileRows];
    alignas(128) float online_sum[kTileRows];
    alignas(128) float prior_scale[kTileRows];
    alignas(128) float subtile_scale[kTileRows];
};

__device__ __forceinline__ int clamp_int(int value, int low, int high) {
    return max(low, min(value, high));
}

// Real-CSC profiling established 4096 rows as the best production split:
// it removes hot-column tail waves while remaining neutral on ordinary CSCs.
constexpr int kQSplitTarget = 4096;

__global__ void build_qsplit_worklist_kernel(
    const int* __restrict__ col_nnz,
    int* __restrict__ worklist,
    int* __restrict__ work_count,
    int64_t column_count,
    int num_q_heads,
    int max_columns,
    int split_target,
    int work_capacity) {
    const int64_t column_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column_index >= column_count) return;
    const int count = col_nnz[column_index];
    if (count <= 0) return;

    const int chunks = 1 + (count - 1) / split_target;
    const int begin = atomicAdd(work_count, chunks);
    const int column = column_index % max_columns;
    const int q_head = (column_index / max_columns) % num_q_heads;
    const int batch =
        column_index / (static_cast<int64_t>(max_columns) * num_q_heads);
    for (int split = 0; split < chunks; ++split) {
        const int work_index = begin + split;
        if (work_index >= work_capacity) return;
        int* record =
            worklist + static_cast<int64_t>(work_index) * kWorkRecordInts;
        const int q_begin = split * split_target;
        record[0] = batch;
        record[1] = q_head;
        record[2] = column;
        record[3] = q_begin;
        // Publish row_count last. Padded worklist records remain zero and are
        // rejected before the expensive pipeline initialization.
        record[4] = min(split_target, count - q_begin);
    }
}

template <
    bool BatchWgmma,
    bool RegisterState,
    bool WeightedPv,
    bool CompensatedPv,
    bool SplitPv,
    bool PartialSpill,
    typename TmaK,
    typename TmaV>
__global__ __launch_bounds__(kThreads, 2)
void phase4b_csc_pipe_kernel(
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    cute::TmaDescriptor* __restrict__ dynamic_tma_descriptors,
    const int64_t* __restrict__ col_offsets,
    const int* __restrict__ col_nnz,
    const int* __restrict__ q_rows,
    const int* __restrict__ worklist,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ cu_seqlens_k,
    float* __restrict__ output,
    float* __restrict__ lse,
    int num_q_heads,
    int num_kv_heads,
    int max_columns,
    float softmax_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    using Pipe = Pipeline<2>;
    using Storage = SharedStorage<2, PartialSpill>;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<Storage*>(shared_bytes);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_idx = tid >> 5;
    const bool is_producer = tid >= kProducerBegin;
    const int work_index = blockIdx.x;
    __shared__ int work_is_valid;
    if (tid == 0) {
        work_is_valid =
            worklist == nullptr ||
            worklist[static_cast<int64_t>(work_index) * kWorkRecordInts + 4] >
                0;
    }
    __syncthreads();
    if (!work_is_valid) {
        return;
    }

    // B2': retain mode56's native D64-stage cp.async contract, but repartition
    // it across the dedicated 32-thread producer warp.  Each copy atom moves
    // eight BF16 values (16 bytes) directly into the WGMMA-compatible swizzled
    // sQ layout, with no intermediate shared-memory materialization.
    using GmemCopyAtomQ = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
    using ProducerTiledCopyQ = decltype(make_tiled_copy(
        GmemCopyAtomQ{},
        Layout<Shape<Int<4>, Int<8>>, Stride<Int<8>, Int<1>>>{},
        Layout<Shape<_1, _8>>{}));
    ProducerTiledCopyQ producer_tiled_copy_q;
    auto producer_thr_copy_q = producer_tiled_copy_q.get_thread_slice(lane);
    Tensor cQ = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    Tensor tQcQ = producer_thr_copy_q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQcQ)));
    #pragma unroll
    for (int copy_k = 0; copy_k < size(tQpQ); ++copy_k) {
        tQpQ(copy_k) = true;
    }

    typename Pipe::Params params;
    params.role = is_producer ? Pipe::ThreadCategory::Producer : Pipe::ThreadCategory::Consumer;
    params.producer_arv_count = 1;
    params.consumer_arv_count = 1;
    params.initializing_warp = 0;
    Pipe pipeline(storage.pipeline, params);
    __syncthreads();

    int batch;
    int q_head;
    int column;
    int q_range_begin;
    int row_count;
    int64_t column_index;
    if (worklist != nullptr) {
        const int* record =
            worklist + static_cast<int64_t>(work_index) * kWorkRecordInts;
        batch = record[0];
        q_head = record[1];
        column = record[2];
        q_range_begin = record[3];
        row_count = record[4];
        column_index =
            (static_cast<int64_t>(batch) * num_q_heads + q_head) *
                max_columns +
            column;
    } else {
        column_index = work_index;
        column = column_index % max_columns;
        q_head = (column_index / max_columns) % num_q_heads;
        batch =
            column_index /
            (static_cast<int64_t>(max_columns) * num_q_heads);
        q_range_begin = 0;
        row_count = col_nnz[column_index];
    }
    if (row_count <= 0) return;
    const int64_t csc_offset = col_offsets[column_index];
    const int q_base = cu_seqlens_q[batch];
    const int q_length = cu_seqlens_q[batch + 1] - q_base;
    const int k_base = cu_seqlens_k[batch];
    const int k_length = cu_seqlens_k[batch + 1] - k_base;
    const int k_start = column * 128;
    if (k_start >= k_length) return;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    Element* overlay_ptr = reinterpret_cast<Element*>(storage.kv_tma_overlay);
    Tensor sK0 = make_tensor(make_smem_ptr(overlay_ptr), typename Traits::SmemLayoutK{});
    Tensor sVTma = make_tensor(make_smem_ptr(overlay_ptr), typename Traits::SmemLayoutK{});
    Tensor sK1 = make_tensor(make_smem_ptr(storage.k1.begin()), typename Traits::SmemLayoutK{});
    Tensor sV0 = make_tensor(make_smem_ptr(storage.v0.begin()), SmemLayoutV{});
    Tensor sV1 = make_tensor(make_smem_ptr(storage.v1.begin()), SmemLayoutV{});
    Tensor mK = tma_k.get_tma_tensor(make_shape(Int<64>{}, Int<128>{}));
    Tensor mV = tma_v.get_tma_tensor(make_shape(Int<64>{}, Int<128>{}));
    auto kv_tiler = make_shape(Int<64>{}, Int<64>{}, Int<64>{});
    Tensor gK = local_tile(mK, kv_tiler, make_coord(0, 0, _), Step<X, _1, _1>{});
    Tensor gV = local_tile(mV, kv_tiler, make_coord(0, 0, _), Step<X, _1, _1>{});
    auto [tKgK, tKsK] = tma_partition(tma_k, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sK0), group_modes<0, 2>(gK));
    auto [tK1gK, tK1sK] = tma_partition(tma_k, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sK1), group_modes<0, 2>(gK));
    auto [tVgV, tVsV] = tma_partition(tma_v, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sVTma), group_modes<0, 2>(gV));
    constexpr int k_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tKsK)));
    constexpr int v_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tVsV)));
    cute::TmaDescriptor* tma_desc_k =
        &dynamic_tma_descriptors[2 * work_index];
    cute::TmaDescriptor* tma_desc_v =
        &dynamic_tma_descriptors[2 * work_index + 1];
    const int elected = cute::elect_one_sync();

        // First materialize both V subtiles while the original K/V carrier is
        // reusable.  Only after V0/V1 are transposed do we reuse the carrier as
        // resident K0; K1 receives its dedicated resident buffer.
        #pragma unroll
        for (int subtile = 0; subtile < 2; ++subtile) {
            const int subtile_k_start = k_start + subtile * 64;
            const int subtile_valid = subtile_k_start < k_length
                ? min(64, k_length - subtile_k_start) : 0;
            if (subtile_valid == 64) {
                if (warp_idx == 0 && elected) {
                    Tensor pV = make_tensor(tma_v.get_tma_descriptor(), Int<1>{}, Int<1>{});
                    Tensor sVDesc = make_tensor(make_smem_ptr(&storage.tma_descriptor_v), Int<1>{}, Int<1>{});
                    copy(recast<uint128_t>(pV), recast<uint128_t>(sVDesc));
                }
                __syncthreads();
                if (warp_idx == 0 && elected) {
                    const int64_t packed_offset =
                        (static_cast<int64_t>(k_base + subtile_k_start) * num_kv_heads + kv_head) * 128;
                    cute::tma_descriptor_replace_addr_in_shared_mem(storage.tma_descriptor_v, v + packed_offset);
                    cute::tma_desc_commit_group();
                    cute::tma_desc_wait_group();
                }
                __syncthreads();
                cute::tma_descriptor_cp_fence_release(tma_desc_v, storage.tma_descriptor_v);
                cute::tma_descriptor_fence_acquire(tma_desc_v);
                if (warp_idx == 0 && elected) {
                    #pragma unroll
                    for (int stage = 0; stage < 2; ++stage) {
                        cutlass::arch::ClusterTransactionBarrier::init(&storage.tma_barrier[stage], 1);
                    }
                }
                cutlass::arch::fence_barrier_init();
                cluster_sync();
                if (warp_idx == 0 && elected) {
                    #pragma unroll
                    for (int stage = 0; stage < 2; ++stage) {
                        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                            &storage.tma_barrier[stage], v_transaction_bytes);
                        copy(tma_v.with(tma_desc_v, storage.tma_barrier[stage]),
                             tVgV(_, stage), tVsV(_, stage));
                    }
                }
                #pragma unroll
                for (int stage = 0; stage < 2; ++stage) {
                    cutlass::arch::ClusterTransactionBarrier::wait(&storage.tma_barrier[stage], 0);
                }
                for (int index = threadIdx.x; index < 64 * 128; index += blockDim.x) {
                    const int row = index / 128;
                    const int dim = index % 128;
                    if (subtile == 0) {
                        sV0(dim, row) = sVTma(row, dim % 64, dim / 64);
                    } else {
                        sV1(dim, row) = sVTma(row, dim % 64, dim / 64);
                    }
                }
            } else {
                // Scalar tail load, including explicit zero fill for an empty
                // second subtile.  The consumer always executes two fixed
                // groups, so both operands must be initialized.
                for (int index = threadIdx.x; index < 64 * 128; index += blockDim.x) {
                    const int row = index / 128;
                    const int dim = index % 128;
                    const bool valid = row < subtile_valid;
                    const int64_t source =
                        (static_cast<int64_t>(k_base + subtile_k_start + row) * num_kv_heads + kv_head) * 128 + dim;
                    if (subtile == 0) {
                        sV0(dim, row) = valid ? v[source] : Element(0);
                    } else {
                        sV1(dim, row) = valid ? v[source] : Element(0);
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int subtile = 0; subtile < 2; ++subtile) {
            const int subtile_k_start = k_start + subtile * 64;
            const int subtile_valid = subtile_k_start < k_length
                ? min(64, k_length - subtile_k_start) : 0;
            if (subtile_valid == 64) {
                if (warp_idx == 0 && elected) {
                    Tensor pK = make_tensor(tma_k.get_tma_descriptor(), Int<1>{}, Int<1>{});
                    Tensor sKDesc = make_tensor(make_smem_ptr(&storage.tma_descriptor_k), Int<1>{}, Int<1>{});
                    copy(recast<uint128_t>(pK), recast<uint128_t>(sKDesc));
                }
                __syncthreads();
                if (warp_idx == 0 && elected) {
                    const int64_t packed_offset =
                        (static_cast<int64_t>(k_base + subtile_k_start) * num_kv_heads + kv_head) * 128;
                    cute::tma_descriptor_replace_addr_in_shared_mem(storage.tma_descriptor_k, k + packed_offset);
                    cute::tma_desc_commit_group();
                    cute::tma_desc_wait_group();
                }
                __syncthreads();
                cute::tma_descriptor_cp_fence_release(tma_desc_k, storage.tma_descriptor_k);
                cute::tma_descriptor_fence_acquire(tma_desc_k);
                if (warp_idx == 0 && elected) {
                    #pragma unroll
                    for (int stage = 0; stage < 2; ++stage) {
                        cutlass::arch::ClusterTransactionBarrier::init(&storage.tma_barrier[stage], 1);
                    }
                }
                cutlass::arch::fence_barrier_init();
                cluster_sync();
                if (warp_idx == 0 && elected) {
                    #pragma unroll
                    for (int stage = 0; stage < 2; ++stage) {
                        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                            &storage.tma_barrier[stage], k_transaction_bytes);
                        if (subtile == 0) {
                            copy(tma_k.with(tma_desc_k, storage.tma_barrier[stage]),
                                 tKgK(_, stage), tKsK(_, stage));
                        } else {
                            copy(tma_k.with(tma_desc_k, storage.tma_barrier[stage]),
                                 tK1gK(_, stage), tK1sK(_, stage));
                        }
                    }
                }
                #pragma unroll
                for (int stage = 0; stage < 2; ++stage) {
                    cutlass::arch::ClusterTransactionBarrier::wait(&storage.tma_barrier[stage], 0);
                }
            } else {
                // Scalar tail load, including explicit zero fill for an empty
                // second subtile.
                for (int index = threadIdx.x; index < 64 * 128; index += blockDim.x) {
                    const int row = index / 128;
                    const int dim = index % 128;
                    const bool valid = row < subtile_valid;
                    const int64_t source =
                        (static_cast<int64_t>(k_base + subtile_k_start + row) * num_kv_heads + kv_head) * 128 + dim;
                    if (subtile == 0) {
                        sK0(row, dim % 64, dim / 64) = valid ? k[source] : Element(0);
                    } else {
                        sK1(row, dim % 64, dim / 64) = valid ? k[source] : Element(0);
                    }
                }
            }
            __syncthreads();
        }
        cutlass::arch::fence_view_async_shared();
    const int tile_count = (row_count + kTileRows - 1) / kTileRows;
    if (is_producer) {
        auto state = cutlass::make_producer_start_state<Pipe>();
        for (int tile = 0; tile < tile_count; ++tile, ++state) {
            if (lane == 0) {
                auto token = pipeline.producer_try_acquire(state);
                pipeline.producer_acquire(state, token);
            }
            __syncwarp();
            const int stage = state.index();
            Tensor sQ = make_tensor(make_smem_ptr(storage.q[stage].begin()),
                                    typename Traits::SmemLayoutQ{});
            for (int row = lane; row < kTileRows; row += 32) {
                const int index = tile * kTileRows + row;
                const int q_position = index < row_count
                    ? q_rows[csc_offset + q_range_begin + index] : -1;
                const bool valid = q_position >= 0 && q_position < q_length;
                const int causal_end = valid ? q_position + k_length - q_length - k_start : -1;
                storage.q_positions[stage][row] = valid ? q_position : 0;
                storage.valid_cols[stage][row][0] = valid
                    ? clamp_int(causal_end + 1, 0, clamp_int(k_length - k_start, 0, 64)) : 0;
                storage.valid_cols[stage][row][1] = valid
                    ? clamp_int(causal_end - 63, 0, clamp_int(k_length - k_start - 64, 0, 64)) : 0;
            }
            __syncwarp();
            const int q_valid = min(kTileRows, row_count - tile * kTileRows);
            const Element* q_head_ptr =
                q + (static_cast<int64_t>(q_base) * num_q_heads + q_head) * kHeadDim;
            #pragma unroll
            for (int q_stage = 0; q_stage < 2; ++q_stage) {
                Tensor sQ_stage = sQ(_, _, q_stage);
                Tensor tQsQ_stage = producer_thr_copy_q.partition_D(sQ_stage);
                Tensor gQ_stage = make_gather_tensor(
                    make_gmem_ptr(q_head_ptr + q_stage * 64),
                    make_shape(Int<64>{}, Int<64>{}),
                    make_stride(num_q_heads * kHeadDim, Int<1>{}),
                    IndexedGather(storage.q_positions[stage]));
                Tensor tQgQ_stage = producer_thr_copy_q.partition_S(gQ_stage);
                LIGHTNING_MOBA_NAMESPACE::copy<false, true, true>(
                    producer_tiled_copy_q, tQgQ_stage, tQsQ_stage, tQcQ, tQpQ, q_valid);
            }
            cute::cp_async_fence();
            LIGHTNING_MOBA_NAMESPACE::cp_async_wait<0>();
            __syncwarp();
            if (lane == 0) {
                cutlass::arch::fence_view_async_shared();
                pipeline.producer_commit(state);
            }
        }
        if (lane == 0) pipeline.producer_tail(state);
    } else {
        QkMma qk_mma;
        auto qk_thr = qk_mma.get_thread_slice(tid);
        Tensor tSrK0 = qk_thr.make_fragment_B(qk_thr.partition_B(sK0));
        Tensor tSrK1 = qk_thr.make_fragment_B(qk_thr.partition_B(sK1));
        Tensor score_coords = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
        Tensor tCcScores = qk_thr.partition_C(score_coords);

        PvMma pv_mma;
        auto pv_thr = pv_mma.get_thread_slice(tid);
        Tensor sPCarrier = make_tensor(make_smem_ptr(storage.v0.begin()), SmemLayoutP{});
        Tensor tSrP = pv_thr.make_fragment_A(pv_thr.partition_A(sPCarrier));
        Tensor tSrPResidual =
            pv_thr.make_fragment_A(pv_thr.partition_A(sPCarrier));
        Tensor tSrV0 = pv_thr.make_fragment_B(pv_thr.partition_B(sV0));
        Tensor tSrV1 = pv_thr.make_fragment_B(pv_thr.partition_B(sV1));
        Tensor output_coords = make_identity_tensor(make_shape(Int<64>{}, Int<128>{}));
        Tensor tCcOutput = pv_thr.partition_C(output_coords);
        static_assert(size(tSrP) == size(tCcScores));

        PvMma64 pv64_mma;
        auto pv64_thr = pv64_mma.get_thread_slice(tid);
        Tensor tSrP64 = pv64_thr.make_fragment_A(pv64_thr.partition_A(sPCarrier));
        Tensor output64_coords =
            make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
        Tensor tCcOutput64 = pv64_thr.partition_C(output64_coords);

        typename Pipe::PipelineState state;
        for (int tile = 0; tile < tile_count; ++tile, ++state) {
            auto token = pipeline.consumer_try_wait(state);
            pipeline.consumer_wait(state, token);
            const int stage = state.index();
            Tensor sQ = make_tensor(make_smem_ptr(storage.q[stage].begin()),
                                    typename Traits::SmemLayoutQ{});
            Tensor tSrQ = qk_thr.make_fragment_A(qk_thr.partition_A(sQ));
            auto online_acc = pv_thr.make_fragment_C(tCcOutput);
            if constexpr (!SplitPv) {
                clear(online_acc);
            }
            float online_max0 = -INFINITY;
            float online_max1 = -INFINITY;
            float online_sum0 = 0.0f;
            float online_sum1 = 0.0f;
            const int row0 = warp_idx * 16 + lane / 4;
            const int row1 = row0 + 8;

            #pragma unroll
            for (int subtile = 0; subtile < 2; ++subtile) {
                auto qk_acc = qk_thr.make_fragment_C(tCcScores);
                clear(qk_acc);
                if constexpr (BatchWgmma) {
                    warpgroup_fence_operand(qk_acc);
                    warpgroup_arrive();
                    #pragma unroll
                    for (int reduction = 0; reduction < 2; ++reduction) {
                        if (subtile == 0) {
                            gemm(qk_mma, tSrQ(_, _, _, reduction),
                                 tSrK0(_, _, _, reduction), qk_acc);
                        } else {
                            gemm(qk_mma, tSrQ(_, _, _, reduction),
                                 tSrK1(_, _, _, reduction), qk_acc);
                        }
                    }
                    warpgroup_commit_batch();
                    warpgroup_wait<0>();
                    warpgroup_fence_operand(qk_acc);
                } else {
                    #pragma unroll
                    for (int reduction = 0; reduction < 2; ++reduction) {
                        warpgroup_fence_operand(qk_acc);
                        warpgroup_arrive();
                        if (subtile == 0) {
                            gemm(qk_mma, tSrQ(_, _, _, reduction),
                                 tSrK0(_, _, _, reduction), qk_acc);
                        } else {
                            gemm(qk_mma, tSrQ(_, _, _, reduction),
                                 tSrK1(_, _, _, reduction), qk_acc);
                        }
                        warpgroup_commit_batch();
                        warpgroup_wait<0>();
                        warpgroup_fence_operand(qk_acc);
                    }
                }

                const int valid_cols0 = storage.valid_cols[stage][row0][subtile];
                const int valid_cols1 = storage.valid_cols[stage][row1][subtile];
                float m0 = -INFINITY;
                float m1 = -INFINITY;
                #pragma unroll
                for (int i = 0; i < size(qk_acc); ++i) {
                    const auto coord = tCcScores(i);
                    const int row = get<0>(coord);
                    const int col = get<1>(coord);
                    const int valid_cols = row == row0 ? valid_cols0 : valid_cols1;
                    const float score = col < valid_cols ? qk_acc(i) * softmax_scale : -INFINITY;
                    if (row == row0) m0 = fmaxf(m0, score);
                    if (row == row1) m1 = fmaxf(m1, score);
                }
                #pragma unroll
                for (int offset = 2; offset > 0; offset >>= 1) {
                    m0 = fmaxf(m0, __shfl_down_sync(0xffffffff, m0, offset, 4));
                    m1 = fmaxf(m1, __shfl_down_sync(0xffffffff, m1, offset, 4));
                }
                const int leader = lane & ~3;
                m0 = __shfl_sync(0xffffffff, m0, leader);
                m1 = __shfl_sync(0xffffffff, m1, leader);
                float z0 = 0.0f;
                float z1 = 0.0f;
                #pragma unroll
                for (int i = 0; i < size(qk_acc); ++i) {
                    const auto coord = tCcScores(i);
                    const int row = get<0>(coord);
                    const int col = get<1>(coord);
                    const int valid_cols = row == row0 ? valid_cols0 : valid_cols1;
                    const float maximum = row == row0 ? m0 : m1;
                    const float probability = col < valid_cols && maximum != -INFINITY
                        ? expf(qk_acc(i) * softmax_scale - maximum) : 0.0f;
                    tSrP(i) = Element(probability);
                    if (row == row0) z0 += probability;
                    if (row == row1) z1 += probability;
                }
                #pragma unroll
                for (int offset = 2; offset > 0; offset >>= 1) {
                    z0 += __shfl_down_sync(0xffffffff, z0, offset, 4);
                    z1 += __shfl_down_sync(0xffffffff, z1, offset, 4);
                }
                z0 = __shfl_sync(0xffffffff, z0, leader);
                z1 = __shfl_sync(0xffffffff, z1, leader);

                float prior_scale0 = 0.0f;
                float prior_scale1 = 0.0f;
                float subtile_scale0 = z0 > 0.0f ? 1.0f : 0.0f;
                float subtile_scale1 = z1 > 0.0f ? 1.0f : 0.0f;
                if constexpr (RegisterState) {
                    if (subtile == 0) {
                        online_max0 = m0;
                        online_max1 = m1;
                        online_sum0 = z0;
                        online_sum1 = z1;
                    } else {
                        const float merged_max0 = fmaxf(online_max0, m0);
                        const float merged_max1 = fmaxf(online_max1, m1);
                        prior_scale0 = online_sum0 > 0.0f
                            ? expf(online_max0 - merged_max0) : 0.0f;
                        prior_scale1 = online_sum1 > 0.0f
                            ? expf(online_max1 - merged_max1) : 0.0f;
                        subtile_scale0 = z0 > 0.0f
                            ? expf(m0 - merged_max0) : 0.0f;
                        subtile_scale1 = z1 > 0.0f
                            ? expf(m1 - merged_max1) : 0.0f;
                        online_max0 = merged_max0;
                        online_max1 = merged_max1;
                        online_sum0 =
                            prior_scale0 * online_sum0 + subtile_scale0 * z0;
                        online_sum1 =
                            prior_scale1 * online_sum1 + subtile_scale1 * z1;
                    }
                } else {
                    if ((lane & 3) == 0) {
                        #pragma unroll
                        for (int pair = 0; pair < 2; ++pair) {
                            const int row = pair == 0 ? row0 : row1;
                            const float subtile_max = pair == 0 ? m0 : m1;
                            const float subtile_sum = pair == 0 ? z0 : z1;
                            if (subtile == 0) {
                                storage.prior_scale[row] = 0.0f;
                                storage.subtile_scale[row] =
                                    subtile_sum > 0.0f ? 1.0f : 0.0f;
                                storage.online_max[row] = subtile_max;
                                storage.online_sum[row] = subtile_sum;
                            } else {
                                const float prior_max = storage.online_max[row];
                                const float prior_sum = storage.online_sum[row];
                                const float merged_max = fmaxf(prior_max, subtile_max);
                                const float prior_scale = prior_sum > 0.0f
                                    ? expf(prior_max - merged_max) : 0.0f;
                                const float subtile_scale = subtile_sum > 0.0f
                                    ? expf(subtile_max - merged_max) : 0.0f;
                                storage.prior_scale[row] = prior_scale;
                                storage.subtile_scale[row] = subtile_scale;
                                storage.online_max[row] = merged_max;
                                storage.online_sum[row] =
                                    prior_scale * prior_sum + subtile_scale * subtile_sum;
                            }
                        }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }
                // Once the second subtile has materialized P in registers,
                // neither shared Q nor this stage's valid-column metadata is
                // referenced again. Release before PV/online merge and the
                // epilogue so the producer can gather the next Q tile.
                if (subtile == 1 && tid == 0) {
                    pipeline.consumer_release(state);
                }

                if constexpr (SplitPv) {
                    // QK no longer references K0/K1 after its corresponding
                    // subtile. Reuse those two 8 KiB buffers as row-major
                    // BF16 P0/P1 scratch without increasing CTA shared memory.
                    Element* p_scratch = subtile == 0
                        ? overlay_ptr : storage.k1.begin();
                    #pragma unroll
                    for (int i = 0; i < size(tSrP); ++i) {
                        const auto coord = tCcScores(i);
                        const int row = get<0>(coord);
                        const int col = get<1>(coord);
                        p_scratch[row * 64 + col] = tSrP(i);
                    }
                } else if constexpr (WeightedPv) {
                    if (subtile == 1) {
                        #pragma unroll
                        for (int i = 0; i < size(tSrP); ++i) {
                            const int row = get<0>(tCcScores(i));
                            const float scale = RegisterState
                                ? (row == row0 ? subtile_scale0 : subtile_scale1)
                                : storage.subtile_scale[row];
                            const float scaled = float(tSrP(i)) * scale;
                            const Element high = Element(scaled);
                            tSrP(i) = high;
                            if constexpr (CompensatedPv) {
                                tSrPResidual(i) =
                                    Element(scaled - float(high));
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < size(online_acc); ++i) {
                            const int row = get<0>(tCcOutput(i));
                            const float scale = RegisterState
                                ? (row == row0 ? prior_scale0 : prior_scale1)
                                : storage.prior_scale[row];
                            online_acc(i) *= scale;
                        }
                    }
                    #pragma unroll
                    for (int reduction = 0; reduction < 4; ++reduction) {
                        warpgroup_fence_operand(tSrP);
                        warpgroup_fence_operand(online_acc);
                        warpgroup_arrive();
                        if (subtile == 0) {
                            gemm(pv_mma, tSrP(_, _, reduction),
                                 tSrV0(_, _, reduction), online_acc);
                        } else {
                            gemm(pv_mma, tSrP(_, _, reduction),
                                 tSrV1(_, _, reduction), online_acc);
                        }
                        warpgroup_commit_batch();
                        warpgroup_wait<0>();
                        warpgroup_fence_operand(online_acc);
                    }
                    if constexpr (CompensatedPv) {
                        if (subtile == 1) {
                            #pragma unroll
                            for (int reduction = 0; reduction < 4; ++reduction) {
                                warpgroup_fence_operand(tSrPResidual);
                                warpgroup_fence_operand(online_acc);
                                warpgroup_arrive();
                                gemm(
                                    pv_mma,
                                    tSrPResidual(_, _, reduction),
                                    tSrV1(_, _, reduction),
                                    online_acc);
                                warpgroup_commit_batch();
                                warpgroup_wait<0>();
                                warpgroup_fence_operand(online_acc);
                            }
                        }
                    }
                } else {
                    auto pv_acc = pv_thr.make_fragment_C(tCcOutput);
                    clear(pv_acc);
                    if constexpr (BatchWgmma) {
                        warpgroup_fence_operand(tSrP);
                        warpgroup_fence_operand(pv_acc);
                        warpgroup_arrive();
                        #pragma unroll
                        for (int reduction = 0; reduction < 4; ++reduction) {
                            if (subtile == 0) {
                                gemm(pv_mma, tSrP(_, _, reduction),
                                     tSrV0(_, _, reduction), pv_acc);
                            } else {
                                gemm(pv_mma, tSrP(_, _, reduction),
                                     tSrV1(_, _, reduction), pv_acc);
                            }
                        }
                        warpgroup_commit_batch();
                        warpgroup_wait<0>();
                        warpgroup_fence_operand(pv_acc);
                    } else {
                        #pragma unroll
                        for (int reduction = 0; reduction < 4; ++reduction) {
                            warpgroup_fence_operand(tSrP);
                            warpgroup_fence_operand(pv_acc);
                            warpgroup_arrive();
                            if (subtile == 0) {
                                gemm(pv_mma, tSrP(_, _, reduction),
                                     tSrV0(_, _, reduction), pv_acc);
                            } else {
                                gemm(pv_mma, tSrP(_, _, reduction),
                                     tSrV1(_, _, reduction), pv_acc);
                            }
                            warpgroup_commit_batch();
                            warpgroup_wait<0>();
                            warpgroup_fence_operand(pv_acc);
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < size(pv_acc); ++i) {
                        const auto coord = tCcOutput(i);
                        const int row = get<0>(coord);
                        if constexpr (RegisterState) {
                            const float prior_scale =
                                row == row0 ? prior_scale0 : prior_scale1;
                            const float subtile_scale =
                                row == row0 ? subtile_scale0 : subtile_scale1;
                            online_acc(i) = subtile == 0 ? pv_acc(i) :
                                prior_scale * online_acc(i) +
                                subtile_scale * pv_acc(i);
                        } else {
                            if constexpr (PartialSpill) {
                                const int dim = get<1>(coord);
                                if (subtile == 0) {
                                    if (dim < kPartialSpillDims) {
                                        storage.pv0[
                                            row * kPartialSpillDims + dim] =
                                            pv_acc(i);
                                    } else {
                                        online_acc(i) = pv_acc(i);
                                    }
                                } else {
                                    const float pv0 = dim < kPartialSpillDims
                                        ? storage.pv0[
                                            row * kPartialSpillDims + dim]
                                        : online_acc(i);
                                    online_acc(i) =
                                        storage.prior_scale[row] * pv0 +
                                        storage.subtile_scale[row] * pv_acc(i);
                                }
                            } else {
                                online_acc(i) = subtile == 0 ? pv_acc(i) :
                                    storage.prior_scale[row] * online_acc(i) +
                                    storage.subtile_scale[row] * pv_acc(i);
                            }
                        }
                    }
                }
                if constexpr (!RegisterState) {
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                }
            }

            if constexpr (SplitPv) {
                static_assert(size(tSrP64) == size(tSrP));
                Element* p0_scratch = overlay_ptr;
                Element* p1_scratch = storage.k1.begin();
                #pragma unroll
                for (int n_half = 0; n_half < 2; ++n_half) {
                    Tensor sV0Half = local_tile(
                        sV0, make_shape(Int<64>{}, Int<64>{}),
                        make_coord(n_half, 0));
                    Tensor sV1Half = local_tile(
                        sV1, make_shape(Int<64>{}, Int<64>{}),
                        make_coord(n_half, 0));
                    Tensor tSrV0Half = pv64_thr.make_fragment_B(
                        pv64_thr.partition_B(sV0Half));
                    Tensor tSrV1Half = pv64_thr.make_fragment_B(
                        pv64_thr.partition_B(sV1Half));

                    #pragma unroll
                    for (int i = 0; i < size(tSrP64); ++i) {
                        const auto coord = tCcScores(i);
                        const int row = get<0>(coord);
                        const int col = get<1>(coord);
                        tSrP64(i) = p0_scratch[row * 64 + col];
                    }
                    auto pv0_acc = pv64_thr.make_fragment_C(tCcOutput64);
                    clear(pv0_acc);
                    #pragma unroll
                    for (int reduction = 0; reduction < 4; ++reduction) {
                        warpgroup_fence_operand(tSrP64);
                        warpgroup_fence_operand(pv0_acc);
                        warpgroup_arrive();
                        gemm(
                            pv64_mma, tSrP64(_, _, reduction),
                            tSrV0Half(_, _, reduction), pv0_acc);
                        warpgroup_commit_batch();
                        warpgroup_wait<0>();
                        warpgroup_fence_operand(pv0_acc);
                    }

                    #pragma unroll
                    for (int i = 0; i < size(tSrP64); ++i) {
                        const auto coord = tCcScores(i);
                        const int row = get<0>(coord);
                        const int col = get<1>(coord);
                        tSrP64(i) = p1_scratch[row * 64 + col];
                    }
                    auto pv1_acc = pv64_thr.make_fragment_C(tCcOutput64);
                    clear(pv1_acc);
                    #pragma unroll
                    for (int reduction = 0; reduction < 4; ++reduction) {
                        warpgroup_fence_operand(tSrP64);
                        warpgroup_fence_operand(pv1_acc);
                        warpgroup_arrive();
                        gemm(
                            pv64_mma, tSrP64(_, _, reduction),
                            tSrV1Half(_, _, reduction), pv1_acc);
                        warpgroup_commit_batch();
                        warpgroup_wait<0>();
                        warpgroup_fence_operand(pv1_acc);
                    }

                    static_assert(size(pv0_acc) % 2 == 0);
                    #pragma unroll
                    for (int i = 0; i < size(pv0_acc); i += 2) {
                        const auto coord = tCcOutput64(i);
                        const int row = get<0>(coord);
                        const int dim = n_half * 64 + get<1>(coord);
                        if (tile * 64 + row < row_count) {
                            const float denominator = storage.online_sum[row];
                            const float prior_scale = storage.prior_scale[row];
                            const float subtile_scale =
                                storage.subtile_scale[row];
                            const float2 value = denominator > 0.0f
                                ? make_float2(
                                    (
                                        prior_scale * pv0_acc(i) +
                                        subtile_scale * pv1_acc(i)
                                    ) / denominator,
                                    (
                                        prior_scale * pv0_acc(i + 1) +
                                        subtile_scale * pv1_acc(i + 1)
                                    ) / denominator)
                                : make_float2(0.0f, 0.0f);
                            float* destination = output +
                                (
                                    csc_offset + q_range_begin +
                                    static_cast<int64_t>(tile) * 64 + row
                                ) * 128 + dim;
                            *reinterpret_cast<float2*>(destination) = value;
                        }
                    }
                }
            } else {
                // The audited PV-C fragment mapping exposes each local
                // (2j, 2j+1) pair as adjacent output dimensions of the row.
                static_assert(size(online_acc) % 2 == 0);
                #pragma unroll
                for (int i = 0; i < size(online_acc); i += 2) {
                    const auto coord = tCcOutput(i);
                    const int row = get<0>(coord);
                    const int dim = get<1>(coord);
                    if (tile * 64 + row < row_count) {
                        const float denominator = RegisterState
                            ? (row == row0 ? online_sum0 : online_sum1)
                            : storage.online_sum[row];
                        const float2 value = denominator > 0.0f
                            ? make_float2(
                                online_acc(i) / denominator,
                                online_acc(i + 1) / denominator)
                            : make_float2(0.0f, 0.0f);
                        float* destination = output +
                            (
                                csc_offset + q_range_begin +
                                static_cast<int64_t>(tile) * 64 + row
                            ) * 128 + dim;
                        *reinterpret_cast<float2*>(destination) = value;
                    }
                }
            }
            if constexpr (RegisterState) {
                if ((lane & 3) == 0) {
                    if (tile * 64 + row0 < row_count) {
                        lse[csc_offset + q_range_begin +
                            static_cast<int64_t>(tile) * 64 + row0] =
                            online_sum0 > 0.0f
                            ? online_max0 + logf(online_sum0) : -INFINITY;
                    }
                    if (tile * 64 + row1 < row_count) {
                        lse[csc_offset + q_range_begin +
                            static_cast<int64_t>(tile) * 64 + row1] =
                            online_sum1 > 0.0f
                            ? online_max1 + logf(online_sum1) : -INFINITY;
                    }
                }
            } else {
                if (tid < 64 && tile * 64 + tid < row_count) {
                    const float sum = storage.online_sum[tid];
                    lse[csc_offset + q_range_begin +
                        static_cast<int64_t>(tile) * 64 + tid] =
                        sum > 0.0f
                        ? storage.online_max[tid] + logf(sum) : -INFINITY;
                }
                asm volatile("bar.sync 0, 128;" ::: "memory");
            }
        }
    }
#endif
}

template <
    bool BatchWgmma,
    bool RegisterState,
    bool WeightedPv,
    bool CompensatedPv,
    bool SplitPv,
    bool PartialSpill,
    typename TmaK,
    typename TmaV>
void launch(
    TmaK const& tma_k, TmaV const& tma_v, const at::Tensor& descriptor_storage,
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& col_offsets, const at::Tensor& col_nnz,
    const at::Tensor& q_rows, const at::Tensor& worklist,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, at::Tensor& output, at::Tensor& lse,
    int grid_size, int num_q_heads, int num_kv_heads, int max_columns,
    float softmax_scale, cudaStream_t stream) {
    const int smem = sizeof(SharedStorage<2, PartialSpill>);
    auto kernel =
        phase4b_csc_pipe_kernel<
            BatchWgmma, RegisterState, WeightedPv, CompensatedPv,
            SplitPv, PartialSpill, TmaK, TmaV>;
    cudaStreamCaptureStatus capture_status{};
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    const bool is_capturing = capture_status != cudaStreamCaptureStatusNone;
    // cudaFuncSetAttribute/occupancy queries are host-side launch setup and
    // invalidate stream capture.  The mandatory eager warmup immediately
    // before capture performs and validates this persistent kernel setup.
    if (!is_capturing) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int active_blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, kernel, kThreads, smem));
        TORCH_CHECK(active_blocks >= 2,
                    "SM90 KV-centric CSC pipeline must preserve two CTAs/SM; got ",
                    active_blocks, " with dynamic smem=", smem, " bytes");
    }
    kernel<<<grid_size, kThreads, smem, stream>>>(
        tma_k, tma_v,
        reinterpret_cast<const Element*>(q.data_ptr()),
        reinterpret_cast<const Element*>(k.data_ptr()),
        reinterpret_cast<const Element*>(v.data_ptr()),
        reinterpret_cast<cute::TmaDescriptor*>(descriptor_storage.data_ptr()),
        reinterpret_cast<const int64_t*>(col_offsets.data_ptr()),
        reinterpret_cast<const int*>(col_nnz.data_ptr()),
        reinterpret_cast<const int*>(q_rows.data_ptr()),
        worklist.defined()
            ? reinterpret_cast<const int*>(worklist.data_ptr()) : nullptr,
        reinterpret_cast<const int*>(cu_seqlens_q.data_ptr()),
        reinterpret_cast<const int*>(cu_seqlens_k.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()), reinterpret_cast<float*>(lse.data_ptr()),
        num_q_heads, num_kv_heads, max_columns, softmax_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> moba_sm90_phase4b_csc_pipe(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& col_offsets, const at::Tensor& col_nnz,
    const at::Tensor& q_rows, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, bool is_causal,
    double softmax_scale) {
#if !defined(LIGHTNING_MOBA_SM90_PIPELINE_AVAILABLE)
    TORCH_CHECK(false,
                "SM90 pipeline was not enabled in this build; rebuild with "
                "LIGHTNING_MOBA_CUDA_ARCHS=90a and CUDA 12.3 or newer");
#endif
    at::cuda::CUDAGuard guard{q.device()};
    TORCH_CHECK(is_causal, "SM90 KV-centric CSC pipeline is causal-only");
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == at::kBFloat16 &&
                    q.dim() == 3 && q.size(2) == kHeadDim &&
                    k.is_cuda() && k.is_contiguous() && k.scalar_type() == at::kBFloat16 &&
                    v.is_cuda() && v.is_contiguous() && v.scalar_type() == at::kBFloat16 &&
                    k.sizes() == v.sizes() && k.dim() == 3 && k.size(2) == kHeadDim,
                "SM90 KV-centric requires contiguous CUDA BF16 packed Q/K/V with head_dim=128");
    TORCH_CHECK(col_offsets.is_cuda() && col_offsets.is_contiguous() &&
                    col_offsets.scalar_type() == at::kLong && col_offsets.dim() == 3 &&
                    col_nnz.is_cuda() && col_nnz.is_contiguous() &&
                    col_nnz.scalar_type() == at::kInt && col_nnz.sizes() == col_offsets.sizes() &&
                    q_rows.is_cuda() && q_rows.is_contiguous() && q_rows.scalar_type() == at::kInt &&
                    q_rows.dim() == 1 && cu_seqlens_q.is_cuda() && cu_seqlens_q.is_contiguous() &&
                    cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_q.dim() == 1 &&
                    cu_seqlens_k.is_cuda() && cu_seqlens_k.is_contiguous() &&
                    cu_seqlens_k.scalar_type() == at::kInt && cu_seqlens_k.dim() == 1,
                "SM90 KV-centric requires contiguous CUDA CSC metadata");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                    q.device() == col_offsets.device() && q.device() == col_nnz.device() &&
                    q.device() == q_rows.device() && q.device() == cu_seqlens_q.device() &&
                    q.device() == cu_seqlens_k.device(),
                "SM90 KV-centric inputs must share one CUDA device");
    TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
                "softmax_scale must be finite and positive");

    const int64_t batch_size = col_nnz.size(0);
    const int64_t num_q_heads = col_nnz.size(1);
    const int64_t max_columns = col_nnz.size(2);
    const int64_t num_kv_heads = k.size(1);
    const int64_t grid_size = batch_size * num_q_heads * max_columns;
    TORCH_CHECK(batch_size > 0 && num_q_heads == q.size(1) && num_q_heads > 0 &&
                    num_kv_heads > 0 && num_q_heads % num_kv_heads == 0 && max_columns > 0 &&
                    cu_seqlens_q.size(0) == batch_size + 1 &&
                    cu_seqlens_k.size(0) == batch_size + 1 && grid_size <= INT_MAX &&
                    q_rows.numel() <= INT_MAX,
                "invalid SM90 KV-centric CSC dimensions");
    const cudaDeviceProp* prop = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(prop->major >= 9, "SM90 KV-centric CSC pipeline requires SM90+");

    const int64_t total_nnz = q_rows.numel();
    auto output = torch::empty({total_nnz, 128}, q.options().dtype(at::kFloat));
    auto lse = torch::empty({total_nnz}, q.options().dtype(at::kFloat));
    if (total_nnz > 0) {
        const auto current_stream = at::cuda::getCurrentCUDAStream();
        const auto stream = current_stream.stream();
        at::Tensor worklist;
        at::Tensor work_count;
        int launch_grid_size = static_cast<int>(grid_size);
        constexpr int qsplit_target = kQSplitTarget;
        {
            const int64_t work_capacity =
                grid_size + (total_nnz + qsplit_target - 1) / qsplit_target;
            TORCH_CHECK(
                work_capacity > 0 && work_capacity <= INT_MAX,
                "SM90 KV-centric q-range split worklist exceeds int grid range");
            worklist = torch::zeros(
                {work_capacity, kWorkRecordInts},
                col_nnz.options().dtype(at::kInt));
            work_count = torch::zeros(
                {1}, col_nnz.options().dtype(at::kInt));
            constexpr int threads = 256;
            const int blocks =
                static_cast<int>((grid_size + threads - 1) / threads);
            build_qsplit_worklist_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const int*>(col_nnz.data_ptr()),
                reinterpret_cast<int*>(worklist.data_ptr()),
                reinterpret_cast<int*>(work_count.data_ptr()),
                grid_size, static_cast<int>(num_q_heads),
                static_cast<int>(max_columns), qsplit_target,
                static_cast<int>(work_capacity));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            launch_grid_size = static_cast<int>(work_capacity);
            c10::cuda::CUDACachingAllocator::recordStream(
                worklist.storage().data_ptr(), current_stream);
            c10::cuda::CUDACachingAllocator::recordStream(
                work_count.storage().data_ptr(), current_stream);
        }

        Tensor k_tma_tensor = make_tensor(
            reinterpret_cast<Element const*>(k.data_ptr()),
            make_layout(make_shape(Int<64>{}, Int<128>{}),
                        make_stride(num_kv_heads * 128, Int<1>{})));
        Tensor v_tma_tensor = make_tensor(
            reinterpret_cast<Element const*>(v.data_ptr()),
            make_layout(make_shape(Int<64>{}, Int<128>{}),
                        make_stride(num_kv_heads * 128, Int<1>{})));
        auto tma_k = make_tma_atom(
            SM90_TMA_LOAD{}, k_tma_tensor, typename Traits::SmemLayoutK{}(_, _, Int<0>{}),
            make_shape(Int<64>{}, Int<64>{}));
        auto tma_v = make_tma_atom(
            SM90_TMA_LOAD{}, v_tma_tensor, typename Traits::SmemLayoutK{}(_, _, Int<0>{}),
            make_shape(Int<64>{}, Int<64>{}));
        auto descriptor_storage = torch::empty(
            {static_cast<int64_t>(launch_grid_size) * 2,
             static_cast<int64_t>(sizeof(cute::TmaDescriptor))},
            q.options().dtype(at::kByte));
        c10::cuda::CUDACachingAllocator::recordStream(
            descriptor_storage.storage().data_ptr(), current_stream);
        launch<false, false, true, false, false, false>(
            tma_k, tma_v, descriptor_storage, q, k, v, col_offsets, col_nnz,
            q_rows, worklist, cu_seqlens_q, cu_seqlens_k,
            output, lse, launch_grid_size,
            num_q_heads, num_kv_heads, max_columns, softmax_scale, stream);
    }
    return {output, lse};
}

}  // namespace LIGHTNING_MOBA_NAMESPACE
