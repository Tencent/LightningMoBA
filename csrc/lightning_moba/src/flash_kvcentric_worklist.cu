// Copyright (c) 2025-2026, FlashMoBA Team.
#include "flash_kvcentric_worklist.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace LIGHTNING_MOBA_NAMESPACE {
namespace {

__global__ void lightning_moba_build_kvcentric_worklist_kernel(
    const int* col_nnz,
    int* worklist,
    int batch_size,
    int num_heads,
    int max_columns) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch_size) * num_heads * max_columns;
    if (index >= total) { return; }
    const int column = index % max_columns;
    const int head = (index / max_columns) % num_heads;
    const int batch = index / (static_cast<int64_t>(max_columns) * num_heads);
    int* record = worklist + index * 5;
    record[0] = batch;
    record[1] = head;
    record[2] = column;
    record[3] = 0;
    const int count = col_nnz[index];
    record[4] = count > 0 ? count : 0;
}

__global__ void lightning_moba_build_kvcentric_msplit_worklist_kernel(
    const int* col_nnz,
    int* worklist,
    int batch_size,
    int num_heads,
    int max_columns,
    int max_q_splits) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t column_count = static_cast<int64_t>(batch_size) * num_heads * max_columns;
    const int64_t total = column_count * max_q_splits;
    if (index >= total) { return; }
    const int split = index % max_q_splits;
    const int64_t column_index = index / max_q_splits;
    const int column = column_index % max_columns;
    const int head = (column_index / max_columns) % num_heads;
    const int batch = column_index / (static_cast<int64_t>(max_columns) * num_heads);
    const int q_begin = split * 64;
    const int remaining = col_nnz[column_index] - q_begin;
    int* record = worklist + index * 5;
    record[0] = batch;
    record[1] = head;
    record[2] = column;
    record[3] = q_begin;
    record[4] = remaining > 64 ? 64 : (remaining > 0 ? remaining : 0);
}
}  // namespace

at::Tensor moba_build_kvcentric_worklist(const at::Tensor& moba_col_nnz) {
    TORCH_CHECK(moba_col_nnz.is_cuda() && moba_col_nnz.is_contiguous(),
                "moba_col_nnz must be a contiguous CUDA tensor");
    TORCH_CHECK(moba_col_nnz.scalar_type() == at::kInt && moba_col_nnz.dim() == 3,
                "moba_col_nnz must be CUDA int32 [batch, heads, max_columns]");
    const int64_t batch_size = moba_col_nnz.size(0);
    const int64_t num_heads = moba_col_nnz.size(1);
    const int64_t max_columns = moba_col_nnz.size(2);
    TORCH_CHECK(batch_size > 0 && num_heads > 0 && max_columns > 0 &&
                    batch_size <= INT_MAX && num_heads <= INT_MAX && max_columns <= INT_MAX,
                "worklist dimensions must be positive int-range values");
    const int64_t work_count = batch_size * num_heads * max_columns;
    TORCH_CHECK(work_count <= INT_MAX, "worklist size exceeds supported int range");
    auto worklist = torch::empty({work_count, 5}, moba_col_nnz.options());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((work_count + threads - 1) / threads);
    lightning_moba_build_kvcentric_worklist_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<int const*>(moba_col_nnz.data_ptr()),
        reinterpret_cast<int*>(worklist.data_ptr()),
        static_cast<int>(batch_size), static_cast<int>(num_heads), static_cast<int>(max_columns));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return worklist;
}

at::Tensor moba_build_kvcentric_msplit_worklist(
    const at::Tensor& moba_col_nnz,
    int64_t max_q_splits) {
    TORCH_CHECK(moba_col_nnz.is_cuda() && moba_col_nnz.is_contiguous(),
                "moba_col_nnz must be a contiguous CUDA tensor");
    TORCH_CHECK(moba_col_nnz.scalar_type() == at::kInt && moba_col_nnz.dim() == 3,
                "moba_col_nnz must be CUDA int32 [batch, heads, max_columns]");
    const int64_t batch_size = moba_col_nnz.size(0);
    const int64_t num_heads = moba_col_nnz.size(1);
    const int64_t max_columns = moba_col_nnz.size(2);
    TORCH_CHECK(batch_size > 0 && num_heads > 0 && max_columns > 0 && max_q_splits > 0 &&
                    batch_size <= INT_MAX && num_heads <= INT_MAX && max_columns <= INT_MAX &&
                    max_q_splits <= INT_MAX / 64,
                "worklist dimensions must be positive int-range values");
    const int64_t work_count = batch_size * num_heads * max_columns * max_q_splits;
    TORCH_CHECK(work_count <= INT_MAX, "worklist size exceeds supported int range");
    auto worklist = torch::empty({work_count, 5}, moba_col_nnz.options());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((work_count + threads - 1) / threads);
    lightning_moba_build_kvcentric_msplit_worklist_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<int const*>(moba_col_nnz.data_ptr()),
        reinterpret_cast<int*>(worklist.data_ptr()),
        static_cast<int>(batch_size), static_cast<int>(num_heads),
        static_cast<int>(max_columns), static_cast<int>(max_q_splits));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return worklist;
}

}  // namespace LIGHTNING_MOBA_NAMESPACE
