#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace flash_dynamic_attention {

struct QcsrScatterParams {
    const int64_t* col_offsets;
    const int* col_nnz;
    const int* csc_rows;
    const int* cu_seqlens_q;
    const int64_t* q_offsets;
    int* q_cursor;
    int* q_csc_indices;
    int batch;
    int heads;
    int columns;
    int total_q;
    int64_t total_nnz;
    int64_t col_batch_stride;
    int64_t col_head_stride;
};

__global__ void scatter_csc_to_qcsr_kernel(QcsrScatterParams params) {
    const int col = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (col >= params.columns || head >= params.heads || batch >= params.batch) return;

    const int64_t meta = static_cast<int64_t>(batch) * params.col_batch_stride
        + static_cast<int64_t>(head) * params.col_head_stride + col;
    const int64_t csc_begin = params.col_offsets[meta];
    const int count = params.col_nnz[meta];
    const int q_base = params.cu_seqlens_q[batch];
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int64_t csc_pos = csc_begin + i;
        if (csc_pos < 0 || csc_pos >= params.total_nnz) continue;
        const int q_global = q_base + params.csc_rows[csc_pos];
        if (q_global < 0 || q_global >= params.total_q) continue;
        const int qh = q_global * params.heads + head;
        const int slot = atomicAdd(params.q_cursor + qh, 1);
        const int64_t dst = params.q_offsets[qh] + slot;
        if (dst < params.q_offsets[qh + 1] && dst < params.total_nnz) {
            params.q_csc_indices[dst] = static_cast<int>(csc_pos);
        }
    }
}

template <typename T>
__device__ inline float partial_to_float(T value);

template <>
__device__ inline float partial_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float partial_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
__device__ inline float partial_to_float<__half>(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ inline T float_to_output(float value);

template <>
__device__ inline __nv_bfloat16 float_to_output<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ inline __half float_to_output<__half>(float value) {
    return __float2half_rn(value);
}

__device__ inline float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ inline float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename Partial, typename Output>
__global__ void combine_qcsr_kernel(
    const Partial* o_partial,
    const float* lse_partial,
    const int64_t* q_offsets,
    const int* q_csc_indices,
    Output* out,
    int64_t partial_row_stride,
    int64_t out_row_stride,
    int64_t out_head_stride,
    int total_q,
    int heads,
    int head_dim) {
    const int q = blockIdx.x;
    const int head = blockIdx.y;
    const int d = threadIdx.x;
    if (q >= total_q || head >= heads || d >= head_dim) return;

    const int qh = q * heads + head;
    const int64_t begin = q_offsets[qh];
    const int64_t end = q_offsets[qh + 1];
    const int64_t out_pos = static_cast<int64_t>(q) * out_row_stride
        + static_cast<int64_t>(head) * out_head_stride + d;
    if (begin >= end) {
        out[out_pos] = float_to_output<Output>(0.0f);
        return;
    }

    __shared__ float shared_max;
    __shared__ float shared_inv_sum;
    __shared__ int shared_csc[32];
    __shared__ float shared_weight[32];

    const int lane = d & 31;
    if (d < 32) {
        float local_max = -INFINITY;
        for (int64_t pos = begin + lane; pos < end; pos += 32) {
            const float value = lse_partial[q_csc_indices[pos]];
            if (isfinite(value)) local_max = fmaxf(local_max, value);
        }
        local_max = warp_max(local_max);
        if (lane == 0) shared_max = local_max;
    }
    __syncthreads();

    if (d < 32) {
        float local_sum = 0.0f;
        for (int64_t pos = begin + lane; pos < end; pos += 32) {
            const float value = lse_partial[q_csc_indices[pos]];
            if (isfinite(value) && isfinite(shared_max)) {
                local_sum += expf(value - shared_max);
            }
        }
        local_sum = warp_sum(local_sum);
        if (lane == 0) shared_inv_sum = local_sum > 0.0f ? 1.0f / local_sum : 0.0f;
    }
    __syncthreads();

    float accumulator = 0.0f;
    for (int64_t tile = begin; tile < end; tile += 32) {
        const int64_t remaining = end - tile;
        const int tile_count = remaining < 32 ? static_cast<int>(remaining) : 32;
        if (d < tile_count) {
            const int csc = q_csc_indices[tile + d];
            const float lse = lse_partial[csc];
            shared_csc[d] = csc;
            shared_weight[d] = isfinite(lse) && isfinite(shared_max)
                ? expf(lse - shared_max) * shared_inv_sum : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i < tile_count) {
                const int csc = shared_csc[i];
                const float value = partial_to_float(o_partial[
                    static_cast<int64_t>(csc) * partial_row_stride + d]);
                accumulator = fmaf(shared_weight[i], value, accumulator);
            }
        }
        __syncthreads();
    }
    out[out_pos] = float_to_output<Output>(accumulator);
}

// Four independent Q/head rows per CTA.  The original kernel launches one
// 128-thread CTA per row and synchronizes all four warps for every 32 edges.
// Here each warp owns a row, computes four output dimensions per lane, and
// broadcasts one CSC position/weight at a time without shared-memory fences.
template <typename Partial, typename Output>
__global__ void combine_qcsr_warp_kernel(
    const Partial* o_partial,
    const float* lse_partial,
    const int64_t* q_offsets,
    const int* q_csc_indices,
    Output* out,
    int64_t partial_row_stride,
    int64_t out_row_stride,
    int64_t out_head_stride,
    int total_q,
    int heads) {
    const int lane = threadIdx.x & 31;
    const int qh = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (qh >= total_q * heads) return;
    const int64_t begin = q_offsets[qh];
    const int64_t end = q_offsets[qh + 1];
    const int q = qh / heads;
    const int head = qh - q * heads;
    const int64_t out_pos = static_cast<int64_t>(q) * out_row_stride
        + static_cast<int64_t>(head) * out_head_stride + lane;
    if (begin == end) {
#pragma unroll
        for (int segment = 0; segment < 4; ++segment) {
            out[out_pos + segment * 32] = float_to_output<Output>(0.0f);
        }
        return;
    }

    float local_max = -INFINITY;
    for (int64_t pos = begin + lane; pos < end; pos += 32) {
        const float value = lse_partial[q_csc_indices[pos]];
        if (isfinite(value)) local_max = fmaxf(local_max, value);
    }
    const float row_max = __shfl_sync(0xffffffffu, warp_max(local_max), 0);
    float local_sum = 0.0f;
    for (int64_t pos = begin + lane; pos < end; pos += 32) {
        const float value = lse_partial[q_csc_indices[pos]];
        if (isfinite(value) && isfinite(row_max)) {
            local_sum += expf(value - row_max);
        }
    }
    const float sum = __shfl_sync(0xffffffffu, warp_sum(local_sum), 0);
    const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int64_t pos = begin; pos < end; ++pos) {
        int csc = 0;
        float weight = 0.0f;
        if (lane == 0) {
            csc = q_csc_indices[pos];
            const float lse = lse_partial[csc];
            weight = isfinite(lse) && isfinite(row_max)
                ? expf(lse - row_max) * inv_sum : 0.0f;
        }
        csc = __shfl_sync(0xffffffffu, csc, 0);
        weight = __shfl_sync(0xffffffffu, weight, 0);
        const int64_t partial_pos = static_cast<int64_t>(csc) * partial_row_stride + lane;
#pragma unroll
        for (int segment = 0; segment < 4; ++segment) {
            accum[segment] = fmaf(weight,
                partial_to_float(o_partial[partial_pos + segment * 32]),
                accum[segment]);
        }
    }
#pragma unroll
    for (int segment = 0; segment < 4; ++segment) {
        out[out_pos + segment * 32] = float_to_output<Output>(accum[segment]);
    }
}

std::vector<at::Tensor> build_q_csr_from_csc(
    const at::Tensor& col_offsets,
    const at::Tensor& col_nnz,
    const at::Tensor& csc_rows,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& row_nnz) {
    at::cuda::CUDAGuard guard{col_offsets.device()};
    TORCH_CHECK(col_offsets.is_cuda() && col_nnz.is_cuda() && csc_rows.is_cuda() &&
                cu_seqlens_q.is_cuda() && row_nnz.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(col_offsets.scalar_type() == at::kLong, "col_offsets must be int64");
    TORCH_CHECK(col_nnz.scalar_type() == at::kInt && csc_rows.scalar_type() == at::kInt &&
                cu_seqlens_q.scalar_type() == at::kInt && row_nnz.scalar_type() == at::kInt,
                "counts, rows, and cu_seqlens_q must be int32");
    TORCH_CHECK(col_offsets.is_contiguous() && col_nnz.is_contiguous() && csc_rows.is_contiguous() &&
                cu_seqlens_q.is_contiguous() && row_nnz.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(col_offsets.dim() == 3 && col_nnz.sizes() == col_offsets.sizes(),
                "column metadata must have matching [batch, heads, columns] shapes");
    TORCH_CHECK(csc_rows.dim() == 1 && row_nnz.dim() == 2,
                "csc_rows must be 1D and row_nnz must be [total_q, heads]");
    const int batch = col_offsets.size(0);
    const int heads = col_offsets.size(1);
    const int columns = col_offsets.size(2);
    const int total_q = row_nnz.size(0);
    TORCH_CHECK(row_nnz.size(1) == heads, "row_nnz head count mismatch");
    TORCH_CHECK(cu_seqlens_q.numel() == batch + 1, "cu_seqlens_q length mismatch");
    TORCH_CHECK(csc_rows.numel() <= std::numeric_limits<int>::max(),
                "q_csc_indices int32 encoding requires total_nnz <= INT_MAX");

    const int64_t qh = static_cast<int64_t>(total_q) * heads;
    auto counts64 = row_nnz.reshape({qh}).to(at::kLong);
    auto prefix = at::cumsum(counts64, 0);
    auto q_offsets = at::empty({qh + 1}, col_offsets.options());
    q_offsets.narrow(0, 0, 1).zero_();
    q_offsets.narrow(0, 1, qh).copy_(prefix);
    auto q_csc_indices = at::empty({csc_rows.numel()}, csc_rows.options());
    auto q_cursor = at::zeros({qh}, csc_rows.options());

    QcsrScatterParams params{};
    params.col_offsets = col_offsets.data_ptr<int64_t>();
    params.col_nnz = col_nnz.data_ptr<int>();
    params.csc_rows = csc_rows.data_ptr<int>();
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.q_offsets = q_offsets.data_ptr<int64_t>();
    params.q_cursor = q_cursor.data_ptr<int>();
    params.q_csc_indices = q_csc_indices.data_ptr<int>();
    params.batch = batch;
    params.heads = heads;
    params.columns = columns;
    params.total_q = total_q;
    params.total_nnz = csc_rows.numel();
    params.col_batch_stride = col_offsets.stride(0);
    params.col_head_stride = col_offsets.stride(1);
    dim3 grid(columns, heads, batch);
    scatter_csc_to_qcsr_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream().stream()>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_offsets, q_csc_indices};
}

at::Tensor combine_q_csr(
    const at::Tensor& o_partial,
    const at::Tensor& lse_partial,
    const at::Tensor& q_offsets,
    const at::Tensor& q_csc_indices,
    at::Tensor& out) {
    at::cuda::CUDAGuard guard{out.device()};
    TORCH_CHECK(o_partial.is_cuda() && lse_partial.is_cuda() && q_offsets.is_cuda() &&
                q_csc_indices.is_cuda() && out.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(o_partial.is_contiguous() && lse_partial.is_contiguous() &&
                q_offsets.is_contiguous() && q_csc_indices.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(o_partial.dim() == 2 && lse_partial.dim() == 1 && out.dim() == 3,
                "expected partial [nnz,D], lse [nnz], and out [Q,H,D]");
    TORCH_CHECK(lse_partial.scalar_type() == at::kFloat && q_offsets.scalar_type() == at::kLong &&
                q_csc_indices.scalar_type() == at::kInt, "invalid metadata dtype");
    TORCH_CHECK(o_partial.size(0) == lse_partial.size(0), "partial NNZ mismatch");
    TORCH_CHECK(q_csc_indices.numel() == o_partial.size(0), "Q-CSR NNZ mismatch");
    TORCH_CHECK(out.size(2) == 128 && o_partial.size(1) == 128,
                "first variable combine implementation requires head_dim=128");
    TORCH_CHECK(q_offsets.numel() == out.size(0) * out.size(1) + 1, "Q offsets shape mismatch");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf,
                "out must be BF16 or FP16");
    TORCH_CHECK(out.stride(2) == 1, "out last dimension must be contiguous");
    TORCH_CHECK(o_partial.scalar_type() == at::kFloat || o_partial.scalar_type() == at::kBFloat16 ||
                o_partial.scalar_type() == at::kHalf, "partial must be FP32, BF16, or FP16");

    const char* warp_env = std::getenv("LIGHTNING_MOBA_COMBINE_WARP");
    // At high edge counts the original four-warp/row kernel is faster: it
    // parallelizes the partial-vector loads within a row.  The warp/row path
    // wins only for short CSR rows, so select it from host-known tensor shapes
    // without reading GPU data or synchronizing the stream.
    const int64_t qh = out.size(0) * out.size(1);
    const bool use_warp = warp_env != nullptr && std::strcmp(warp_env, "1") == 0 &&
        out.size(0) >= 32768 && o_partial.size(0) <= qh * 15;
    dim3 grid(out.size(0), out.size(1));
    dim3 block(128);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
#define LAUNCH_COMBINE(PARTIAL_T, OUTPUT_T) do { \
    if (use_warp) { \
        combine_qcsr_warp_kernel<PARTIAL_T, OUTPUT_T><<< \
            (out.size(0) * out.size(1) + 3) / 4, block.x, 0, stream>>>( \
            reinterpret_cast<const PARTIAL_T*>(o_partial.data_ptr()), \
            lse_partial.data_ptr<float>(), q_offsets.data_ptr<int64_t>(), \
            q_csc_indices.data_ptr<int>(), reinterpret_cast<OUTPUT_T*>(out.data_ptr()), \
            o_partial.stride(0), out.stride(0), out.stride(1), \
            out.size(0), out.size(1)); \
    } else { \
        combine_qcsr_kernel<PARTIAL_T, OUTPUT_T><<<grid, block, 0, stream>>>( \
        reinterpret_cast<const PARTIAL_T*>(o_partial.data_ptr()), \
        lse_partial.data_ptr<float>(), q_offsets.data_ptr<int64_t>(), \
        q_csc_indices.data_ptr<int>(), reinterpret_cast<OUTPUT_T*>(out.data_ptr()), \
        o_partial.stride(0), out.stride(0), out.stride(1), \
        out.size(0), out.size(1), out.size(2)); \
    } \
} while (false)
    if (out.scalar_type() == at::kBFloat16) {
        if (o_partial.scalar_type() == at::kFloat) LAUNCH_COMBINE(float, __nv_bfloat16);
        else if (o_partial.scalar_type() == at::kBFloat16) LAUNCH_COMBINE(__nv_bfloat16, __nv_bfloat16);
        else LAUNCH_COMBINE(__half, __nv_bfloat16);
    } else {
        if (o_partial.scalar_type() == at::kFloat) LAUNCH_COMBINE(float, __half);
        else if (o_partial.scalar_type() == at::kBFloat16) LAUNCH_COMBINE(__nv_bfloat16, __half);
        else LAUNCH_COMBINE(__half, __half);
    }
#undef LAUNCH_COMBINE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace flash_dynamic_attention

void init_flash_dynamic_attention(pybind11::module& m) {
    m.def("moba_build_q_csr_from_csc", &flash_dynamic_attention::build_q_csr_from_csc,
          "Build capacity-free Q-to-CSC reverse CSR");
    m.def("moba_combine_q_csr", &flash_dynamic_attention::combine_q_csr,
          "Combine KV-centric partials through variable-NNZ Q CSR");
}
