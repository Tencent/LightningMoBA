// Copyright (c) 2025-2026, FlashMoBA Team.
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>
#include "flash.h"
#include "flash_fwd_kernel.h"
#include "flash_fwd_kvcentric_kernel.h"
#include "lightning_moba_static_switch.h"

namespace LIGHTNING_MOBA_NAMESPACE {
namespace {

template <typename Element, bool IsCausal, bool IsEvenK>
__global__ void kv_forward_stream(Lightning_moba_fwd_params p, int64_t begin, int64_t end) {
    using Traits = Lightning_moba_fwd_kernel_traits<128, 64, 64, 4, 1024, false, false, Element>;
    compute_moba_attn_kv_centric_worklist<Traits, IsCausal, IsEvenK>(p, begin, end);
}

__global__ void accumulate_stream_kernel(
    const float* partial, const float* local_lse, const int64_t* q_offsets,
    const int* q_csc, int64_t* cursor, float* numerator, float* maximum,
    float* denominator, int64_t begin, int64_t end, int64_t num_rows, int dim) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= num_rows) return;
    int64_t pos = cursor[row];
    const int64_t row_end = q_offsets[row + 1];
    float m = maximum[row], z = denominator[row];
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        acc[i] = d < dim ? numerator[row * dim + d] : 0.0f;
    }
    while (pos < row_end) {
        const int64_t edge = q_csc[pos];
        if (edge >= end) break;
        if (edge >= begin) {
            const int64_t local = edge - begin;
            const float lse = local_lse[local];
            if (isfinite(lse)) {
                const float next_m = fmaxf(m, lse);
                const float old_scale = isfinite(m) ? expf(m - next_m) : 0.0f;
                const float new_scale = expf(lse - next_m);
                z = z * old_scale + new_scale;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int d = lane + i * 32;
                    if (d < dim) acc[i] = fmaf(new_scale, partial[local * dim + d], acc[i] * old_scale);
                }
                m = next_m;
            }
        }
        ++pos;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        if (d < dim) numerator[row * dim + d] = acc[i];
    }
    if (lane == 0) {
        maximum[row] = m;
        denominator[row] = z;
        cursor[row] = pos;
    }
}

template <typename Element>
__global__ void finish_stream_kernel(
    const float* numerator, const float* maximum, const float* denominator,
    Element* out, float* lse, int64_t num_rows, int heads, int dim) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= num_rows) return;
    const float z = denominator[row];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        if (d < dim) out[row * dim + d] = Element(z > 0 ? numerator[row * dim + d] / z : 0.0f);
    }
    if (lane == 0) lse[static_cast<int64_t>(row % heads) * (num_rows / heads) + row / heads] =
        z > 0 ? maximum[row] + logf(z) : INFINITY;
}

// Live 64-edge records are ordered by CSC offset and tile [0, nnz).
at::Tensor stream_work_ranges(const at::Tensor& offsets, const at::Tensor& work,
                              int64_t nnz, int64_t chunk_edges) {
    const int64_t chunks = (nnz + chunk_edges - 1) / chunk_edges;
    auto column = (work.select(1, 0).to(at::kLong) * offsets.size(1) + work.select(1, 1).to(at::kLong))
        * offsets.size(2) + work.select(1, 2).to(at::kLong);
    auto starts = offsets.flatten().index_select(0, column) + work.select(1, 3).to(at::kLong);
    auto bounds = at::arange(chunks + 1, starts.options()).mul_(chunk_edges).clamp_max_(nnz);
    auto first = at::searchsorted(starts, bounds.narrow(0, 0, chunks), false, true).sub_(1).clamp_min_(0);
    auto last = at::searchsorted(starts, bounds.narrow(0, 1, chunks), false, false);
    return at::stack({first, last}).cpu();
}

template <typename Element, bool IsCausal, bool IsEvenK>
void launch_stream(
    Lightning_moba_fwd_params& p, const at::Tensor& offsets, const at::Tensor& work,
    const at::Tensor& q_offsets, const at::Tensor& q_csc,
    at::Tensor& partial, at::Tensor& partial_lse, at::Tensor& numerator,
    at::Tensor& maximum, at::Tensor& denominator, at::Tensor& cursor,
    at::Tensor& output, at::Tensor& lse, int64_t nnz, int64_t chunk_edges,
    bool restrict_grid, cudaStream_t stream) {
    using Traits = Lightning_moba_fwd_kernel_traits<128, 64, 64, 4, 1024, false, false, Element>;
    auto kernel = &kv_forward_stream<Element, IsCausal, IsEvenK>;
    if (nnz > 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    }
    const bool ranged = restrict_grid && nnz > 0 && work.size(0) > 0;
    const auto ranges = ranged ? stream_work_ranges(offsets, work, nnz, chunk_edges) : at::Tensor();
    const int64_t* range_first = ranged ? ranges.data_ptr<int64_t>() : nullptr;
    const int64_t chunk_count = ranged ? ranges.size(1) : 0;
    const int* work_base = work.data_ptr<int>();
    const int64_t rows = output.size(0) * output.size(1);
    for (int64_t begin = 0, chunk = 0; begin < nnz; begin += chunk_edges, ++chunk) {
        const int64_t end = std::min(begin + chunk_edges, nnz);
        int64_t first = 0, last = work.size(0);
        if (ranged) {
            first = range_first[chunk];
            last = range_first[chunk_count + chunk];
        }
        if (last > first) {
            p.kv_worklist_ptr = work_base + first * 5;
            p.kv_work_count = static_cast<int>(last - first);
            kernel<<<last - first, Traits::kNThreads, Traits::kSmemSize, stream>>>(p, begin, end);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (rows > 0) {
            accumulate_stream_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
                partial.data_ptr<float>(), partial_lse.data_ptr<float>(), q_offsets.data_ptr<int64_t>(),
                q_csc.data_ptr<int>(), cursor.data_ptr<int64_t>(), numerator.data_ptr<float>(),
                maximum.data_ptr<float>(), denominator.data_ptr<float>(), begin, end, rows, p.d);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    if (rows > 0) {
        finish_stream_kernel<Element><<<(rows + 3) / 4, 128, 0, stream>>>(
            numerator.data_ptr<float>(), maximum.data_ptr<float>(), denominator.data_ptr<float>(),
            reinterpret_cast<Element*>(output.data_ptr()), lse.data_ptr<float>(), rows, p.h, p.d);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace

std::vector<at::Tensor> moba_varlen_fwd_kvcentric_stream(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& offsets, const at::Tensor& counts, const at::Tensor& csc,
    const at::Tensor& work, const at::Tensor& cuq, const at::Tensor& cuk,
    const at::Tensor& q_offsets, const at::Tensor& q_csc,
    int maxq, int maxk, double scale, bool causal, int block_size,
    int64_t chunk_edges, bool restrict_grid) {
    c10::cuda::CUDAGuard guard(q.device());
    for (const auto& t : {q, k, v, offsets, counts, csc, work, cuq, cuk, q_offsets, q_csc}) {
        TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.is_contiguous(),
                    "streaming inputs must be contiguous CUDA tensors on the same device");
    }
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.sizes() == k.sizes(), "invalid Q/K/V dimensions");
    const int64_t rows = q.size(0) * q.size(1), nnz = csc.numel();
    TORCH_CHECK((q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf)
                && k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "streaming requires matching FP16/BF16 Q/K/V");
    TORCH_CHECK(q.size(2) > 0 && q.size(2) <= 128 && q.size(2) % 8 == 0
                && k.size(2) == q.size(2) && k.size(1) > 0 && q.size(1) > 0
                && q.size(1) % k.size(1) == 0, "streaming requires D<=128 and a valid GQA ratio");
    TORCH_CHECK(cuq.dim() == 1 && cuk.sizes() == cuq.sizes() && cuq.numel() >= 2
                && cuq.scalar_type() == at::kInt && cuk.scalar_type() == at::kInt,
                "invalid packed sequence lengths");
    TORCH_CHECK(offsets.dim() == 3 && counts.sizes() == offsets.sizes()
                && offsets.size(0) == cuq.numel() - 1 && offsets.size(1) == q.size(1)
                && offsets.scalar_type() == at::kLong && counts.scalar_type() == at::kInt
                && csc.dim() == 1 && csc.scalar_type() == at::kInt, "invalid CSC metadata");
    TORCH_CHECK(work.dim() == 2 && work.size(1) == 5 && work.scalar_type() == at::kInt
                && q_offsets.dim() == 1 && q_offsets.numel() == rows + 1 && q_offsets.scalar_type() == at::kLong
                && q_csc.dim() == 1 && q_csc.numel() == nnz && q_csc.scalar_type() == at::kInt,
                "invalid streaming worklist or sorted Q-CSR metadata");
    TORCH_CHECK(chunk_edges > 0 && chunk_edges <= INT_MAX && nnz <= INT_MAX
                && work.size(0) <= INT_MAX && rows <= INT_MAX && q.size(0) <= INT_MAX && k.size(0) <= INT_MAX
                && maxq >= 0 && maxq <= INT_MAX - 127 && maxk > 0 && maxk <= INT_MAX - 127
                && block_size > 0 && block_size % 64 == 0 && std::isfinite(scale) && scale > 0,
                "invalid streaming capacity, block size or scale");
    TORCH_CHECK(nnz == 0 || (rows > 0 && work.size(0) > 0), "nonempty CSC requires queries and live work records");
    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaStreamCaptureStatus status{};
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    TORCH_CHECK(status == cudaStreamCaptureStatusNone, "streaming prefill is eager-only; keep decode CUDA Graph on its existing path");
    const auto* prop = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(prop->major >= 8, "streaming requires an SM80+ GPU");
    const int64_t capacity = std::min(chunk_edges, nnz);
    auto opts = q.options().dtype(at::kFloat);
    auto partial = at::empty({capacity, q.size(2)}, opts);
    auto partial_lse = at::empty({capacity}, opts);
    auto numerator = at::zeros({rows, q.size(2)}, opts);
    auto maximum = at::full({rows}, -INFINITY, opts);
    auto denominator = at::zeros({rows}, opts);
    auto cursor = q_offsets.narrow(0, 0, rows).clone();
    auto output = at::empty(q.sizes(), q.options());
    auto lse = at::empty({q.size(1), q.size(0)}, opts);
    Lightning_moba_fwd_params p{};
    p.q_ptr = q.data_ptr(); p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr();
    p.q_row_stride = q.stride(0); p.q_head_stride = q.stride(1);
    p.k_row_stride = k.stride(0); p.k_head_stride = k.stride(1);
    p.v_row_stride = v.stride(0); p.v_head_stride = v.stride(1);
    p.cu_seqlens_q = cuq.data_ptr<int>(); p.cu_seqlens_k = cuk.data_ptr<int>();
    p.b = cuq.numel() - 1; p.h = q.size(1); p.h_k = k.size(1); p.h_h_k_ratio = p.h / p.h_k;
    p.d = q.size(2); p.d_rounded = (p.d + 31) / 32 * 32;
    p.total_q = q.size(0); p.total_k = k.size(0); p.seqlen_q = maxq; p.seqlen_k = maxk;
    p.seqlen_q_rounded = (maxq + 127) / 128 * 128; p.seqlen_k_rounded = (maxk + 127) / 128 * 128;
    p.col_offsets_ptr = offsets.data_ptr(); p.col_nnz_ptr = counts.data_ptr(); p.indices_ptr = csc.data_ptr();
    p.max_lg_col_num = offsets.size(2); p.col_offsets_batch_stride = offsets.stride(0);
    p.col_offsets_head_stride = offsets.stride(1); p.m_lg_block_dim = 128; p.n_lg_block_dim = block_size;
    p.is_seqlens_k_cumulative = true; p.is_bf16 = q.scalar_type() == at::kBFloat16;
    p.is_causal = causal; p.unpadded_lse = true; p.kv_causal_unsorted = true; p.use_moba_sparse = true;
    p.scale_softmax = scale; p.scale_softmax_log2 = scale * M_LOG2E;
    p.p_dropout = 1; p.rp_dropout = 1; p.scale_softmax_rp_dropout = scale;
    p.p_dropout_in_uint8_t = 255; p.window_size_left = maxk; p.window_size_right = causal ? 0 : maxk;
    p.o_tmp_ptr = partial.data_ptr(); p.softmax_lseaccum_ptr = partial_lse.data_ptr();
    p.kv_partial_dtype = 0; p.kv_partial_v_dim = p.d;
    FP16_SWITCH(!p.is_bf16, [&] {
        BOOL_SWITCH(causal, IsCausal, [&] {
            BOOL_SWITCH(p.d == 128, IsEvenK, [&] {
                launch_stream<elem_type, IsCausal, IsEvenK>(p, offsets, work, q_offsets, q_csc,
                    partial, partial_lse, numerator, maximum, denominator, cursor, output, lse,
                    nnz, chunk_edges, restrict_grid, stream);
            });
        });
    });
    return {output, lse};
}

}  // namespace LIGHTNING_MOBA_NAMESPACE

void init_flash_kvcentric_stream(pybind11::module& m) {
    m.def("moba_varlen_fwd_kvcentric_stream", &LIGHTNING_MOBA_NAMESPACE::moba_varlen_fwd_kvcentric_stream,
          "Bounded FP32 partial buffers and online O/LSE merge; sorted Q-CSR and live ordered worklist required");
}
