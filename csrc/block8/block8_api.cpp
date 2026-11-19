#include <torch/python.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <tuple>

#include "block8_launch.h"

namespace {

void check_sm90(const at::Tensor& reference) {
    TORCH_CHECK(reference.is_cuda(), "block8 requires CUDA tensors");
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, reference.get_device()));
    TORCH_CHECK(prop.major == 9 && prop.minor == 0,
                "block8 register kernel requires SM90");
}

void check_tensor(const at::Tensor& tensor, const at::Tensor& reference,
                  at::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == reference.device(),
                name, " must be on the same CUDA device as Q");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> block8_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& keep, const at::Tensor& shift,
    const at::Tensor& offsets, const at::Tensor& nnz,
    const at::Tensor& indices, const at::Tensor& slots,
    const at::Tensor& tiles, int64_t warps) {
    TORCH_CHECK(q.is_cuda(), "Q must be CUDA");
    const c10::cuda::CUDAGuard guard(q.device());
    check_sm90(q);
    check_tensor(q, q, at::kBFloat16, "Q");
    check_tensor(k, q, at::kBFloat16, "K");
    check_tensor(v, q, at::kBFloat16, "V");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
                "block8 is inference-only");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.sizes() == k.sizes(),
                "Q/K/V must be head-major rank-3 tensors with matching K/V");
    const int64_t h = q.size(0), s = q.size(1), d = q.size(2);
    const int64_t hk = k.size(0), n = k.size(1);
    TORCH_CHECK(h > 0 && hk > 0 && h % hk == 0 && s <= n &&
                d == 128 && k.size(2) == 128 && n % 256 == 0,
                "block8 requires D128 GQA, Sq <= Sk and 256-aligned K");
    TORCH_CHECK(warps == 1 || warps == 2 || warps == 4 || warps == 8,
                "warps must be 1, 2, 4 or 8");
    const int64_t columns = n / 256;
    for (int64_t dimension : {h, s, hk, n, h * columns}) {
        TORCH_CHECK(dimension <= std::numeric_limits<int>::max(),
                    "block8 dimensions exceed CUDA launch indexing");
    }
    check_tensor(keep, q, at::kInt, "keep");
    check_tensor(shift, q, at::kFloat, "shift");
    check_tensor(offsets, q, at::kLong, "offsets");
    check_tensor(nnz, q, at::kInt, "nnz");
    check_tensor(indices, q, at::kInt, "indices");
    check_tensor(slots, q, at::kInt, "work_slot");
    check_tensor(tiles, q, at::kInt, "work_tile");
    TORCH_CHECK(keep.sizes() == at::IntArrayRef({h, s, columns}) &&
                shift.sizes() == at::IntArrayRef({h, s}) &&
                offsets.sizes() == at::IntArrayRef({h, columns}) &&
                nnz.sizes() == at::IntArrayRef({h, columns}),
                "block8 pattern shapes do not match Q/K");
    TORCH_CHECK(indices.dim() == 1 && slots.dim() == 1 && tiles.dim() == 1 &&
                slots.numel() == tiles.numel() &&
                slots.numel() <= std::numeric_limits<int>::max(),
                "block8 worklist must have matching one-dimensional slot/tile arrays");

    auto out = at::zeros({h, s, d}, q.options().dtype(at::kFloat));
    auto den = at::zeros({h, s}, q.options().dtype(at::kFloat));
    auto count = at::zeros({h, s}, q.options().dtype(at::kInt));
    if (slots.numel() > 0) {
        // The pooled-key maximum is used only for selection. It is not a safe
        // exponential shift for token attention. Keep the shift argument for
        // the existing binding ABI, and compute the true masked maximum here.
        auto row_max = at::full({h, s}, -std::numeric_limits<float>::infinity(),
                                q.options().dtype(at::kFloat));
        auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
        const int error = lightning_moba_block8_launch(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), keep.data_ptr(),
            row_max.data_ptr(), offsets.data_ptr(), nnz.data_ptr(),
            indices.data_ptr(), slots.data_ptr(), tiles.data_ptr(),
            out.data_ptr(), den.data_ptr(), count.data_ptr(),
            s, n, h, hk, columns, columns, slots.numel(), warps,
            q.get_device(), reinterpret_cast<void*>(stream.stream()));
        C10_CUDA_CHECK(static_cast<cudaError_t>(error));
    }
    return {out, den, count};
}

at::Tensor block8_layout_probe(const at::Tensor& reference) {
    TORCH_CHECK(reference.is_cuda(), "layout probe requires a CUDA reference tensor");
    const c10::cuda::CUDAGuard guard(reference.device());
    check_sm90(reference);
    auto out = at::empty({16, 16}, reference.options().dtype(at::kFloat));
    auto stream = c10::cuda::getCurrentCUDAStream(reference.get_device());
    const int error = lightning_moba_block8_probe(
        out.data_ptr(), reinterpret_cast<void*>(stream.stream()), reference.get_device());
    C10_CUDA_CHECK(static_cast<cudaError_t>(error));
    return out;
}

}  // namespace

void init_block8(pybind11::module& m) {
    m.def("moba_block8_forward", &block8_forward,
          "Block8 SM90 CUDA attention with per-row fine masks",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("keep"), pybind11::arg("shift"),
          pybind11::arg("offsets"), pybind11::arg("nnz"),
          pybind11::arg("indices"), pybind11::arg("work_slot"),
          pybind11::arg("work_tile"), pybind11::arg("warps") = 4);
    m.def("moba_block8_layout_probe", &block8_layout_probe,
          "Verify the SM90 WMMA accumulator layout on the current stream");
}
