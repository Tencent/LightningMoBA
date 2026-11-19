#pragma once

#include <torch/python.h>

#include <tuple>

#include "namespace_config.h"

namespace LIGHTNING_MOBA_NAMESPACE {

std::tuple<at::Tensor, at::Tensor> moba_sm90_phase4b_csc_pipe(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& col_offsets,
    const at::Tensor& col_nnz,
    const at::Tensor& q_rows,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    bool is_causal,
    double softmax_scale);

}  // namespace LIGHTNING_MOBA_NAMESPACE
