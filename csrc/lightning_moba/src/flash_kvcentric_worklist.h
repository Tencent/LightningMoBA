// Copyright (c) 2025-2026, FlashMoBA Team.
#pragma once
#include <torch/python.h>
#include "namespace_config.h"

namespace LIGHTNING_MOBA_NAMESPACE {
// Build the existing KV-centric device worklist ABI directly from CSC counts.
// The returned CUDA int32 tensor is [B * H * C, 5] with records
// [batch, head, column, q_begin=0, q_count]. Empty columns remain present
// with q_count=0 so consumers can skip them without host-side compaction.
at::Tensor moba_build_kvcentric_worklist(const at::Tensor& moba_col_nnz);

// M-split variant of the same ABI. Each CSC column reserves max_q_splits
// 64-row records; q_count is zero for unused records. Callers must provide
// enough splits to cover their selected-row capacity. This is suitable for a
// direct 64-row WGMMA consumer without host-side CSC count inspection.
at::Tensor moba_build_kvcentric_msplit_worklist(
    const at::Tensor& moba_col_nnz,
    int64_t max_q_splits);

}  // namespace LIGHTNING_MOBA_NAMESPACE
