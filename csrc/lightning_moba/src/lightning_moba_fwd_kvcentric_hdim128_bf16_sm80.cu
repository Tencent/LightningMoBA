// Copyright (c) 2025-2026, FlashMoBA Team.
// KV-Centric forward kernel instantiation: bf16, hdim128, non-causal
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace LIGHTNING_MOBA_NAMESPACE {

template void run_moba_fwd_kvcentric_hdim128<cutlass::bfloat16_t, false>(Lightning_moba_fwd_params &params, cudaStream_t stream);

} // namespace LIGHTNING_MOBA_NAMESPACE
