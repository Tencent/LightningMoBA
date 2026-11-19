// Copyright (c) 2024, Tri Dao.
// Copyright (c) 2025, FlashMoBA Team.
// Splitting the different head dimensions to different files to speed up compilation.
// Explicit template instantiation.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace LIGHTNING_MOBA_NAMESPACE {

template<>
void run_moba_fwd_<cutlass::bfloat16_t, 192, true>(Lightning_moba_fwd_params &params, cudaStream_t stream) {
    run_moba_fwd_hdim192<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace LIGHTNING_MOBA_NAMESPACE