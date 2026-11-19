// Copyright (c) 2024, Tri Dao.
// Copyright (c) 2025, FlashMoBA Team.
// Splitting the different head dimensions to different files to speed up compilation.
// Explicit template instantiation.
#include "namespace_config.h"
#include "flash_bwd_launch_template.h"

namespace LIGHTNING_MOBA_NAMESPACE {

template<>
void run_moba_bwd_<cutlass::half_t, 128, false>(Lightning_moba_bwd_params &params, cudaStream_t stream) {
    run_moba_bwd_hdim128<cutlass::half_t, false>(params, stream);
}

} // namespace LIGHTNING_MOBA_NAMESPACE