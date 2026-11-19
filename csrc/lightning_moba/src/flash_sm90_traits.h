// Copyright (c) 2025-2026, FlashMoBA Team.
#pragma once

#include "namespace_config.h"
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm90.h>

namespace LIGHTNING_MOBA_NAMESPACE {
using namespace cute;

template <typename Element>
struct Sm90WgmmaQk64Traits {
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockK = 64;
    static constexpr int kStages = 2;
    static constexpr int kThreads = 128;
    using ElementAccum = float;
    using SmemLayoutQ = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<Element>{},
        Shape<Int<kBlockM>, Int<kBlockK>, Int<kStages>>{}));
    using SmemLayoutK = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<Element>{},
        Shape<Int<kBlockN>, Int<kBlockK>, Int<kStages>>{}));
    using TiledMma = decltype(make_tiled_mma(
        SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
};

}  // namespace LIGHTNING_MOBA_NAMESPACE
