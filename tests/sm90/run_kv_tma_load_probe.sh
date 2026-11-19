#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUTLASS_ROOT="${PROJECT_ROOT}/csrc/cutlass"
FLASH_SRC="${PROJECT_ROOT}/csrc/lightning_moba/src"
SOURCE="${SCRIPT_DIR}/kv_tma_load_probe.cu"
BUILD_DIR="${BUILD_DIR:-/tmp/lightning_moba_sm90_kv_tma_probe}"
BINARY="${BUILD_DIR}/kv_tma_load_probe"
SASS="${BUILD_DIR}/kv_tma_load_probe.sass"

if [[ -n "${NVCC:-}" ]]; then
    NVCC_BIN="${NVCC}"
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
else
    NVCC_BIN="$(command -v nvcc || true)"
fi

if [[ -z "${NVCC_BIN}" || ! -x "${NVCC_BIN}" ]]; then
    echo "[FAIL] nvcc was not found. Set CUDA_HOME or NVCC." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

echo "=========================================="
echo "  LightningMoBA SM90 KV TMA load probe"
echo "  nvcc:   ${NVCC_BIN}"
echo "  source: ${SOURCE}"
echo "  build:  ${BUILD_DIR}"
echo "=========================================="
"${NVCC_BIN}" --version

"${NVCC_BIN}" \
    -std=c++17 \
    -O3 \
    -lineinfo \
    -arch=sm_90a \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
    -I"${CUTLASS_ROOT}/include" \
    -I"${CUTLASS_ROOT}/tools/util/include" \
    -I"${FLASH_SRC}" \
    "${SOURCE}" \
    -o "${BINARY}"

CUOBJDUMP_BIN="$(dirname "${NVCC_BIN}")/cuobjdump"
if [[ ! -x "${CUOBJDUMP_BIN}" ]]; then
    CUOBJDUMP_BIN="$(command -v cuobjdump || true)"
fi
if [[ -n "${CUOBJDUMP_BIN}" && -x "${CUOBJDUMP_BIN}" ]]; then
    "${CUOBJDUMP_BIN}" --dump-sass "${BINARY}" > "${SASS}"
    TMA_COUNT="$(grep -Ec 'UTMALDG|CPASYNC.*BULK.*TENSOR' "${SASS}" || true)"
    echo "[SASS] TMA-like instructions: ${TMA_COUNT}"
    if [[ "${TMA_COUNT}" -eq 0 ]]; then
        echo "[WARN] TMA mnemonic was not recognized; inspect ${SASS} manually."
    fi
else
    echo "[WARN] cuobjdump not found; skipping SASS check."
fi

"${BINARY}"

echo "=========================================="
echo "  KV TMA load probe complete"
echo "  binary: ${BINARY}"
echo "  sass:   ${SASS}"
echo "=========================================="
