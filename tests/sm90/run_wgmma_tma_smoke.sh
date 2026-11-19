#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUTLASS_ROOT="${PROJECT_ROOT}/csrc/cutlass"
SOURCE="${SCRIPT_DIR}/wgmma_tma_smoke.cu"
BUILD_DIR="${BUILD_DIR:-/tmp/lightning_moba_sm90_smoke}"
BINARY="${BUILD_DIR}/wgmma_tma_smoke"
SASS="${BUILD_DIR}/wgmma_tma_smoke.sass"
PV_SASS="${BUILD_DIR}/wgmma_pv_rs_kernel.sass"

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

if [[ ! -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]]; then
    echo "[FAIL] CUTLASS submodule is missing: ${CUTLASS_ROOT}" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

echo "=========================================="
echo "  LightningMoBA SM90A TMA/WGMMA smoke test"
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
    "${SOURCE}" \
    -o "${BINARY}"

CUOBJDUMP_BIN="$(dirname "${NVCC_BIN}")/cuobjdump"
if [[ ! -x "${CUOBJDUMP_BIN}" ]]; then
    CUOBJDUMP_BIN="$(command -v cuobjdump || true)"
fi
if [[ -z "${CUOBJDUMP_BIN}" || ! -x "${CUOBJDUMP_BIN}" ]]; then
    echo "[FAIL] cuobjdump was not found; cannot verify generated WGMMA SASS." >&2
    exit 1
fi

"${CUOBJDUMP_BIN}" --dump-sass "${BINARY}" > "${SASS}"
WGMMA_COUNT="$(grep -Ec 'WGMMA|HGMMA' "${SASS}" || true)"
TMA_COUNT="$(grep -Ec 'UTMALDG|CPASYNC.*BULK.*TENSOR' "${SASS}" || true)"
echo "[SASS] WGMMA/HGMMA instructions: ${WGMMA_COUNT}"
echo "[SASS] TMA-like instructions: ${TMA_COUNT}"
if [[ "${WGMMA_COUNT}" -eq 0 ]]; then
    echo "[FAIL] No WGMMA/HGMMA instruction found. Confirm the sm_90a target." >&2
    grep -E 'MMA|HMMA|GMMA' "${SASS}" | head -20 >&2 || true
    exit 1
fi
if [[ "${TMA_COUNT}" -eq 0 ]]; then
    echo "[WARN] TMA mnemonic was not recognized; inspect ${SASS} manually."
fi

# C7510 can be conservative, so inspect the generated PV function rather than
# treating the compiler diagnostic alone as proof of a serialized pipeline.
awk '
    /Function :/ { in_pv = index($0, "wgmma_pv_rs_kernel") != 0 }
    in_pv { print }
' "${SASS}" > "${PV_SASS}"
if [[ -s "${PV_SASS}" ]]; then
    PV_CALL_COUNT="$(grep -Ec '[[:space:]]CALL([.;[:space:]]|$)' "${PV_SASS}" || true)"
    PV_DEPBAR_COUNT="$(grep -c 'WARPGROUP.DEPBAR' "${PV_SASS}" || true)"
    PV_HGMMA_COUNT="$(grep -Ec 'WGMMA|HGMMA' "${PV_SASS}" || true)"
    echo "[SASS PV] HGMMA=${PV_HGMMA_COUNT} CALL=${PV_CALL_COUNT} DEPBAR=${PV_DEPBAR_COUNT}"
    if [[ "${PV_CALL_COUNT}" -gt 0 ]]; then
        echo "[SASS PV] Device calls found:"
        grep -n -B3 -A3 -E '[[:space:]]CALL([.;[:space:]]|$)' "${PV_SASS}" || true
    fi
else
    echo "[WARN] Could not isolate PV kernel SASS; inspect ${SASS} manually."
fi

"${BINARY}"

echo "=========================================="
echo "  SM90A smoke test complete"
echo "  binary: ${BINARY}"
echo "  sass:   ${SASS}"
echo "  pv sass:${PV_SASS}"
echo "=========================================="
