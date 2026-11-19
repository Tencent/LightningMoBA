#!/bin/bash
# Test script for KV-Centric Triton kernel verification
# Run this on a GPU machine with lightning_moba installed

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "KV-Centric Triton Kernel Verification"
echo "============================================"
echo ""
echo "Running: python -m lightning_moba.triton_kv_centric"
echo ""

python -m lightning_moba.triton_kv_centric

echo ""
echo "Done."
