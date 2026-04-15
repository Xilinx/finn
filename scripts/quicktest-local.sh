#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# quicktest-local.sh - Quick validation tests for FINN local installation
#
# This script runs a minimal set of tests to verify the FINN installation.
#
# Usage:
#   ./scripts/quicktest-local.sh [MODE]
#
# Modes:
#   (default)  Run import test + basic transformation tests
#   vivado     Also run a sanity test to check if Vivado integration works (requires Vivado)

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

gecho() {
    echo -e "${GREEN}$1${NC}"
}

yecho() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

recho() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Determine FINN_ROOT
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
FINN_ROOT=$(dirname "$SCRIPTPATH")

# Parse mode
MODE="${1:-default}"

echo "=============================================="
echo "FINN Local Installation Validation"
echo "=============================================="
echo ""

# Test 1: Python imports
gecho "Test 1: Python imports..."

python3 -c "
import sys
print(f'  Python version: {sys.version}')

import finn
print(f'  finn: OK')

import qonnx
print(f'  qonnx: OK')

import brevitas
print(f'  brevitas: OK')

import onnx
print(f'  onnx: OK')

import numpy
print(f'  numpy: OK')

print('  All core imports successful!')
"

if [ $? -ne 0 ]; then
    recho "Import test failed!"
    exit 1
fi
gecho "  PASSED"
echo ""

# Test 2: Basic transformation tests (no Vivado required)
gecho "Test 2: Basic transformation tests..."

cd "$FINN_ROOT"
pytest tests/transformation/ \
    -m "not vivado and not slow and not vitis" \
    --maxfail=3 \
    -q \
    --tb=short

if [ $? -ne 0 ]; then
    recho "Transformation tests failed!"
    exit 1
fi
gecho "  PASSED"
echo ""

# Test 3: Utility tests
gecho "Test 3: Utility tests..."

pytest tests/util/ \
    -m "not vivado and not slow" \
    --maxfail=3 \
    -q \
    --tb=short

if [ $? -ne 0 ]; then
    recho "Utility tests failed!"
    exit 1
fi
gecho "  PASSED"
echo ""

# Test 4: Vivado integration sanity test (optional)
if [ "$MODE" = "vivado" ]; then
    if [ -n "$XILINX_VIVADO" ]; then
        gecho "Test 4a: Vivado cppsim test (HLS LayerNorm)..."

        # cppsim test - tests HLS C++ simulation
        pytest "tests/fpgadataflow/test_fpgadataflow_layernorm.py::test_fpgadataflow_hls_layernorm[FLOAT32-ishape1-1-cppsim]" \
            -v \
            --maxfail=1 \
            --tb=short

        if [ $? -ne 0 ]; then
            recho "cppsim test failed!"
            exit 1
        fi
        gecho "  PASSED"
        echo ""

        gecho "Test 4b: Vivado node-by-node rtlsim test (HLS LayerNorm)..."

        # node_by_node rtlsim test - tests RTL simulation per node
        pytest "tests/fpgadataflow/test_fpgadataflow_layernorm.py::test_fpgadataflow_hls_layernorm[FLOAT32-ishape1-1-node_by_node]" \
            -v \
            --maxfail=1 \
            --tb=short

        if [ $? -ne 0 ]; then
            recho "node-by-node rtlsim test failed!"
            exit 1
        fi
        gecho "  PASSED"
        echo ""

        gecho "Test 4c: Vivado stitched IP rtlsim test (HLS LayerNorm)..."

        # stitched_ip rtlsim test - tests RTL simulation of stitched IP
        pytest "tests/fpgadataflow/test_fpgadataflow_layernorm.py::test_fpgadataflow_hls_layernorm[FLOAT32-ishape1-1-stitched_ip]" \
            -v \
            --maxfail=1 \
            --tb=short

        if [ $? -ne 0 ]; then
            recho "stitched IP rtlsim test failed!"
            exit 1
        fi
        gecho "  PASSED"
    else
        yecho "Test 4: Skipping Vivado tests (XILINX_VIVADO not set)"
    fi
    echo ""
fi

echo "=============================================="
gecho "All validation tests passed!"
echo "=============================================="
echo ""
echo "Your FINN installation is working correctly."
echo ""
if [ -z "$XILINX_VIVADO" ]; then
    echo "Note: Vivado integration was not tested. To test Vivado:"
    echo "  1. Set FINN_XILINX_PATH and FINN_XILINX_VERSION"
    echo "  2. Run: ./scripts/quicktest-local.sh vivado"
fi
