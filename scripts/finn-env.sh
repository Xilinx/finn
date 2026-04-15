#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# finn-env.sh - Environment activation script for FINN local installation
#
# Source this script to set up the FINN environment:
#   source scripts/finn-env.sh
#
# This script:
#   - Activates the Python virtual environment
#   - Sets FINN-specific environment variables
#   - Sources Xilinx tools if configured
#   - Sets up library paths for finn_xsi

# Color definitions
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

_finn_gecho() {
    echo -e "${GREEN}$1${NC}"
}

_finn_yecho() {
    echo -e "${YELLOW}$1${NC}"
}

# Determine FINN_ROOT from script location
if [ -n "${BASH_SOURCE[0]}" ]; then
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export FINN_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
else
    # Fallback if BASH_SOURCE is not available
    if [ -z "$FINN_ROOT" ]; then
        echo "ERROR: Could not determine FINN_ROOT. Please set it manually."
        return 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "$FINN_ROOT/.venv" ]; then
    echo "ERROR: Virtual environment not found at $FINN_ROOT/.venv"
    echo "Please run ./setup-local.sh first."
    return 1
fi

# Activate virtual environment
source "$FINN_ROOT/.venv/bin/activate"
_finn_gecho "Activated FINN environment at $FINN_ROOT"

# Set FINN environment variables
export FINN_BUILD_DIR="${FINN_BUILD_DIR:-/tmp/finn_local_$(whoami)}"
mkdir -p "$FINN_BUILD_DIR"

# Add oh-my-xilinx to PATH
export OHMYXILINX="$FINN_ROOT/deps/oh-my-xilinx"
if [ -d "$OHMYXILINX" ]; then
    export PATH="$PATH:$OHMYXILINX"
fi

# Board files path
export FINN_BOARD_FILES_PATH="$FINN_ROOT/deps/board_files"

# Xilinx tools setup
if [ -n "$FINN_XILINX_PATH" ] && [ -n "$FINN_XILINX_VERSION" ]; then
    # Determine and export tool paths
    export VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"
    export VITIS_PATH="$FINN_XILINX_PATH/Vitis/$FINN_XILINX_VERSION"
    export HLS_PATH="$FINN_XILINX_PATH/Vitis_HLS/$FINN_XILINX_VERSION"

    # Source Vitis (includes Vivado settings)
    if [ -f "$VITIS_PATH/settings64.sh" ]; then
        source "$VITIS_PATH/settings64.sh"
        _finn_gecho "Sourced Vitis at $VITIS_PATH"

        # Source XRT if available (for Alveo support)
        export XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"
        if [ -f "$XILINX_XRT/setup.sh" ]; then
            source "$XILINX_XRT/setup.sh"
            _finn_gecho "Sourced XRT at $XILINX_XRT"
        else
            _finn_yecho "XRT not found at $XILINX_XRT (optional, for Alveo)"
        fi
    elif [ -f "$VIVADO_PATH/settings64.sh" ]; then
        # Fall back to Vivado only
        source "$VIVADO_PATH/settings64.sh"
        _finn_gecho "Sourced Vivado at $VIVADO_PATH"
    else
        _finn_yecho "Vivado not found at $VIVADO_PATH"
    fi

    # Source Vitis HLS
    if [ -f "$HLS_PATH/settings64.sh" ]; then
        source "$HLS_PATH/settings64.sh"
        _finn_gecho "Sourced Vitis HLS at $HLS_PATH"
    else
        _finn_yecho "Vitis HLS not found at $HLS_PATH"
    fi

    # Set LD_LIBRARY_PATH for finn_xsi
    if [ -n "$XILINX_VIVADO" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lib/x86_64-linux-gnu/:${XILINX_VIVADO}/lib/lnx64.o"
    fi

    # FPO tools library path (for floating point operations)
    if [ -d "$VITIS_PATH/lnx64/tools/fpo_v7_1" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$VITIS_PATH/lnx64/tools/fpo_v7_1"
    fi
    if [ -d "$HLS_PATH/lnx64/tools/fpo_v7_1" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$HLS_PATH/lnx64/tools/fpo_v7_1"
    fi
else
    _finn_yecho "FINN_XILINX_PATH and/or FINN_XILINX_VERSION not set"
    _finn_yecho "Xilinx tools not available. Set these for synthesis support."
fi

# FlexLM workaround (may help with some Vivado licensing issues)
export LD_PRELOAD="${LD_PRELOAD}:/lib/x86_64-linux-gnu/libudev.so.1"

# Multiple Vivado instances workaround
export XILINX_LOCAL_USER_DATA=no

# Handle Xilinx init scripts if present
if [ -d "$FINN_ROOT/.Xilinx" ]; then
    if [ ! -d "$HOME/.Xilinx" ]; then
        mkdir -p "$HOME/.Xilinx"
    fi
    if [ -f "$FINN_ROOT/.Xilinx/HLS_init.tcl" ]; then
        cp "$FINN_ROOT/.Xilinx/HLS_init.tcl" "$HOME/.Xilinx/"
    fi
    if [ -f "$FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl" ]; then
        mkdir -p "$HOME/.Xilinx/Vivado"
        cp "$FINN_ROOT/.Xilinx/Vivado/Vivado_init.tcl" "$HOME/.Xilinx/Vivado/"
    fi
fi

# Vivado IP cache directory
export VIVADO_IP_CACHE="${VIVADO_IP_CACHE:-$FINN_BUILD_DIR/vivado_ip_cache}"
mkdir -p "$VIVADO_IP_CACHE"

echo ""
_finn_gecho "FINN environment ready!"
echo "  FINN_ROOT=$FINN_ROOT"
echo "  FINN_BUILD_DIR=$FINN_BUILD_DIR"
if [ -n "$XILINX_VIVADO" ]; then
    echo "  XILINX_VIVADO=$XILINX_VIVADO"
fi
echo ""
