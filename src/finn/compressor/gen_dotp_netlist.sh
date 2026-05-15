#!/bin/bash
#############################################################################
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Generate standalone dotp compressor netlist for inspection or integration.
# Output is a self-contained RTL directory that can be simulated or synthesized.
#
# Usage: Edit parameters below, then run: ./gen_dotp_netlist.sh
#############################################################################

# === Configuration ===
SIMD=256
WW=4
AW=4
ACCU_WIDTH=16
SIGNED_WEIGHTS=0      # 0=unsigned, 1=signed
SIGNED_ACT=0          # 0=unsigned, 1=signed
TARGET="Versal"       # Versal, 7-Series, UltraScale
# =====================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$SCRIPT_DIR/../../.." && pwd):${PYTHONPATH:-}"

# Build output directory name from config
LABEL="simd${SIMD}_w${WW}_a${AW}"
[ "$SIGNED_WEIGHTS" -eq 0 ] && LABEL="${LABEL}_uw"
[ "$SIGNED_ACT" -eq 1 ] && LABEL="${LABEL}_sa"
LABEL="${LABEL}_$(echo "$TARGET" | tr '[:upper:]' '[:lower:]' | tr -d '-')"
OUT_DIR="$SCRIPT_DIR/gen/$LABEL"
mkdir -p "$OUT_DIR"

echo "Generating dotp compressor netlist"
echo "  Config: SIMD=$SIMD, WW=$WW, AW=$AW, ACCU=$ACCU_WIDTH"
echo "  Target: $TARGET"
echo "  Output: $OUT_DIR"
echo ""

# Build flags
FLAGS=""
[ "$SIGNED_WEIGHTS" -eq 0 ] && FLAGS="--unsigned_weights"
[ "$SIGNED_ACT" -eq 1 ] && FLAGS="$FLAGS --signed_activations"

# Generate compressor core and dotp wrapper
python3 -m finn.compressor.src.dotp_finn \
    --simd "$SIMD" --ww "$WW" --aw "$AW" \
    --accu_width "$ACCU_WIDTH" $FLAGS \
    --target "$TARGET" \
    --dotp-template "$SCRIPT_DIR/hdl/dotp_comp_template.sv" \
    --dotp-output-name dotp_comp.sv \
    -o "$OUT_DIR"

# Include mul_comp_map for complete netlist
cp "$SCRIPT_DIR/hdl/mul_comp_map.sv" "$OUT_DIR/"

echo ""
echo "Generated files:"
ls -1 "$OUT_DIR"/*.sv
echo ""
echo "Done. Netlist ready in: $OUT_DIR"
