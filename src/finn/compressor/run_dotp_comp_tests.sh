#!/bin/bash
#############################################################################
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief    Test runner for dot product compressor verification
# @author    Simon Gerber <simon.gerber@amd.com>
#############################################################################

# Usage: ./run_dotp_comp_tests.sh [target]
#   target: versal, 7series, ultrascale (default: versal)

((${KEEP_LOG:=0}))
((${MAX_WORKERS:=12}))

# Parse target argument
TARGET="${1:-versal}"

# Paths (all absolute for portability)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HDL_DIR="$SCRIPT_DIR/hdl"
GEN_DIR="$SCRIPT_DIR/gen"
FINN_SRC="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# PYTHONPATH needs to point to where finn.compressor can be imported from (src/)
export PYTHONPATH="$FINN_SRC/src${PYTHONPATH:+:$PYTHONPATH}"

# Vivado working directory (isolated temp, unique per invocation)
WORK_DIR="/tmp/finn_compressor_tests_$$"

if ! command -v vivado >/dev/null 2>&1; then
	echo "ERROR: vivado not found in PATH." >&2
	exit 1
fi

echo "Vivado: $(command -v vivado)"
echo "Settings: KEEP_LOG=$KEEP_LOG MAX_WORKERS=$MAX_WORKERS"
echo "Target: $TARGET"

source "$SCRIPT_DIR/lib/test_common.sh"

# Test configs: PE SIMD WW AW ACCU SIGNED_ACT
# Format: "PE SIMD WW AW ACCU SIGNED" where SIGNED=1 for signed activations, 0 otherwise
# Target is set via script argument, applied to all tests
TESTS=(
	"2 8 1 1 16 0"
	"2 8 1 1 16 1"
	"2 8 2 1 16 0"
	"2 8 2 2 16 1"
	"2 4 2 2 16 1"
	"2 16 2 2 16 1"
	"1 8 2 2 16 1"
	"4 8 2 2 16 1"
)

# Set FPGA part based on TARGET variable
function get_fpga_part {
	if [[ "$TARGET" == "7series" ]]; then
		echo "xc7z020clg400-1"  # Pynq-Z1
	elif [[ "$TARGET" == "ultrascale" ]]; then
		echo "xczu9eg-ffvb1156-2-e"  # ZCU102
	else
		echo "xcvc1902-vsva2197-2MP-e-S"  # Versal VCK190
	fi
}

# Build label from config
function make_label {
	local pe=$1 simd=$2 ww=$3 aw=$4 accu=$5 signed=$6
	local label="pe${pe}_simd${simd}_ww${ww}_aw${aw}_accu${accu}"
	[ "$signed" -eq 1 ] && label="${label}_sa"
	echo "${label//-/_}"  # Sanitize for SystemVerilog
}

function run_sim {
	local label="$1"
	local work="$WORK_DIR/$label"
	local tcl="$GEN_DIR/$label/dotp_comp_${label}.tcl"
	local out="$GEN_DIR/$label/dotp_comp_${label}.runner.out"
	local log=(-nolog); [ "$KEEP_LOG" -gt 0 ] && log=(-log "$GEN_DIR/$label/sim.log")

	mkdir -p "$work"
	(cd "$work" && vivado "${log[@]}" -nojournal -mode batch -source "$tcl" >"$out" 2>&1)
	check_vivado_errors "$out" "$label"
	exit $?
}

# Phase 1: Generate
LABELS=()
FPGA_PART=$(get_fpga_part)
echo -e "Generating configs:\n"
for test in "${TESTS[@]}"; do
	read -r pe simd ww aw accu signed <<< "$test"
	label=$(make_label "$pe" "$simd" "$ww" "$aw" "$accu" "$signed")
	LABELS+=("$label")
	out_dir="$GEN_DIR/$label"
	mkdir -p "$out_dir"

	echo "  $label ..."

	# Build target flag (Versal is default, no flag needed)
	target_flag=""
	[[ "$TARGET" == "7series" ]] && target_flag="--target 7-Series"
	[[ "$TARGET" == "ultrascale" ]] && target_flag="--target UltraScale"

	# Build signed activations flag
	signed_flag=""
	[ "$signed" -eq 1 ] && signed_flag="--signed_activations"

	# Generate compressor
	# shellcheck disable=SC2086
	gen_out=$(python3 -m finn.compressor.src.dotp_finn \
		--simd "$simd" --ww "$ww" --aw "$aw" --accu_width "$accu" \
		$signed_flag $target_flag \
		--dotp-template hdl/dotp_comp_template.sv \
		--dotp-output-name dotp_comp.sv \
		-o "$out_dir" 2>&1)
	if [ $? -ne 0 ]; then
		echo "GENERATION FAILED: $gen_out" >&2; exit 1
	fi

	comp_depth=$(echo "$gen_out" | sed -n 's/^ *Pipeline depth:[[:space:]]*//p' | head -n 1 | grep -Eo '[0-9]+' || true)
	[ -z "$comp_depth" ] && { echo "ERROR: No depth for $label" >&2; exit 1; }

	# Extract dotp module name from generated file
	dotp_module=$(grep "^module" "$out_dir/dotp_comp.sv" | sed 's/module \([^ #]*\).*/\1/')
	[ -z "$dotp_module" ] && { echo "ERROR: No dotp module name for $label" >&2; exit 1; }

	# Expand TB
	sed -e "s/{pe}/$pe/g" -e "s/{simd}/$simd/g" \
	    -e "s/{ww}/$ww/g" -e "s/{aw}/$aw/g" \
	    -e "s/{accu_width}/$accu/g" \
	    -e "s/{signed_act}/$signed/g" \
	    -e "s/{full_sig}/$label/g" -e "s/{comp_depth}/$comp_depth/g" \
	    -e "s/{dotp_module}/$dotp_module/g" \
	    "$HDL_DIR/dotp_comp_tb_template.sv" > "$out_dir/dotp_comp_${label}_tb.sv"

	# Expand TCL
	sed -e "s/{label}/$label/g" -e "s|{src_dir}|$SCRIPT_DIR|g" -e "s/{part}/$FPGA_PART/g" \
	    "$HDL_DIR/dotp_comp_template.tcl" > "$out_dir/dotp_comp_${label}.tcl"
done
echo

# Phase 2: Simulate
echo -e "Running simulations with $MAX_WORKERS parallel workers:\n"
for label in "${LABELS[@]}"; do
	collect_workers $((MAX_WORKERS - 1))
	start_worker "$label" run_sim
done
collect_workers 0
echo

print_summary
exit $?
