#!/bin/bash
#############################################################################
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief    Test runner for add_multi compressor verification
# @author    Simon Gerber <simon.gerber@amd.com>
#############################################################################

# Usage: ./run_add_multi_comp_tests.sh [target]
#   target: versal, 7series, ultrascale (default: versal)

((${KEEP_LOG:=0}))
((${MAX_WORKERS:=12}))

# Parse target argument
TARGET="${1:-versal}"

if ! command -v vivado >/dev/null 2>&1; then
	echo "ERROR: vivado not found in PATH." >&2
	exit 1
fi

echo "Vivado: $(command -v vivado)"
echo "Settings: KEEP_LOG=$KEEP_LOG MAX_WORKERS=$MAX_WORKERS"
echo "Target: $TARGET"

# Paths (all absolute for portability)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HDL_DIR="$SCRIPT_DIR/hdl"
GEN_DIR="$SCRIPT_DIR/gen"
FINN_SRC="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$FINN_SRC/src${PYTHONPATH:+:$PYTHONPATH}"

# Vivado working directory (isolated temp, unique per invocation)
WORK_DIR="/tmp/finn_compressor_tests_$$"

source "$SCRIPT_DIR/lib/test_common.sh"

# Test configs: N ARG_WIDTH PIPELINE_EVERY
# Format: "N W P" where P is pipeline_every (0 = no pipelining)
TESTS=(
	"8  4  0"
	"8  4  2"
	"16 3  0"
	"16 6  2"
	"32 6  2"
	"32 16 2"
	"47 5  2"
	"56 8  2"
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
	local n=$1 w=$2 p=$3
	local label="n${n}_w${w}"
	[ "$p" -ne 0 ] && label="${label}_p${p}"
	echo "$label"
}

function run_sim {
	local label="$1"
	local work="$WORK_DIR/$label"
	local tcl="$GEN_DIR/$label/add_multi_comp_${label}.tcl"
	local out="$GEN_DIR/$label/add_multi_comp_${label}.runner.out"
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
	read -r n w p <<< "$test"
	label=$(make_label "$n" "$w" "$p")
	LABELS+=("$label")
	gen_dir="$GEN_DIR/$label"
	mkdir -p "$gen_dir"

	echo "  $label ..."

	# Build target flag (Versal is default, no flag needed)
	target_flag=""
	[[ "$TARGET" == "7series" ]] && target_flag="--target 7-Series"
	[[ "$TARGET" == "ultrascale" ]] && target_flag="--target UltraScale"

	# Build pipeline flag
	pipeline_flag=""
	[ "$p" -ne 0 ] && pipeline_flag="-p $p"

	# Generate compressor
	# shellcheck disable=SC2086
	if ! gen_out=$(python3 -m finn.compressor.src.add_multi_finn \
		--n "$n" --arg_width "$w" $pipeline_flag $target_flag \
		-o "$gen_dir" 2>&1); then
		echo "GENERATION FAILED: $gen_out" >&2; exit 1
	fi

	comp_name=$(echo "$gen_out" | sed -n 's/^ *Module name:[[:space:]]*//p' | head -n 1)
	comp_depth=$(echo "$gen_out" | sed -n 's/^ *Pipeline depth:[[:space:]]*//p' | head -n 1 | grep -Eo '[0-9]+' || true)
	[ -z "$comp_name" ] && { echo "ERROR: No module name for $label" >&2; exit 1; }
	[ -z "$comp_depth" ] && { echo "ERROR: No depth for $label" >&2; exit 1; }

	# Expand TB
	sed -e "s/{n}/$n/g" -e "s/{arg_width}/$w/g" \
	    -e "s/{depth}/$comp_depth/g" -e "s/{label}/$label/g" \
	    -e "s/{comp_module}/$comp_name/g" \
	    "$HDL_DIR/add_multi_comp_tb_template.sv" > "$gen_dir/add_multi_comp_${label}_tb.sv"

	# Expand TCL
	sed -e "s|{label}|$label|g" -e "s|{tb}|add_multi_comp_${label}_tb|g" \
	    -e "s|{gen_dir}|$gen_dir|g" -e "s|{part}|$FPGA_PART|g" \
	    "$HDL_DIR/add_multi_comp_template.tcl" > "$gen_dir/add_multi_comp_${label}.tcl"
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
