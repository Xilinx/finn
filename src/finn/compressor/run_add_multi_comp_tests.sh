#!/bin/bash
#
# Run standalone add_multi compressor tests.
# For each (N, ARG_WIDTH) configuration:
#   1. Generate comp_NuW_dD.sv via add_multi_finn.py
#   2. Expand TB and TCL templates
#   3. Run XSim via Vivado
#
# Usage: ./run_add_multi_comp_tests.sh [versal|7series]
# Prerequisites: Vivado on PATH

((${KEEP_LOG:=0}))
((${MAX_WORKERS:=12}))
TARGET="${1:-versal}"  # Default to versal

if ! command -v vivado >/dev/null 2>&1; then
	echo "ERROR: vivado not found in PATH." >&2
	exit 1
fi

echo "Vivado: $(command -v vivado)"
echo "Settings: KEEP_LOG=$KEEP_LOG MAX_WORKERS=$MAX_WORKERS"
echo "Target: $TARGET"

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HDL_DIR="$SCRIPT_DIR/hdl"
GEN_BASE="$SCRIPT_DIR/gen"
FINN_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$FINN_SRC${PYTHONPATH:+:$PYTHONPATH}"
: "${WORK_DIR:=${FINN_HOST_BUILD_DIR:-/tmp/finn_compressor_tests}}"

source "$SCRIPT_DIR/lib/test_common.sh"

# Test configs: --n N --arg_width W [-p pipeline_every]
TESTS=(
	"--n 8  --arg_width 4"
	"--n 8  --arg_width 4  -p 2"
	"--n 16 --arg_width 3"
	"--n 16 --arg_width 6  -p 2"
	"--n 32 --arg_width 6  -p 2"
	"--n 32 --arg_width 16 -p 2"
	"--n 47 --arg_width 5  -p 2"
	"--n 56 --arg_width 8  -p 2"
)

function parse_config {
	local n="" w="" p=""
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--n)         n="$2"; shift 2;;
			--arg_width) w="$2"; shift 2;;
			-p)          p="$2"; CFG_P_FLAG="-p $2"; shift 2;;
			*)           shift;;
		esac
	done
	CFG_N="$n"; CFG_W="$w"
	CFG_LABEL="n${n}_w${w}"; [ -n "$p" ] && CFG_LABEL="${CFG_LABEL}_p${p}"
	# Set FPGA part based on TARGET variable
	if [[ "$TARGET" == "7series" ]]; then
		CFG_PART="xc7z020clg400-1"  # Pynq-Z1
	else
		CFG_PART="xcvc1902-vsva2197-2MP-e-S"  # Versal VCK190
	fi
}

function run_sim {
	local label="$1"
	local tcl="$GEN_BASE/$label/add_multi_comp_${label}.tcl"
	local out="$GEN_BASE/$label/add_multi_comp_${label}.runner.out"
	local log=(-nolog); [ "$KEEP_LOG" -gt 0 ] && log=(-log "$GEN_BASE/$label/sim.log")

	vivado "${log[@]}" -nojournal -mode batch -source "$tcl" >"$out" 2>&1
	check_vivado_errors "$out" "$label"
	exit $?
}

# Phase 1: Generate
LABELS=()
echo -e "Generating configs:\n"
for args in "${TESTS[@]}"; do
	CFG_P_FLAG=""
	# shellcheck disable=SC2086
	parse_config $args
	label="$CFG_LABEL"
	LABELS+=("$label")
	gen_dir="$GEN_BASE/$label"
	mkdir -p "$gen_dir"

	echo "  $label ..."

	# Generate compressor
	# shellcheck disable=SC2086
	if ! gen_out=$(python3 -m finn.compressor.src.add_multi_finn \
		--n "$CFG_N" --arg_width "$CFG_W" $CFG_P_FLAG -o "$gen_dir" 2>&1); then
		echo "GENERATION FAILED: $gen_out" >&2; exit 1
	fi

	comp_name=$(echo "$gen_out" | sed -n 's/^ *Module name:[[:space:]]*//p' | head -n 1)
	comp_depth=$(echo "$gen_out" | sed -n 's/^ *Pipeline depth:[[:space:]]*//p' | head -n 1 | grep -Eo '[0-9]+' || true)
	[ -z "$comp_name" ] && { echo "ERROR: No module name for $label" >&2; exit 1; }
	[ -z "$comp_depth" ] && { echo "ERROR: No depth for $label" >&2; exit 1; }

	# Expand TB
	sed -e "s/{n}/$CFG_N/g" -e "s/{arg_width}/$CFG_W/g" \
	    -e "s/{depth}/$comp_depth/g" -e "s/{label}/$label/g" \
	    -e "s/{comp_module}/$comp_name/g" \
	    "$HDL_DIR/add_multi_comp_tb_template.sv" > "$gen_dir/add_multi_comp_${label}_tb.sv"

	# Expand TCL
	sed -e "s|{label}|$label|g" -e "s|{tb}|add_multi_comp_${label}_tb|g" \
	    -e "s|{gen_dir}|$gen_dir|g" -e "s|{part}|$CFG_PART|g" \
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
