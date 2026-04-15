#!/bin/bash
#
# Run dotp_comp integration tests for multiple configurations.
# Uses dotp_finn.py to generate the compressor core (comp.sv),
# then instantiates it from the static dotp_comp template via XSim.
#
# Usage: ./run_dotp_comp_tests.sh [versal|7series]

((${KEEP_LOG:=0}))
((${MAX_WORKERS:=12}))
TARGET="${1:-versal}"  # Default to versal

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
FINN_SRC="$(cd "$SRC_DIR/../.." && pwd)"
export PYTHONPATH="$FINN_SRC${PYTHONPATH:+:$PYTHONPATH}"
: "${WORK_DIR:=${FINN_HOST_BUILD_DIR:-/tmp/finn_compressor_tests}}"

if ! command -v vivado >/dev/null 2>&1; then
	echo "ERROR: vivado not found in PATH." >&2
	exit 1
fi

echo "Vivado: $(command -v vivado)"
echo "Settings: KEEP_LOG=$KEEP_LOG MAX_WORKERS=$MAX_WORKERS WORK_DIR=$WORK_DIR"
echo "Target: $TARGET"

source "$SRC_DIR/lib/test_common.sh"

# Test configs: --pe PE --simd SIMD --ww WW --aw AW --accu_width ACCU [--signed_activations]
# Target is set via script argument, applied to all tests
TESTS=(
	"--pe 2 --simd 8 --ww 1 --aw 1 --accu_width 16"
	"--pe 2 --simd 8 --ww 1 --aw 1 --accu_width 16 --signed_activations"
	"--pe 2 --simd 8 --ww 2 --aw 1 --accu_width 16"
	"--pe 2 --simd 8 --ww 2 --aw 2 --accu_width 16 --signed_activations"
	"--pe 2 --simd 4 --ww 2 --aw 2 --accu_width 16 --signed_activations"
	"--pe 2 --simd 16 --ww 2 --aw 2 --accu_width 16 --signed_activations"
	"--pe 1 --simd 8 --ww 2 --aw 2 --accu_width 16 --signed_activations"
	"--pe 4 --simd 8 --ww 2 --aw 2 --accu_width 16 --signed_activations"
)

function parse_config {
	local pe="" simd="" ww="" aw="" accu="" signed_act=""
	CFG_SIGNED_FLAG=""
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--pe)    pe="$2"; shift 2;;
			--simd)  simd="$2"; shift 2;;
			--ww)    ww="$2"; shift 2;;
			--aw)    aw="$2"; shift 2;;
			--accu_width) accu="$2"; shift 2;;
			--signed_activations) signed_act="_sa"; CFG_SIGNED_FLAG="--signed_activations"; shift;;
			*) shift;;
		esac
	done
	CFG_PE="$pe"; CFG_SIMD="$simd"; CFG_WW="$ww"; CFG_AW="$aw"; CFG_ACCU="$accu"
	CFG_LABEL="pe${pe}_simd${simd}_ww${ww}_aw${aw}_accu${accu}${signed_act}"
	# Sanitize label for SystemVerilog identifiers
	CFG_LABEL="${CFG_LABEL//-/_}"
	# Set FPGA part and target flag based on TARGET variable
	if [[ "$TARGET" == "7series" ]]; then
		CFG_PART="xc7z020clg400-1"  # Pynq-Z1
		CFG_TARGET_FLAG="--target 7-Series"
	else
		CFG_PART="xcvc1902-vsva2197-2MP-e-S"  # Versal VCK190
		CFG_TARGET_FLAG=""
	fi
}

function run_sim {
	local label="$1"
	local tcl="$SRC_DIR/gen/$label/dotp_comp_${label}.tcl"
	local out="$SRC_DIR/gen/$label/dotp_comp_${label}.runner.out"
	local log=(-nolog); [ "$KEEP_LOG" -gt 0 ] && log=(-log "$SRC_DIR/gen/$label/sim.log")

	mkdir -p "$WORK_DIR"
	(cd "$WORK_DIR" && vivado "${log[@]}" -nojournal -mode batch -source "$tcl" >"$out" 2>&1)
	check_vivado_errors "$out" "$label"
	exit $?
}

# Phase 1: Generate
LABELS=()
echo -e "Generating configs:\n"
for args in "${TESTS[@]}"; do
	CFG_SIGNED_FLAG=""
	# shellcheck disable=SC2086
	parse_config $args
	label="$CFG_LABEL"
	LABELS+=("$label")
	out_dir="gen/$label"
	mkdir -p "$out_dir"

	echo "  $label ..."

	# Generate compressor
	# shellcheck disable=SC2086
	gen_out=$(python3 -m finn.compressor.src.dotp_finn \
		--simd "$CFG_SIMD" --ww "$CFG_WW" --aw "$CFG_AW" \
		--accu_width "$CFG_ACCU" $CFG_SIGNED_FLAG $CFG_TARGET_FLAG \
		--dotp-template hdl/dotp_comp_template.sv \
		--dotp-output-name dotp_comp.sv \
		-o "$out_dir" 2>&1)
	if [ $? -ne 0 ]; then
		echo "GENERATION FAILED: $gen_out" >&2; exit 1
	fi

	comp_depth=$(echo "$gen_out" | sed -n 's/^ *Pipeline depth:[[:space:]]*//p' | head -n 1 | grep -Eo '[0-9]+' || true)
	[ -z "$comp_depth" ] && { echo "ERROR: No depth for $label" >&2; exit 1; }

	# Expand TB
	sed -e "s/{pe}/$CFG_PE/g" -e "s/{simd}/$CFG_SIMD/g" \
	    -e "s/{ww}/$CFG_WW/g" -e "s/{aw}/$CFG_AW/g" \
	    -e "s/{accu_width}/$CFG_ACCU/g" \
	    -e "s/{signed_act}/$([ -n "$CFG_SIGNED_FLAG" ] && echo 1 || echo 0)/g" \
	    -e "s/{full_sig}/$label/g" -e "s/{comp_depth}/$comp_depth/g" \
	    hdl/dotp_comp_tb_template.sv > "$out_dir/dotp_comp_${label}_tb.sv"

	# Expand TCL
	sed -e "s/{label}/$label/g" -e "s|{src_dir}|$SRC_DIR|g" -e "s/{part}/$CFG_PART/g" \
	    hdl/dotp_comp_template.tcl > "$out_dir/dotp_comp_${label}.tcl"
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
