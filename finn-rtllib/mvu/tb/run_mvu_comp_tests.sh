#!/bin/bash
#
# Run MVU compressor integration tests (dotp_comp path).
# For each (MH, MW, PE, SIMD, WW, AW) configuration:
#   1. Generate comp_<sig>.sv via dotp_finn.py
#   2. Expand dotp_comp template and testbench
#   3. Run full mvu_vvu_axi simulation via XSim

((${KEEP_LOG:=0}))
((${MAX_WORKERS:=12}))

if ! command -v vivado >/dev/null 2>&1; then
	echo "ERROR: vivado not found in PATH." >&2
	exit 1
fi

echo "Vivado: $(command -v vivado)"
echo "Settings: KEEP_LOG=$KEEP_LOG MAX_WORKERS=$MAX_WORKERS"

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MVU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GEN_BASE="$SCRIPT_DIR/gen"
FINN_SRC="$(cd "$MVU_DIR/../../src" && pwd)"
export PYTHONPATH="$FINN_SRC${PYTHONPATH:+:$PYTHONPATH}"
: "${WORK_DIR:=${FINN_HOST_BUILD_DIR:-/tmp/finn_compressor_tests}}"

COMP_SRC_DIR="$FINN_SRC/finn/compressor/src"
echo "Compressor source: $COMP_SRC_DIR"

source "$FINN_SRC/finn/compressor/lib/test_common.sh"

# Test configs
TESTS=(
	"--mh 16 --mw  8 --pe 2 --simd 8 --ww 2 --aw 2 --accu_width 16 --signed_activations"
	"--mh 16 --mw 16 --pe 2 --simd 8 --ww 2 --aw 2 --accu_width 16 --signed_activations"
	"--mh 16 --mw  8 --pe 4 --simd 8 --ww 1 --aw 1 --accu_width 16"
	"--mh  8 --mw  8 --pe 2 --simd 4 --ww 3 --aw 3 --accu_width 16 --signed_activations"
)

function parse_config {
	local mh="" mw="" pe="" simd="" ww="" aw="" accu="" signed_act=""
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--mh)    mh="$2"; shift 2;;
			--mw)    mw="$2"; shift 2;;
			--pe)    pe="$2"; shift 2;;
			--simd)  simd="$2"; shift 2;;
			--ww)    ww="$2"; shift 2;;
			--aw)    aw="$2"; shift 2;;
			--accu_width) accu="$2"; shift 2;;
			--signed_activations) signed_act="_sa"; CFG_SIGNED_FLAG="--signed_activations"; shift;;
			*) shift;;
		esac
	done
	CFG_MH="$mh"; CFG_MW="$mw"; CFG_PE="$pe"; CFG_SIMD="$simd"
	CFG_WW="$ww"; CFG_AW="$aw"; CFG_ACCU="$accu"
	CFG_SIGNED_ACT="$([ -n "$CFG_SIGNED_FLAG" ] && echo 1 || echo 0)"
	CFG_LABEL="mh${mh}_mw${mw}_pe${pe}_simd${simd}_ww${ww}_aw${aw}${signed_act}"
}

function run_sim {
	local label="$1"
	local tcl="$GEN_BASE/$label/mvu_comp_${label}.tcl"
	local out="$GEN_BASE/$label/mvu_comp_${label}.runner.out"
	local log=(-nolog); [ "$KEEP_LOG" -gt 0 ] && log=(-log "$GEN_BASE/$label/sim.log")

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
	gen_dir="$GEN_BASE/$label"
	mkdir -p "$gen_dir"

	echo "  $label ..."

	# Generate compressor
	# shellcheck disable=SC2086
	gen_out=$(python3 -m finn.compressor.src.dotp_finn \
		--simd "$CFG_SIMD" --ww "$CFG_WW" --aw "$CFG_AW" \
		--accu_width "$CFG_ACCU" $CFG_SIGNED_FLAG \
		--dotp-template "$FINN_SRC/finn/compressor/hdl/dotp_comp_template.sv" \
		--dotp-output-name dotp_comp.sv \
		-o "$gen_dir" 2>&1)
	if [ $? -ne 0 ]; then
		echo "GENERATION FAILED: $gen_out" >&2; exit 1
	fi

	comp_name=$(echo "$gen_out" | sed -n 's/^ *Module name:[[:space:]]*//p' | head -n 1)
	comp_depth=$(echo "$gen_out" | sed -n 's/^ *Pipeline depth:[[:space:]]*//p' | head -n 1 | grep -Eo '[0-9]+' || true)
	echo "    comp_name=$comp_name  comp_depth=$comp_depth"

	# Expand TB
	sed -e "s/{mh}/$CFG_MH/g" -e "s/{mw}/$CFG_MW/g" \
	    -e "s/{pe}/$CFG_PE/g" -e "s/{simd}/$CFG_SIMD/g" \
	    -e "s/{ww}/$CFG_WW/g" -e "s/{aw}/$CFG_AW/g" \
	    -e "s/{accu_width}/$CFG_ACCU/g" -e "s/{signed_act}/$CFG_SIGNED_ACT/g" \
	    -e "s/{comp_depth}/$comp_depth/g" -e "s/{label}/$label/g" \
	    "$SCRIPT_DIR/mvu_comp_tb_template.sv" > "$gen_dir/mvu_comp_${label}_tb.sv"

	# Expand TCL
	sed -e "s|{label}|$label|g" -e "s|{mvu_dir}|$MVU_DIR|g" \
	    -e "s|{comp_dir}|$COMP_SRC_DIR|g" -e "s|{gen_dir}|$gen_dir|g" \
	    "$SCRIPT_DIR/mvu_comp_tb_template.tcl" > "$gen_dir/mvu_comp_${label}.tcl"
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
