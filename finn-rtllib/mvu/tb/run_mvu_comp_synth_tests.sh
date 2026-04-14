#!/bin/bash
#
# Run MVU compressor synthesis tests.
# For each configuration:
#   1. Generate comp_<sig>.sv via dotp_finn.py
#   2. Expand dotp_comp template
#   3. Run Vivado synthesis and report utilization/timing
#
# Prerequisites:
#   - Vivado on PATH
#   - compressor-python source (COMP_SRC_DIR)

if ! command -v vivado >/dev/null 2>&1; then
	echo "ERROR: vivado not found in PATH." >&2
	echo "  Source Vivado settings first, e.g. settings64.sh." >&2
	exit 1
fi

echo "Vivado: $(command -v vivado)"
echo "Vivado version: $(vivado -version | head -n 1)"

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MVU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GEN_BASE="$SCRIPT_DIR/gen"
: "${WORK_DIR:=${FINN_BUILD_DIR:-/tmp}/finn_synth_tests}"

# Compressor source directory and PYTHONPATH
FINN_SRC="$(cd "$MVU_DIR/../../src" && pwd)"
export PYTHONPATH="$FINN_SRC${PYTHONPATH:+:$PYTHONPATH}"

COMP_SRC_DIR="$FINN_SRC/finn/compressor/src"
if [ ! -f "$COMP_SRC_DIR/dotp_finn.py" ]; then
	echo "ERROR: Cannot find compressor source." >&2
	echo "  Expected at $COMP_SRC_DIR/" >&2
	exit 1
fi
echo "Compressor source: $COMP_SRC_DIR"

# Synthesis configs — same as simulation configs
TESTS=(
	"--mh 16 --mw  8 --pe 2 --simd 8 --ww 2 --aw 2 --accu_width 16 --signed_activations"
	"--mh 16 --mw  8 --pe 4 --simd 8 --ww 1 --aw 1 --accu_width 16"
)

function parse_config {
	local  mh="" mw="" pe="" simd="" ww="" aw="" accu="" signed_act="" signed_flag=""
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--mh)    mh="$2"; shift 2;;
			--mw)    mw="$2"; shift 2;;
			--pe)    pe="$2"; shift 2;;
			--simd)  simd="$2"; shift 2;;
			--ww)    ww="$2"; shift 2;;
			--aw)    aw="$2"; shift 2;;
			--accu_width) accu="$2"; shift 2;;
			--signed_activations) signed_act="_sa"; signed_flag="--signed_activations"; shift;;
			*) shift;;
		esac
	done
	CFG_MH="$mh"
	CFG_MW="$mw"
	CFG_PE="$pe"
	CFG_SIMD="$simd"
	CFG_WW="$ww"
	CFG_AW="$aw"
	CFG_ACCU="$accu"
	CFG_SIGNED_FLAG="$signed_flag"
	CFG_SIGNED_ACT="$([ -n "$signed_flag" ] && echo 1 || echo 0)"
	CFG_LABEL="mh${mh}_mw${mw}_pe${pe}_simd${simd}_ww${ww}_aw${aw}${signed_act}"
}

overall=0
for i in "${!TESTS[@]}"; do
	args="${TESTS[$i]}"
	# shellcheck disable=SC2086
	parse_config $args
	label="$CFG_LABEL"

	gen_dir="$GEN_BASE/$label"
	mkdir -p "$gen_dir"

	echo "=== $label ==="

	# Generate compressor core
	# Run from compressor source dir so bare imports resolve correctly.
	# shellcheck disable=SC2086
	gen_out=$(python3 -m finn.compressor.src.dotp_finn \
		--simd "$CFG_SIMD" --ww "$CFG_WW" --aw "$CFG_AW" \
		--accu_width "$CFG_ACCU" $CFG_SIGNED_FLAG \
		--dotp-template "$FINN_SRC/finn/compressor/hdl/dotp_comp_template.sv" \
		--dotp-output-name dotp_comp.sv \
		-o "$gen_dir" 2>&1)
	if [ $? -ne 0 ]; then
		echo "GENERATION FAILED for $label:" >&2
		echo "$gen_out" >&2
		overall=1
		continue
	fi

	comp_depth=$(echo "$gen_out" | sed -n 's/^ *Pipeline depth:[[:space:]]*//p' | head -n 1 | grep -Eo '[0-9]+' || true)
	if [ -z "$comp_depth" ]; then comp_depth=1; fi

	# Expand synthesis TCL template
	sed -e "s|{label}|$label|g" \
	    -e "s|{mvu_dir}|$MVU_DIR|g" \
	    -e "s|{comp_dir}|$FINN_SRC|g" \
	    -e "s|{gen_dir}|$gen_dir|g" \
	    -e "s|{mh}|$CFG_MH|g" \
	    -e "s|{mw}|$CFG_MW|g" \
	    -e "s|{pe}|$CFG_PE|g" \
	    -e "s|{simd}|$CFG_SIMD|g" \
	    -e "s|{ww}|$CFG_WW|g" \
	    -e "s|{aw}|$CFG_AW|g" \
	    -e "s|{accu_width}|$CFG_ACCU|g" \
	    -e "s|{signed_act}|$CFG_SIGNED_ACT|g" \
	    -e "s|{comp_depth}|$comp_depth|g" \
	    "$SCRIPT_DIR/mvu_comp_synth_tb_template.tcl" > "$gen_dir/mvu_comp_synth_${label}.tcl"

	# Run Vivado synthesis
	out="$gen_dir/mvu_comp_synth_${label}.out"
	mkdir -p "$WORK_DIR"
	if (cd "$WORK_DIR" && vivado -nolog -nojournal -mode batch -source "$gen_dir/mvu_comp_synth_${label}.tcl" >"$out" 2>&1); then
		echo "  PASS — see $gen_dir for utilization/timing reports"
	else
		echo "  FAIL — see $out"
		overall=1
	fi
	echo
done

exit "$overall"
