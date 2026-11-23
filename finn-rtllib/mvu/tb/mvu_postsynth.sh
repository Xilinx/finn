#!/bin/bash
# Post-synthesis simulation of mvu_vvu_axi (DSP58) with mvu_postsynth_tb.sv.
#   1. synthesizes mvu_vvu_axi out of context and writes a functional netlist,
#   2. simulates the netlist with xsim and checks the outputs against the integer matrix product.
#
# usage: mvu_postsynth.sh [WORK_DIR]            (default: ./mvu_postsynth_work)
#   RTL_ROOT=<dir>   finn-rtllib directory to take the RTL from (default: the one next to this script)
#   PART=<part>      target part (default: VCK190, xcvc1902-vsva2197-2MP-e-S)
# Exit code is 0 only if the testbench prints "TEST PASSED".
set -eo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
RTL="$(readlink -f "${RTL_ROOT:-$HERE/../..}")"
PART="${PART:-xcvc1902-vsva2197-2MP-e-S}"
WORK="$(mkdir -p "${1:-mvu_postsynth_work}" && cd "${1:-mvu_postsynth_work}" && pwd)"

for t in vivado xvlog xelab xsim; do
	command -v "$t" > /dev/null || { echo "ERROR: $t not on PATH" >&2; exit 1; }
done
VIVADO_DIR="$(dirname "$(dirname "$(readlink -f "$(command -v xvlog)")")")"

SOURCES=(
	"$RTL/mvu/mvu_pkg.sv"
	"$RTL/mvu/add_multi.sv"
	"$RTL/skid/skid.sv"
	"$RTL/input_gen/hdl/input_gen.sv"
	"$RTL/mvu/replay_buffer.sv"
	"$RTL/mvu/weights_buff_tile.sv"
	"$RTL/mvu/mvu_vvu_8sx9_dsp58.sv"
	"$RTL/mvu/mvu.sv"
	"$RTL/mvu/mvu_vvu_axi.sv"
)

cd "$WORK"
rm -rf xsim.dir netlist.v

# Generics must match the localparams in mvu_postsynth_tb.sv (SEGMENTLEN only matters here)
cat > synth.tcl << EOF
read_verilog -sv {${SOURCES[*]}}
synth_design -top mvu_vvu_axi -part $PART -mode out_of_context \\
	-generic IS_MVU=1 -generic VERSION=3 -generic MW=32 -generic MH=4 -generic PE=2 -generic SIMD=8 \\
	-generic SEGMENTLEN=3 -generic ACTIVATION_WIDTH=6 -generic WEIGHT_WIDTH=4 -generic ACCU_WIDTH=18 \\
	-generic NARROW_WEIGHTS=1 -generic SIGNED_ACTIVATIONS=1 -generic TH=1 -generic FORCE_BEHAVIORAL=0
write_verilog -mode funcsim -force netlist.v
EOF

XILINX_LOCAL_USER_DATA=no vivado -mode batch -source synth.tcl -nojournal -log synth.log > /dev/null
xvlog -sv netlist.v "$HERE/mvu_postsynth_tb.sv" > xvlog.out
xvlog "$VIVADO_DIR/data/verilog/src/glbl.v" >> xvlog.out
xelab -timescale 1ns/1ps -L unisims_ver -L unimacro_ver -L secureip mvu_postsynth_tb glbl -s sim > xelab.out
# report the log even if xsim exits non-zero; the grep below decides
xsim sim -runall -log sim.log > /dev/null || true

grep -E "MISMATCH|TEST (PASSED|FAILED)" sim.log | head -20
grep -q "TEST PASSED" sim.log
