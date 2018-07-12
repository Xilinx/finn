#!/bin/sh

GENDIR=$(pwd)
MLBP="$XILINX_BNN_ROOT/mlbp"
VERILATOR_TARGET="$GENDIR/verilator-flow"
HLS_VERILOG_DIR="$GENDIR/hls_syn/sol1/syn/verilog"
FINN_FPGA_SIMTOOLS="$FINN_ROOT/backend/fpga/simtools"
#DONUT_WRAPPERS="$FINN_FPGA_SIMTOOLS/sim-ondevice-mlbp.cpp $FINN_FPGA_SIMTOOLS/cnpy.cpp"
DRIVER="$VERILATOR_TARGET/sw-driver"
INCLUDES="-I$DRIVER -I$VERILATOR_TARGET/verilated -I$VERILATOR_ROOT/include"
EMU_LIBDIR="$VERILATOR_TARGET/output"

# clean any old files
rm -rf $VERILATOR_TARGET
# set up sources
cp -r "$MLBP/verilator-flow" $VERILATOR_TARGET
cp -r "$HLS_VERILOG_DIR" "$VERILATOR_TARGET/jam-verilog"
cd $VERILATOR_TARGET
# build emulation library
./update-hardware.sh

# build emulation executable
g++ -std=c++11 $DRIVER/*.cpp $INCLUDES -L$EMU_LIBDIR -lmlbpaccel-virtual -o $EMU_LIBDIR/emu # $DONUT_WRAPPERS
