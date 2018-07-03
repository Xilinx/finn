#!/bin/sh

GENDIR=$(pwd)
SYN_TCL="$FINN_ROOT/backend/fpga/scripts/make-pynq-vivado-proj.tcl"
XDC_DIR="$FINN_ROOT/backend/fpga/misc"
JAM_REPO="$GENDIR/hls_syn/sol1/impl/ip"
PROJ_NAME="finnaccel"
PROJ_DIR="$GENDIR/finnaccel"

# first, run HLS synthesis
vivado_hls -f hls_syn.tcl
# now run bitstream generation
vivado -mode batch -source $SYN_TCL -tclargs $JAM_REPO $PROJ_NAME $PROJ_DIR $XDC_DIR
# copy resulting bitfile
cp $PROJ_DIR/$PROJ_NAME.runs/impl_1/procsys_wrapper.bit $GENDIR/finnaccel.bit
