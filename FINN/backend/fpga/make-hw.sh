#!/bin/bash
set -e

RPNN_PATH=$XILINX_RPNN_ROOT/example

NETWORKS=$(ls $RPNN_PATH | cut -f1 -d'_' | tr "\n" " ")
PRECISIONS=$(ls $RPNN_PATH | cut -f2 -d'_' | tr "\n" " ")
MODES=$(ls $RPNN_PATH | cut -f3 -d'_' | tr "\n" " ")

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <network> <precision> <target> <platform> <mode>" >&2
  echo "where <network> = network name (available $NETWORKS)" >&2
  echo "where <precision> = NN precision (available $PRECISIONS)" >&2
  echo "where <target> = target donut (available $MODES sdx)" >&2
  echo "where <platform> = ku115 zc706" >&2
  echo "where <mode> = generate (h)ls only" >&2
  exit 1
fi


NETWORK=$1
PRECISION=$2
if [[ ("$2" == "sdx") ]]; then
	TARGET_DONUT=axi-full
else
	TARGET_DONUT=$3
fi
PLATFORM=$4
MODE=$5
PATH_TO_VIVADO=$(which vivado)
PATH_TO_VIVADO_HLS=$(which vivado_hls)


if [ -z "$XILINX_RPNN_ROOT" ]; then
    echo "Need to set XILINX_RPNN_ROOT"
    exit 1
fi  

if [ -z "$PATH_TO_VIVADO" ]; then
    echo "vivado not found in path"
    exit 1
fi 

if [ -z "$PATH_TO_VIVADO_HLS" ]; then
    echo "vivado_hls not found in path"
    exit 1
fi 

export RPNN_LIBRARY=$XILINX_RPNN_ROOT/hls
HLS_DIR=$XILINX_RPNN_ROOT/example/${NETWORK}_${PRECISION}_$TARGET_DONUT/
if [[ ("$PRECISION" == "rpnn") ]]; then
	
	HLS_SCRIPT=$HLS_DIR/$NETWORK.tcl

	HLS_IP_REPO=$HLS_DIR/hls-syn/sol1/impl/ip

	# regenerate HLS jam if requested
	if [[ ("$MODE" == "h") ]]; then
	  OLD_DIR=$(pwd)
	  echo "Calling Vivado HLS for hardware synthesis..."
	  cd $HLS_DIR
	  vivado_hls -f $HLS_SCRIPT 
	  echo "HLS synthesis complete"
	  cd $OLD_DIR
	fi
elif [[ ("$PRECISION" == "bnn") ]]; then
	mkdir -p "$HLS_DIR/report"
	HLS_OUT_DIR="$HLS_DIR/$NETWORK"
	mkdir -p "$HLS_OUT_DIR"
	HLS_REPORT_PATH="$HLS_OUT_DIR/sol1/syn/report/BlackBoxJam_csynth.rpt"
	HLS_LOG_PATH="$HLS_OUT_DIR/sol1/sol1.log"
	REPORT_OUT_DIR="$HLS_DIR/report"
	OLDDIR=$(pwd)
	cd $HLS_OUT_DIR/..
	HLS_SCRIPT=$XILINX_RPNN_ROOT/scripts/hls-syn_bnn.tcl
	HLS_SRC_DIR=$HLS_DIR/hw/
	vivado_hls -f $HLS_SCRIPT -tclargs $NETWORK $HLS_SRC_DIR
	cat $HLS_REPORT_PATH | grep "Utilization Estimates" -A 20 > $REPORT_OUT_DIR/hls.txt
	echo "Non-II-1 components: " >> $REPORT_OUT_DIR/hls.txt
	cat $HLS_LOG_PATH | grep "Final II:" | grep -v "Final II: 1," >> $REPORT_OUT_DIR/hls.txt
	cat $REPORT_OUT_DIR/hls.txt
	echo "HLS synthesis complete"
	echo "HLS-generated IP is at $HLS_IP_REPO"
	cd $OLDDIR	
else
	echo "Precision $PRECISION not supported"
	exit 1
fi

echo "Done!"
