#!/bin/bash
set -e

export XILINX_RPNN_ROOT=$FINN_ROOT/backend/fpga
RPNN_PATH=$XILINX_RPNN_ROOT/example

NETWORKS=$(ls $RPNN_PATH | cut -f1 -d'_' | tr "\n" " ")
PRECISIONS=$(ls $RPNN_PATH | cut -f2 -d'_' | tr "\n" " ")
MODES=$(ls $RPNN_PATH | cut -f3 -d'_' | tr "\n" " ")

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <network> <precision> <mode>" >&2
  echo "where <network> = network name (available $NETWORKS)" >&2
  echo "where <precision> = NN precision (available $PRECISIONS)" >&2
  echo "where <mode> = target donut (available $MODES sdx)" >&2
  exit 1
fi


NETWORK=$1
PRECISION=$2
COMPILE_MODE=$3

if [ -z "$XILINX_RPNN_ROOT" ]; then
    echo "Need to set XILINX_RPNN_ROOT"
    echo "It should be set to the FPGA backend root"
    exit 1
fi  


COMPILE_DIR=$XILINX_RPNN_ROOT/scripts/

OLD_DIR=$(pwd)

if [[ ("$PRECISION" == "rpnn") ]]; then
	cd $COMPILE_DIR
	./compile_rpnn.sh ${NETWORK} $COMPILE_MODE
	cd $OLD_DIR
elif [[ ("$PRECISION" == "bnn") ]]; then
	cd $COMPILE_DIR
	./compile_bnn.sh ${NETWORK} $COMPILE_MODE
	cd $OLD_DIR		
else
	echo "Precision $PRECISION not supported"
	exit 1
fi

echo "Done!"
