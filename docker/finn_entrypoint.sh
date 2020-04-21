#!/bin/bash
set -e

export XILINX_VIVADO=$VIVADO_PATH
export SHELL=/bin/bash
export FINN_ROOT=/workspace/finn

# source Vivado env.vars
source $VIVADO_PATH/settings64.sh

exec "$@"
