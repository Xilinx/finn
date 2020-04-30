#!/bin/bash

export XILINX_VIVADO=$VIVADO_PATH
export SHELL=/bin/bash
export FINN_ROOT=/workspace/finn

# checkout the correct dependency repo commits
# the repos themselves are cloned in the Dockerfile
# Brevitas
git -C /workspace/brevitas checkout 215cf44c76d562339fca368c8c3afee3110033e8
# Brevitas examples
git -C /workspace/brevitas_cnv_lfc checkout 2059f96bd576bf71f32c757e7f92617a70190c90
# CNPY
git -C /workspace/cnpy checkout 4e8810b1a8637695171ed346ce68f6984e585ef4
# FINN hlslib
git -C /workspace/finn-hlslib checkout b139bf051ac8f8e0a3625509247f714127cf3317
# PyVerilator
git -C /workspace/pyverilator checkout 307fc5c82db748620836307a2002fdc9fe170226
# PYNQ-HelloWorld
git -C /workspace/PYNQ-HelloWorld checkout db7e418767ce2a8e08fe732ddb3aa56ee79b7560


# source Vivado env.vars
source $VIVADO_PATH/settings64.sh

exec "$@"
