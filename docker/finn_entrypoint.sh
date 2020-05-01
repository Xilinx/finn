#!/bin/bash

export XILINX_VIVADO=$VIVADO_PATH
export SHELL=/bin/bash
export FINN_ROOT=/workspace/finn

GREEN='\033[0;32m'
NC='\033[0m' # No Color

gecho () {
  echo -e "${GREEN}$1${NC}"
}

# checkout the correct dependency repo commits
# the repos themselves are cloned in the Dockerfile
BREVITAS_COMMIT=215cf44c76d562339fca368c8c3afee3110033e8
BREVITAS_EXAMPLES_COMMIT=2059f96bd576bf71f32c757e7f92617a70190c90
CNPY_COMMIT=4e8810b1a8637695171ed346ce68f6984e585ef4
HLSLIB_COMMIT=b139bf051ac8f8e0a3625509247f714127cf3317
PYVERILATOR_COMMIT=307fc5c82db748620836307a2002fdc9fe170226
PYNQSHELL_COMMIT=db7e418767ce2a8e08fe732ddb3aa56ee79b7560


gecho "Setting up known-good commit versions for FINN dependencies"
# Brevitas
gecho "brevitas @ $BREVITAS_COMMIT"
git -C /workspace/brevitas pull --quiet
git -C /workspace/brevitas checkout $BREVITAS_COMMIT --quiet
# Brevitas examples
gecho "brevitas_cnv_lfc @ $BREVITAS_EXAMPLES_COMMIT"
git -C /workspace/brevitas_cnv_lfc pull --quiet
git -C /workspace/brevitas_cnv_lfc checkout $BREVITAS_EXAMPLES_COMMIT --quiet
# CNPY
gecho "cnpy @ $CNPY_COMMIT"
git -C /workspace/cnpy pull --quiet
git -C /workspace/cnpy checkout $CNPY_COMMIT --quiet
# FINN hlslib
gecho "finn-hlslib @ $HLSLIB_COMMIT"
git -C /workspace/finn-hlslib pull --quiet
git -C /workspace/finn-hlslib checkout $HLSLIB_COMMIT --quiet
# PyVerilator
gecho "PyVerilator @ $PYVERILATOR_COMMIT"
git -C /workspace/pyverilator pull --quiet
git -C /workspace/pyverilator checkout $PYVERILATOR_COMMIT --quiet
# PYNQ-HelloWorld
gecho "PYNQ shell @ $PYNQSHELL_COMMIT"
git -C /workspace/PYNQ-HelloWorld pull --quiet
git -C /workspace/PYNQ-HelloWorld checkout $PYNQSHELL_COMMIT --quiet

# source Vivado env.vars
source $VIVADO_PATH/settings64.sh

exec "$@"
