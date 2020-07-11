#!/bin/bash

export SHELL=/bin/bash
export FINN_ROOT=/workspace/finn

GREEN='\033[0;32m'
NC='\033[0m' # No Color

gecho () {
  echo -e "${GREEN}$1${NC}"
}

# checkout the correct dependency repo commits
# the repos themselves are cloned in the Dockerfile
BREVITAS_COMMIT=172e423164402a07826877fa9730063bee10a208
CNPY_COMMIT=4e8810b1a8637695171ed346ce68f6984e585ef4
HLSLIB_COMMIT=cfafe11a93b79ab1af7529d68f08886913a6466e
PYVERILATOR_COMMIT=c97a5ba41bbc7c419d6f25c74cdf3bdc3393174f
PYNQSHELL_COMMIT=0c82a61b0ec1a07fa275a14146233824ded7a13d
OMX_COMMIT=1bae737669901e762f581af73348332b5c4b2ada


gecho "Setting up known-good commit versions for FINN dependencies"
# Brevitas
gecho "brevitas @ $BREVITAS_COMMIT"
git -C /workspace/brevitas pull --quiet
git -C /workspace/brevitas checkout $BREVITAS_COMMIT --quiet
pip install --user -e /workspace/brevitas
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
# oh-my-xilinx
gecho "oh-my-xilinx @ $OMX_COMMIT"
git -C /workspace/oh-my-xilinx pull --quiet
git -C /workspace/oh-my-xilinx checkout $OMX_COMMIT --quiet

if [ ! -z "$VIVADO_PATH" ];then
  # source Vivado env.vars
  export XILINX_VIVADO=$VIVADO_PATH
  source $VIVADO_PATH/settings64.sh
fi
if [ ! -z "$VITIS_PATH" ];then
  # source Vitis env.vars
  export XILINX_VITIS=$VITIS_PATH
  source $VITIS_PATH/settings64.sh
fi
exec "$@"
