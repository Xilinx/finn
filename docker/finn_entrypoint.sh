#!/bin/bash

export XILINX_VIVADO=$VIVADO_PATH
export XILINX_VITIS=$VITIS_PATH
export SHELL=/bin/bash
export FINN_ROOT=/workspace/finn

GREEN='\033[0;32m'
NC='\033[0m' # No Color

gecho () {
  echo -e "${GREEN}$1${NC}"
}

# checkout the correct dependency repo commits
# the repos themselves are cloned in the Dockerfile
BREVITAS_COMMIT=f9a27226d4acf1661dd38bc449f71f89e0983cce
CNPY_COMMIT=4e8810b1a8637695171ed346ce68f6984e585ef4
HLSLIB_COMMIT=8f9f2018762f654f196b666838aeaf6fc730ad9a
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

# source Vivado env.vars
source $VIVADO_PATH/settings64.sh

# source Vitis env.vars
source $VITIS_PATH/settings64.sh

exec "$@"
