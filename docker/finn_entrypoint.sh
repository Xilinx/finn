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
FINN_BASE_COMMIT=ac0b86a63eb937b869bfa453a996a8a8b8506546
FINN_EXP_COMMIT=e9f97dcdb4db2f889b0f36af079a6a1792b7d4de
BREVITAS_COMMIT=d7ded80fa9557da2998ea310669edee7fb2d9526
CNPY_COMMIT=4e8810b1a8637695171ed346ce68f6984e585ef4
HLSLIB_COMMIT=4d74baefa79df48b5a0348d63f39a26df075de51
PYVERILATOR_COMMIT=e2ff74030de3992dcac54bf1b6aad2915946e8cb
OMX_COMMIT=1bae737669901e762f581af73348332b5c4b2ada

gecho "Setting up known-good commit versions for FINN dependencies"
# finn-base
gecho "finn-base @ $FINN_BASE_COMMIT"
git -C /workspace/finn-base pull --quiet
git -C /workspace/finn-base checkout $FINN_BASE_COMMIT --quiet
pip install --user -e /workspace/finn-base
# finn-experimental
gecho "finn-experimental @ $FINN_EXP_COMMIT"
git -C /workspace/finn-experimental pull --quiet
git -C /workspace/finn-experimental checkout $FINN_EXP_COMMIT --quiet
pip install --user -e /workspace/finn-experimental
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
pip install --user -e /workspace/pyverilator
# oh-my-xilinx
gecho "oh-my-xilinx @ $OMX_COMMIT"
git -C /workspace/oh-my-xilinx pull --quiet
git -C /workspace/oh-my-xilinx checkout $OMX_COMMIT --quiet
# remove old version egg-info, if any
rm -rf $FINN_ROOT/src/FINN.egg-info
# run pip install for finn
pip install --user -e $FINN_ROOT

if [ ! -z "$VIVADO_PATH" ];then
  # source Vivado env.vars
  export XILINX_VIVADO=$VIVADO_PATH
  source $VIVADO_PATH/settings64.sh
fi

# download PYNQ board files if not already there
if [ ! -d "/workspace/finn/board_files" ]; then
    gecho "Downloading PYNQ board files for Vivado"
    OLD_PWD=$(pwd)
    cd /workspace/finn
    wget -q https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip
    wget -q https://d2m32eurp10079.cloudfront.net/Download/pynq-z2.zip
    unzip -q pynq-z1.zip
    unzip -q pynq-z2.zip
    mkdir /workspace/finn/board_files
    mv pynq-z1/ board_files/
    mv pynq-z2/ board_files/
    rm pynq-z1.zip
    rm pynq-z2.zip
    cd $OLD_PWD
fi
if [ ! -d "/workspace/finn/board_files/ultra96v2" ]; then
    gecho "Downloading Avnet BDF files into board_files"
    OLD_PWD=$(pwd)
    cd /workspace/finn
    git clone https://github.com/Avnet/bdf.git
    mv /workspace/finn/bdf/* /workspace/finn/board_files/
    rm -rf /workspace/finn/bdf
    cd $OLD_PWD
fi
if [ ! -z "$VITIS_PATH" ];then
  # source Vitis env.vars
  export XILINX_VITIS=$VITIS_PATH
  source $VITIS_PATH/settings64.sh
  if [ ! -z "$XILINX_XRT" ];then
    # source XRT
    source $XILINX_XRT/setup.sh
  fi
fi
exec "$@"
