#!/bin/bash
# Copyright (c) 2021, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


export FINN_ROOT=/workspace/finn
export HOME=/tmp/home_dir
export SHELL=/bin/bash
# colorful terminal output
export PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

yecho () {
  echo -e "${YELLOW}WARNING: $1${NC}"
}

gecho () {
  echo -e "${GREEN}$1${NC}"
}

recho () {
  echo -e "${RED}ERROR: $1${NC}"
}

if [ -f "$FINN_ROOT/setup.py" ];then
  # run pip install for finn
  pip install --user -e $FINN_ROOT
else
  recho "Unable to find FINN source code in /workspace/finn"
  recho "Ensure you have passed -v <path-to-finn-repo>:/workspace/finn to the docker run command"
  exit -1
fi

if [ -f "$VITIS_PATH/settings64.sh" ];then
  # source Vitis env.vars
  export XILINX_VITIS=$VITIS_PATH
  export XILINX_XRT=/opt/xilinx/xrt
  source $VITIS_PATH/settings64.sh
  gecho "Found Vitis at $VITIS_PATH"
  if [ -f "$XILINX_XRT/setup.sh" ];then
    # source XRT
    source $XILINX_XRT/setup.sh
    gecho "Found XRT at $XILINX_XRT"
  else
    recho "XRT not found on $XILINX_XRT, did the installation fail?"
    exit -1
  fi
else
  yecho "Unable to find $VITIS_PATH/settings64.sh"
  yecho "Functionality dependent on Vitis will not be available."
  yecho "If you need Vitis, ensure VITIS_PATH is set correctly and mounted into the Docker container."
  if [ -f "$VIVADO_PATH/settings64.sh" ];then
    # source Vivado env.vars
    export XILINX_VIVADO=$VIVADO_PATH
    source $VIVADO_PATH/settings64.sh
    gecho "Found Vivado at $VIVADO_PATH"
  else
    yecho "Unable to find $VIVADO_PATH/settings64.sh"
    yecho "Functionality dependent on Vivado will not be available."
    yecho "If you need Vivado, ensure VIVADO_PATH is set correctly and mounted into the Docker container."
  fi
fi

# execute the provided command(s) as root
exec "$@"
