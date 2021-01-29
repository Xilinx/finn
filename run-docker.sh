#!/bin/bash
# Copyright (c) 2020, Xilinx
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

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# green echo
gecho () {
  echo -e "${GREEN}$1${NC}"
}

# red echo
recho () {
  echo -e "${RED}$1${NC}"
}

if [ -z "$VIVADO_PATH" ];then
  recho "Please set the VIVADO_PATH that contains the path to your Vivado installation directory."
  recho "FINN functionality depending on Vivado or Vivado HLS will not be available."
fi

if [ -z "$PYNQ_IP" ];then
  recho "Please set the PYNQ_IP env.var. to enable PYNQ deployment tests."
fi

if [ -z "$VITIS_PATH" ];then
  recho "Please set the VITIS_PATH that contains the path to your Vitis installation directory."
  recho "FINN functionality depending on Vitis will not be available."
else
  if [ -z "$PLATFORM_REPO_PATHS" ];then
    recho "Please set PLATFORM_REPO_PATHS pointing to Vitis platform files (DSAs)."
    recho "This is required to be able to use Vitis."
    exit -1
  fi
  if [ -z "$XILINX_XRT" ];then
    recho "Please set XILINX_XRT pointing to your XRT installation."
    recho "This is required to be able to use Vitis."
    exit -1
  fi
fi

DOCKER_GID=$(id -g)
DOCKER_GNAME=$(id -gn)
DOCKER_UNAME=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_PASSWD="finn"
# generate a random number per-run to allow multiple
# containers from the same user
DOCKER_RND=$(shuf -i0-32768 -n1)
DOCKER_TAG="finn_dev_${DOCKER_UNAME}"
# uncomment to run multiple instances with different names
# DOCKER_INST_NAME="finn_${DOCKER_UNAME}_${DOCKER_RND}"
DOCKER_INST_NAME="finn_dev_${DOCKER_UNAME}"
# ensure Docker tag and inst. name are all lowercase
DOCKER_TAG=$(echo "$DOCKER_TAG" | tr '[:upper:]' '[:lower:]')
DOCKER_INST_NAME=$(echo "$DOCKER_INST_NAME" | tr '[:upper:]' '[:lower:]')
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

# the settings below will be taken from environment variables if available,
# otherwise the defaults below will be used
: ${JUPYTER_PORT=8888}
: ${NETRON_PORT=8081}
: ${PYNQ_USERNAME="xilinx"}
: ${PYNQ_PASSWORD="xilinx"}
: ${PYNQ_BOARD="Pynq-Z1"}
: ${PYNQ_TARGET_DIR="/home/xilinx/$DOCKER_INST_NAME"}
: ${NUM_DEFAULT_WORKERS=1}
: ${FINN_SSH_KEY_DIR="$SCRIPTPATH/ssh_keys"}
: ${ALVEO_USERNAME="alveo_user"}
: ${ALVEO_PASSWORD=""}
: ${ALVEO_BOARD="U250"}
: ${ALVEO_TARGET_DIR="/tmp"}
: ${XILINX_XRT="/opt/xilinx/xrt"}
: ${PLATFORM_REPO_PATHS="/opt/xilinx/platforms"}
: ${FINN_HOST_BUILD_DIR="/tmp/$DOCKER_INST_NAME"}

DOCKER_INTERACTIVE=""
DOCKER_EXTRA=""

if [ "$1" = "test" ]; then
  gecho "Running test suite (all tests)"
  DOCKER_CMD="python setup.py test"
elif [ "$1" = "quicktest" ]; then
  gecho "Running test suite (non-Vivado, non-slow tests)"
  DOCKER_CMD="quicktest.sh"
elif [ "$1" = "notebook" ]; then
  gecho "Running Jupyter notebook server"
  DOCKER_CMD="jupyter notebook --ip=0.0.0.0 --port $JUPYTER_PORT notebooks"
  DOCKER_EXTRA+="-e JUPYTER_PORT=$JUPYTER_PORT "
  DOCKER_EXTRA+="-e NETRON_PORT=$NETRON_PORT "
  DOCKER_EXTRA+="-p $JUPYTER_PORT:$JUPYTER_PORT "
  DOCKER_EXTRA+="-p $NETRON_PORT:$NETRON_PORT "
elif [ "$1" = "build_dataflow" ]; then
  BUILD_DATAFLOW_DIR=$(readlink -f "$2")
  DOCKER_EXTRA="-v $BUILD_DATAFLOW_DIR:$BUILD_DATAFLOW_DIR"
  DOCKER_INTERACTIVE="-it"
  #FINN_HOST_BUILD_DIR=$BUILD_DATAFLOW_DIR/build
  gecho "Running build_dataflow for folder $BUILD_DATAFLOW_DIR"
  DOCKER_CMD="build_dataflow $BUILD_DATAFLOW_DIR"
elif [ "$1" = "build_custom" ]; then
  BUILD_CUSTOM_DIR=$(readlink -f "$2")
  DOCKER_EXTRA="-v $BUILD_CUSTOM_DIR:$BUILD_CUSTOM_DIR -w $BUILD_CUSTOM_DIR"
  DOCKER_INTERACTIVE="-it"
  #FINN_HOST_BUILD_DIR=$BUILD_DATAFLOW_DIR/build
  gecho "Running build_custom: $BUILD_CUSTOM_DIR/build.py"
  DOCKER_CMD="python -mpdb -cc -cq build.py"
else
  gecho "Running container only"
  DOCKER_CMD="bash"
  DOCKER_INTERACTIVE="-it"
fi

VIVADO_HLS_LOCAL=$VIVADO_PATH
VIVADO_IP_CACHE=$FINN_HOST_BUILD_DIR/vivado_ip_cache
INSTALL_XRT_DEPS=0

# ensure build dir exists locally
mkdir -p $FINN_HOST_BUILD_DIR
mkdir -p $FINN_SSH_KEY_DIR

gecho "Docker container is named $DOCKER_INST_NAME"
gecho "Mounting $FINN_HOST_BUILD_DIR into $FINN_HOST_BUILD_DIR"
gecho "Mounting $VIVADO_PATH into $VIVADO_PATH"
if [ ! -z "$VITIS_PATH" ];then
  gecho "Mounting $VITIS_PATH into $VITIS_PATH"
  INSTALL_XRT_DEPS=1
fi
gecho "Port-forwarding for Jupyter $JUPYTER_PORT:$JUPYTER_PORT"
gecho "Port-forwarding for Netron $NETRON_PORT:$NETRON_PORT"
gecho "Vivado IP cache dir is at $VIVADO_IP_CACHE"
gecho "Using default PYNQ board $PYNQ_BOARD"

# Build the FINN Docker image
# Need to ensure this is done within the finn/ root folder:
OLD_PWD=$(pwd)
cd $SCRIPTPATH
docker build -f docker/Dockerfile.finn_dev --tag=$DOCKER_TAG \
             --build-arg GID=$DOCKER_GID \
             --build-arg GNAME=$DOCKER_GNAME \
             --build-arg UNAME=$DOCKER_UNAME \
             --build-arg UID=$DOCKER_UID \
             --build-arg PASSWD=$DOCKER_PASSWD \
             --build-arg INSTALL_XRT_DEPS=$INSTALL_XRT_DEPS \
             .
cd $OLD_PWD
# Launch container with current directory mounted
# important to pass the --init flag here for correct Vivado operation, see:
# https://stackoverflow.com/questions/55733058/vivado-synthesis-hangs-in-docker-container-spawned-by-jenkins
DOCKER_EXEC="docker run -t --rm $DOCKER_INTERACTIVE --init "
DOCKER_EXEC+="--hostname $DOCKER_INST_NAME "
DOCKER_EXEC+="-e SHELL=/bin/bash "
DOCKER_EXEC+="-v $SCRIPTPATH:/workspace/finn "
DOCKER_EXEC+="-v $FINN_HOST_BUILD_DIR:$FINN_HOST_BUILD_DIR "
DOCKER_EXEC+="-v $FINN_SSH_KEY_DIR:/home/$DOCKER_UNAME/.ssh "
DOCKER_EXEC+="-e FINN_BUILD_DIR=$FINN_HOST_BUILD_DIR "
DOCKER_EXEC+="-e FINN_ROOT="/workspace/finn" "
DOCKER_EXEC+="-e VIVADO_IP_CACHE=$VIVADO_IP_CACHE "
DOCKER_EXEC+="-e PYNQ_BOARD=$PYNQ_BOARD "
DOCKER_EXEC+="-e PYNQ_IP=$PYNQ_IP "
DOCKER_EXEC+="-e PYNQ_USERNAME=$PYNQ_USERNAME "
DOCKER_EXEC+="-e PYNQ_PASSWORD=$PYNQ_PASSWORD "
DOCKER_EXEC+="-e PYNQ_TARGET_DIR=$PYNQ_TARGET_DIR "
DOCKER_EXEC+="-e NUM_DEFAULT_WORKERS=$NUM_DEFAULT_WORKERS "
if [ ! -z "$IMAGENET_VAL_PATH" ];then
  DOCKER_EXEC+="-v $IMAGENET_VAL_PATH:$IMAGENET_VAL_PATH "
  DOCKER_EXEC+="-e IMAGENET_VAL_PATH=$IMAGENET_VAL_PATH "
fi
if [ ! -z "$VIVADO_PATH" ];then
  DOCKER_EXEC+="-e "XILINX_VIVADO=$VIVADO_PATH" "
  DOCKER_EXEC+="-v $VIVADO_PATH:$VIVADO_PATH "
  DOCKER_EXEC+="-e VIVADO_PATH=$VIVADO_PATH "
fi
if [ ! -z "$VITIS_PATH" ];then
  if [ -z "$PLATFORM_REPO_PATHS" ];then
    recho "PLATFORM_REPO_PATHS must be set for Vitis/Alveo flows"
    exit -1
  fi
  if [ -z "$XILINX_XRT" ];then
    recho "XILINX_XRT must be set for Vitis/Alveo flows"
    exit -1
  fi
  DOCKER_EXEC+="-v $VITIS_PATH:$VITIS_PATH "
  DOCKER_EXEC+="-v $PLATFORM_REPO_PATHS:$PLATFORM_REPO_PATHS "
  DOCKER_EXEC+="-v $XILINX_XRT:$XILINX_XRT "
  DOCKER_EXEC+="-e VITIS_PATH=$VITIS_PATH "
  DOCKER_EXEC+="-e PLATFORM_REPO_PATHS=$PLATFORM_REPO_PATHS "
  DOCKER_EXEC+="-e XILINX_XRT=$XILINX_XRT "
  DOCKER_EXEC+="-e ALVEO_IP=$ALVEO_IP "
  DOCKER_EXEC+="-e ALVEO_USERNAME=$ALVEO_USERNAME "
  DOCKER_EXEC+="-e ALVEO_PASSWORD=$ALVEO_PASSWORD "
  DOCKER_EXEC+="-e ALVEO_BOARD=$ALVEO_BOARD "
  DOCKER_EXEC+="-e ALVEO_TARGET_DIR=$ALVEO_TARGET_DIR "
fi
DOCKER_EXEC+="$DOCKER_EXTRA "
DOCKER_EXEC+="$DOCKER_TAG $DOCKER_CMD"

$DOCKER_EXEC