#!/bin/bash
# Copyright (c) 2020-2022, Xilinx, Inc.
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

if [ -z "$FINN_XILINX_PATH" ];then
  recho "Please set the FINN_XILINX_PATH environment variable to the path to your Xilinx tools installation directory (e.g. /opt/Xilinx)."
  recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
fi

if [ -z "$FINN_XILINX_VERSION" ];then
  recho "Please set the FINN_XILINX_VERSION to the version of the Xilinx tools to use (e.g. 2020.1)"
  recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
fi

if [ -z "$PLATFORM_REPO_PATHS" ];then
  recho "Please set PLATFORM_REPO_PATHS pointing to Vitis platform files (DSAs)."
  recho "This is required to be able to use Alveo PCIe cards."
fi

DOCKER_GID=$(id -g)
DOCKER_GNAME=$(id -gn)
DOCKER_UNAME=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_PASSWD="finn"
DOCKER_INST_NAME="finn_dev_${DOCKER_UNAME}"
# ensure Docker inst. name is all lowercase
DOCKER_INST_NAME=$(echo "$DOCKER_INST_NAME" | tr '[:upper:]' '[:lower:]')
# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

# the settings below will be taken from environment variables if available,
# otherwise the defaults below will be used
: ${JUPYTER_PORT=8888}
: ${JUPYTER_PASSWD_HASH=""}
: ${NETRON_PORT=8081}
: ${LOCALHOST_URL="localhost"}
: ${PYNQ_USERNAME="xilinx"}
: ${PYNQ_PASSWORD="xilinx"}
: ${PYNQ_BOARD="Pynq-Z1"}
: ${PYNQ_TARGET_DIR="/home/xilinx/$DOCKER_INST_NAME"}
: ${NUM_DEFAULT_WORKERS=4}
: ${FINN_SSH_KEY_DIR="$SCRIPTPATH/ssh_keys"}
: ${ALVEO_USERNAME="alveo_user"}
: ${ALVEO_PASSWORD=""}
: ${ALVEO_BOARD="U250"}
: ${ALVEO_TARGET_DIR="/tmp"}
: ${PLATFORM_REPO_PATHS="/opt/xilinx/platforms"}
: ${XRT_DEB_VERSION="xrt_202220.2.14.354_22.04-amd64-xrt"}
: ${FINN_HOST_BUILD_DIR="/tmp/$DOCKER_INST_NAME"}
: ${FINN_DOCKER_TAG="xilinx/finn:$(git describe --always --tags --dirty).$XRT_DEB_VERSION"}
: ${FINN_DOCKER_PREBUILT="0"}
: ${FINN_DOCKER_RUN_AS_ROOT="0"}
: ${FINN_DOCKER_GPU="$(docker info | grep nvidia | wc -m)"}
: ${FINN_DOCKER_EXTRA=""}
: ${FINN_DOCKER_BUILD_EXTRA=""}
: ${FINN_SKIP_DEP_REPOS="0"}
: ${FINN_SKIP_BOARD_FILES="0"}
: ${OHMYXILINX="${SCRIPTPATH}/deps/oh-my-xilinx"}
: ${NVIDIA_VISIBLE_DEVICES=""}
: ${DOCKER_BUILDKIT="1"}
: ${FINN_SINGULARITY=""}

DOCKER_INTERACTIVE=""

# Catch FINN_DOCKER_EXTRA options being passed in without a trailing space
FINN_DOCKER_EXTRA+=" "

if [ "$1" = "test" ]; then
  gecho "Running test suite (all tests)"
  DOCKER_CMD="pytest"
elif [ "$1" = "quicktest" ]; then
  gecho "Running test suite (non-Vivado, non-slow tests)"
  DOCKER_CMD="quicktest.sh"
elif [ "$1" = "notebook" ]; then
  gecho "Running Jupyter notebook server"
  if [ -z "$JUPYTER_PASSWD_HASH" ]; then
    JUPYTER_PASSWD_ARG=""
  else
    JUPYTER_PASSWD_ARG="--NotebookApp.password='$JUPYTER_PASSWD_HASH'"
  fi
  DOCKER_CMD="jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port $JUPYTER_PORT $JUPYTER_PASSWD_ARG notebooks"
  FINN_DOCKER_EXTRA+="-e JUPYTER_PORT=$JUPYTER_PORT "
  FINN_DOCKER_EXTRA+="-e NETRON_PORT=$NETRON_PORT "
  if [ -z "$FINN_SINGULARITY" ]; then
    FINN_DOCKER_EXTRA+="-p $JUPYTER_PORT:$JUPYTER_PORT "
    FINN_DOCKER_EXTRA+="-p $NETRON_PORT:$NETRON_PORT "
  fi
elif [ "$1" = "build_dataflow" ]; then
  BUILD_DATAFLOW_DIR=$(readlink -f "$2")
  FINN_DOCKER_EXTRA+="-v $BUILD_DATAFLOW_DIR:$BUILD_DATAFLOW_DIR "
  DOCKER_INTERACTIVE="-it"
  #FINN_HOST_BUILD_DIR=$BUILD_DATAFLOW_DIR/build
  gecho "Running build_dataflow for folder $BUILD_DATAFLOW_DIR"
  DOCKER_CMD="build_dataflow $BUILD_DATAFLOW_DIR"
elif [ "$1" = "build_custom" ]; then
  BUILD_CUSTOM_DIR=$(readlink -f "$2")
  FLOW_NAME=${3:-build}
  FINN_DOCKER_EXTRA+="-v $BUILD_CUSTOM_DIR:$BUILD_CUSTOM_DIR -w $BUILD_CUSTOM_DIR "
  DOCKER_INTERACTIVE="-it"
  #FINN_HOST_BUILD_DIR=$BUILD_DATAFLOW_DIR/build
  gecho "Running build_custom: $BUILD_CUSTOM_DIR/$FLOW_NAME.py"
  DOCKER_CMD="python -mpdb -cc -cq $FLOW_NAME.py"
elif [ -z "$1" ]; then
   gecho "Running container only"
   DOCKER_CMD="bash"
   DOCKER_INTERACTIVE="-it"
else
  gecho "Running container with passed arguments"
  DOCKER_CMD="$@"
fi


if [ "$FINN_DOCKER_GPU" != 0 ] && [ -z "$FINN_SINGULARITY" ];then
  gecho "nvidia-docker detected, enabling GPUs"
  if [ ! -z "$NVIDIA_VISIBLE_DEVICES" ];then
    FINN_DOCKER_EXTRA+="--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES "
  else
    FINN_DOCKER_EXTRA+="--gpus all "
  fi
fi

VIVADO_HLS_LOCAL=$VIVADO_PATH
VIVADO_IP_CACHE=$FINN_HOST_BUILD_DIR/vivado_ip_cache

# ensure build dir exists locally
mkdir -p $FINN_HOST_BUILD_DIR
mkdir -p $FINN_SSH_KEY_DIR

gecho "Docker container is named $DOCKER_INST_NAME"
gecho "Docker tag is named $FINN_DOCKER_TAG"
gecho "Mounting $FINN_HOST_BUILD_DIR into $FINN_HOST_BUILD_DIR"
gecho "Mounting $FINN_XILINX_PATH into $FINN_XILINX_PATH"
gecho "Port-forwarding for Jupyter $JUPYTER_PORT:$JUPYTER_PORT"
gecho "Port-forwarding for Netron $NETRON_PORT:$NETRON_PORT"
gecho "Vivado IP cache dir is at $VIVADO_IP_CACHE"
gecho "Using default PYNQ board $PYNQ_BOARD"

# Ensure git-based deps are checked out at correct commit
if [ "$FINN_SKIP_DEP_REPOS" = "0" ]; then
  ./fetch-repos.sh
fi

# Build the FINN Docker image
if [ "$FINN_DOCKER_PREBUILT" = "0" ] && [ -z "$FINN_SINGULARITY" ]; then
  # Need to ensure this is done within the finn/ root folder:
  OLD_PWD=$(pwd)
  cd $SCRIPTPATH
  docker build -f docker/Dockerfile.finn --build-arg XRT_DEB_VERSION=$XRT_DEB_VERSION --tag=$FINN_DOCKER_TAG $FINN_DOCKER_BUILD_EXTRA .
  cd $OLD_PWD
fi
# Launch container with current directory mounted
# important to pass the --init flag here for correct Vivado operation, see:
# https://stackoverflow.com/questions/55733058/vivado-synthesis-hangs-in-docker-container-spawned-by-jenkins
DOCKER_BASE="docker run -t --rm $DOCKER_INTERACTIVE --tty --init --hostname $DOCKER_INST_NAME "
DOCKER_EXEC="-e SHELL=/bin/bash "
DOCKER_EXEC+="-w $SCRIPTPATH "
DOCKER_EXEC+="-v $SCRIPTPATH:$SCRIPTPATH "
DOCKER_EXEC+="-v $FINN_HOST_BUILD_DIR:$FINN_HOST_BUILD_DIR "
DOCKER_EXEC+="-e FINN_BUILD_DIR=$FINN_HOST_BUILD_DIR "
DOCKER_EXEC+="-e FINN_ROOT="$SCRIPTPATH" "
DOCKER_EXEC+="-e LOCALHOST_URL=$LOCALHOST_URL "
DOCKER_EXEC+="-e VIVADO_IP_CACHE=$VIVADO_IP_CACHE "
DOCKER_EXEC+="-e PYNQ_BOARD=$PYNQ_BOARD "
DOCKER_EXEC+="-e PYNQ_IP=$PYNQ_IP "
DOCKER_EXEC+="-e PYNQ_USERNAME=$PYNQ_USERNAME "
DOCKER_EXEC+="-e PYNQ_PASSWORD=$PYNQ_PASSWORD "
DOCKER_EXEC+="-e PYNQ_TARGET_DIR=$PYNQ_TARGET_DIR "
DOCKER_EXEC+="-e OHMYXILINX=$OHMYXILINX "
DOCKER_EXEC+="-e NUM_DEFAULT_WORKERS=$NUM_DEFAULT_WORKERS "
# Workaround for FlexLM issue, see:
# https://community.flexera.com/t5/InstallAnywhere-Forum/Issues-when-running-Xilinx-tools-or-Other-vendor-tools-in-docker/m-p/245820#M10647
DOCKER_EXEC+="-e LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 "
if [ "$FINN_DOCKER_RUN_AS_ROOT" = "0" ] && [ -z "$FINN_SINGULARITY" ];then
  DOCKER_EXEC+="-v /etc/group:/etc/group:ro "
  DOCKER_EXEC+="-v /etc/passwd:/etc/passwd:ro "
  DOCKER_EXEC+="-v /etc/shadow:/etc/shadow:ro "
  DOCKER_EXEC+="-v /etc/sudoers.d:/etc/sudoers.d:ro "
  DOCKER_EXEC+="-v $FINN_SSH_KEY_DIR:$HOME/.ssh "
  DOCKER_EXEC+="--user $DOCKER_UID:$DOCKER_GID "
else
  DOCKER_EXEC+="-v $FINN_SSH_KEY_DIR:/root/.ssh "
fi
if [ ! -z "$IMAGENET_VAL_PATH" ];then
  DOCKER_EXEC+="-v $IMAGENET_VAL_PATH:$IMAGENET_VAL_PATH "
  DOCKER_EXEC+="-e IMAGENET_VAL_PATH=$IMAGENET_VAL_PATH "
fi
if [ ! -z "$FINN_XILINX_PATH" ];then
  VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"
  VITIS_PATH="$FINN_XILINX_PATH/Vitis/$FINN_XILINX_VERSION"
  HLS_PATH="$FINN_XILINX_PATH/Vitis_HLS/$FINN_XILINX_VERSION"
  DOCKER_EXEC+="-v $FINN_XILINX_PATH:$FINN_XILINX_PATH "
  if [ -d "$VIVADO_PATH" ];then
    DOCKER_EXEC+="-e "XILINX_VIVADO=$VIVADO_PATH" "
    DOCKER_EXEC+="-e VIVADO_PATH=$VIVADO_PATH "
  fi
  if [ -d "$HLS_PATH" ];then
    DOCKER_EXEC+="-e HLS_PATH=$HLS_PATH "
  fi
  if [ -d "$VITIS_PATH" ];then
    DOCKER_EXEC+="-e VITIS_PATH=$VITIS_PATH "
  fi
  if [ -d "$PLATFORM_REPO_PATHS" ];then
    DOCKER_EXEC+="-v $PLATFORM_REPO_PATHS:$PLATFORM_REPO_PATHS "
    DOCKER_EXEC+="-e PLATFORM_REPO_PATHS=$PLATFORM_REPO_PATHS "
    DOCKER_EXEC+="-e ALVEO_IP=$ALVEO_IP "
    DOCKER_EXEC+="-e ALVEO_USERNAME=$ALVEO_USERNAME "
    DOCKER_EXEC+="-e ALVEO_PASSWORD=$ALVEO_PASSWORD "
    DOCKER_EXEC+="-e ALVEO_BOARD=$ALVEO_BOARD "
    DOCKER_EXEC+="-e ALVEO_TARGET_DIR=$ALVEO_TARGET_DIR "
  fi
fi
DOCKER_EXEC+="$FINN_DOCKER_EXTRA "

if [ -z "$FINN_SINGULARITY" ];then
  CMD_TO_RUN="$DOCKER_BASE $DOCKER_EXEC $FINN_DOCKER_TAG $DOCKER_CMD"
else
  SINGULARITY_BASE="singularity exec"
  # Replace command options for Singularity
  SINGULARITY_EXEC="${DOCKER_EXEC//"-e "/"--env "}"
  SINGULARITY_EXEC="${SINGULARITY_EXEC//"-v "/"-B "}"
  SINGULARITY_EXEC="${SINGULARITY_EXEC//"-w "/"--pwd "}"
  CMD_TO_RUN="$SINGULARITY_BASE $SINGULARITY_EXEC $FINN_SINGULARITY /usr/local/bin/finn_entrypoint.sh $DOCKER_CMD"
  gecho "FINN_SINGULARITY is set, launching Singularity container instead of Docker"
fi

$CMD_TO_RUN
