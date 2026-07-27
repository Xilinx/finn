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
: ${NUM_DEFAULT_WORKERS=4}
: ${FINN_SSH_KEY_DIR="$SCRIPTPATH/ssh_keys"}
: ${PLATFORM_REPO_PATHS="/opt/xilinx/platforms"}
: ${XRT_DEB_VERSION="xrt_202220.2.14.354_22.04-amd64-xrt"}
: ${V80PP_DEB_PACKAGE=""}
: ${FINN_HOST_BUILD_DIR="/tmp/$DOCKER_INST_NAME"}
: ${FINN_DOCKER_TAG="xilinx/finn:$(OLD_PWD=$(pwd); cd $SCRIPTPATH; git describe --always --tags --dirty; cd $OLD_PWD).$XRT_DEB_VERSION"}
: ${FINN_DOCKER_PREBUILT="0"}
: ${FINN_DOCKER_RUN_AS_ROOT="0"}
: ${FINN_DOCKER_EXTRA=""}
: ${FINN_DOCKER_BUILD_EXTRA=""}
: ${FINN_SKIP_DEP_REPOS="0"}
: ${FINN_SKIP_BOARD_FILES="0"}
: ${COMPRESSOR_ROOT="${SCRIPTPATH}/../arith-finaladder"}
: ${NVIDIA_VISIBLE_DEVICES=""}
: ${DOCKER_BUILDKIT="1"}
: ${FINN_SINGULARITY=""}
: ${FINN_SKIP_XRT_DOWNLOAD=""}
: ${FINN_XRT_PATH=""}
: ${FINN_DOCKER_NO_CACHE="0"}

# print-tag emits the Docker image tag and exits, so the Jenkins publish step
# has one source of truth for the tag (FINN_DOCKER_TAG). Placed before any
# side effects so the invocation is read-only.
if [ "$1" = "print-tag" ]; then
  if [ "$#" -ne 1 ]; then
    echo "Usage: $0 print-tag" >&2
    exit 2
  fi
  echo "$FINN_DOCKER_TAG"
  exit 0
fi

DOCKER_INTERACTIVE=""

# Catch FINN_DOCKER_EXTRA options being passed in without a trailing space
FINN_DOCKER_EXTRA+=" "

if [ -z "$FINN_XILINX_PATH" ];then
  recho "Please set the FINN_XILINX_PATH environment variable to the path to your Xilinx tools installation directory (e.g. /opt/Xilinx)."
  recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
fi

if [ -z "$FINN_XILINX_VERSION" ];then
  recho "Please set the FINN_XILINX_VERSION to the version of the Xilinx tools to use (e.g. 2022.2)"
  recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
fi

if [ -z "$PLATFORM_REPO_PATHS" ];then
  recho "Please set PLATFORM_REPO_PATHS pointing to Vitis platform files (DSAs)."
  recho "This is required to be able to use Vitis-based Alveo PCIe cards."
fi

if [ -z "$V80PP_DEB_PACKAGE" ];then
  recho "Please set V80PP_DEB_PACKAGE pointing to the SLASH v80++ .deb package."
  recho "This is required to be able to use the Alveo V80 card."
fi

# Mirror the Jenkinsfile's local-fallback banner, but only inside a real
# Jenkins run (JENKINS_URL + BUILD_NUMBER) so unrelated CI systems and
# developer shells that happen to export BUILD_NUMBER stay quiet.
if [ -n "$JENKINS_URL" ] && [ -n "$BUILD_NUMBER" ] \
   && [ -z "$FINN_CI_NFS_ROOT" ] && [ -z "$FINN_DOCKER_SHARED_IMAGE_DIR" ]; then
  recho "FINN_CI_NFS_ROOT and FINN_DOCKER_SHARED_IMAGE_DIR are unset. Running in local-fallback mode."
  recho "  - no shared Docker image cache (this agent will build locally)"
  recho "  - no build-to-HW artifact handoff (the HW pipeline cannot test this build)"
  recho "Set FINN_CI_NFS_ROOT in the Jenkins job DSL to enable the shared cache."
fi

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
  DOCKER_CMD="python -mpdb -cc -cq $FLOW_NAME.py ${@:4}"
elif [ -z "$1" ]; then
   gecho "Running container only"
   DOCKER_CMD="bash"
   DOCKER_INTERACTIVE="-it"
else
  gecho "Running container with passed arguments"
  DOCKER_CMD="$@"
fi

# ensure build dir exists locally
mkdir -p $FINN_HOST_BUILD_DIR
mkdir -p $FINN_SSH_KEY_DIR

gecho "Docker container is named $DOCKER_INST_NAME"
gecho "Docker tag is named $FINN_DOCKER_TAG"
gecho "Mounting $FINN_HOST_BUILD_DIR into $FINN_HOST_BUILD_DIR"
gecho "Mounting $FINN_XILINX_PATH into $FINN_XILINX_PATH"
gecho "Port-forwarding for Jupyter $JUPYTER_PORT:$JUPYTER_PORT"
gecho "Port-forwarding for Netron $NETRON_PORT:$NETRON_PORT"

# Ensure git-based deps are checked out at correct commit
if [ "$FINN_SKIP_DEP_REPOS" = "0" ]; then
  ./fetch-repos.sh || exit 1
fi

# If xrt path given, copy .deb file to this repo. Gate on the .deb
# itself, not the dir. Otherwise an empty cache dir trips LOCAL_XRT=1
# without producing a build-context .deb, and the docker build then
# fails because the wget branch is also skipped.
if [ -f "$FINN_XRT_PATH/$XRT_DEB_VERSION.deb" ]; then
  cp "$FINN_XRT_PATH/$XRT_DEB_VERSION.deb" .
  export LOCAL_XRT=1
fi

# If v80++ deb package given, copy it to repo root for docker build
if [ -n "$V80PP_DEB_PACKAGE" ] && [ -f "$V80PP_DEB_PACKAGE" ]; then
  cp "$V80PP_DEB_PACKAGE" ./v80pp.deb
fi

if [ "$FINN_DOCKER_NO_CACHE" = "1" ]; then
  FINN_DOCKER_BUILD_EXTRA+="--no-cache "
fi

# fail fast on PREBUILT=1 with no usable image source: with no shared dir
# configured and no local image, docker run further down would fail with
# a generic "Unable to find image" much later in the pipeline.
if [ "$FINN_DOCKER_PREBUILT" = "1" ] && [ -z "$FINN_DOCKER_SHARED_IMAGE_DIR" ] \
   && ! docker image inspect "$FINN_DOCKER_TAG" > /dev/null 2>&1; then
  recho "FINN_DOCKER_PREBUILT=1 but FINN_DOCKER_SHARED_IMAGE_DIR is unset and tag $FINN_DOCKER_TAG is not loaded locally"
  recho "Set FINN_DOCKER_SHARED_IMAGE_DIR to a directory containing finn-docker-image.tar.gz, or unset FINN_DOCKER_PREBUILT to build locally."
  exit 1
fi

# If a shared-image dir is configured, load from there. In prebuilt mode
# the shared image is authoritative and any same-tag local image is ignored.
if [ -n "$FINN_DOCKER_SHARED_IMAGE_DIR" ] && \
   { [ "$FINN_DOCKER_PREBUILT" = "1" ] || ! docker image inspect "$FINN_DOCKER_TAG" > /dev/null 2>&1; }; then
  SHARED_DIR="$FINN_DOCKER_SHARED_IMAGE_DIR"
  SHARED_LOADED="0"
  SHARED_IMG="$SHARED_DIR/finn-docker-image.tar.gz"
  SHARED_TAG_FILE="$SHARED_DIR/finn-docker-tag.txt"
  if [ -f "$SHARED_IMG" ] && [ -f "$SHARED_TAG_FILE" ]; then
    gecho "Loading Docker image from shared storage ($SHARED_DIR)..."
    SHARED_TAG=$(cat "$SHARED_TAG_FILE")
    if [ "$FINN_DOCKER_PREBUILT" = "1" ] && [ "$SHARED_TAG" != "$FINN_DOCKER_TAG" ]; then
      recho "Shared Docker tag $SHARED_TAG does not match requested tag $FINN_DOCKER_TAG"
      exit 1
    fi
    # local /tmp lock to serialise concurrent loads on the same host
    if flock /tmp/finn-docker-load.lock \
         bash -c 'set -o pipefail; gunzip -c "$1" | docker load' _ "$SHARED_IMG"; then
      SHARED_LOADED="1"
      if [ "$SHARED_TAG" != "$FINN_DOCKER_TAG" ]; then
        gecho "Tagging $SHARED_TAG as $FINN_DOCKER_TAG"
        docker tag "$SHARED_TAG" "$FINN_DOCKER_TAG"
      fi
    else
      gecho "WARNING: Failed to load Docker image from shared storage ($SHARED_DIR)"
    fi
  fi
  if [ "$SHARED_LOADED" != "1" ] && [ "$FINN_DOCKER_PREBUILT" != "1" ]; then
    gecho "WARNING: No usable shared Docker image found at FINN_DOCKER_SHARED_IMAGE_DIR=$SHARED_DIR. Falling back to local build"
  fi
  if [ "$FINN_DOCKER_PREBUILT" = "1" ] && [ "$SHARED_LOADED" != "1" ]; then
    recho "FINN_DOCKER_PREBUILT=1 but no usable shared Docker image at FINN_DOCKER_SHARED_IMAGE_DIR=$SHARED_DIR (expected finn-docker-image.tar.gz and finn-docker-tag.txt)"
    exit 1
  fi
fi

# Build the FINN Docker image
if [ "$FINN_DOCKER_PREBUILT" = "0" ] && [ -z "$FINN_SINGULARITY" ]; then
  # Need to ensure this is done within the finn/ root folder:
  OLD_PWD=$(pwd)
  cd $SCRIPTPATH
  # Export DOCKER_BUILDKIT to enable BuildKit features
  export DOCKER_BUILDKIT
  docker build \
    -f docker/Dockerfile.finn \
    --build-arg XRT_DEB_VERSION=$XRT_DEB_VERSION \
    --build-arg SKIP_XRT=$FINN_SKIP_XRT_DOWNLOAD \
    --build-arg LOCAL_XRT=$LOCAL_XRT \
    --build-arg V80PP_DEB_PACKAGE=$V80PP_DEB_PACKAGE \
    --tag=$FINN_DOCKER_TAG $FINN_DOCKER_BUILD_EXTRA \
    --build-arg GROUP_ID=$DOCKER_GID \
    --build-arg GROUPNAME=$DOCKER_GNAME \
    --build-arg USERNAME=$DOCKER_UNAME \
    --build-arg USER_UID=$DOCKER_UID \
    . || { recho "docker build failed"; exit 1; }
  cd $OLD_PWD
fi

# Remove local xrt.deb file from repo
if [ ! -z "$LOCAL_XRT" ];then
  rm $XRT_DEB_VERSION.deb
fi

# Remove local v80pp.deb file from repo
if [ -f "./v80pp.deb" ]; then
  rm ./v80pp.deb
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
# Mount the compressor HDL repo into the container.
if [ -d "$COMPRESSOR_ROOT" ];then
  COMPRESSOR_ROOT=$(cd "$COMPRESSOR_ROOT" && pwd)   # normalize the ../ path
  DOCKER_EXEC+="-v $COMPRESSOR_ROOT:$COMPRESSOR_ROOT "
  DOCKER_EXEC+="-e COMPRESSOR_ROOT=$COMPRESSOR_ROOT "
fi
DOCKER_EXEC+="-e LOCALHOST_URL=$LOCALHOST_URL "
DOCKER_EXEC+="-e NUM_DEFAULT_WORKERS=$NUM_DEFAULT_WORKERS "
# Workaround for FlexLM issue, see:
# https://community.flexera.com/t5/InstallAnywhere-Forum/Issues-when-running-Xilinx-tools-or-Other-vendor-tools-in-docker/m-p/245820#M10647
DOCKER_EXEC+="-e LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 "
# Workaround for running multiple Vivado instances simultaneously, see:
# https://adaptivesupport.amd.com/s/article/63253?language=en_US
DOCKER_EXEC+="-e XILINX_LOCAL_USER_DATA=no "
# Optional host cache for torch.hub / huggingface weights to avoid CDN 504s
# on parallel CI runs. Bind target is /finn_cache (NOT $HOME, because docker
# creates bind parents as root and that would break pip install --user).
: ${FINN_DOCKER_CACHE_DIR=""}
if [ -n "$FINN_DOCKER_CACHE_DIR" ]; then
  mkdir -p "$FINN_DOCKER_CACHE_DIR/torch" "$FINN_DOCKER_CACHE_DIR/huggingface"
  DOCKER_EXEC+="-v $FINN_DOCKER_CACHE_DIR:/finn_cache "
  DOCKER_EXEC+="-e TORCH_HOME=/finn_cache/torch "
  DOCKER_EXEC+="-e HF_HOME=/finn_cache/huggingface "
fi
if [ "$FINN_DOCKER_RUN_AS_ROOT" = "0" ] && [ -z "$FINN_SINGULARITY" ];then
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
  if [[ "$FINN_XILINX_VERSION" =~ ^20([0-9]{2})\.(1|2)$ ]]; then
    year="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"

    # Convert to integers for comparison
    year=$((10#$year))
    minor=$((10#$minor))

    if (( year > 24 )) || { (( year == 24 )) && (( minor > 2 )); }; then
      VIVADO_PATH="$FINN_XILINX_PATH/$FINN_XILINX_VERSION/Vivado"
      VITIS_PATH="$FINN_XILINX_PATH/$FINN_XILINX_VERSION/Vitis"
      HLS_PATH="$FINN_XILINX_PATH/$FINN_XILINX_VERSION/Vitis"
    else
      VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"
      VITIS_PATH="$FINN_XILINX_PATH/Vitis/$FINN_XILINX_VERSION"
      HLS_PATH="$FINN_XILINX_PATH/Vitis_HLS/$FINN_XILINX_VERSION"
    fi
  else
    echo "FINN_XILINX_VERSION ($FINN_XILINX_VERSION) is not in the correct format (YYYY.1 or YYYY.2)"
  fi
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
  fi
fi

# This part is used for internal ci for finn-examples
# if using build verification for finn-examples ci, set up the necessary Docker variables
if [ "$VERIFICATION_EN" = 1 ]; then
  if [ -z "$FINN_EXAMPLES_ROOT" ]; then
    recho "FINN_EXAMPLES_ROOT path has not been set."
    recho "Please set FINN_EXAMPLES_ROOT path to enable verification."
    exit -1
  elif [ ! -d "${FINN_EXAMPLES_ROOT}/ci" ]; then
    recho "ci folder not found in ${FINN_EXAMPLES_ROOT}."
    recho "Please ensure the FINN-examples repo has been set up correctly, and FINN_EXAMPLES_ROOT path is set correctly, to enable verification."
    exit -1
  elif [ -z "$VERIFICATION_IO" ]; then
    recho "VERIFICATION_IO paths has not been set."
    recho "Please ensure the path to the input and expected output files has been set correctly to eneable verification."
    exit -1
  elif [ ! -d "$VERIFICATION_IO" ]; then
    recho "${VERIFICATION_IO} is not a directory."
    recho "Please ensure the VERIFICATION_IO path has been set to the directory containing the input and expected output files for verification."
    exit -1
  else
    DOCKER_EXEC+="-e VERIFICATION_EN=$VERIFICATION_EN "
    DOCKER_EXEC+="-e FINN_EXAMPLES_ROOT=$FINN_EXAMPLES_ROOT "
    DOCKER_EXEC+="-e VERIFICATION_IO=$VERIFICATION_IO "
    FINN_DOCKER_EXTRA+="-v $FINN_EXAMPLES_ROOT/ci:$FINN_EXAMPLES_ROOT/ci "
    FINN_DOCKER_EXTRA+="-v $VERIFICATION_IO:$VERIFICATION_IO "
  fi
fi


DOCKER_EXEC+="$FINN_DOCKER_EXTRA "

if [ -z "$FINN_SINGULARITY" ];then
  CMD_TO_RUN="$DOCKER_BASE $DOCKER_EXEC $FINN_DOCKER_TAG $DOCKER_CMD"
else
  SINGULARITY_BASE="singularity exec"
  # Replace command options for Singularity
  SINGULARITY_EXEC="${DOCKER_EXEC//-e /--env }"
  SINGULARITY_EXEC="${SINGULARITY_EXEC//-v /-B }"
  SINGULARITY_EXEC="${SINGULARITY_EXEC//-w /--pwd }"
  CMD_TO_RUN="$SINGULARITY_BASE $SINGULARITY_EXEC $FINN_SINGULARITY /usr/local/bin/finn_entrypoint.sh $DOCKER_CMD"
  gecho "FINN_SINGULARITY is set, launching Singularity container instead of Docker"
fi

echo $CMD_TO_RUN
$CMD_TO_RUN
