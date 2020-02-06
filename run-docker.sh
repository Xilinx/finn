#!/bin/bash

if [ -z "$VIVADO_PATH" ];then
        echo "For correct implementation please set an environment variable VIVADO_PATH that contains the path to your vivado installation directory"
        exit 1
fi

DOCKER_GID=$(id -g)
DOCKER_GNAME=$(id -gn)
DOCKER_UNAME=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_PASSWD="finn"
DOCKER_TAG="finn_$DOCKER_UNAME"

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

BREVITAS_REPO=https://github.com/Xilinx/brevitas.git
EXAMPLES_REPO=https://github.com/maltanar/brevitas_cnv_lfc.git
CNPY_REPO=https://github.com/rogersce/cnpy.git
FINN_HLS_REPO=https://github.com/Xilinx/finn-hlslib.git
PYVERILATOR_REPO=https://github.com/maltanar/pyverilator
PYNQSHELL_REPO=https://github.com/maltanar/PYNQ-HelloWorld.git

BREVITAS_LOCAL=$SCRIPTPATH/brevitas
EXAMPLES_LOCAL=$SCRIPTPATH/brevitas_cnv_lfc
CNPY_LOCAL=$SCRIPTPATH/cnpy
FINN_HLS_LOCAL=$SCRIPTPATH/finn-hlslib
PYVERILATOR_LOCAL=$SCRIPTPATH/pyverilator
PYNQSHELL_LOCAL=$SCRIPTPATH/PYNQ-HelloWorld
BUILD_LOCAL=/tmp/finn
VIVADO_HLS_LOCAL=$VIVADO_PATH

# clone dependency repos
git clone --branch feature/finn_onnx_export $BREVITAS_REPO $BREVITAS_LOCAL ||  git -C "$BREVITAS_LOCAL" pull
git clone $EXAMPLES_REPO $EXAMPLES_LOCAL ||  git -C "$EXAMPLES_LOCAL" pull
git clone $CNPY_REPO $CNPY_LOCAL ||  git -C "$CNPY_LOCAL" pull
git clone $FINN_HLS_REPO $FINN_HLS_LOCAL; git -C "$FINN_HLS_LOCAL" checkout b5dc957a16017b8356a7010144b0a4e2f8cfd124 ||  git -C "$FINN_HLS_LOCAL" checkout b5dc957a16017b8356a7010144b0a4e2f8cfd124
git clone $PYVERILATOR_REPO $PYVERILATOR_LOCAL ||  git -C "$PYVERILATOR_LOCAL" pull
git clone $PYNQSHELL_REPO $PYNQSHELL_LOCAL ||  git -C "$PYNQSHELL_LOCAL" pull

# ensure build dir exists locally
mkdir -p $BUILD_LOCAL

echo "Mounting $SCRIPTPATH into /workspace/finn"
echo "Mounting $SCRIPTPATH/brevitas into /workspace/brevitas"
echo "Mounting $SCRIPTPATH/brevitas_cnv_lfc into /workspace/brevitas_cnv_lfc"
echo "Mounting $SCRIPTPATH/cnpy into /workspace/cnpy"
echo "Mounting $SCRIPTPATH/finn-hlslib into /workspace/finn-hlslib"
echo "Mounting $SCRIPTPATH/pyverilator into /workspace/pyverilator"
echo "Mounting $SCRIPTPATH/PYNQ-HelloWorld into /workspace/PYNQ-HelloWorld"
echo "Mounting $BUILD_LOCAL into $BUILD_LOCAL"
echo "Mounting $VIVADO_PATH into $VIVADO_PATH"

if [ "$1" = "test" ]; then
        echo "Running test suite"
        DOCKER_CMD="python setup.py test"
elif [ "$1" = "notebook" ]; then
        echo "Running Jupyter notebook server"
        DOCKER_CMD="source ~/.bashrc; jupyter notebook --ip=0.0.0.0 notebooks"
else
        echo "Running container only"
        DOCKER_CMD="bash"
fi

# Build the FINN Docker image
docker build --tag=$DOCKER_TAG \
             --build-arg GID=$DOCKER_GID \
             --build-arg GNAME=$DOCKER_GNAME \
             --build-arg UNAME=$DOCKER_UNAME \
             --build-arg UID=$DOCKER_UID \
             --build-arg PASSWD=$DOCKER_PASSWD \
             .
# Launch container with current directory mounted
docker run -t --rm --name finn_dev_$DOCKER_UNAME -it \
-e "XILINX_VIVADO=$VIVADO_PATH" \
-e "SHELL=/bin/bash" \
-v $SCRIPTPATH:/workspace/finn \
-v $SCRIPTPATH/brevitas:/workspace/brevitas \
-v $SCRIPTPATH/brevitas_cnv_lfc:/workspace/brevitas_cnv_lfc \
-v $SCRIPTPATH/cnpy:/workspace/cnpy \
-v $SCRIPTPATH/finn-hlslib:/workspace/finn-hlslib \
-v $SCRIPTPATH/pyverilator:/workspace/pyverilator \
-v $SCRIPTPATH/PYNQ-HelloWorld:/workspace/PYNQ-HelloWorld \
-v $BUILD_LOCAL:$BUILD_LOCAL \
-v $VIVADO_PATH:$VIVADO_PATH \
-e VIVADO_PATH=$VIVADO_PATH \
-p 8888:8888 -p 8081:8081 \
$DOCKER_TAG bash -c "$DOCKER_CMD"
