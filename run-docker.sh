#!/bin/sh

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

BREVITAS_LOCAL=$SCRIPTPATH/brevitas
EXAMPLES_LOCAL=$SCRIPTPATH/brevitas_cnv_lfc
CNPY_LOCAL=$SCRIPTPATH/cnpy
FINN_HLS_LOCAL=$SCRIPTPATH/finn-hlslib
VIVADO_HLS_LOCAL=$VIVADO_PATH/include

# clone dependency repos
git clone --branch feature/finn_onnx_export $BREVITAS_REPO $BREVITAS_LOCAL ||  git -C "$BREVITAS_LOCAL" pull
git clone $EXAMPLES_REPO $EXAMPLES_LOCAL ||  git -C "$EXAMPLES_LOCAL" pull
git clone $CNPY_REPO $CNPY_LOCAL ||  git -C "$CNPY_LOCAL" pull
git clone $FINN_HLS_REPO $FINN_HLS_LOCAL ||  git -C "$FINN_HLS_LOCAL" checkout b5dc957a16017b8356a7010144b0a4e2f8cfd124 

echo "Mounting $SCRIPTPATH into /workspace/finn"
echo "Mounting $SCRIPTPATH/brevitas into /workspace/brevitas"
echo "Mounting $SCRIPTPATH/brevitas_cnv_lfc into /workspace/brevitas_cnv_lfc"
echo "Mounting $SCRIPTPATH/cnpy into /workspace/cnpy"
echo "Mounting $SCRIPTPATH/finn-hlslib into /workspace/finn-hlslib"
echo "Mounting $VIVADO_PATH/include into /workspace/vivado-hlslib"

# Build the FINN Docker image
docker build --tag=$DOCKER_TAG \
             --build-arg GID=$DOCKER_GID \
             --build-arg GNAME=$DOCKER_GNAME \
             --build-arg UNAME=$DOCKER_UNAME \
             --build-arg UID=$DOCKER_UID \
             --build-arg PASSWD=$DOCKER_PASSWD \
             .
# Launch container with current directory mounted
docker run --rm --name finn_dev -it \
-v $SCRIPTPATH:/workspace/finn \
-v $SCRIPTPATH/brevitas:/workspace/brevitas \
-v $SCRIPTPATH/brevitas_cnv_lfc:/workspace/brevitas_cnv_lfc \
-v $SCRIPTPATH/cnpy:/workspace/cnpy \
-v $SCRIPTPATH/finn-hlslib:/workspace/finn-hlslib \
-v $VIVADO_PATH/include:/workspace/vivado-hlslib \
$DOCKER_TAG bash
