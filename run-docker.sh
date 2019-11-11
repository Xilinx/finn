#!/bin/sh

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

BREVITAS_LOCAL=$SCRIPTPATH/brevitas
EXAMPLES_LOCAL=$SCRIPTPATH/brevitas_cnv_lfc
CNPY_LOCAL=$SCRIPTPATH/cnpy

# clone dependency repos
git clone --branch feature/finn_onnx_export $BREVITAS_REPO $BREVITAS_LOCAL ||  git -C "$BREVITAS_LOCAL" pull
git clone $EXAMPLES_REPO $EXAMPLES_LOCAL ||  git -C "$EXAMPLES_LOCAL" pull

echo "Mounting $SCRIPTPATH into /workspace/finn"
echo "Mounting $SCRIPTPATH/brevitas into /workspace/brevitas"
echo "Mounting $SCRIPTPATH/brevitas_cnv_lfc into /workspace/brevitas_cnv_lfc"
echo "Mounting $SCRIPTPATH/cnpy into /workspace/cnpy"

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
$DOCKER_TAG bash
