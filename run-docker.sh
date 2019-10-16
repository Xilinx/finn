#!/bin/sh

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
echo "Mounting $SCRIPTPATH into /workspace/finn"
echo "Mounting $SCRIPTPATH/../brevitas into /workspace/brevitas"
# Build the FINN Docker image
docker build --tag=finn .
# Launch container with current directory mounted
docker run --rm --name finn_dev -it \
-v $SCRIPTPATH:/workspace/finn \
-v $SCRIPTPATH/../brevitas:/workspace/brevitas \
finn bash
