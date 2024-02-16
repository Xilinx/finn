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

QONNX_COMMIT="04e24583fb5c1895744801480db3ced8a5b6a914"
FINN_EXP_COMMIT="de99347e936d51715f5356a1b6c64e37b91c23c2"
BREVITAS_COMMIT="9bb26bf2798de210a267d1e4aed4c20087e0e8a5"
PYVERILATOR_COMMIT="766e457465f5c0dd315490d7b9cc5d74f9a76f4f"
CNPY_COMMIT="4e8810b1a8637695171ed346ce68f6984e585ef4"
HLSLIB_COMMIT="c17aa478ae574971d115afa9fa4d9c215857d1ac"
OMX_COMMIT="0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a"
AVNET_BDF_COMMIT="2d49cfc25766f07792c0b314489f21fe916b639b"
XIL_BDF_COMMIT="8cf4bb674a919ac34e3d99d8d71a9e60af93d14e"
RFSOC4x2_BDF_COMMIT="13fb6f6c02c7dfd7e4b336b18b959ad5115db696"
KV260_BDF_COMMIT="98e0d3efc901f0b974006bc4370c2a7ad8856c79"
EXP_BOARD_FILES_MD5="226ca927a16ea4ce579f1332675e9e9a"

QONNX_URL="https://github.com/fastmachinelearning/qonnx.git"
FINN_EXP_URL="https://github.com/Xilinx/finn-experimental.git"
BREVITAS_URL="https://github.com/Xilinx/brevitas.git"
PYVERILATOR_URL="https://github.com/maltanar/pyverilator.git"
CNPY_URL="https://github.com/rogersce/cnpy.git"
HLSLIB_URL="https://github.com/Xilinx/finn-hlslib.git"
OMX_URL="https://github.com/maltanar/oh-my-xilinx.git"
AVNET_BDF_URL="https://github.com/Avnet/bdf.git"
XIL_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"
RFSOC4x2_BDF_URL="https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git"
KV260_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"

QONNX_DIR="qonnx"
FINN_EXP_DIR="finn-experimental"
BREVITAS_DIR="brevitas"
PYVERILATOR_DIR="pyverilator"
CNPY_DIR="cnpy"
HLSLIB_DIR="finn-hlslib"
OMX_DIR="oh-my-xilinx"
AVNET_BDF_DIR="avnet-bdf"
XIL_BDF_DIR="xil-bdf"
RFSOC4x2_BDF_DIR="rfsoc4x2-bdf"
KV260_SOM_BDF_DIR="kv260-som-bdf"

# absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

fetch_repo() {
    # URL for git repo to be cloned
    REPO_URL=$1
    # commit hash for repo
    REPO_COMMIT=$2
    # directory to clone to under deps/
    REPO_DIR=$3
    # absolute path for the repo local copy
    CLONE_TO=$SCRIPTPATH/deps/$REPO_DIR

    # clone repo if dir not found
    if [ ! -d "$CLONE_TO" ]; then
        git clone $REPO_URL $CLONE_TO
    fi
    # verify and try to pull repo if not at correct commit
    CURRENT_COMMIT=$(git -C $CLONE_TO rev-parse HEAD)
    if [ $CURRENT_COMMIT != $REPO_COMMIT ]; then
        git -C $CLONE_TO pull
        # checkout the expected commit
        git -C $CLONE_TO checkout $REPO_COMMIT
    fi
    # verify one last time
    CURRENT_COMMIT=$(git -C $CLONE_TO rev-parse HEAD)
    if [ $CURRENT_COMMIT == $REPO_COMMIT ]; then
        echo "Successfully checked out $REPO_DIR at commit $CURRENT_COMMIT"
    else
        echo "Could not check out $REPO_DIR. Check your internet connection and try again."
    fi
}

fetch_board_files() {
    echo "Downloading and extracting board files..."
    mkdir -p "$SCRIPTPATH/deps/board_files"
    OLD_PWD=$(pwd)
    cd "$SCRIPTPATH/deps/board_files"
    wget -q https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip
    wget -q https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip
    unzip -q pynq-z1.zip
    unzip -q pynq-z2.zip
    cp -r $SCRIPTPATH/deps/$AVNET_BDF_DIR/* $SCRIPTPATH/deps/board_files/
    cp -r $SCRIPTPATH/deps/$XIL_BDF_DIR/boards/Xilinx/rfsoc2x2 $SCRIPTPATH/deps/board_files/;
    cp -r $SCRIPTPATH/deps/$RFSOC4x2_BDF_DIR/board_files/rfsoc4x2 $SCRIPTPATH/deps/board_files/;
    cp -r $SCRIPTPATH/deps/$KV260_SOM_BDF_DIR/boards/Xilinx/kv260_som $SCRIPTPATH/deps/board_files/;
    cd $OLD_PWD
}

fetch_repo $QONNX_URL $QONNX_COMMIT $QONNX_DIR
fetch_repo $FINN_EXP_URL $FINN_EXP_COMMIT $FINN_EXP_DIR
fetch_repo $BREVITAS_URL $BREVITAS_COMMIT $BREVITAS_DIR
fetch_repo $PYVERILATOR_URL $PYVERILATOR_COMMIT $PYVERILATOR_DIR
fetch_repo $CNPY_URL $CNPY_COMMIT $CNPY_DIR
fetch_repo $HLSLIB_URL $HLSLIB_COMMIT $HLSLIB_DIR
fetch_repo $OMX_URL $OMX_COMMIT $OMX_DIR
fetch_repo $AVNET_BDF_URL $AVNET_BDF_COMMIT $AVNET_BDF_DIR
fetch_repo $XIL_BDF_URL $XIL_BDF_COMMIT $XIL_BDF_DIR
fetch_repo $RFSOC4x2_BDF_URL $RFSOC4x2_BDF_COMMIT $RFSOC4x2_BDF_DIR
fetch_repo $KV260_BDF_URL $KV260_BDF_COMMIT $KV260_SOM_BDF_DIR

# download extra Pynq board files and extract if needed
if [ ! -d "$SCRIPTPATH/deps/board_files" ]; then
    fetch_board_files
else
    cd $SCRIPTPATH
    BOARD_FILES_MD5=$(find deps/board_files/ -type f -exec md5sum {} \; | sort -k 2 | md5sum | cut -d' ' -f 1)
    if [ "$BOARD_FILES_MD5" = "$EXP_BOARD_FILES_MD5" ]; then
        echo "Verified board files folder content md5: $BOARD_FILES_MD5"
    else
        echo "Board files folder content mismatch, removing and re-downloading"
        rm -rf deps/board_files/
        fetch_board_files
    fi
fi
