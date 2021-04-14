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

import torch
from brevitas.nn import QuantLinear, QuantReLU
import torch.nn as nn
import numpy as np
from brevitas.core.quant import QuantType
from brevitas.nn import QuantIdentity
import brevitas.onnx as bo
from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil
from finn.util.test import get_build_env, load_test_checkpoint_or_skip
import pytest
from finn.util.basic import make_build_dir
import pkg_resources as pk
import json
import wget
import subprocess

target_clk_ns = 10
build_kind = "zynq"
build_dir = os.environ["FINN_BUILD_DIR"]
onnx_zip_url = "https://github.com/Xilinx/finn-examples/releases/download/v0.0.1a/onnx-models-bnn-pynq.zip"
onnx_zip_local = build_dir + "/onnx-models-bnn-pynq.zip"

def get_checkpoint_name(step):
    if step == "build":
        # checkpoint for build step is an entire dir
        return build_dir + "/end2end_ext_weights_build"
    else:
        # other checkpoints are onnx files
        return build_dir + "/end2end_ext_weights_%s.onnx" % (step)


def test_end2end_ext_weights_download():
    if not os.path.isfile(onnx_zip_local):
        wget.download(onnx_zip_url, out=onnx_zip_local)
    assert os.path.isfile(onnx_zip_local)
    subprocess.check_output(['unzip', onnx_zip_local])
