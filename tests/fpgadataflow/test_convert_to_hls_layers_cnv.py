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

import os
import pkg_resources as pk

import brevitas.onnx as bo
import numpy as np

import finn.core.onnx_exec as oxe
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.util.test import get_test_model_trained
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul

export_onnx_path_cnv = "test_output_cnv.onnx"


def test_convert_to_hls_layers_cnv_w1a1():
    cnv = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(cnv, (1, 3, 32, 32), export_onnx_path_cnv)
    model = ModelWrapper(export_onnx_path_cnv)
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(Streamline())
    model.save("cnv-streamline.onnx")
    # load one of the test vectors
    fn = pk.resource_filename("finn", "data/cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    assert input_tensor.shape == (1, 3, 32, 32)
    # generate expected value from streamlined net
    input_dict = {"global_in": input_tensor}
    expected_ctx = oxe.execute_onnx(model, input_dict, True)
    expected = expected_ctx[model.graph.output[0].name]

    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model.save("cnv-lower.onnx")
    produced_ctx = oxe.execute_onnx(model, input_dict, True)
    produced = produced_ctx[model.graph.output[0].name]
    assert np.isclose(expected, produced, atol=1e-3).all()
    os.remove(export_onnx_path_cnv)
