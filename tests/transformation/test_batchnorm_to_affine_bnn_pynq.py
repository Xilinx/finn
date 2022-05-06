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

import pytest

import pkg_resources as pk

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import os
from pkgutil import get_data

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.batchnorm_to_affine import BatchNormToAffine
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
from finn.util.test import get_test_model_trained

export_onnx_path = "test_output_bn2affine.onnx"

@pytest.mark.transform
def test_batchnorm_to_affine_cnv_w1a1():
    lfc = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(lfc, (1, 3, 32, 32), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    fn = pk.resource_filename("finn.qnn-data", "cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    input_tensor = input_tensor / 255
    assert input_tensor.shape == (1, 3, 32, 32)
    input_dict = {"0": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    expected = output_dict[list(output_dict.keys())[0]]
    new_model = model.transform(BatchNormToAffine())
    # check that there are no BN nodes left
    op_types = list(map(lambda x: x.op_type, new_model.graph.node))
    assert "BatchNormalization" not in op_types
    output_dict_p = oxe.execute_onnx(new_model, input_dict)
    produced = output_dict_p[list(output_dict_p.keys())[0]]
    assert np.isclose(expected, produced).all()
    assert np.argmax(produced) == 3
    os.remove(export_onnx_path)


@pytest.mark.transform
def test_batchnorm_to_affine_lfc_w1a1():
    lfc = get_test_model_trained("LFC", 1, 1)
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    new_model = model.transform(BatchNormToAffine())
    # load one of the test vectors
    raw_i = get_data("finn.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_dict = {"0": nph.to_array(input_tensor)}
    assert oxe.compare_execution(model, new_model, input_dict)
    os.remove(export_onnx_path)
