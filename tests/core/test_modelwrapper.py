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
from collections import Counter

import brevitas.onnx as bo
import numpy as np

from finn.core.modelwrapper import ModelWrapper
from finn.util.test import get_test_model_trained

export_onnx_path = "test_output_lfc.onnx"


def test_modelwrapper():
    lfc = get_test_model_trained("LFC", 1, 1)
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    assert model.check_all_tensor_shapes_specified() is False
    inp_name = model.graph.input[0].name
    inp_shape = model.get_tensor_shape(inp_name)
    assert inp_shape == [1, 1, 28, 28]
    # find first matmul node
    l0_mat_tensor_name = ""
    l0_inp_tensor_name = ""
    for node in model.graph.node:
        if node.op_type == "MatMul":
            l0_inp_tensor_name = node.input[0]
            l0_mat_tensor_name = node.input[1]
            break
    assert l0_mat_tensor_name != ""
    l0_weights = model.get_initializer(l0_mat_tensor_name)
    assert l0_weights.shape == (784, 1024)
    l0_weights_hist = Counter(l0_weights.flatten())
    assert (l0_weights_hist[1.0] + l0_weights_hist[-1.0]) == 784 * 1024
    l0_weights_rand = np.random.randn(784, 1024)
    model.set_initializer(l0_mat_tensor_name, l0_weights_rand)
    assert (model.get_initializer(l0_mat_tensor_name) == l0_weights_rand).all()
    assert l0_inp_tensor_name != ""
    inp_cons = model.find_consumer(l0_inp_tensor_name)
    assert inp_cons.op_type == "MatMul"
    out_prod = model.find_producer(l0_inp_tensor_name)
    assert out_prod.op_type == "Sign"
    os.remove(export_onnx_path)
