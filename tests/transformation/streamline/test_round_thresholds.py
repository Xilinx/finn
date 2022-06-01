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

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

import finn.core.onnx_exec as oxe
from finn.transformation.streamline import RoundAndClipThresholds


@pytest.mark.streamline
def test_round_thresholds():
    v = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 4])
    thresholds = helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [4, 1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    node_def = helper.make_node(
        "MultiThreshold", ["v", "thresholds"], ["out"], domain="finn.custom_op.general"
    )
    graph_def = helper.make_graph([node_def], "test_model", [v, thresholds], [out])
    model_def = helper.make_model(graph_def)
    model = ModelWrapper(model_def)
    threshold_val = np.asarray([[-1.1], [0.7], [2.3], [5.1]], dtype=np.float32)
    model.set_initializer("thresholds", threshold_val)
    model.set_tensor_datatype("v", DataType["INT8"])
    inp_dict_f = {"v": np.floor(threshold_val).T}
    inp_dict_n = {"v": np.round(threshold_val).T}
    inp_dict_c = {"v": np.ceil(threshold_val).T}
    orig_f = oxe.execute_onnx(model, inp_dict_f)["out"]
    orig_n = oxe.execute_onnx(model, inp_dict_n)["out"]
    orig_c = oxe.execute_onnx(model, inp_dict_c)["out"]
    assert model.get_tensor_datatype("thresholds") == DataType["FLOAT32"]
    new_model = model.transform(RoundAndClipThresholds())
    # rounded up thresholds should have same dtype as input
    assert new_model.get_tensor_datatype("thresholds") == DataType["INT8"]
    new_f = oxe.execute_onnx(new_model, inp_dict_f)["out"]
    new_n = oxe.execute_onnx(new_model, inp_dict_n)["out"]
    new_c = oxe.execute_onnx(new_model, inp_dict_c)["out"]
    assert np.isclose(orig_f, new_f, atol=1e-3).all()
    assert np.isclose(orig_n, new_n, atol=1e-3).all()
    assert np.isclose(orig_c, new_c, atol=1e-3).all()
