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
import qonnx.core.data_layout as DataLayout
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveTransposePastScalarMul


@pytest.mark.streamline
# permutation of transpose node
@pytest.mark.parametrize("perm", [[0, 2, 3, 1], [0, 1, 3, 2], [3, 2, 0, 1]])
# scalar mul
@pytest.mark.parametrize("scalar", [True, False])
# data layout
@pytest.mark.parametrize("data_layout", [None, DataLayout.NHWC, DataLayout.NCHW])
def test_move_transpose_past_scalar_mul(perm, scalar, data_layout):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 2, 3, 4])
    # to determine out_size we need to calculate with "perm" for this test case
    dummy_in = np.random.uniform(low=0, high=1, size=(1, 2, 3, 4)).astype(np.float32)
    out_size = dummy_in.transpose(tuple(perm)).shape

    if scalar is True:
        a0_size = []
    else:
        a0_size = out_size
    a0 = helper.make_tensor_value_info("a0", TensorProto.FLOAT, a0_size)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_size)
    transp_node = helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm)
    mul_node = helper.make_node("Mul", ["transp_out", "a0"], ["outp"])

    graph = helper.make_graph(
        nodes=[transp_node, mul_node],
        name="mv-transpose-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[a0],
    )

    model = qonnx_make_model(graph, producer_name="mv_transpose_model")
    model = ModelWrapper(model)

    # initialize values
    a0_values = np.random.uniform(low=0, high=1, size=tuple(a0_size)).astype(np.float32)
    model.set_initializer("a0", a0_values)
    if data_layout is not None:
        model.set_tensor_layout("inp", data_layout)
        model = model.transform(InferDataLayouts())

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # compare execution before and after transformation
    inp_values = np.random.uniform(low=0, high=1, size=(1, 2, 3, 4)).astype(np.float32)
    idict = {model.graph.input[0].name: inp_values}
    model_transformed = model.transform(MoveTransposePastScalarMul())
    assert oxe.compare_execution(model, model_transformed, idict)

    # check if order changed
    if scalar is True and data_layout is not None:
        assert model_transformed.graph.node[0] != model.graph.node[0]
        assert model_transformed.graph.node[1] != model.graph.node[1]
        assert model_transformed.graph.node[0].op_type == "Mul"
        assert model_transformed.graph.node[1].op_type == "Transpose"
        mul_input = model_transformed.graph.node[0].input[0]
        mul_output = model_transformed.graph.node[0].output[0]
        assert model_transformed.get_tensor_layout(mul_input) == data_layout
        assert model_transformed.get_tensor_layout(mul_output) == data_layout
    else:
        assert model_transformed.graph.node[0] == model.graph.node[0]
        assert model_transformed.graph.node[1] == model.graph.node[1]
        if data_layout is not None:
            mul_input = model_transformed.graph.node[1].input[0]
            mul_output = model_transformed.graph.node[1].output[0]
            assert model_transformed.get_tensor_layout(mul_input) != data_layout
            assert model_transformed.get_tensor_layout(mul_output) != data_layout
