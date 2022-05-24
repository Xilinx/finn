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

import finn.core.data_layout as DataLayout
import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline.reorder import MoveFlattenPastAffine
from finn.util.basic import gen_finn_dt_tensor


@pytest.mark.streamline
# data layout
@pytest.mark.parametrize("data_layout", [DataLayout.NHWC, DataLayout.NCHW])
# batch size
@pytest.mark.parametrize("batch_size", [1, 2])
def test_move_flatten_past_affine(data_layout, batch_size):
    if data_layout == DataLayout.NHWC:
        ishape = [batch_size, 1, 1, 1024]
        oshape = [batch_size, 1000]
    else:
        ishape = [batch_size, 1024, 1, 1]
        oshape = [batch_size, 1000]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    a0 = helper.make_tensor_value_info("a1", TensorProto.FLOAT, [1024, 1000])
    a1 = helper.make_tensor_value_info("a2", TensorProto.FLOAT, [])
    a2 = helper.make_tensor_value_info("a3", TensorProto.FLOAT, [1000])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)

    flatten_node = helper.make_node("Flatten", ["inp"], ["flatten_out"])
    matmul_node = helper.make_node("MatMul", ["flatten_out", "a0"], ["matmul_out"])
    mul_node = helper.make_node("Mul", ["matmul_out", "a1"], ["mul_out"])
    add_node = helper.make_node("Add", ["mul_out", "a2"], ["outp"])

    graph = helper.make_graph(
        nodes=[flatten_node, matmul_node, mul_node, add_node],
        name="move-reshape-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[a0, a1, a2],
    )

    model = helper.make_model(graph, producer_name="move_reshape_model")
    model = ModelWrapper(model)

    # initialize values
    a0_values = gen_finn_dt_tensor(DataType["TERNARY"], [1024, 1000])
    model.set_initializer("a0", a0_values)
    a1_values = np.random.uniform(low=0.1, high=0.99, size=(1)).astype(np.float32)
    model.set_initializer("a1", a1_values)
    a2_values = np.random.uniform(low=-1, high=1, size=(1000)).astype(np.float32)
    model.set_initializer("a2", a2_values)

    model.set_tensor_datatype("inp", DataType["INT2"])
    model.set_tensor_layout("inp", data_layout)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # compare execution before and after transformation
    inp_values = gen_finn_dt_tensor(DataType["INT2"], ishape)
    idict = {model.graph.input[0].name: inp_values}
    model_transformed = model.transform(MoveFlattenPastAffine())
    assert oxe.compare_execution(model, model_transformed, idict)

    # depending on data layout check if graph is transformed or not
    if data_layout == DataLayout.NHWC:
        # check if nodes have new order in transformed graph
        assert model.graph != model_transformed.graph
        assert model_transformed.graph.node[-1].op_type == "Flatten"
    else:
        assert model.graph == model_transformed.graph
