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
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.absorb import AbsorbScalarMulAddIntoTopK


@pytest.mark.streamline
# parameter to indicate if mul parameter is negative or positive
@pytest.mark.parametrize("mul_positive", [True, False])
# parameter to indicate if mul parameter is scalar or not
@pytest.mark.parametrize("scalar", [True, False])
def test_absorb_mul_into_topk(mul_positive, scalar):
    if scalar is True:
        shape = [1]
    else:
        shape = [1, 1, 1, 1000]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 1, 1, 1000])
    a0 = helper.make_tensor_value_info("a0", TensorProto.FLOAT, shape)
    b0 = helper.make_tensor_value_info("b0", TensorProto.FLOAT, [1, 1, 1, 1000])
    c0 = helper.make_tensor_value_info("c0", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 1, 1, 1000])

    mul_node = helper.make_node("Mul", ["inp", "a0"], ["b0"])
    add_node = helper.make_node("Add", ["b0", "c0"], ["outp"])
    mul_graph = helper.make_graph(
        nodes=[mul_node, add_node],
        name="mul-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[a0, b0, c0],
    )

    model = helper.make_model(mul_graph, producer_name="mul_model")
    model = ModelWrapper(model)
    # initialize values
    # for mul
    if mul_positive is True:
        a0_values = np.random.uniform(low=0.1, high=1, size=tuple(shape)).astype(
            np.float32
        )
    else:
        a0_values = np.random.uniform(low=-1, high=-0.1, size=tuple(shape)).astype(
            np.float32
        )
    model.set_initializer("a0", a0_values)
    # for add
    c0_values = np.random.uniform(low=-1, high=-0.1, size=tuple(shape)).astype(
        np.float32
    )
    model.set_initializer("c0", c0_values)
    model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model_transformed = model.transform(AbsorbScalarMulAddIntoTopK())

    # compare execution results
    inp_values = np.random.uniform(low=-10, high=10, size=(1, 1, 1, 1000)).astype(
        np.float32
    )
    idict = {"global_in": inp_values}
    odict = oxe.execute_onnx(model, idict, True)
    y_indices = odict["global_out"]
    y_values = odict["TopK_0_out0"]
    odict = oxe.execute_onnx(model_transformed, idict, True)
    y_tr_indices = odict["global_out"]
    y_tr_values = odict["TopK_0_out0"]

    # the indices stay the same, if the model is transformed or not
    assert (y_indices == y_tr_indices).all()

    if scalar is True and mul_positive is True:
        # the values change if the model was transformed
        assert (y_values != y_tr_values).all()

        # check for new order
        assert model.graph != model_transformed.graph
        assert len(model.graph.node) - 2 == len(model_transformed.graph.node)
        assert model_transformed.graph.node[0].op_type == "TopK"
