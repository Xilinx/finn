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

from onnx import TensorProto, helper

from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
import finn.core.data_layout as DataLayout
from finn.util.basic import gen_finn_dt_tensor
from finn.transformation.insert_topk import InsertTopK
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.transformation.streamline.reorder import MoveFlattenPastTopK
import finn.core.onnx_exec as oxe

# data layout
@pytest.mark.parametrize("data_layout", [DataLayout.NHWC, DataLayout.NCHW])
# batch size
@pytest.mark.parametrize("batch_size", [1, 2])
def test_move_flatten_past_affine(data_layout, batch_size):
    if data_layout == DataLayout.NHWC:
        ishape = [batch_size, 1, 1, 1024]
        oshape = [batch_size, 1024]
    else:
        ishape = [batch_size, 1024, 1, 1]
        oshape = [batch_size, 1024]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)

    flatten_node = helper.make_node("Flatten", ["inp"], ["outp"])

    graph = helper.make_graph(
        nodes=[flatten_node], name="move-flatten-graph", inputs=[inp], outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="move_flatten_model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", DataType.INT2)
    model.set_tensor_layout("inp", data_layout)
    model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # compare execution before and after transformation
    inp_values = gen_finn_dt_tensor(DataType.INT2, ishape)
    idict = {model.graph.input[0].name: inp_values}
    model_transformed = model.transform(MoveFlattenPastTopK())
    assert oxe.compare_execution(model, model_transformed, idict)

    # depending on data layout check if graph is transformed or not
    if data_layout == DataLayout.NHWC:
        # check if nodes have new order in transformed graph
        assert model.graph != model_transformed.graph
        assert model_transformed.graph.node[-1].op_type == "Flatten"
    else:
        assert model.graph == model_transformed.graph
