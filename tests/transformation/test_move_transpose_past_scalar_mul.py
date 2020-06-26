import pytest

import numpy as np
from onnx import TensorProto, helper

from finn.core.modelwrapper import ModelWrapper
import finn.core.data_layout as DataLayout
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.transformation.streamline.reorder import MoveTransposePastScalarMul
import finn.core.onnx_exec as oxe

# permutation of transpose node
@pytest.mark.parametrize("perm", [[0, 2, 3, 1], [0, 1, 3, 2], [3, 2, 0, 1]])
# scalar mul
@pytest.mark.parametrize("scalar", [True, False])
# data layout
@pytest.mark.parametrize("data_layout", [DataLayout.NHWC, DataLayout.NCHW])
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

    model = helper.make_model(graph, producer_name="mv_transpose_model")
    model = ModelWrapper(model)

    # initialize values
    a0_values = np.random.uniform(low=0, high=1, size=tuple(a0_size)).astype(np.float32)
    model.set_initializer("a0", a0_values)
    model.set_tensor_layout("inp", data_layout)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # compare execution before and after transformation
    inp_values = np.random.uniform(low=0, high=1, size=(1, 2, 3, 4)).astype(np.float32)
    idict = {model.graph.input[0].name: inp_values}
    model_transformed = model.transform(MoveTransposePastScalarMul())
    assert oxe.compare_execution(model, model_transformed, idict)

    # check if order changed
    if scalar is True:
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
        mul_input = model_transformed.graph.node[1].input[0]
        mul_output = model_transformed.graph.node[1].output[0]
        assert model_transformed.get_tensor_layout(mul_input) != data_layout
        assert model_transformed.get_tensor_layout(mul_output) != data_layout
