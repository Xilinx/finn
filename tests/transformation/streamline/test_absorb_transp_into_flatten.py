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
from finn.transformation.streamline.absorb import AbsorbTransposeIntoFlatten


@pytest.mark.streamline
# permutation of transpose node
@pytest.mark.parametrize("perm", [[0, 2, 3, 1], [0, 1, 3, 2], [3, 2, 0, 1]])
# reshape or flatten
@pytest.mark.parametrize("shape", [None, [1, -1], [-1, 1]])
# input shape
@pytest.mark.parametrize("ishape", [[1, 1, 1, 4], [2, 4, 1, 1], [1, 2, 2, 4]])
# datalayout
@pytest.mark.parametrize("data_layout", ["NCHW", "NHWC"])
def test_absorb_transp_into_flatten(perm, shape, ishape, data_layout):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    transp_node = helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm)
    dummy_in = np.random.uniform(low=0, high=1, size=tuple(ishape)).astype(np.float32)
    if shape is None:
        shape_node = helper.make_node("Flatten", ["transp_out"], ["outp"])
        dummy_in = dummy_in.transpose(tuple(perm))
        oshape = dummy_in.reshape(dummy_in.shape[0], -1).shape
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)
        shape0 = None
    else:
        shape0 = helper.make_tensor_value_info("shape0", TensorProto.FLOAT, shape)
        shape_node = helper.make_node("Reshape", ["transp_out", "shape0"], ["outp"])
        oshape = dummy_in.transpose(tuple(perm)).reshape(tuple(shape)).shape
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)

    graph = helper.make_graph(
        nodes=[transp_node, shape_node],
        name="absorb-transpose-graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="absorb_transpose_model")
    model = ModelWrapper(model)
    if shape is not None:
        model.graph.value_info.append(shape0)
        model.set_initializer("shape0", np.asarray(shape))
    if data_layout == "NCHW":
        model.set_tensor_layout("inp", DataLayout.NCHW)
    else:
        model.set_tensor_layout("inp", DataLayout.NHWC)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # model.save("test.onnx")
    model_transformed = model.transform(AbsorbTransposeIntoFlatten())
    # model_transformed.save("test2.onnx")

    # verify transformation
    inp_values = np.random.uniform(low=-1, high=1, size=tuple(ishape)).astype(
        np.float32
    )
    idict = {model.graph.input[0].name: inp_values}
    assert oxe.compare_execution(model, model_transformed, idict)

    # only some of the parameter combinations lead to a graph that will be changed when
    # AbsorbTransposeIntoFlatten is applied

    if shape == [-1, 1]:  # not a flatten operation, so the graph will not be changed
        assert model.graph == model_transformed.graph

    elif perm == [
        3,
        2,
        0,
        1,
    ]:  # the first dimension is also part of the transpose operation
        # so the graph will not be changed
        assert model.graph == model_transformed.graph

    # the following cases are the ones in which the model is transformed
    # because we tested the parameters shape and perm befire we can only consider ishape
    # and data_layout (the transformed model should only contain a "Flatten" node)
    elif ishape == [1, 1, 1, 4] and data_layout == "NHWC":
        assert model_transformed.graph.node[0].op_type == "Flatten"

    elif ishape == [2, 4, 1, 1] and data_layout == "NCHW" and shape is None:
        # If the first  dimension of the input tensor is not 1, flatten and
        # reshape (with shape = [1, -1]) would lead to different results
        assert model_transformed.graph.node[0].op_type == "Flatten"

    # all other cases lead to an unchanged model
    else:
        assert model.graph == model_transformed.graph
