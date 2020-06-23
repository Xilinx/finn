import numpy as np
from onnx import TensorProto, helper

from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.util.basic import gen_finn_dt_tensor
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.transformation.streamline.reorder import MoveFlatten
import finn.core.onnx_exec as oxe


def test_move_flatten():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 1, 1, 1024])
    a0 = helper.make_tensor_value_info("a0", TensorProto.FLOAT, [])
    shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
    a1 = helper.make_tensor_value_info("a1", TensorProto.FLOAT, [1024, 1000])
    a2 = helper.make_tensor_value_info("a2", TensorProto.FLOAT, [])
    a3 = helper.make_tensor_value_info("a3", TensorProto.FLOAT, [1000])
    k = helper.make_tensor_value_info("k", TensorProto.INT64, [1])

    outp_values = helper.make_tensor_value_info(
        "outp_values", TensorProto.FLOAT, [1, 5]
    )
    outp_indices = helper.make_tensor_value_info(
        "outp_indices", TensorProto.INT64, [1, 5]
    )
    mul0_node = helper.make_node("Mul", ["inp", "a0"], ["mul0_out"])
    reshape_node = helper.make_node("Reshape", ["mul0_out", "shape"], ["reshape_out"])
    matmul_node = helper.make_node("MatMul", ["reshape_out", "a1"], ["matmul_out"])
    mul1_node = helper.make_node("Mul", ["matmul_out", "a2"], ["mul1_out"])
    add_node = helper.make_node("Add", ["mul1_out", "a3"], ["add_out"])
    topk_node = helper.make_node(
        "TopK", ["add_out", "k"], ["outp_values", "outp_indices"]
    )

    graph = helper.make_graph(
        nodes=[mul0_node, reshape_node, matmul_node, mul1_node, add_node, topk_node],
        name="move-reshape-graph",
        inputs=[inp],
        outputs=[outp_values, outp_indices],
        value_info=[a0, shape, a1, a2, a3, k],
    )

    model = helper.make_model(graph, producer_name="move_reshape_model")
    model = ModelWrapper(model)

    # initialize values
    a0_values = np.random.uniform(low=0.1, high=0.99, size=(1)).astype(np.float32)
    model.set_initializer("a0", a0_values)
    shape_values = np.asarray([1, -1]).astype(np.int64)
    model.set_initializer("shape", shape_values)
    a1_values = gen_finn_dt_tensor(DataType.TERNARY, [1024, 1000])
    model.set_initializer("a1", a1_values)
    a2_values = 1.0 / a0_values
    model.set_initializer("a2", a2_values)
    a3_values = np.random.uniform(low=-1, high=1, size=(1000)).astype(np.float32)
    model.set_initializer("a3", a3_values)
    model.set_initializer("k", np.asarray([5]).astype(np.int64))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # compare execution before and after transformation
    inp_values = gen_finn_dt_tensor(DataType.INT2, [1, 1, 1, 1024])
    idict = {"inp": inp_values}
    model_transformed = model.transform(MoveFlatten())
    assert oxe.compare_execution(model, model_transformed, idict)
