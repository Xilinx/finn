import pytest

import numpy as np
from onnx import helper, TensorProto
import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline.remove import RemoveIdentityOps
from finn.util.basic import gen_finn_dt_tensor


def insert_identity_op(model, op):
    if op in ["Add", "Sub"]:
        val = np.asarray([0.0], dtype=np.float32)
    elif op in ["Mul", "Div"]:
        val = np.asarray([1.0], dtype=np.float32)
    else:
        return

    identity_node = helper.make_node(op, ["div_out", "value"], ["ident_out"])
    graph = model.graph
    graph.node.insert(3, identity_node)
    graph.node[-1].input[0] = "ident_out"
    model.set_initializer("value", val)

    return model


# identity operations to be inserted
@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div"])
def test_remove_identity_ops(op):

    # set up onnx model
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 1, 1])
    mul = helper.make_tensor_value_info("mul", TensorProto.FLOAT, [])
    shape = helper.make_tensor_value_info("shape", TensorProto.FLOAT, [2])
    div = helper.make_tensor_value_info("div", TensorProto.FLOAT, [])
    matmul = helper.make_tensor_value_info("matmul", TensorProto.FLOAT, [4, 2])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 2])

    mul_node = helper.make_node("Mul", ["inp", "mul"], ["mul_out"])
    reshape_node = helper.make_node("Reshape", ["mul_out", "shape"], ["reshape_out"])
    div_node = helper.make_node("Div", ["reshape_out", "div"], ["div_out"])
    matmul_node = helper.make_node("MatMul", ["div_out", "matmul"], ["outp"])

    graph = helper.make_graph(
        nodes=[mul_node, reshape_node, div_node, matmul_node],
        name="identity-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[mul, shape, div, matmul],
    )

    model = helper.make_model(graph, producer_name="mulpastconv-model")
    model = ModelWrapper(model)
    inp_values = gen_finn_dt_tensor(DataType.INT2, [1, 4, 1, 1])
    mul_values = np.random.uniform(low=0.1, high=0.99, size=(1)).astype(np.float32)
    shape_values = np.asarray([1, -1], dtype=np.int64)
    div_values = np.random.uniform(low=0.1, high=0.99, size=(1)).astype(np.float32)
    matmul_values = gen_finn_dt_tensor(DataType.INT2, [4, 2])
    model.set_initializer("mul", mul_values)
    model.set_initializer("shape", shape_values)
    model.set_initializer("div", div_values)
    model.set_initializer("matmul", matmul_values)
    insert_identity_op(model, op)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    idict = {"inp": inp_values}
    odict = oxe.execute_onnx(model, idict)
    out_before = odict["outp"]
    num_of_nodes_before = len(model.graph.node)

    model = model.transform(RemoveIdentityOps())
    num_of_nodes_after = len(model.graph.node)
    assert num_of_nodes_before - 1 == num_of_nodes_after

    odict = oxe.execute_onnx(model, idict)
    out_after = odict["outp"]
    assert (out_before == out_after).all()
