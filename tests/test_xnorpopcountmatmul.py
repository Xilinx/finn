import numpy as np
import onnx.helper as helper
from onnx import TensorProto

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes


def test_xnorpopcountmatmul():
    M = 1
    K = 3
    N = 3
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [K, N])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["x", "y"])
    node_def = helper.make_node(
        "XnorPopcountMatMul", ["x", "W"], ["out"], domain="finn"
    )
    modelproto = helper.make_model(
        helper.make_graph([node_def], "test_model", [x], [out], value_info=[W])
    )
    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("x", DataType.BINARY)
    model.set_tensor_datatype("W", DataType.BINARY)
    W_data = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    model.set_initializer("W", W_data)
    # test shape inference
    model = model.transform(InferShapes())
    assert model.get_tensor_shape("out") == [M, N]
    # test datatype inference
    assert model.get_tensor_datatype("out") is DataType.FLOAT32
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("out") is DataType.UINT32
    # test execution
    x_data = np.asarray([[1, 0, 0]], dtype=np.float32)
    inp_dict = {"x": x_data}
    out_dict = oxe.execute_onnx(model, inp_dict)
    Wb = 2 * W_data - 1
    xb = 2 * x_data - 1
    rb = np.matmul(xb, Wb)
    assert (2 * out_dict["out"] - K == rb).all()
