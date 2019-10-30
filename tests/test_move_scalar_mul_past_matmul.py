import numpy as np
import onnx.helper as oh
from onnx import TensorProto

import finn.core.onnx_exec as ox
import finn.transformation.infer_shapes as si
import finn.transformation.streamline as tx
from finn.core.modelwrapper import ModelWrapper


def test_move_scalar_mul_past_matmul():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [1, 2])
    mul_param = oh.make_tensor_value_info("mul_param", TensorProto.FLOAT, [1, 1])
    matmul_param = oh.make_tensor_value_info("matmul_param", TensorProto.FLOAT, [2, 2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [1, 2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[mul_param, matmul_param],
            nodes=[
                oh.make_node("Mul", ["top_in", "mul_param"], ["middle"]),
                oh.make_node("MatMul", ["middle", "matmul_param"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform_single(si.infer_shapes)
    model.set_initializer("mul_param", np.asarray([[3]], dtype=np.float32))
    model.set_initializer(
        "matmul_param", np.asarray([[2, 4], [-1, 1]], dtype=np.float32)
    )
    new_model = model.transform_repeated(tx.move_scalar_mul_past_matmul)
    inp_dict = {"top_in": np.asarray([[-1.0, 1.0]], dtype=np.float32)}
    out_orig = ox.execute_onnx(model, inp_dict)["top_out"]
    out_transformed = ox.execute_onnx(new_model, inp_dict)["top_out"]
    assert np.isclose(out_orig, out_transformed).all()
    assert new_model.graph.node[0].op_type == "MatMul"
    assert new_model.graph.node[1].op_type == "Mul"
    assert new_model.graph.node[0].output[0] == new_model.graph.node[1].input[0]
