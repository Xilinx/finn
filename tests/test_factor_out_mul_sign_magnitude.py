import numpy as np
import onnx.helper as oh
from onnx import TensorProto

import finn.core.onnx_exec as ox
import finn.transformation.infer_shapes as si
import finn.transformation.streamline as tx
from finn.core.modelwrapper import ModelWrapper


def test_factor_out_mul_sign_magnitude():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [1, 2])
    mul_param = oh.make_tensor_value_info("mul_param", TensorProto.FLOAT, [1, 2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [1, 2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[mul_param],
            nodes=[oh.make_node("Mul", ["top_in", "mul_param"], ["top_out"])],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform_single(si.infer_shapes)
    model.set_initializer("mul_param", np.asarray([[-1, 4]], dtype=np.float32))
    new_model = model.transform_repeated(tx.factor_out_mul_sign_magnitude)
    inp_dict = {"top_in": np.asarray([[-1.0, 1.0]], dtype=np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)
