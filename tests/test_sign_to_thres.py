import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
import finn.transformation.infer_shapes as si
import finn.transformation.streamline as sl
from finn.core.modelwrapper import ModelWrapper


def test_sign_to_thres():
    out0 = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [6, 3, 2, 2])
    graph_def = helper.make_graph(
        nodes=[
            helper.make_node("Sign", ["v"], ["out0"]),
            helper.make_node("Relu", ["out0"], ["out1"]),
        ],
        name="test-model",
        inputs=[helper.make_tensor_value_info("v", TensorProto.FLOAT, [6, 3, 2, 2])],
        value_info=[out0],
        outputs=[
            helper.make_tensor_value_info("out1", TensorProto.FLOAT, [6, 3, 2, 2])
        ],
    )
    model_def = helper.make_model(graph_def, producer_name="finn-test")
    model = ModelWrapper(model_def)
    model = model.transform_single(si.infer_shapes)
    input_dict = {}
    input_dict["v"] = np.random.randn(*[6, 3, 2, 2]).astype(np.float32)
    expected = oxe.execute_onnx(model, input_dict)["out1"]
    model = model.transform_single(sl.convert_sign_to_thres)
    assert model.graph.node[0].op_type == "MultiThreshold"
    produced = oxe.execute_onnx(model, input_dict)["out1"]
    assert np.isclose(expected, produced, atol=1e-3).all()
