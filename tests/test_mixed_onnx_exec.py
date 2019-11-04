import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper


def test_execute_mixed_model():

    out0 = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [6, 3, 2, 2])

    graph_def = helper.make_graph(
        nodes=[
            helper.make_node(
                "MultiThreshold", ["v", "thresholds"], ["out0"], domain="finn"
            ),
            helper.make_node("Relu", ["out0"], ["out1"]),
        ],
        name="test-model",
        inputs=[
            helper.make_tensor_value_info("v", TensorProto.FLOAT, [6, 3, 2, 2]),
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [3, 7]),
        ],
        outputs=[
            helper.make_tensor_value_info("out1", TensorProto.FLOAT, [6, 3, 2, 2])
        ],
        value_info=[out0],
    )
    model_def = helper.make_model(graph_def, producer_name="onnx-example")

    model = ModelWrapper(model_def)
    model = model.transform_single(si.infer_shapes)

    inputs = np.asarray(
        [
            4.8,
            3.2,
            1.2,
            4.9,
            7.8,
            2.4,
            3.1,
            4.7,
            6.2,
            5.1,
            4.9,
            2.2,
            6.2,
            0.0,
            0.8,
            4.7,
            0.2,
            5.6,
            8.9,
            9.2,
            9.1,
            4.0,
            3.3,
            4.9,
            2.3,
            1.7,
            1.3,
            2.2,
            4.6,
            3.4,
            3.7,
            9.8,
            4.7,
            4.9,
            2.8,
            2.7,
            8.3,
            6.7,
            4.2,
            7.1,
            2.8,
            3.1,
            0.8,
            0.6,
            4.4,
            2.7,
            6.3,
            6.1,
            1.4,
            5.3,
            2.3,
            1.9,
            4.7,
            8.1,
            9.3,
            3.7,
            2.7,
            5.1,
            4.2,
            1.8,
            4.1,
            7.3,
            7.1,
            0.4,
            0.2,
            1.3,
            4.3,
            8.9,
            1.4,
            1.6,
            8.3,
            9.4,
        ],
        dtype=np.float32,
    ).reshape(6, 3, 2, 2)

    threshold_values = np.asarray(
        [
            0.8,
            1.4,
            1.7,
            3.5,
            5.2,
            6.8,
            8.2,
            0.2,
            2.2,
            3.5,
            4.5,
            6.6,
            8.6,
            9.2,
            1.3,
            4.1,
            4.5,
            6.5,
            7.8,
            8.1,
            8.9,
        ],
        dtype=np.float32,
    ).reshape(3, 7)

    input_dict = {}
    input_dict["v"] = inputs
    input_dict["thresholds"] = threshold_values

    output_dict = oxe.execute_onnx(model, input_dict)

    outputs = np.asarray(
        [
            4.0,
            3.0,
            1.0,
            4.0,
            5.0,
            2.0,
            2.0,
            4.0,
            3.0,
            3.0,
            3.0,
            1.0,
            5.0,
            0.0,
            1.0,
            4.0,
            1.0,
            4.0,
            6.0,
            7.0,
            7.0,
            1.0,
            1.0,
            3.0,
            3.0,
            3.0,
            1.0,
            3.0,
            4.0,
            2.0,
            3.0,
            7.0,
            3.0,
            3.0,
            1.0,
            1.0,
            7.0,
            5.0,
            4.0,
            6.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            1.0,
            3.0,
            3.0,
            2.0,
            5.0,
            3.0,
            3.0,
            4.0,
            5.0,
            7.0,
            3.0,
            1.0,
            3.0,
            2.0,
            1.0,
            4.0,
            6.0,
            6.0,
            0.0,
            1.0,
            1.0,
            3.0,
            6.0,
            1.0,
            1.0,
            6.0,
            7.0,
        ],
        dtype=np.float32,
    ).reshape(6, 3, 2, 2)

    assert (output_dict["out1"] == outputs).all()
