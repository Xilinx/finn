import numpy as np
from onnx import TensorProto, helper


def test_execute_mixed_model():

    out0 = helper.make_tensor_value_info("out", TensorProto.FLOAT, [6, 3, 2, 2])

    graph_def = helper.make_graph(
        nodes=[
            helper.make_node(
                "MultiThreshold", ["v", "thresholds"], ["out"], domain="finn"
            ),
            helper.make_node("Relu", ["out0"], ["out1"]),
        ],
        name="test-model",
        inputs=[
            helper.make_tensor_value_info("v", TensorProto.FLOAT, [6, 3, 2, 2]),
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [3, 7]),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [6, 3, 2, 2])],
    )
