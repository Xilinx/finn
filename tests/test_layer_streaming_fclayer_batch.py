# import onnx
import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper


def test_fclayer_batch():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 2, 8])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4, 4])

    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inp", "weights", "thresh"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        MW=16,
        MH=16,
        SIMD=8,
        PE=4,
        resDataType="Recast<XnorMul>",
    )

    graph = helper.make_graph(
        nodes=[FCLayer_node],
        name="fclayer_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [8, 4, 16]),
            helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [16, 4, 3]),
        ],
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    # set the tensor datatypes (in this case: all to bipolar)
    for tensor in graph.input:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])
    for tensor in graph.output:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])

    # onnx.save(model.model, "fclayer-model.onnx")

    # generate input data
    input_tensor = np.random.randint(2, size=16)
    input_tensor = (np.asarray(input_tensor, dtype=np.float32)).reshape(1, 2, 8)
    input_dict = {"inp": input_tensor}

    # generate weights
    weights_tensor = np.random.randint(2, size=512)
    weights_tensor = (np.asarray(weights_tensor, dtype=np.float32)).reshape(8, 4, 16)
    input_dict["weights"] = weights_tensor

    # generate threshold activation
    thresh_tensor = np.random.randint(2, size=192)
    thresh_tensor = (np.asarray(thresh_tensor, dtype=np.float32)).reshape(16, 4, 3)
    input_dict["thresh"] = thresh_tensor

    output_dict = oxe.execute_onnx(model, input_dict)
    print(output_dict)
