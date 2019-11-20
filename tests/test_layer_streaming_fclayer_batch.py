import onnx
from onnx import TensorProto, helper
import numpy as np

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper


def test_fclayer_batch():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 13, 64])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 32, 32])

    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inp", "weights", "thresh"],
        ["outp"],
        domain='finn',
        backend='fpgadataflow',
        resType="ap_resource_lut()",
        MW=832,
        MH=1024,
        SIMD=64,
        PE=32,
        resDataType="Recast<XnorMul>",
    )

    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp], value_info=[
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [64, 32, 416]),
            helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [32, 32, 1, 16, 1])]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    # set the tensor datatypes (in this case: all to bipolar)
    for tensor in graph.input:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])
    for tensor in graph.output:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])

    onnx.save(model.model, "fclayer-model.onnx")

    # generate input data
    input_tensor = np.random.randint(2, size=832)
    input_tensor = (np.asarray(input_tensor, dtype=np.float32)).reshape(1,13,64)
    print(input_tensor)

