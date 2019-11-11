# import onnx
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper


def test_layer_streaming_maxpool():
    inp = helper.make_tensor_value_info("in", TensorProto.FLOAT, [2, 4, 4])
    outp = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2, 2])

    MaxPool_node = helper.make_node(
        "StreamingMaxPool",
        ["in"],
        ["out"],
        domain="finn",
        backend="fpgadataflow",
        ImgDim=4,
        PoolDim=2,
        NumChannels=2,
    )

    graph = helper.make_graph(
        nodes=[MaxPool_node], name="max_pool_graph", inputs=[inp], outputs=[outp],
    )
    model = helper.make_model(graph, producer_name="finn-hls-onnx-model")
    model = ModelWrapper(model)

    # set the tensor datatypes (in this case: all to bipolar)
    for tensor in graph.input:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])
    for tensor in graph.output:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])

    # onnx.save(model.model, "max-pool-model.onnx")
