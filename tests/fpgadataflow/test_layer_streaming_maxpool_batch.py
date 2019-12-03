import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.code_gen_transformation import CodeGen
from finn.transformation.fpgadataflow.compilation_transformation import Compilation


def test_layer_streaming_maxpool_batch():
    inp = helper.make_tensor_value_info("in", TensorProto.FLOAT, [2, 2, 4, 4])
    outp = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2, 2, 2])

    MaxPool_batch_node = helper.make_node(
        "StreamingMaxPool_Batch",
        ["in"],
        ["out"],
        domain="finn",
        backend="fpgadataflow",
        ImgDim=4,
        PoolDim=2,
        NumChannels=2,
    )

    graph = helper.make_graph(
        nodes=[MaxPool_batch_node],
        name="max_pool_batch_graph",
        inputs=[inp],
        outputs=[outp],
    )
    model = helper.make_model(graph, producer_name="finn-hls-onnx-model")
    model = ModelWrapper(model)

    # set the tensor datatypes (in this case: all to bipolar)
    for tensor in graph.input:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])
    for tensor in graph.output:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])

    # onnx.save(model.model, "max-pool-model.onnx")

    input_tensor = np.asarray(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=np.float32,
    ).reshape(2, 2, 4, 4)
    print(input_tensor)

    model = model.transform(CodeGen())
    model = model.transform(Compilation())

    input_dict = {"in": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    print(output_dict)
