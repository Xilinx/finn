import numpy as np
from onnx import TensorProto, helper

import finn.transformation.code_gen_transformation as cg_trafo
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper


def test_code_gen_trafo():
    inp = helper.make_tensor_value_info("in", TensorProto.FLOAT, [2, 2, 4, 4])
    outp = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2, 2, 2])

    MaxPool_batch_node = helper.make_node(
        "StreamingMaxPool_Batch",
        ["in"],
        ["out"],
        domain="finn",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
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

    input_dict = {"in": input_tensor}
    # cg_trafo.
