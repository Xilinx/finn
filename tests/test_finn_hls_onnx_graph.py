import onnx
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper


def test_manually_construct_onnx_graph():

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 13, 64])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 1, 64])

    memInStrm_node = helper.make_node(
        "FIFO", ["inp"], ["memInStrm"], "memInStrm", depth=1024
    )
    FCLayer0_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["memInStrm", "weights0", "thresh0"],
        ["out1"],
        resType="ap_resource_lut()",
        MW=832,
        MH=1024,
        SIMD=64,
        PE=32,
        resDataType="Recast<XnorMul>",
    )
    inter0_node = helper.make_node("FIFO", ["out1"], ["inter0"], "inter0", depth=16)
    FCLayer1_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inter0", "weights1", "thresh1"],
        ["out2"],
        resType="ap_resource_lut()",
        MW=1024,
        MH=1024,
        SIMD=32,
        PE=64,
        resDataType="Recast<XnorMul>",
    )
    inter1_node = helper.make_node("FIFO", ["out2"], ["inter1"], "inter1", depth=16)
    FCLayer2_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inter1", "weights2", "thresh2"],
        ["out3"],
        resType="ap_resource_lut()",
        MW=1024,
        MH=1024,
        SIMD=64,
        PE=32,
        resDataType="Recast<XnorMul>",
    )
    inter2_node = helper.make_node("FIFO", ["out3"], ["inter2"], "inter2", depth=8)
    FCLayer3_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inter2", "weights3", "thresh3"],
        ["out4"],
        resType="ap_resource_lut()",
        MW=1024,
        MH=64,
        SIMD=8,
        PE=16,
        resDataType="Recast<XnorMul>",
    )
    memOutStrm_node = helper.make_node(
        "FIFO", ["out4"], ["outp"], "memOutStrm", depth=1024
    )

    graph = helper.make_graph(
        nodes=[
            memInStrm_node,
            FCLayer0_node,
            inter0_node,
            FCLayer1_node,
            inter1_node,
            FCLayer2_node,
            inter2_node,
            FCLayer3_node,
            memOutStrm_node,
        ],
        name="finn_hls_onnx_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[
            helper.make_tensor_value_info("memInStrm", TensorProto.FLOAT, [1, 13, 64]),
            helper.make_tensor_value_info("weights0", TensorProto.FLOAT, [64, 32, 416]),
            helper.make_tensor_value_info(
                "thresh0", TensorProto.FLOAT, [32, 32, 1, 16, 1]
            ),
            helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info("inter0", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info("weights1", TensorProto.FLOAT, [32, 64, 512]),
            helper.make_tensor_value_info(
                "thresh1", TensorProto.FLOAT, [16, 64, 1, 16, 1]
            ),
            helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 16, 64]),
            helper.make_tensor_value_info("inter1", TensorProto.FLOAT, [1, 16, 64]),
            helper.make_tensor_value_info("weights2", TensorProto.FLOAT, [64, 32, 512]),
            helper.make_tensor_value_info(
                "thresh2", TensorProto.FLOAT, [32, 32, 1, 16, 1]
            ),
            helper.make_tensor_value_info("out3", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info("inter2", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info("weights3", TensorProto.FLOAT, [8, 16, 512]),
            helper.make_tensor_value_info(
                "thresh3", TensorProto.FLOAT, [4, 16, 1, 16, 1]
            ),
            helper.make_tensor_value_info("out4", TensorProto.FLOAT, [1, 1, 64]),
        ],
    )
    model = helper.make_model(graph, producer_name="finn-hls-onnx-model")
    model = ModelWrapper(model)

    # set the tensor datatypes (in this case: all to bipolar)
    for tensor in graph.input:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])
    for tensor in graph.output:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])
    for tensor in graph.value_info:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])

    onnx.save(model.model, "finn-hls-onnx-model.onnx")
