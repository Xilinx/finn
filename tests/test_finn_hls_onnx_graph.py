import onnx
from onnx import TensorProto, helper


def test_manually_construct_onnx_graph():

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 64])

    Mem2Stream0_node = helper.make_node(
        "Mem2Stream_Batch", ["inp"], ["out0"], numReps=4, DataWidth=64, numBytes=104
    )
    memInStrm_node = helper.make_node("StreamingNode", ["out0"], ["memInStrm"])
    FCLayer0_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["memInStrm", "weights0", "thresh0"],
        ["out1"],
        numReps=4,
        resType="ap_resource_lut()",
        L_MW=832,
        L_MH=1024,
        L_SIMD=64,
        L_PE=32,
        resDataType="Recast<XnorMul>",
    )
    inter0_node = helper.make_node("StreamingNode", ["out1"], ["inter0"])
    FCLayer1_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inter0", "weights1", "thresh1"],
        ["out2"],
        numReps=4,
        resType="ap_resource_lut()",
        L_MW=1024,
        L_MH=1024,
        L_SIMD=32,
        L_PE=64,
        resDataType="Recast<XnorMul>",
    )
    inter1_node = helper.make_node("StreamingNode", ["out2"], ["inter1"])
    FCLayer2_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inter1", "weights2", "thresh2"],
        ["out3"],
        numReps=4,
        resType="ap_resource_lut()",
        L_MW=1024,
        L_MH=1024,
        L_SIMD=64,
        L_PE=32,
        resDataType="Recast<XnorMul>",
    )
    inter2_node = helper.make_node("StreamingNode", ["out3"], ["inter2"])
    FCLayer3_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inter2", "weights3", "thresh3"],
        ["out4"],
        numReps=4,
        resType="ap_resource_lut()",
        L_MW=1024,
        L_MH=64,
        L_SIMD=8,
        L_PE=16,
        resDataType="Recast<XnorMul>",
    )
    memOutStrm_node = helper.make_node("StreamingNode", ["out4"], ["memOutStrm"])
    Mem2Stream1_node = helper.make_node(
        "Mem2Stream_Batch",
        ["memOutStrm"],
        ["outp"],
        numReps=4,
        DataWidth=64,
        numBytes=8,
    )

    graph = helper.make_graph(
        nodes=[
            Mem2Stream0_node,
            memInStrm_node,
            FCLayer0_node,
            inter0_node,
            FCLayer1_node,
            inter1_node,
            FCLayer2_node,
            inter2_node,
            FCLayer3_node,
            memOutStrm_node,
            Mem2Stream1_node,
        ],
        name="finn_hls_onnx_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[
            helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info("memInStrm", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info("weights0", TensorProto.FLOAT, [64, 32, 416]),
            helper.make_tensor_value_info(
                "thresh0", TensorProto.FLOAT, [32, 32, 1, 16, 1]
            ),
            helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 16, 64]),
            helper.make_tensor_value_info("inter0", TensorProto.FLOAT, [1, 16, 64]),
            helper.make_tensor_value_info("weights1", TensorProto.FLOAT, [32, 64, 512]),
            helper.make_tensor_value_info(
                "thresh1", TensorProto.FLOAT, [16, 64, 1, 16, 1]
            ),
            helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info("inter1", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info("weights2", TensorProto.FLOAT, [64, 32, 512]),
            helper.make_tensor_value_info(
                "thresh2", TensorProto.FLOAT, [32, 32, 1, 16, 1]
            ),
            helper.make_tensor_value_info("out3", TensorProto.FLOAT, [1, 64, 16]),
            helper.make_tensor_value_info("inter2", TensorProto.FLOAT, [1, 64, 16]),
            helper.make_tensor_value_info("weights3", TensorProto.FLOAT, [8, 16, 512]),
            helper.make_tensor_value_info(
                "thresh3", TensorProto.FLOAT, [4, 16, 1, 16, 1]
            ),
            helper.make_tensor_value_info("out4", TensorProto.FLOAT, [1, 64]),
            helper.make_tensor_value_info("memOutStrm", TensorProto.FLOAT, [1, 64]),
        ],
    )
    model = helper.make_model(graph, producer_name="finn-hls-onnx-model")
    onnx.save(model, "finn-hls-onnx-model.onnx")
