from onnx import TensorProto, helper

from finn.analysis.verify_custom_nodes import verify_nodes
from finn.core.modelwrapper import ModelWrapper


def check_two_dict_for_equality(dict1, dict2):
    for key in dict1:
        assert key in dict2, "Key: {} is not in both dictionaries".format(key)
        assert (
            dict1[key] == dict2[key]
        ), """Values for key {} are not the same
        in both dictionaries""".format(
            key
        )

    return True


def test_verify_custom_nodes():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 13, 64])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 1, 64])

    # MultiThreshold
    m_node = helper.make_node(
        "MultiThreshold",
        ["xnor_out", "threshs"],
        ["outp"],
        domain="finn",
        out_scale=2.0,
        out_bias=-1.0,
        out_dtype="",
    )

    # XnorPopcountMatMul
    xnor_node = helper.make_node(
        "XnorPopcountMatMul",
        ["fclayer_out0", "fclayer_out1"],
        ["xnor_out"],
        domain="finn",
    )

    # StreamingMaxPool_Batch
    MaxPool_batch_node = helper.make_node(
        "StreamingMaxPool_Batch",
        ["inp"],
        ["max_out"],
        domain="finn",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
        ImgDim=4,
        PoolDim=2,
        NumChannels=2,
    )

    # StreamingFCLayer_Batch - no activation
    FCLayer0_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["max_out", "weights"],
        ["fclayer_out0"],
        domain="finn",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
        resType="ap_resource_lut()",
        MW=8,
        MH=8,
        SIMD=4,
        PE=4,
        inputDataType="<FINN DataType>",
        weightDataType="<FINN DataType>",
        outputDataType="<FINN DataType>",
        ActVal=0,
        binaryXnorMode=1,
        noActivation=1,
    )

    # StreamingFCLayer_Batch - with activation
    FCLayer1_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["fclayer_out0", "weights", "threshs"],
        ["fclayer_out1"],
        domain="finn",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
        resType="ap_resource_lut()",
        MW=8,
        MH=8,
        SIMD=4,
        PE=4,
        inputDataType="<FINN DataType>",
        weightDataType="<FINN DataType>",
        outputDataType="<FINN DataType>",
        ActVal=0,
        binaryXnorMode=1,
        noActivation=0,
    )

    graph = helper.make_graph(
        nodes=[MaxPool_batch_node, FCLayer0_node, FCLayer1_node, xnor_node, m_node],
        name="custom_op_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[
            helper.make_tensor_value_info("max_out", TensorProto.FLOAT, [1, 13, 64]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [64, 32, 416]),
            helper.make_tensor_value_info("threshs", TensorProto.FLOAT, [32, 32, 16]),
            helper.make_tensor_value_info("xnor_out", TensorProto.FLOAT, [1, 32, 32]),
            helper.make_tensor_value_info(
                "fclayer_out0", TensorProto.FLOAT, [1, 32, 32]
            ),
            helper.make_tensor_value_info(
                "fclayer_out1", TensorProto.FLOAT, [32, 64, 512]
            ),
        ],
    )
    model = helper.make_model(graph, producer_name="custom-op-model")
    model = ModelWrapper(model)

    produced = model.analysis(verify_nodes)

    expected = {
        "StreamingMaxPool_Batch": [
            "The number of attributes is correct",
            "Attribute domain is set correctly",
            "Attribute backend is set correctly",
            "All necessary attributes exist",
            "The number of inputs is correct",
        ],
        "StreamingFCLayer_Batch": [
            "The number of attributes is correct",
            "Attribute domain is set correctly",
            "Attribute backend is set correctly",
            "All necessary attributes exist",
            "The number of inputs is correct",
        ],
        "XnorPopcountMatMul": [
            "The number of attributes is correct",
            "Attribute domain is set correctly",
            "XnorPopcountMatMul should not have any attributes",
            "The number of inputs is correct",
        ],
        "MultiThreshold": [
            "The number of attributes is correct",
            "Attribute domain is set correctly",
            "All necessary attributes exist",
            "The number of inputs is correct",
        ],
    }

    assert check_two_dict_for_equality(
        produced, expected
    ), """The produced output of
    the verification analysis pass is not equal to the expected one"""
