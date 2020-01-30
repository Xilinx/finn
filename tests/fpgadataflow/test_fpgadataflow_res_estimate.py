from onnx import TensorProto, helper

from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveUniqueNodeNames


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


def test_res_estimate():
    mw = mh = 4
    simd = 1
    pe = 1
    idt = DataType.INT2
    wdt = DataType.INT2
    odt = DataType.INT32
    actval = odt.min()

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    node_inp_list = ["inp", "weights", "thresh"]

    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=0,
        noActivation=0,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)

    model = model.transform(GiveUniqueNodeNames())
    prod_resource_estimation = model.analysis(res_estimation)
    expect_resource_estimation = {
        "StreamingFCLayer_Batch_0": ["BRAMs: 1", "LUTs: 304.4"]
    }

    assert check_two_dict_for_equality(
        prod_resource_estimation, expect_resource_estimation
    ), """The produced output of
    the resource estimation analysis pass is not equal to the expected one"""
