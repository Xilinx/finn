from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.utils import get_by_name


def test_set_attribute():
    mw = 8
    mh = 8
    pe = 4
    simd = 4
    wmem = mw * mh // (pe * simd)
    nf = mh // pe
    sf = mw // simd
    idt = odt = wdt = DataType.BIPOLAR

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, sf, simd])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, nf, pe])
    node_inp_list = ["inp", "weights"]

    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="",
        code_gen_dir="",
        executable_path="",
        resType="ap_resource_lut()",
        MW=0,
        MH=mh,
        SIMD=simd,
        PE=pe,
        WMEM=wmem,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    # test cases
    value_to_set = "fpgadataflow"
    model.set_attribute(FCLayer_node, "backend", value_to_set)
    value = get_by_name(FCLayer_node.attribute, "backend")
    assert value.s.decode("UTF-8") == value_to_set

    value_to_set = mw
    model.set_attribute(FCLayer_node, "MW", value_to_set)
    value = get_by_name(FCLayer_node.attribute, "MW")
    assert value.i == value_to_set

    value_to_set = idt.name
    model.set_attribute(FCLayer_node, "inputDataType", value_to_set)
    value = get_by_name(FCLayer_node.attribute, "inputDataType")
    assert value.s.decode("UTF-8") == value_to_set
