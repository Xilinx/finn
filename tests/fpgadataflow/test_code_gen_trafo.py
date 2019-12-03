import os

from onnx import TensorProto, helper

import finn.core.utils as util
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.codegen import CodeGen


def test_code_gen_trafo():
    idt = wdt = odt = DataType.BIPOLAR
    mw = 8
    mh = 8
    pe = 4
    simd = 4
    wmem = mw * mh // (pe * simd)
    nf = mh // pe
    sf = mw // simd

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, sf, simd])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, nf, pe])
    node_inp_list = ["inp", "weights", "thresh"]
    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        code_gen_dir="",
        executable_path="",
        resType="ap_resource_lut()",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        WMEM=wmem,
        TMEM=0,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    W = util.gen_finn_dt_tensor(wdt, (mw, mh))
    model.set_initializer("weights", W)

    model = model.transform(CodeGen())
    for node in model.graph.node:
        code_gen_attribute = util.get_by_name(node.attribute, "code_gen_dir")
        tmp_dir = code_gen_attribute.s.decode("UTF-8")
        assert os.path.isdir(
            tmp_dir
        ), """Code generation directory of node with
            op type {} does not exist!""".format(
            node.op_type
        )
        assert (
            len(os.listdir(tmp_dir)) != 0
        ), """Code generation directory of node with
            op type {} is empty!""".format(
            node.op_type
        )
