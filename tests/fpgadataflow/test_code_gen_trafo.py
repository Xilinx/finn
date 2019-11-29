import numpy as np
from onnx import TensorProto, helper

import finn.core.utils as util
import finn.transformation.code_gen_transformation as cg_trafo
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper


def test_code_gen_trafo():
    idt = wdt = odt = DataType.BIPOLAR
    tdt = DataType.UINT32
    mw = 8
    mh = 8
    pe = 4
    simd = 4
    wmem = mw * mh // (pe * simd)
    assert mw * mh == wmem * pe * simd
    nf = mh // pe
    sf = mw // simd
    tmem = nf

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
        TMEM=tmem,
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
    W = util.gen_finn_dt_tensor(wdt, (mh, mw))
    model.set_initializer("weights", W)
    model.set_tensor_datatype("thresh", tdt)
    T = np.zeros((1, 1))
    model.set_initializer("thresh", T)

    context = {}
    context["weights"] = W
    context["threshs"] = T
    for node in model.graph.node:
        cg_trafo.code_gen_transformation(node, context)
