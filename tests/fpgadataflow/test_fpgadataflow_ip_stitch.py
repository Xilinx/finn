import os.path

import numpy as np
import pytest
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.utils import calculate_signed_dot_prod_range, gen_finn_dt_tensor
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.codegen_ipstitch import CodeGen_ipstitch
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
from finn.transformation.fpgadataflow.make_pynq_proj import MakePYNQProject
from finn.transformation.fpgadataflow.synth_pynq_proj import SynthPYNQProject
from finn.transformation.general import GiveUniqueNodeNames


def create_two_fc_model():
    # create a model with two StreamingFCLayer instances
    wdt = DataType.INT2
    idt = DataType.INT2
    odt = DataType.INT2
    act = DataType.INT2
    m = 4
    tdt = DataType.INT32
    actval = odt.min()
    no_act = 0
    binary_xnor_mode = 0

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])
    outp_tlast = helper.make_tensor_value_info("outp_tlast", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inp", "w0", "t0"],
        ["mid"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        MW=m,
        MH=m,
        SIMD=1,
        PE=1,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )

    fc1 = helper.make_node(
        "StreamingFCLayer_Batch",
        ["mid", "w1", "t1"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        MW=m,
        MH=m,
        SIMD=1,
        PE=1,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )

    tlastmarker = helper.make_node(
        "TLastMarker",
        ["outp"],
        ["outp_tlast"],
        domain="finn",
        backend="fpgadataflow",
        NumIters=m,
        StreamWidth=2,
    )

    graph = helper.make_graph(
        nodes=[fc0, fc1, tlastmarker],
        name="fclayer_graph",
        inputs=[inp],
        outputs=[outp_tlast],
        value_info=[mid, outp],
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("mid", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("outp_tlast", odt)
    model.set_tensor_datatype("w0", wdt)
    model.set_tensor_datatype("w1", wdt)

    # generate weights
    w0 = gen_finn_dt_tensor(wdt, (m, m))
    w1 = gen_finn_dt_tensor(wdt, (m, m))
    model.set_initializer("w0", w0)
    model.set_initializer("w1", w1)

    # generate thresholds
    (min, max) = calculate_signed_dot_prod_range(idt, wdt, m)
    n_steps = act.get_num_possible_values() - 1
    t0 = np.random.randint(min, max - 1, (m, n_steps)).astype(np.float32)
    t1 = np.random.randint(min, max - 1, (m, n_steps)).astype(np.float32)
    # provide non-decreasing thresholds
    t0 = np.sort(t0, axis=1)
    t1 = np.sort(t1, axis=1)

    model.set_initializer("t0", t0)
    model.set_initializer("t1", t1)
    model.set_tensor_datatype("t0", tdt)
    model.set_tensor_datatype("t1", tdt)
    return model


@pytest.mark.dependency()
def test_fpgadataflow_ip_gen_two_fc_model():
    model = create_two_fc_model()
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(CodeGen_ipgen("xc7z020clg400-1", 5))
    model = model.transform(HLSSynth_IPGen())
    assert model.graph.node[0].op_type == "StreamingFCLayer_Batch"
    assert model.graph.node[1].op_type == "StreamingFCLayer_Batch"
    assert model.graph.node[2].op_type == "TLastMarker"
    model.save("/tmp/test_fpgadataflow_ip_gen_two_fc_model.onnx")

@pytest.mark.dependency(depends=["test_fpgadataflow_ip_gen_two_fc_model"])
def test_fpgadataflow_ip_stitch():
    model = ModelWrapper("/tmp/test_fpgadataflow_ip_gen_two_fc_model.onnx")
    model = model.transform(CodeGen_ipstitch("xc7z020clg400-1"))
    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    assert vivado_stitch_proj_dir is not None
    assert os.path.isdir(vivado_stitch_proj_dir)
    assert os.path.isfile(vivado_stitch_proj_dir + "/ip/component.xml")
    vivado_stitch_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
    assert vivado_stitch_vlnv is not None
    assert vivado_stitch_vlnv == "xilinx_finn:finn:finn_design:1.0"
    model.save("/tmp/test_fpgadataflow_ip_stitch.onnx")


@pytest.mark.dependency(depends=["test_fpgadataflow_ip_stitch"])
def test_fpgadataflow_pynq_projgen():
    model = ModelWrapper("/tmp/test_fpgadataflow_ip_stitch.onnx")
    model = model.transform(MakePYNQProject("Pynq-Z1"))
    vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
    assert vivado_pynq_proj_dir is not None
    assert os.path.isdir(vivado_pynq_proj_dir)
    model.save("/tmp/test_fpgadataflow_pynq_projgen.onnx")

@pytest.mark.dependency(depends=["test_fpgadataflow_pynq_projgen"])
def test_fpgadataflow_pynq_synth():
    model = ModelWrapper("/tmp/test_fpgadataflow_pynq_projgen.onnx")
    model = model.transform(SynthPYNQProject())
    bitfile = model.get_metadata_prop("vivado_pynq_bitfile")
    assert bitfile is not None
    assert os.path.isfile(bitfile)
    model = model.transform(CleanUp())
