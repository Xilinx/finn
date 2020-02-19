import os

import pytest

import numpy as np
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.codegen_ipstitch import CodeGen_ipstitch
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_pynq_proj import MakePYNQProject
from finn.transformation.fpgadataflow.synth_pynq_proj import SynthPYNQProject
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    make_build_dir,
    pynq_part_map,
)
from finn.util.fpgadataflow import pyverilate_stitched_ip

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]

ip_stitch_model_dir = make_build_dir("test_fpgadataflow_ipstitch_")


def create_one_fc_model():
    # create a model with a StreamingFCLayer instance with no activation
    # the wider range of the full accumulator makes debugging a bit easier
    wdt = DataType.INT2
    idt = DataType.INT2
    odt = DataType.INT32
    m = 4
    no_act = 1
    binary_xnor_mode = 0
    actval = 0
    simd = 2
    pe = 2

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inp", "w0"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )

    graph = helper.make_graph(
        nodes=[fc0], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("w0", wdt)

    # generate weights
    w0 = np.eye(m, dtype=np.float32)
    model.set_initializer("w0", w0)

    model = model.transform(CreateDataflowPartition())
    return model


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
    pe = 2
    simd = 2

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inp", "w0", "t0"],
        ["mid"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
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
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )

    graph = helper.make_graph(
        nodes=[fc0, fc1],
        name="fclayer_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[mid],
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("mid", idt)
    model.set_tensor_datatype("outp", odt)
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
# exec_mode of StreamingDataflowPartition
# @pytest.mark.parametrize("exec_mode", ["remote_pynq"]) #, "rtlsim"])
def test_fpgadataflow_ipstitch_gen_model():  # exec_mode):
    model = create_one_fc_model()
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = ModelWrapper(sdp_node.get_nodeattr("model"))
        model.set_metadata_prop("exec_mode", "remote_pynq")
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(CodeGen_ipgen(test_fpga_part, 5))
    model = model.transform(HLSSynth_IPGen())
    assert model.graph.node[0].op_type == "StreamingFCLayer_Batch"
    # assert model.graph.node[1].op_type == "StreamingFCLayer_Batch"
    assert model.graph.node[-1].op_type == "TLastMarker"
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model.onnx")


@pytest.mark.dependency(depends=["test_fpgadataflow_ipstitch_gen_model"])
def test_fpgadataflow_ipstitch_do_stitch():
    model = ModelWrapper(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model.onnx"
    )
    model = model.transform(CodeGen_ipstitch(test_fpga_part))
    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    assert vivado_stitch_proj_dir is not None
    assert os.path.isdir(vivado_stitch_proj_dir)
    assert os.path.isfile(vivado_stitch_proj_dir + "/ip/component.xml")
    vivado_stitch_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
    assert vivado_stitch_vlnv is not None
    assert vivado_stitch_vlnv == "xilinx_finn:finn:finn_design:1.0"
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch.onnx")


@pytest.mark.dependency(depends=["test_fpgadataflow_ipstitch_do_stitch"])
def test_fpgadataflow_ipstitch_rtlsim():
    model = ModelWrapper(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch.onnx")
    sim = pyverilate_stitched_ip(model)
    exp_io = [
        "ap_clk_0",
        "ap_rst_n_0",
        "in0_V_V_0_tdata",
        "in0_V_V_0_tready",
        "in0_V_V_0_tvalid",
        "out_r_0_tdata",
        "out_r_0_tkeep",
        "out_r_0_tlast",
        "out_r_0_tready",
        "out_r_0_tvalid",
    ]
    assert dir(sim.io) == exp_io


@pytest.mark.dependency(depends=["test_fpgadataflow_ipstitch_do_stitch"])
def test_fpgadataflow_ipstitch_pynq_projgen():
    model = ModelWrapper(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch.onnx")
    model = model.transform(MakePYNQProject(test_pynq_board))
    vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
    assert vivado_pynq_proj_dir is not None
    assert os.path.isdir(vivado_pynq_proj_dir)
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_pynq_projgen.onnx")


@pytest.mark.dependency(depends=["test_fpgadataflow_ipstitch_pynq_projgen"])
def test_fpgadataflow_ipstitch_pynq_synth():
    model = ModelWrapper(ip_stitch_model_dir + "/test_fpgadataflow_pynq_projgen.onnx")
    model = model.transform(SynthPYNQProject())
    bitfile = model.get_metadata_prop("vivado_pynq_bitfile")
    assert bitfile is not None
    assert os.path.isfile(bitfile)
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_synth.onnx")


@pytest.mark.dependency(depends=["test_fpgadataflow_ipstitch_pynq_projgen"])
def test_fpgadataflow_ipstitch_pynq_driver():
    model = ModelWrapper(ip_stitch_model_dir + "/test_fpgadataflow_pynq_projgen.onnx")
    model = model.transform(MakePYNQDriver())
    driver_dir = model.get_metadata_prop("pynq_driver_dir")
    assert driver_dir is not None
    assert os.path.isdir(driver_dir)
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_driver.onnx")


@pytest.mark.dependency(depends=["test_fpgadataflow_ipstitch_pynq_driver"])
def test_fpgadataflow_ipstitch_pynq_deployment_folder():
    model = ModelWrapper(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_driver.onnx"
    )
    ip = os.getenv("PYNQ_IP", "192.168.3.1")
    username = os.getenv("PYNQ_USERNAME", "xilinx")
    password = os.getenv("PYNQ_PASSWORD", "xilinx")
    target_dir = os.getenv("PYNQ_TARGET_DIR", "/home/xilinx/finn")
    model = model.transform(DeployToPYNQ(ip, username, password, target_dir))
    pynq_ip = model.get_metadata_prop("pynq_ip")
    pynq_username = model.get_metadata_prop("pynq_username")
    pynq_password = model.get_metadata_prop("pynq_password")
    pynq_target_dir = model.get_metadata_prop("pynq_target_dir")

    assert pynq_ip == ip
    assert pynq_username == username
    assert pynq_password == password
    assert pynq_target_dir == target_dir

    deployment_dir = model.get_metadata_prop("pynq_deploy_dir")
    assert deployment_dir is not None
    assert os.path.isdir(deployment_dir)

    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_deployment.onnx")


@pytest.mark.dependency(depends=["test_fpgadataflow_ipstitch_pynq_deployment_folder"])
def test_fpgadataflow_ipstitch_remote_execution():
    model = ModelWrapper(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_deployment.onnx"
    )
    idt = DataType.INT2
    x = gen_finn_dt_tensor(idt, (1, 4))
    input_dict = {"inp": x}
    outp = execute_onnx(model, input_dict)
    print(outp)
