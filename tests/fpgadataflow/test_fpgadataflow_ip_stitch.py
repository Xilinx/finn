# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
import finn.transformation.fpgadataflow.replace_verilog_relpaths as rvp
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor, pynq_part_map
from finn.util.fpgadataflow import pyverilate_stitched_ip

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]

ip_stitch_model_dir = "/tmp/" + os.environ["FINN_INST_NAME"]


def create_one_fc_model():
    # create a model with a StreamingFCLayer instance with no activation
    # the wider range of the full accumulator makes debugging a bit easier
    wdt = DataType.INT2
    idt = DataType.INT32
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
    idt = DataType.INT32
    odt = DataType.INT32
    m = 4
    actval = 0
    no_act = 1
    binary_xnor_mode = 0
    pe = 2
    simd = 2

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inp", "w0"],
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
        mem_mode="decoupled",
    )

    fc1 = helper.make_node(
        "StreamingFCLayer_Batch",
        ["mid", "w1"],
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
        mem_mode="decoupled",
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
    w0 = np.eye(m, dtype=np.float32)
    w1 = np.eye(m, dtype=np.float32)
    model.set_initializer("w0", w0)
    model.set_initializer("w1", w1)

    model = model.transform(CreateDataflowPartition())
    return model


# exec_mode of StreamingDataflowPartition
# @pytest.mark.parametrize("exec_mode", ["remote_pynq"]) #, "rtlsim"])
def test_fpgadataflow_ipstitch_gen_model():  # exec_mode):
    model = create_two_fc_model()
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
    assert model.graph.node[-1].op_type == "TLastMarker"
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model.onnx")


def test_fpgadataflow_ipstitch_do_stitch():
    model = ModelWrapper(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model.onnx"
    )
    model = model.transform(rvp.ReplaceVerilogRelPaths())
    model = model.transform(CodeGen_ipstitch(test_fpga_part))
    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    assert vivado_stitch_proj_dir is not None
    assert os.path.isdir(vivado_stitch_proj_dir)
    assert os.path.isfile(vivado_stitch_proj_dir + "/ip/component.xml")
    vivado_stitch_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
    assert vivado_stitch_vlnv is not None
    assert vivado_stitch_vlnv == "xilinx_finn:finn:finn_design:1.0"
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch.onnx")


def test_fpgadataflow_ipstitch_rtlsim():
    model = ModelWrapper(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch.onnx")
    model.set_metadata_prop("rtlsim_trace", "whole_trace.vcd")
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
    model.set_metadata_prop("exec_mode", "rtlsim")
    idt = model.get_tensor_datatype("inp")
    ishape = model.get_tensor_shape("inp")
    x = gen_finn_dt_tensor(idt, ishape)
    # x = np.zeros(ishape, dtype=np.float32)
    # x = np.asarray([[-2, -1, 0, 1]], dtype=np.float32)
    rtlsim_res = execute_onnx(model, {"inp": x})["outp"]
    assert (rtlsim_res == x).all()


def test_fpgadataflow_ipstitch_pynq_projgen():
    model = ModelWrapper(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch.onnx")
    model = model.transform(MakePYNQProject(test_pynq_board))
    vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
    assert vivado_pynq_proj_dir is not None
    assert os.path.isdir(vivado_pynq_proj_dir)
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_pynq_projgen.onnx")


def test_fpgadataflow_ipstitch_pynq_synth():
    model = ModelWrapper(ip_stitch_model_dir + "/test_fpgadataflow_pynq_projgen.onnx")
    model = model.transform(SynthPYNQProject())
    bitfile = model.get_metadata_prop("vivado_pynq_bitfile")
    assert bitfile is not None
    assert os.path.isfile(bitfile)
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_synth.onnx")


def test_fpgadataflow_ipstitch_pynq_driver():
    model = ModelWrapper(ip_stitch_model_dir + "/test_fpgadataflow_pynq_projgen.onnx")
    model = model.transform(MakePYNQDriver())
    driver_dir = model.get_metadata_prop("pynq_driver_dir")
    assert driver_dir is not None
    assert os.path.isdir(driver_dir)
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_driver.onnx")


def test_fpgadataflow_ipstitch_pynq_deployment_folder():
    model = ModelWrapper(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_driver.onnx"
    )
    try:
        ip = os.environ["PYNQ_IP"]  # no default for this one; skip if not defined
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
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

        model.save(
            ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_deployment.onnx"
        )
    except KeyError:
        pytest.skip("PYNQ board IP address not specified")


def test_fpgadataflow_ipstitch_remote_execution():
    model = ModelWrapper(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_pynq_deployment.onnx"
    )
    try:
        ip = os.environ["PYNQ_IP"]  # NOQA
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
        idt = DataType.INT2
        x = gen_finn_dt_tensor(idt, (1, 4))
        input_dict = {"inp": x}
        outp = execute_onnx(model, input_dict)
        assert np.isclose(outp["outp"], x).all()
    except KeyError:
        pytest.skip("PYNQ board IP address not specified")
