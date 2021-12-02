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

import pytest

import numpy as np
import os
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.util.basic import (
    alveo_default_platform,
    alveo_part_map,
    gen_finn_dt_tensor,
    pynq_part_map,
)
from finn.util.pyverilator import pyverilate_stitched_ip
from finn.util.test import load_test_checkpoint_or_skip

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]

ip_stitch_model_dir = os.environ["FINN_BUILD_DIR"]


def create_one_fc_model(mem_mode="const"):
    # create a model with a StreamingFCLayer instance with no activation
    # the wider range of the full accumulator makes debugging a bit easier
    wdt = DataType["INT2"]
    idt = DataType["INT32"]
    odt = DataType["INT32"]
    m = 4
    no_act = 1
    binary_xnor_mode = 0
    actval = 0
    simd = 4
    pe = 4

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inp", "w0"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
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
        mem_mode=mem_mode,
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


def create_two_fc_model(mem_mode="decoupled"):
    # create a model with two StreamingFCLayer instances
    wdt = DataType["INT2"]
    idt = DataType["INT32"]
    odt = DataType["INT32"]
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
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
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
        mem_mode=mem_mode,
    )

    fc1 = helper.make_node(
        "StreamingFCLayer_Batch",
        ["mid", "w1"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
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
        mem_mode=mem_mode,
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


@pytest.mark.parametrize("mem_mode", ["const", "decoupled"])
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_gen_model(mem_mode):
    model = create_one_fc_model(mem_mode)
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
        model.set_metadata_prop("exec_mode", "remote_pynq")
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, 5))
    model = model.transform(HLSSynthIP())
    assert model.graph.node[0].op_type == "StreamingFCLayer_Batch"
    assert model.graph.node[-1].op_type == "TLastMarker"
    model.save(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model_%s.onnx" % mem_mode
    )


@pytest.mark.parametrize("mem_mode", ["const", "decoupled"])
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_do_stitch(mem_mode):
    model = load_test_checkpoint_or_skip(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model_%s.onnx" % mem_mode
    )
    model = model.transform(CreateStitchedIP(test_fpga_part, 5))
    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    assert vivado_stitch_proj_dir is not None
    assert os.path.isdir(vivado_stitch_proj_dir)
    assert os.path.isfile(vivado_stitch_proj_dir + "/ip/component.xml")
    vivado_stitch_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
    assert vivado_stitch_vlnv is not None
    assert vivado_stitch_vlnv == "xilinx_finn:finn:finn_design:1.0"
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch_%s.onnx" % mem_mode)


@pytest.mark.parametrize("mem_mode", ["const", "decoupled"])
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_rtlsim(mem_mode):
    model = load_test_checkpoint_or_skip(
        ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch_%s.onnx" % mem_mode
    )
    model.set_metadata_prop("rtlsim_trace", "whole_trace.vcd")
    sim = pyverilate_stitched_ip(model)
    exp_io = [
        "ap_clk",
        "ap_rst_n",
        "s_axis_0_tdata",
        "s_axis_0_tready",
        "s_axis_0_tvalid",
        "m_axis_0_tdata",
        "m_axis_0_tkeep",
        "m_axis_0_tlast",
        "m_axis_0_tready",
        "m_axis_0_tvalid",
        "s_axi_control_0_araddr",
        "s_axi_control_0_arready",
        "s_axi_control_0_arvalid",
        "s_axi_control_0_awaddr",
        "s_axi_control_0_awready",
        "s_axi_control_0_awvalid",
        "s_axi_control_0_bready",
        "s_axi_control_0_bresp",
        "s_axi_control_0_bvalid",
        "s_axi_control_0_rdata",
        "s_axi_control_0_rready",
        "s_axi_control_0_rresp",
        "s_axi_control_0_rvalid",
        "s_axi_control_0_wdata",
        "s_axi_control_0_wready",
        "s_axi_control_0_wstrb",
        "s_axi_control_0_wvalid",
    ]
    assert sorted(dir(sim.io)) == sorted(exp_io)
    model.set_metadata_prop("exec_mode", "rtlsim")
    idt = model.get_tensor_datatype("inp")
    ishape = model.get_tensor_shape("inp")
    x = gen_finn_dt_tensor(idt, ishape)
    # x = np.zeros(ishape, dtype=np.float32)
    # x = np.asarray([[-2, -1, 0, 1]], dtype=np.float32)
    rtlsim_res = execute_onnx(model, {"inp": x})["outp"]
    assert (rtlsim_res == x).all()


@pytest.mark.parametrize("mem_mode", ["const", "decoupled"])
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_ipstitch_synth_ooc(mem_mode):
    model = load_test_checkpoint_or_skip(
        ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch_%s.onnx" % mem_mode
    )
    model = model.transform(SynthOutOfContext(test_fpga_part, 5))
    ret = model.get_metadata_prop("res_total_ooc_synth")
    assert ret is not None
    # example expected output: (details may differ based on Vivado version etc)
    # "{'vivado_proj_folder': ...,
    # 'LUT': 708.0, 'FF': 1516.0, 'DSP': 0.0, 'BRAM': 0.0, 'WNS': 0.152, '': 0,
    # 'fmax_mhz': 206.27062706270627}"
    ret = eval(ret)
    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    assert ret["BRAM"] == 0
    assert ret["fmax_mhz"] > 100


def test_fpgadataflow_ipstitch_iodma_floorplan():
    model = create_one_fc_model()
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(InferDataLayouts())
    model = model.transform(InsertIODMA())
    model = model.transform(Floorplan())
    assert getCustomOp(model.graph.node[0]).get_nodeattr("partition_id") == 0
    assert getCustomOp(model.graph.node[1]).get_nodeattr("partition_id") == 2
    assert getCustomOp(model.graph.node[2]).get_nodeattr("partition_id") == 1
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_iodma_floorplan.onnx")


# board
@pytest.mark.parametrize("board", ["U250"])
# clock period
@pytest.mark.parametrize("period_ns", [5])
# override mem_mode to external
@pytest.mark.parametrize("extw", [True, False])
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.vitis
def test_fpgadataflow_ipstitch_vitis_end2end(board, period_ns, extw):
    if "VITIS_PATH" not in os.environ:
        pytest.skip("VITIS_PATH not set")
    platform = alveo_default_platform[board]
    fpga_part = alveo_part_map[board]
    model = create_two_fc_model("external" if extw else "decoupled")
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(VitisBuild(fpga_part, period_ns, platform))
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_vitis.onnx")
    assert model.get_metadata_prop("platform") == "alveo"
    assert os.path.isdir(model.get_metadata_prop("vitis_link_proj"))
    assert os.path.isfile(model.get_metadata_prop("bitfile"))


# board
@pytest.mark.parametrize("board", ["Pynq-Z1"])
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_zynqbuild_end2end(board):
    model = create_two_fc_model()
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    # bitfile using ZynqBuild
    model = model.transform(ZynqBuild(board, 10))
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_customzynq.onnx")

    bitfile_name = model.get_metadata_prop("bitfile")
    assert bitfile_name is not None
    assert os.path.isfile(bitfile_name)
