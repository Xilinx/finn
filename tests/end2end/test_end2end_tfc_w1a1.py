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
from pkgutil import get_data

import pytest

import numpy as np

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA
import onnx.numpy_helper as nph

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.registry import getCustomOp
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_pynq_proj import MakePYNQProject
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.synth_pynq_proj import SynthPYNQProject
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import pynq_part_map
from finn.util.test import get_test_model_trained, load_test_checkpoint_or_skip
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.core.throughput_test import throughput_test_rtlsim
import finn.util.vcd as vcd

build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10
mem_mode = "decoupled"


def test_end2end_tfc_w1a1_export():
    import brevitas.onnx as bo

    tfc = get_test_model_trained("TFC", 1, 1)
    bo.export_finn_onnx(
        tfc, (1, 1, 28, 28), build_dir + "/end2end_tfc_w1a1_export.onnx"
    )


def test_end2end_tfc_w1a1_import_and_tidy():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_export.onnx")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model.save(build_dir + "/end2end_tfc_w1a1_tidy.onnx")


def test_end2end_tfc_w1a1_streamline():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_tidy.onnx")
    model = model.transform(Streamline())
    model.save(build_dir + "/end2end_tfc_w1a1_streamlined.onnx")


def test_end2end_tfc_w1a1_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_streamlined.onnx"
    )
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    model.save(build_dir + "/end2end_tfc_w1a1_hls_layers.onnx")


def test_end2end_tfc_w1a1_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_hls_layers.onnx"
    )
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_tfc_w1a1_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_tfc_w1a1_dataflow_model.onnx")


def test_end2end_tfc_w1a1_fold_and_tlastmarker():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_dataflow_model.onnx"
    )
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # (PE, SIMD, in_fifo_depth, out_fifo_depth, ramstyle) for each layer
    config = [
        (16, 49, 16, 64, "block"),
        (8, 8, 64, 64, "auto"),
        (8, 8, 64, 64, "auto"),
        (10, 8, 64, 10, "distributed"),
    ]
    for fcl, (pe, simd, ififo, ofifo, ramstyle) in zip(fc_layers, config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififo)
        fcl_inst.set_nodeattr("outFIFODepth", ofifo)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO())
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(AnnotateResources("estimate"))
    model.save(build_dir + "/end2end_tfc_w1a1_folded.onnx")


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_tfc_w1a1_gen_hls_ip():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_folded.onnx")
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(AnnotateResources("hls"))
    model.save(build_dir + "/end2end_tfc_w1a1_ipgen.onnx")


@pytest.mark.vivado
def test_end2end_tfc_w1a1_ip_stitch():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_ipgen.onnx")
    model = model.transform(ReplaceVerilogRelPaths())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.save(build_dir + "/end2end_tfc_w1a1_ipstitch.onnx")


@pytest.mark.vivado
def test_end2end_tfc_w1a1_verify_dataflow_part():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_ipstitch.onnx")
    x = np.zeros((1, 784), dtype=np.float32)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp_dict = {inp_name: x}
    # cppsim
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))
    model.save(build_dir + "/end2end_tfc_w1a1_ipstitch_cppsim.onnx")
    ret_cppsim = execute_onnx(model, inp_dict, True)
    res_cppsim = ret_cppsim[out_name]
    # node-by-node rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    model.save(build_dir + "/end2end_tfc_w1a1_ipstitch_nodebynode_rtlsim.onnx")
    ret_rtlsim_nodebynode = execute_onnx(model, inp_dict, True)
    res_rtlsim_nodebynode = ret_rtlsim_nodebynode[out_name]
    # whole-network (ip-stitched) rtlsim
    model.set_metadata_prop("exec_mode", "rtlsim")
    model.set_metadata_prop("rtlsim_trace", build_dir + "/tfc_w1a1.vcd")
    os.environ["RTLSIM_TRACE_DEPTH"] = "3"
    model.save(build_dir + "/end2end_tfc_w1a1_ipstitch_whole_rtlsim.onnx")
    ret_rtlsim_whole = execute_onnx(model, inp_dict, True)
    res_rtlsim_whole = ret_rtlsim_whole[out_name]
    assert np.isclose(res_cppsim, res_rtlsim_nodebynode).all()
    assert np.isclose(res_cppsim, res_rtlsim_whole).all()


def test_end2end_tfc_w1a1_verify_fifo_fullness():
    vcdf = build_dir + "/tfc_w1a1.vcd"
    if not os.path.isfile(vcdf):
        pytest.skip("Cannot find %s, skipping" % vcdf)
    stream_ifs = vcd.list_stream_if(vcdf)
    fifos = vcd.list_fifo_count_signals(vcdf)
    assert len(stream_ifs) == 37
    assert len(fifos) == 6
    fifo_max = vcd.get_all_fifo_count_max(vcdf)
    assert fifo_max[0][0] == "TOP.v.finn_design_i.StreamingFIFO_0.count[3:0]"
    assert fifo_max[0][1] == 3
    stream_stat = vcd.get_all_stream_if_stats(vcdf)
    assert (
        stream_stat[0][0]
        == "TOP.v.finn_design_i.StreamingDataWidthConverter_Batch_0_out_V_V_"
    )


@pytest.mark.vivado
def test_end2end_tfc_w1a1_throughput_test_rtlsim():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_ipstitch_whole_rtlsim.onnx"
    )
    # run through IP-stitched rtlsim with increasing batch sizes and
    # check the number of cycles it takes to execute
    ret = throughput_test_rtlsim(model, 1)
    assert ret["cycles"] == 205
    ret = throughput_test_rtlsim(model, 10)
    assert ret["cycles"] == 844
    ret = throughput_test_rtlsim(model, 100)
    assert ret["cycles"] == 7234


@pytest.mark.vivado
def test_end2end_tfc_w1a1_verify_all():
    # use the streamlined model as the "golden" model for right answers
    golden = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_streamlined.onnx"
    )
    iname = golden.graph.input[0].name
    oname = golden.graph.output[0].name
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    x = nph.to_array(input_tensor)
    # x = np.zeros(ishape, dtype=np.float32)
    ret_golden = execute_onnx(golden, {iname: x}, True)
    y_golden = ret_golden[oname]
    # set up parent+child graph to test
    # we'll use models from the previous step as the child model
    parent_model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_dataflow_parent.onnx"
    )
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    # produce results with cppsim
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_ipstitch_cppsim.onnx")
    sdp_node.set_nodeattr("model", build_dir + "/end2end_tfc_w1a1_ipstitch_cppsim.onnx")
    ret_cppsim = execute_onnx(parent_model, {iname: x}, True)
    y_cppsim = ret_cppsim[oname]
    # produce results with node-by-node rtlsim
    load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_ipstitch_nodebynode_rtlsim.onnx"
    )
    sdp_node.set_nodeattr(
        "model", build_dir + "/end2end_tfc_w1a1_ipstitch_nodebynode_rtlsim.onnx"
    )
    ret_nodebynode_rtlsim = execute_onnx(parent_model, {iname: x}, True)
    y_nodebynode_rtlsim = ret_nodebynode_rtlsim[oname]
    # produce results with whole-network (stitched ip) rtlsim
    load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_ipstitch_whole_rtlsim.onnx"
    )
    sdp_node.set_nodeattr(
        "model", build_dir + "/end2end_tfc_w1a1_ipstitch_whole_rtlsim.onnx"
    )
    ret_whole_rtlsim = execute_onnx(parent_model, {iname: x}, True)
    y_whole_rtlsim = ret_whole_rtlsim[oname]
    assert np.isclose(y_golden, y_cppsim).all()
    assert np.isclose(y_golden, y_nodebynode_rtlsim).all()
    assert np.isclose(y_golden, y_whole_rtlsim).all()


@pytest.mark.vivado
def test_end2end_tfc_w1a1_make_pynq_proj():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_ipstitch.onnx")
    model = model.transform(MakePYNQProject(test_pynq_board))
    model.save(build_dir + "/end2end_tfc_w1a1_pynq_project.onnx")


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_tfc_w1a1_synth_pynq_project():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_pynq_project.onnx"
    )
    model = model.transform(SynthPYNQProject())
    model = model.transform(AnnotateResources("synth"))
    model.save(build_dir + "/end2end_tfc_w1a1_synth.onnx")


def test_end2end_tfc_w1a1_make_driver():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_synth.onnx")
    model = model.transform(MakePYNQDriver())
    model.save(build_dir + "/end2end_tfc_w1a1_pynq_driver.onnx")


def test_end2end_tfc_w1a1_deploy_on_pynq():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_pynq_driver.onnx"
    )
    try:
        ip = os.environ["PYNQ_IP"]  # no fault for this one; skip if not defined
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
        username = os.getenv("PYNQ_USERNAME", "xilinx")
        password = os.getenv("PYNQ_PASSWORD", "xilinx")
        port = os.getenv("PYNQ_PORT", 22)
        target_dir = os.getenv("PYNQ_TARGET_DIR", "/home/xilinx/finn")
        model = model.transform(DeployToPYNQ(ip, port, username, password, target_dir))
        # save the model to be able to link it to the parent
        model.save(build_dir + "/end2end_tfc_w1a1_pynq_deploy.onnx")
    except KeyError:
        pytest.skip("PYNQ board IP address not specified")


def test_end2end_tfc_w1a1_run_on_pynq():
    # use the streamlined model as the "golden" model for right answers
    golden = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_streamlined.onnx"
    )
    iname = golden.graph.input[0].name
    oname = golden.graph.output[0].name
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    x = nph.to_array(input_tensor)
    # x = np.zeros(ishape, dtype=np.float32)
    # run using FINN-based execution
    ret_golden = execute_onnx(golden, {iname: x}, True)
    y_golden = ret_golden[oname]
    # set up parent+child graph to test
    # we'll use models from the previous step as the child model
    parent_model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_tfc_w1a1_dataflow_parent.onnx"
    )
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    try:
        ip = os.environ["PYNQ_IP"]  # NOQA
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
        # produce results with cppsim
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        load_test_checkpoint_or_skip(build_dir + "/end2end_tfc_w1a1_pynq_deploy.onnx")
        sdp_node.set_nodeattr("model", build_dir + "/end2end_tfc_w1a1_pynq_deploy.onnx")
        ret = execute_onnx(parent_model, {iname: x}, True)
        y = ret[oname]
        assert np.isclose(y, y_golden).all()

    except KeyError:
        pytest.skip("PYNQ board IP address not specified")
