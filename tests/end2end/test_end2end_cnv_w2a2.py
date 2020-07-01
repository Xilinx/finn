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

import numpy as np

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA

import pytest
import pkg_resources as pk
from finn.custom_op.registry import getCustomOp
from finn.core.onnx_exec import execute_onnx
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.streamline import Streamline
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_pynq_proj import MakePYNQProject
from finn.transformation.fpgadataflow.synth_pynq_proj import SynthPYNQProject
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ
from finn.util.basic import pynq_part_map
from finn.util.test import get_test_model_trained, load_test_checkpoint_or_skip
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.core.throughput_test import throughput_test_rtlsim

build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10
mem_mode = "decoupled"


def test_end2end_cnv_w2a2_export():
    import brevitas.onnx as bo

    cnv = get_test_model_trained("CNV", 2, 2)
    bo.export_finn_onnx(
        cnv, (1, 3, 32, 32), build_dir + "/end2end_cnv_w2a2_export.onnx"
    )


def test_end2end_cnv_w2a2_import_and_tidy():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_export.onnx")
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/end2end_cnv_w2a2_tidy.onnx")


def test_end2end_cnv_w2a2_streamline():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_tidy.onnx")
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(Streamline())
    model.save(build_dir + "/end2end_cnv_w2a2_streamlined.onnx")


def test_end2end_cnv_w2a2_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_streamlined.onnx"
    )
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(RemoveCNVtoFCFlatten())
    model.save(build_dir + "/end2end_cnv_w2a2_hls_layers.onnx")


def test_end2end_cnv_w2a2_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_hls_layers.onnx"
    )
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_cnv_w2a2_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_cnv_w2a2_dataflow_model.onnx")


def test_end2end_cnv_w2a2_fold_and_tlastmarker():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_dataflow_model.onnx"
    )
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # each tuple is (PE, SIMD, in_fifo_depth) for a layer
    folding = [
        (8, 3, 256, "auto"),
        (16, 16, 256, "auto"),
        (8, 16, 256, "auto"),
        (8, 16, 256, "block"),
        (4, 8, 214, "auto"),
        (1, 8, 2, "auto"),
        (1, 2, 126, "distributed"),
        (2, 2, 62, "block"),
        (5, 1, 6, "distributed"),
    ]
    for fcl, (pe, simd, ififodepth, ramstyle) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififodepth)
        fcl_inst.set_nodeattr("ram_style", ramstyle)

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    swg_idepth = [2, 51, 9, 106, 2, 2]
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
        swg_inst.set_nodeattr("inFIFODepth", swg_idepth[i])

    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO())
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(AnnotateResources("estimate"))
    model.save(build_dir + "/end2end_cnv_w2a2_folded.onnx")


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_cnv_w2a2_gen_hls_ip():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_folded.onnx")
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(AnnotateResources("hls"))
    model.save(build_dir + "/end2end_cnv_w2a2_ipgen.onnx")


@pytest.mark.vivado
def test_end2end_cnv_w2a2_ip_stitch():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_ipgen.onnx")
    model = model.transform(ReplaceVerilogRelPaths())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.save(build_dir + "/end2end_cnv_w2a2_ipstitch.onnx")


@pytest.mark.vivado
def test_end2end_cnv_w2a2_verify_dataflow_part():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_ipstitch.onnx")
    x = np.zeros((1, 32, 32, 3), dtype=np.float32)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp_dict = {inp_name: x}
    # cppsim
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))
    model.save(build_dir + "/end2end_cnv_w2a2_ipgen_cppsim.onnx")
    ret_cppsim = execute_onnx(model, inp_dict, True)
    res_cppsim = ret_cppsim[out_name]
    # node-by-node rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    model.save(build_dir + "/end2end_cnv_w2a2_ipgen_nodebynode_rtlsim.onnx")
    ret_rtlsim_nodebynode = execute_onnx(model, inp_dict, True)
    res_rtlsim_nodebynode = ret_rtlsim_nodebynode[out_name]
    # whole-network (ip-stitched) rtlsim
    model.set_metadata_prop("exec_mode", "rtlsim")
    model.save(build_dir + "/end2end_cnv_w2a2_ipstitch_whole_rtlsim.onnx")
    # this is a particularly long-running test, set liveness thr. to unlimited
    os.environ["LIVENESS_THRESHOLD"] = "-1"
    ret_rtlsim_whole = execute_onnx(model, inp_dict, True)
    res_rtlsim_whole = ret_rtlsim_whole[out_name]
    assert np.isclose(res_cppsim, res_rtlsim_nodebynode).all()
    assert np.isclose(res_cppsim, res_rtlsim_whole).all()


@pytest.mark.vivado
def test_end2end_cnv_w2a2_throughput_test_rtlsim():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_ipstitch_whole_rtlsim.onnx"
    )
    model.set_metadata_prop("rtlsim_trace", "rtlsim_trace.vcd")
    # os.environ["RTLSIM_TRACE_DEPTH"] = "4"
    # run through IP-stitched rtlsim with increasing batch sizes and
    # check the number of cycles it takes to execute
    ret = throughput_test_rtlsim(model, 10)
    # TODO check for expected performance
    assert ret["cycles"] > 0


@pytest.mark.vivado
def test_end2end_cnv_w2a2_verify_all():
    # use the streamlined model as the "golden" model for right answers
    golden = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_streamlined.onnx"
    )
    iname = golden.graph.input[0].name
    oname = golden.graph.output[0].name
    # load one of the test vectors
    fn = pk.resource_filename("finn", "data/cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    input_tensor = input_tensor / 255
    assert input_tensor.shape == (1, 3, 32, 32)
    x = input_tensor
    # x = np.zeros(ishape, dtype=np.float32)
    ret_golden = execute_onnx(golden, {iname: x}, True)
    y_golden = ret_golden[oname]
    # set up parent+child graph to test
    # we'll use models from the previous step as the child model
    parent_model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_dataflow_parent.onnx"
    )
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    # produce results with cppsim
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_ipgen_cppsim.onnx")
    sdp_node.set_nodeattr("model", build_dir + "/end2end_cnv_w2a2_ipgen_cppsim.onnx")
    ret_cppsim = execute_onnx(parent_model, {iname: x}, True)
    y_cppsim = ret_cppsim[oname]
    # produce results with node-by-node rtlsim
    load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_ipgen_nodebynode_rtlsim.onnx"
    )
    sdp_node.set_nodeattr(
        "model", build_dir + "/end2end_cnv_w2a2_ipgen_nodebynode_rtlsim.onnx"
    )
    ret_nodebynode_rtlsim = execute_onnx(parent_model, {iname: x}, True)
    y_nodebynode_rtlsim = ret_nodebynode_rtlsim[oname]
    # produce results with whole-network (stitched ip) rtlsim
    load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_ipstitch_whole_rtlsim.onnx"
    )
    sdp_node.set_nodeattr(
        "model", build_dir + "/end2end_cnv_w2a2_ipstitch_whole_rtlsim.onnx"
    )
    # this is a particularly long-running test, set liveness thr. to unlimited
    os.environ["LIVENESS_THRESHOLD"] = "-1"
    ret_whole_rtlsim = execute_onnx(parent_model, {iname: x}, True)
    y_whole_rtlsim = ret_whole_rtlsim[oname]
    assert np.isclose(y_golden, y_cppsim).all()
    assert np.isclose(y_golden, y_nodebynode_rtlsim).all()
    assert np.isclose(y_golden, y_whole_rtlsim).all()
    assert np.argmax(y_golden) == 3


@pytest.mark.vivado
def test_end2end_cnv_w2a2_make_pynq_proj():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_ipstitch.onnx")
    model = model.transform(MakePYNQProject(test_pynq_board))
    model.save(build_dir + "/end2end_cnv_w2a2_pynq_project.onnx")


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_cnv_w2a2_synth_pynq_project():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_pynq_project.onnx"
    )
    model = model.transform(SynthPYNQProject())
    model = model.transform(AnnotateResources("synth"))
    model.save(build_dir + "/end2end_cnv_w2a2_synth.onnx")


def test_end2end_cnv_w2a2_make_driver():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_synth.onnx")
    model = model.transform(MakePYNQDriver())
    model.save(build_dir + "/end2end_cnv_w2a2_pynq_driver.onnx")


def test_end2end_cnv_w2a2_deploy_on_pynq():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_pynq_driver.onnx"
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
        model.save(build_dir + "/end2end_cnv_w2a2_pynq_deploy.onnx")
    except KeyError:
        pytest.skip("PYNQ board IP address not specified")


def test_end2end_cnv_w2a2_run_on_pynq():
    # use the streamlined model as the "golden" model for right answers
    golden = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_streamlined.onnx"
    )
    iname = golden.graph.input[0].name
    oname = golden.graph.output[0].name
    # load one of the test vectors
    fn = pk.resource_filename("finn", "data/cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    input_tensor = input_tensor / 255
    assert input_tensor.shape == (1, 3, 32, 32)
    x = input_tensor
    # run using FINN-based execution
    ret_golden = execute_onnx(golden, {iname: x}, True)
    y_golden = ret_golden[oname]
    # set up parent+child graph to test
    # we'll use models from the previous step as the child model
    parent_model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_cnv_w2a2_dataflow_parent.onnx"
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
        load_test_checkpoint_or_skip(build_dir + "/end2end_cnv_w2a2_pynq_deploy.onnx")
        sdp_node.set_nodeattr("model", build_dir + "/end2end_cnv_w2a2_pynq_deploy.onnx")
        ret = execute_onnx(parent_model, {iname: x}, True)
        y = ret[oname]
        assert np.isclose(y, y_golden).all()
        assert np.argmax(y) == 3

    except KeyError:
        pytest.skip("PYNQ board IP address not specified")
