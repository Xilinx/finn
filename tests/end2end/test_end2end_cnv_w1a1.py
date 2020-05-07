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
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.core.onnx_exec import execute_onnx
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.move_reshape import MoveReshape
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.streamline import Streamline
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
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
from finn.util.test import get_test_model_trained
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO

build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 5
mem_mode = "decoupled"


def test_end2end_cnv_w1a1_export():
    import brevitas.onnx as bo

    cnv = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(
        cnv, (1, 3, 32, 32), build_dir + "/end2end_cnv_w1a1_export.onnx"
    )


def test_end2end_cnv_w1a1_import_and_tidy():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_export.onnx")
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/end2end_cnv_w1a1_tidy.onnx")


def test_end2end_cnv_w1a1_streamline():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_tidy.onnx")
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model.save(build_dir + "/end2end_cnv_w1a1_streamlined.onnx")


def test_end2end_cnv_w1a1_convert_to_hls_layers():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_streamlined.onnx")
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(MoveReshape())
    model.save(build_dir + "/end2end_cnv_w1a1_hls_layers.onnx")


def test_end2end_cnv_w1a1_create_dataflow_partition():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_hls_layers.onnx")
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_cnv_w1a1_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = ModelWrapper(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_cnv_w1a1_dataflow_model.onnx")


def test_end2end_cnv_w1a1_fold_and_tlastmarker():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_dataflow_model.onnx")
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # each tuple is (PE, SIMD, in_fifo_depth) for a layer
    folding = [
        (16, 3, 128),
        (32, 32, 128),
        (16, 32, 128),
        (16, 32, 128),
        (4, 32, 81),
        (1, 32, 2),
        (1, 4, 2),
        (1, 8, 128),
        (5, 1, 3),
    ]
    for fcl, (pe, simd, ififodepth) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififodepth)

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)

    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO())
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(AnnotateResources("estimate"))
    model.save(build_dir + "/end2end_cnv_w1a1_folded.onnx")


def test_end2end_cnv_w1a1_gen_hls_ip():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_folded.onnx")
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynth_IPGen())
    model = model.transform(AnnotateResources("hls"))
    model.save(build_dir + "/end2end_cnv_w1a1_ipgen.onnx")


def test_end2end_cnv_w1a1_ip_stitch():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_ipgen.onnx")
    model = model.transform(ReplaceVerilogRelPaths())
    model = model.transform(CreateStitchedIP(test_fpga_part))
    model.save(build_dir + "/end2end_cnv_w1a1_ipstitch.onnx")


def test_end2end_cnv_w1a1_verify_dataflow_part():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_ipstitch.onnx")
    x = np.zeros((1, 32, 32, 3), dtype=np.float32)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp_dict = {inp_name: x}
    # npysim
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("npysim"))
    model.save(build_dir + "/end2end_cnv_w1a1_ipgen_npysim.onnx")
    ret_npysim = execute_onnx(model, inp_dict, True)
    res_npysim = ret_npysim[out_name]
    # node-by-node rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    model.save(build_dir + "/end2end_cnv_w1a1_ipgen_nodebynode_rtlsim.onnx")
    ret_rtlsim_nodebynode = execute_onnx(model, inp_dict, True)
    res_rtlsim_nodebynode = ret_rtlsim_nodebynode[out_name]
    # whole-network (ip-stitched) rtlsim
    model.set_metadata_prop("exec_mode", "rtlsim")
    model.save(build_dir + "/end2end_cnv_w1a1_ipstitch_whole_rtlsim.onnx")
    # this is a particularly long-running test, set liveness thr. to unlimited
    os.environ["LIVENESS_THRESHOLD"] = "-1"
    ret_rtlsim_whole = execute_onnx(model, inp_dict, True)
    res_rtlsim_whole = ret_rtlsim_whole[out_name]
    assert np.isclose(res_npysim, res_rtlsim_nodebynode).all()
    assert np.isclose(res_npysim, res_rtlsim_whole).all()


def test_end2end_cnv_w1a1_verify_all():
    # use the streamlined model as the "golden" model for right answers
    golden = ModelWrapper(build_dir + "/end2end_cnv_w1a1_streamlined.onnx")
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
    parent_model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_dataflow_parent.onnx")
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    # produce results with npysim
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    sdp_node.set_nodeattr("model", build_dir + "/end2end_cnv_w1a1_ipgen_npysim.onnx")
    ret_npysim = execute_onnx(parent_model, {iname: x}, True)
    y_npysim = ret_npysim[oname]
    # produce results with node-by-node rtlsim
    sdp_node.set_nodeattr(
        "model", build_dir + "/end2end_cnv_w1a1_ipgen_nodebynode_rtlsim.onnx"
    )
    ret_nodebynode_rtlsim = execute_onnx(parent_model, {iname: x}, True)
    y_nodebynode_rtlsim = ret_nodebynode_rtlsim[oname]
    # produce results with whole-network (stitched ip) rtlsim
    sdp_node.set_nodeattr(
        "model", build_dir + "/end2end_cnv_w1a1_ipstitch_whole_rtlsim.onnx"
    )
    # this is a particularly long-running test, set liveness thr. to unlimited
    os.environ["LIVENESS_THRESHOLD"] = "-1"
    ret_whole_rtlsim = execute_onnx(parent_model, {iname: x}, True)
    y_whole_rtlsim = ret_whole_rtlsim[oname]
    assert np.isclose(y_golden, y_npysim).all()
    assert np.isclose(y_golden, y_nodebynode_rtlsim).all()
    assert np.isclose(y_golden, y_whole_rtlsim).all()
    assert np.argmax(y_golden) == 3


def test_end2end_cnv_w1a1_make_pynq_proj():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_ipstitch.onnx")
    model = model.transform(MakePYNQProject(test_pynq_board))
    model.save(build_dir + "/end2end_cnv_w1a1_pynq_project.onnx")


def test_end2end_cnv_w1a1_synth_pynq_project():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_pynq_project.onnx")
    model = model.transform(SynthPYNQProject())
    model = model.transform(AnnotateResources("synth"))
    model.save(build_dir + "/end2end_cnv_w1a1_synth.onnx")


def test_end2end_cnv_w1a1_make_driver():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_synth.onnx")
    model = model.transform(MakePYNQDriver())
    model.save(build_dir + "/end2end_cnv_w1a1_pynq_driver.onnx")


def test_end2end_cnv_w1a1_deploy_on_pynq():
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_pynq_driver.onnx")
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
        model.save(build_dir + "/end2end_cnv_w1a1_pynq_deploy.onnx")
    except KeyError:
        pytest.skip("PYNQ board IP address not specified")


def test_end2end_cnv_w1a1_run_on_pynq():
    # use the streamlined model as the "golden" model for right answers
    golden = ModelWrapper(build_dir + "/end2end_cnv_w1a1_streamlined.onnx")
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
    parent_model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_dataflow_parent.onnx")
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    try:
        ip = os.environ["PYNQ_IP"]  # NOQA
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
        # produce results with npysim
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        sdp_node.set_nodeattr("model", build_dir + "/end2end_cnv_w1a1_pynq_deploy.onnx")
        ret = execute_onnx(parent_model, {iname: x}, True)
        y = ret[oname]
        assert np.isclose(y, y_golden).all()
        assert np.argmax(y) == 3

    except KeyError:
        pytest.skip("PYNQ board IP address not specified")
