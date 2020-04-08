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
from finn.core.modelwrapper import ModelWrapper
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.registry import getCustomOp
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.codegen_ipstitch import CodeGen_ipstitch
from finn.transformation.fpgadataflow.codegen_npysim import CodeGen_npysim
from finn.transformation.fpgadataflow.compile import Compile
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
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
from finn.util.basic import pynq_part_map
from finn.util.test import get_test_model_trained

build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 5
mem_mode = "decoupled"


def test_end2end_tfc_w2a2_export():
    import brevitas.onnx as bo

    tfc = get_test_model_trained("TFC", 2, 2)
    bo.export_finn_onnx(
        tfc, (1, 1, 28, 28), build_dir + "/end2end_tfc_w2a2_export.onnx"
    )


def test_end2end_tfc_w2a2_import_and_tidy():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_export.onnx")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model.save(build_dir + "/end2end_tfc_w2a2_tidy.onnx")


def test_end2end_tfc_w2a2_streamline():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_tidy.onnx")
    model = model.transform(Streamline())
    model.save(build_dir + "/end2end_tfc_w2a2_streamlined.onnx")


def test_end2end_tfc_w2a2_convert_to_hls_layers():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_streamlined.onnx")
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    model.save(build_dir + "/end2end_tfc_w2a2_hls_layers.onnx")


def test_end2end_tfc_w2a2_create_dataflow_partition():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_hls_layers.onnx")
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_tfc_w2a2_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = ModelWrapper(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_tfc_w2a2_dataflow_model.onnx")


def test_end2end_tfc_w2a2_fold_and_tlastmarker():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_dataflow_model.onnx")
    model = model.transform(GiveUniqueNodeNames())
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    fc0w = getCustomOp(fc_layers[0])
    fc1w = getCustomOp(fc_layers[1])
    fc2w = getCustomOp(fc_layers[2])
    fc3w = getCustomOp(fc_layers[3])
    fc0w.set_nodeattr("inFIFODepth", 50)
    fc0w.set_nodeattr("SIMD", 8)
    fc0w.set_nodeattr("PE", 16)
    fc0w.set_nodeattr("outFIFODepth", 4)
    fc1w.set_nodeattr("SIMD", 16)
    fc1w.set_nodeattr("PE", 16)
    fc1w.set_nodeattr("outFIFODepth", 4)
    fc2w.set_nodeattr("SIMD", 16)
    fc2w.set_nodeattr("PE", 16)
    fc2w.set_nodeattr("outFIFODepth", 4)
    fc3w.set_nodeattr("SIMD", 16)
    fc3w.set_nodeattr("PE", 10)
    fc3w.set_nodeattr("outFIFODepth", 50)
    model = model.transform(InsertTLastMarker())
    model.save(build_dir + "/end2end_tfc_w2a2_folded.onnx")


def test_end2end_tfc_w2a2_gen_hls_ip():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_folded.onnx")
    model = model.transform(CodeGen_ipgen(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynth_IPGen())
    model.save(build_dir + "/end2end_tfc_w2a2_ipgen.onnx")


def test_end2end_tfc_w2a2_ip_stitch():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_ipgen.onnx")
    model = model.transform(ReplaceVerilogRelPaths())
    model = model.transform(CodeGen_ipstitch(test_fpga_part))
    model.save(build_dir + "/end2end_tfc_w2a2_ipstitch.onnx")


def test_end2end_tfc_w2a2_verify_dataflow_part():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_ipstitch.onnx")
    x = np.zeros((1, 784), dtype=np.float32)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp_dict = {inp_name: x}
    # npysim
    model = model.transform(CodeGen_npysim())
    model = model.transform(Compile())
    model = model.transform(SetExecMode("npysim"))
    model.save(build_dir + "/end2end_tfc_w2a2_ipstitch_npysim.onnx")
    ret_npysim = execute_onnx(model, inp_dict, True)
    res_npysim = ret_npysim[out_name]
    # node-by-node rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    for fcl in fc_layers:
        getCustomOp(fcl).set_nodeattr("rtlsim_trace", "default")
    model.save(build_dir + "/end2end_tfc_w2a2_ipstitch_nodebynode_rtlsim.onnx")
    ret_rtlsim_nodebynode = execute_onnx(model, inp_dict, True)
    res_rtlsim_nodebynode = ret_rtlsim_nodebynode[out_name]
    # whole-network (ip-stitched) rtlsim
    model.set_metadata_prop("exec_mode", "rtlsim")
    model.set_metadata_prop("rtlsim_trace", "whole_trace.vcd")
    model.save(build_dir + "/end2end_tfc_w2a2_ipstitch_whole_rtlsim.onnx")
    ret_rtlsim_whole = execute_onnx(model, inp_dict, True)
    res_rtlsim_whole = ret_rtlsim_whole[out_name]
    assert np.isclose(res_npysim, res_rtlsim_nodebynode).all()
    assert np.isclose(res_npysim, res_rtlsim_whole).all()


def test_end2end_tfc_w2a2_verify_all():
    # use the streamlined model as the "golden" model for right answers
    golden = ModelWrapper(build_dir + "/end2end_tfc_w2a2_streamlined.onnx")
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
    parent_model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_dataflow_parent.onnx")
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    # produce results with npysim
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    sdp_node.set_nodeattr("model", build_dir + "/end2end_tfc_w2a2_ipstitch_npysim.onnx")
    ret_npysim = execute_onnx(parent_model, {iname: x}, True)
    y_npysim = ret_npysim[oname]
    # produce results with node-by-node rtlsim
    sdp_node.set_nodeattr(
        "model", build_dir + "/end2end_tfc_w2a2_ipstitch_nodebynode_rtlsim.onnx"
    )
    ret_nodebynode_rtlsim = execute_onnx(parent_model, {iname: x}, True)
    y_nodebynode_rtlsim = ret_nodebynode_rtlsim[oname]
    # produce results with whole-network (stitched ip) rtlsim
    sdp_node.set_nodeattr(
        "model", build_dir + "/end2end_tfc_w2a2_ipstitch_whole_rtlsim.onnx"
    )
    ret_whole_rtlsim = execute_onnx(parent_model, {iname: x}, True)
    y_whole_rtlsim = ret_whole_rtlsim[oname]
    assert np.isclose(y_golden, y_npysim).all()
    assert np.isclose(y_golden, y_nodebynode_rtlsim).all()
    assert np.isclose(y_golden, y_whole_rtlsim).all()


def test_end2end_tfc_w2a2_make_pynq_proj():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_ipstitch.onnx")
    model = model.transform(MakePYNQProject(test_pynq_board))
    model.save(build_dir + "/end2end_tfc_w2a2_pynq_project.onnx")


def test_end2end_tfc_w2a2_synth_pynq_project():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_pynq_project.onnx")
    model = model.transform(SynthPYNQProject())
    model.save(build_dir + "/end2end_tfc_w2a2_synth.onnx")


def test_end2end_tfc_w2a2_make_driver():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_synth.onnx")
    model = model.transform(MakePYNQDriver())
    model.save(build_dir + "/end2end_tfc_w2a2_pynq_driver.onnx")


def test_end2end_tfc_w2a2_deploy_on_pynq():
    model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_pynq_driver.onnx")
    try:
        ip = os.environ["PYNQ_IP"]  # no fault for this one; skip if not defined
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
        username = os.getenv("PYNQ_USERNAME", "xilinx")
        password = os.getenv("PYNQ_PASSWORD", "xilinx")
        target_dir = os.getenv("PYNQ_TARGET_DIR", "/home/xilinx/finn")
        model = model.transform(DeployToPYNQ(ip, username, password, target_dir))
        # save the model to be able to link it to the parent
        model.save(build_dir + "/end2end_tfc_w2a2_pynq_deploy.onnx")
    except KeyError:
        pytest.skip("PYNQ board IP address not specified")


def test_end2end_tfc_w2a2_run_on_pynq():
    # use the streamlined model as the "golden" model for right answers
    golden = ModelWrapper(build_dir + "/end2end_tfc_w2a2_streamlined.onnx")
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
    parent_model = ModelWrapper(build_dir + "/end2end_tfc_w2a2_dataflow_parent.onnx")
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    try:
        ip = os.environ["PYNQ_IP"]  # NOQA
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
        # produce results with npysim
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        sdp_node.set_nodeattr("model", build_dir + "/end2end_tfc_w2a2_pynq_deploy.onnx")
        ret = execute_onnx(parent_model, {iname: x}, True)
        y = ret[oname]
        assert np.isclose(y, y_golden).all()

    except KeyError:
        pytest.skip("PYNQ board IP address not specified")
