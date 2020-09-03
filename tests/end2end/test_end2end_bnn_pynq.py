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

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA
import torch
import brevitas.onnx as bo

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.registry import getCustomOp
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ
from finn.transformation.general import (
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.util.test import (
    get_build_env,
    load_test_checkpoint_or_skip,
    get_example_input,
    get_trained_network_and_ishape,
    execute_parent,
)
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
import warnings
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.core.modelwrapper import ModelWrapper


build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
target_clk_ns = 10
mem_mode = "decoupled"
rtlsim_trace = False


def get_checkpoint_name(topology, wbits, abits, step):
    return build_dir + "/end2end_%s_w%da%d_%s.onnx" % (topology, wbits, abits, step)


def fold_tfc(model):
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
    return model


def fold_cnv_large(model):
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # each tuple is (PE, SIMD, in_fifo_depth) for a layer
    folding = [
        (16, 3, 256),
        (32, 32, 256),
        (16, 32, 256),
        (16, 32, 256),
        (4, 32, 214),
        (1, 32, 2),
        (1, 4, 126),
        (1, 8, 62),
        (5, 1, 6),
    ]
    for fcl, (pe, simd, ififodepth) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififodepth)

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    swg_idepth = [2, 51, 9, 106, 2, 2]
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
        swg_inst.set_nodeattr("inFIFODepth", swg_idepth[i])
    return model


def fold_cnv_small(model):
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
    return model


def get_folding_function(topology, wbits, abits):
    if "tfc" in topology:
        return fold_tfc
    elif "cnv" in topology:
        if wbits == 1 and abits == 1:
            return fold_cnv_large
        else:
            return fold_cnv_small
    else:
        raise Exception("Unknown topology/quantization combo for predefined folding")


def get_golden_io_pair(topology, wbits, abits):
    (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    input_tensor_npy = get_example_input(topology)
    input_tensor_torch = torch.from_numpy(input_tensor_npy).float()
    output_tensor_npy = model.forward(input_tensor_torch).detach().numpy()
    return (input_tensor_npy, output_tensor_npy)


@pytest.mark.parametrize("wbits", [1, 2])
@pytest.mark.parametrize("abits", [1, 2])
@pytest.mark.parametrize("topology", ["tfc", "cnv"])
class TestEnd2End:
    def test_export(self, topology, wbits, abits):
        if wbits > abits:
            pytest.skip("No wbits > abits end2end network configs for now")
        (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
        chkpt_name = get_checkpoint_name(topology, wbits, abits, "export")
        bo.export_finn_onnx(model, ishape, chkpt_name)
        assert os.path.isfile(chkpt_name)

    def test_import_and_tidy(self, topology, wbits, abits):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "export")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(DoubleToSingleFloat())
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(RemoveStaticGraphInputs())
        model.save(get_checkpoint_name(topology, wbits, abits, "import_and_tidy"))

    def test_streamline(self, topology, wbits, abits):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "import_and_tidy")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(Streamline())
        if "fc" not in topology:
            model = model.transform(LowerConvsToMatMul())
            model = model.transform(MakeMaxPoolNHWC())
            model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(ConvertBipolarMatMulToXnorPopcount())
        model = model.transform(Streamline())
        model = model.transform(RemoveUnusedTensors())
        model.save(get_checkpoint_name(topology, wbits, abits, "streamline"))

    def test_convert_to_hls_layers(self, topology, wbits, abits):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "streamline")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        # needed for bipolar MatMul layers
        model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
        # needed for non-bipolar MatMul layers
        model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
        # needed for convolutions
        if "fc" not in topology:
            model = model.transform(to_hls.InferConvInpGen())
            model = model.transform(to_hls.InferStreamingMaxPool())
            model = model.transform(RemoveCNVtoFCFlatten())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataLayouts())
        model.save(get_checkpoint_name(topology, wbits, abits, "convert_to_hls_layers"))

    def test_create_dataflow_partition(self, topology, wbits, abits):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, "convert_to_hls_layers"
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        parent_model = model.transform(CreateDataflowPartition())
        parent_model_chkpt = get_checkpoint_name(
            topology, wbits, abits, "dataflow_parent"
        )
        parent_model.save(parent_model_chkpt)
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
        dataflow_model_chkpt = get_checkpoint_name(
            topology, wbits, abits, "dataflow_model"
        )
        dataflow_model.save(dataflow_model_chkpt)

    def test_fold(self, topology, wbits, abits):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "dataflow_model")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        folding_fxn = get_folding_function(topology, wbits, abits)
        model = folding_fxn(model)
        model.save(get_checkpoint_name(topology, wbits, abits, "fold"))

    def test_cppsim(self, topology, wbits, abits):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "fold")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        cppsim_chkpt = get_checkpoint_name(topology, wbits, abits, "cppsim")
        model.save(cppsim_chkpt)
        parent_chkpt = get_checkpoint_name(topology, wbits, abits, "dataflow_parent")
        (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
            topology, wbits, abits
        )
        y = execute_parent(parent_chkpt, cppsim_chkpt, input_tensor_npy)
        assert np.isclose(y, output_tensor_npy).all()

    def test_ipgen(self, topology, wbits, abits):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "fold")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        test_fpga_part = get_build_env("zynq", target_clk_ns)["part"]
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model.save(get_checkpoint_name(topology, wbits, abits, "ipgen"))

    def test_ipstitch_rtlsim(self, topology, wbits, abits):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "ipgen")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        test_fpga_part = get_build_env("zynq", target_clk_ns)["part"]
        model = model.transform(InsertDWC())
        model = model.transform(InsertFIFO())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        perf = model.analysis(dataflow_performance)
        latency = perf["critical_path_cycles"]
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
        model = model.transform(PrepareRTLSim())
        model.set_metadata_prop("exec_mode", "rtlsim")
        os.environ["LIVENESS_THRESHOLD"] = str(int(latency * 1.1))
        if rtlsim_trace:
            model.set_metadata_prop(
                "rtlsim_trace", "%s_w%da%d.vcd" % (topology, wbits, abits)
            )
            os.environ["RTLSIM_TRACE_DEPTH"] = "3"
        rtlsim_chkpt = get_checkpoint_name(topology, wbits, abits, "ipstitch_rtlsim")
        model.save(rtlsim_chkpt)
        parent_chkpt = get_checkpoint_name(topology, wbits, abits, "dataflow_parent")
        (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
            topology, wbits, abits
        )
        y = execute_parent(parent_chkpt, rtlsim_chkpt, input_tensor_npy)
        model = ModelWrapper(rtlsim_chkpt)
        perf["cycles_rtlsim"] = model.get_metadata_prop("cycles_rtlsim")
        warnings.warn("Estimated & rtlsim performance: " + str(perf))
        assert np.isclose(y, output_tensor_npy).all()

    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_build(self, topology, wbits, abits, kind):
        if kind == "alveo" and ("VITIS_PATH" not in os.environ):
            pytest.skip("VITIS_PATH not set")
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "ipgen")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        cfg = get_build_env(kind, target_clk_ns)
        model = model.transform(cfg["build_fxn"])
        model = model.transform(AnnotateResources("synth"))
        warnings.warn(
            "Post-synthesis resources (excluding shell): "
            + model.get_metadata_prop("res_total_synth")
        )
        warnings.warn(
            "Post-synthesis resources (all inclusive): "
            + model.get_metadata_prop("res_total_top_synth")
        )
        model.save(get_checkpoint_name(topology, wbits, abits, "build_" + kind))

    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_deploy(self, topology, wbits, abits, kind):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "build_" + kind)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        cfg = get_build_env(kind, target_clk_ns)
        if cfg["ip"] == "":
            pytest.skip("PYNQ board IP address not specified")
        model = model.transform(
            DeployToPYNQ(
                cfg["ip"],
                cfg["port"],
                cfg["username"],
                cfg["password"],
                cfg["target_dir"],
            )
        )
        # save the model to be able to link it to the parent
        model.save(get_checkpoint_name(topology, wbits, abits, "deploy_" + kind))

    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_run_on_pynq(self, topology, wbits, abits, kind):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "deploy_" + kind)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)  # NOQA
        cfg = get_build_env(kind, target_clk_ns)
        if cfg["ip"] == "":
            pytest.skip("PYNQ board IP address not specified")
        (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
            topology, wbits, abits
        )
        parent_model = load_test_checkpoint_or_skip(
            get_checkpoint_name(topology, wbits, abits, "dataflow_parent")
        )
        iname = parent_model.graph.input[0].name
        oname = parent_model.graph.output[0].name
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        sdp_chkpt = get_checkpoint_name(topology, wbits, abits, "deploy")
        load_test_checkpoint_or_skip(sdp_chkpt)
        sdp_node.set_nodeattr("model", sdp_chkpt)
        ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
        y = ret[oname]
        assert np.isclose(y, output_tensor_npy).all()
