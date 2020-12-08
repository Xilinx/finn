# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

from finn.core.modelwrapper import ModelWrapper
import os
import json
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
)
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)
from shutil import copy, copytree
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    RemoveShallowFIFOs,
)
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.custom_op.registry import getCustomOp
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from finn.analysis.fpgadataflow.op_and_param_counts import (
    aggregate_dict_keys,
    op_and_param_counts,
)
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.util.config import extract_model_config_to_json
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.core.onnx_exec import execute_onnx
import numpy as np
from finn.util.test import execute_parent
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim


def verify_step(
    model: ModelWrapper, cfg: DataflowBuildConfig, step_name: str, need_parent: bool
):
    print("Running verification for " + step_name)
    verify_out_dir = cfg.output_dir + "/verification_output"
    intermediate_models_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(verify_out_dir, exist_ok=True)
    (in_npy, exp_out_npy) = cfg._resolve_verification_io_pair()
    if need_parent:
        assert (
            cfg.save_intermediate_models
        ), "Enable save_intermediate_models for verification"
        parent_model_fn = intermediate_models_dir + "/dataflow_parent.onnx"
        child_model_fn = intermediate_models_dir + "/verify_%s.onnx" % step_name
        model.save(child_model_fn)
        out_npy = execute_parent(parent_model_fn, child_model_fn, in_npy)
    else:
        inp_tensor_name = model.graph.input[0].name
        out_tensor_name = model.graph.output[0].name
        inp_dict = {inp_tensor_name: in_npy}
        out_dict = execute_onnx(model, inp_dict)
        out_npy = out_dict[out_tensor_name]
    res = np.isclose(exp_out_npy, out_npy, atol=1e-3).all()
    res_to_str = {True: "SUCCESS", False: "FAIL"}
    res_str = res_to_str[res]
    verification_output_fn = verify_out_dir + "/verify_%s_%s.npy" % (step_name, res_str)
    np.save(verification_output_fn, out_npy)
    print("Verification for %s : %s" % (step_name, res_str))


def step_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run the tidy-up step on given model. This includes shape and datatype
    inference, constant folding, and giving nodes and tensors better names.
    """

    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    if VerificationStepType.TIDY_UP_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "initial_python", need_parent=False)

    return model


def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run streamlining on given model. Streamlining involves moving floating point
    scale/shift parameters around, collapsing adjacent ones into a single parameter,
    then absorbing the scale/shift into the following `MultiThreshold` node.
    Streamlining requires careful topology design and cannot be applied to all
    topologies.
    """

    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    need_lowering = len(model.get_nodes_by_op_type("Conv")) > 0
    if need_lowering:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    # absorb final add-mul nodes into TopK
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model


def step_convert_to_hls(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert eligible nodes to `HLSCustomOp` subclasses that represent HLS
    layers. Which nodes and particular configurations can be converted to HLS
    is limited, see the source code of the `convert_to_hls` module for more. """

    mem_mode = cfg.default_mem_mode.value
    if cfg.standalone_thresholds:
        # doing this first causes all threshold layers to be standalone
        model = model.transform(to_hls.InferThresholdingLayer())
    # needed for bipolar MatMul layers
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    # needed for non-bipolar MatMul layers
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    # TopK to LabelSelect
    model = model.transform(to_hls.InferLabelSelectLayer())
    # input quantization (if any) as standalone threshold
    model = model.transform(to_hls.InferThresholdingLayer())
    # needed for convolutions -- TODO always exec?
    need_conv = len(model.get_nodes_by_op_type("Im2Col")) > 0
    if need_conv:
        model = model.transform(to_hls.InferConvInpGen())
        model = model.transform(to_hls.InferStreamingMaxPool())
        model = model.transform(RemoveCNVtoFCFlatten())
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    return model


def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Separate consecutive groups of HLSCustomOp nodes into StreamingDataflowPartition
    nodes, which point to a separate ONNX file. Dataflow accelerator synthesis
    can only be performed on those HLSCustomOp sub-graphs."""

    parent_model = model.transform(CreateDataflowPartition())
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 1, "Only a single StreamingDataflowPartition supported."
    sdp_node = sdp_nodes[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    if cfg.save_intermediate_models:
        parent_model.save(cfg.output_dir + "/intermediate_models/dataflow_parent.onnx")
    model = ModelWrapper(dataflow_model_filename)
    return model


def step_target_fps_parallelization(model: ModelWrapper, cfg: DataflowBuildConfig):
    """If target_fps was specified, use the SetFolding transformation to determine
    parallelization attributes."""

    target_cycles_per_frame = cfg._resolve_cycles_per_frame()
    if target_cycles_per_frame is not None:
        model = model.transform(
            SetFolding(target_cycles_per_frame, mvau_wwidth_max=cfg.mvau_wwidth_max)
        )
    return model


def step_apply_folding_config(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Apply the folding configuration file onto the model to set folding (parallelization)
    and other attributes, if config file is specified."""

    if cfg.folding_config_file is not None:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.folding_config_file))

    if VerificationStepType.FOLDED_HLS_CPPSIM in cfg._resolve_verification_steps():
        # prepare cppsim
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        verify_step(model, cfg, "folded_hls_cppsim", need_parent=True)
    return model


def step_generate_estimate_reports(model: ModelWrapper, cfg: DataflowBuildConfig):
    "Generate per-layer resource and cycle estimates using analytical models."

    if DataflowOutputType.ESTIMATE_REPORTS in cfg.generate_outputs:
        report_dir = cfg.output_dir + "/report"
        os.makedirs(report_dir, exist_ok=True)
        ops_and_params = model.analysis(op_and_param_counts)
        with open(report_dir + "/op_and_param_counts.json", "w") as f:
            json.dump(ops_and_params, f, indent=2)
        estimate_layer_cycles = model.analysis(exp_cycles_per_layer)
        with open(report_dir + "/estimate_layer_cycles.json", "w") as f:
            json.dump(estimate_layer_cycles, f, indent=2)
        estimate_layer_resources = model.analysis(res_estimation)
        estimate_layer_resources["total"] = aggregate_dict_keys(
            estimate_layer_resources
        )
        with open(report_dir + "/estimate_layer_resources.json", "w") as f:
            json.dump(estimate_layer_resources, f, indent=2)
        estimate_layer_resources_complete = model.analysis(res_estimation_complete)
        with open(report_dir + "/estimate_layer_config_alternatives.json", "w") as f:
            json.dump(estimate_layer_resources_complete, f, indent=2)
        # need to call AnnotateCycles before dataflow_performance
        model = model.transform(AnnotateCycles())
        estimate_network_performance = model.analysis(dataflow_performance)
        # add some more metrics to estimated performance
        n_clock_cycles_per_sec = (10 ** 9) / cfg.synth_clk_period_ns
        est_fps = n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
        estimate_network_performance["estimated_throughput_fps"] = est_fps
        est_latency_ns = (
            estimate_network_performance["critical_path_cycles"]
            * cfg.synth_clk_period_ns
        )
        estimate_network_performance["estimated_latency_ns"] = est_latency_ns
        with open(report_dir + "/estimate_network_performance.json", "w") as f:
            json.dump(estimate_network_performance, f, indent=2)
    return model


def step_hls_ipgen(model: ModelWrapper, cfg: DataflowBuildConfig):
    "Run Vivado HLS synthesis on any HLSCustomOp nodes to generate IP blocks."

    model = model.transform(
        PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period())
    )
    model = model.transform(HLSSynthIP())
    model = model.transform(ReplaceVerilogRelPaths())
    report_dir = cfg.output_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)
    estimate_layer_resources_hls = model.analysis(hls_synth_res_estimation)
    with open(report_dir + "/estimate_layer_resources_hls.json", "w") as f:
        json.dump(estimate_layer_resources_hls, f, indent=2)
    return model


def step_set_fifo_depths(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Depending on the auto_fifo_depths setting, do one of the following:
    * if auto_fifo_depths=True:  Run the `InsertAndSetFIFODepths` transformation
    to attempt to determine the FIFO sizes that provide full throughput. Involves
    running stitched-IP rtlsim and may take a long time.
    * if auto_fifo_depths=False:  Assume the folding config file contains FIFO
    sizes as well. Runs the `InsertFIFO` transformation, then
    `ApplyConfig(cfg.folding_config_file)`, and finally `RemoveShallowFIFOs`.
    Coherency with config file node naming is ensured by calling
    `GiveUniqueNodeNames`.
    """

    if cfg.auto_fifo_depths:
        model = model.transform(
            InsertAndSetFIFODepths(
                cfg._resolve_fpga_part(),
                cfg._resolve_hls_clk_period(),
                vivado_ram_style=cfg.large_fifo_mem_style.value,
            )
        )
    else:
        # assume folding cfg json contains FIFO sizes too
        # insert DWCs, FIFOs and run ApplyConfig once more
        model = model.transform(InsertDWC())
        # need to make sure all FIFOs are created so that their depth can be
        # set by ApplyConfig, so create_shallow_fifos=True
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        if cfg.folding_config_file is not None:
            model = model.transform(ApplyConfig(cfg.folding_config_file))
        # remove any shallow FIFOs
        model = model.transform(RemoveShallowFIFOs())

    # extract the final configuration and save it as json
    hw_attrs = [
        "PE",
        "SIMD",
        "ram_style",
        "depth",
        "impl_style",
        "resType",
        "mem_mode",
        "runtime_writeable_weights",
    ]
    extract_model_config_to_json(
        model, cfg.output_dir + "/final_hw_config.json", hw_attrs
    )

    # after FIFOs are ready to go, call PrepareIP and HLSSynthIP again
    # this will only run for the new nodes (e.g. FIFOs and DWCs)
    model = model.transform(
        PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period())
    )
    model = model.transform(HLSSynthIP())
    return model


def step_create_stitched_ip(model: ModelWrapper, cfg: DataflowBuildConfig):
    "Create stitched IP for a graph after all HLS IP blocks have been generated."

    if DataflowOutputType.STITCHED_IP in cfg.generate_outputs:
        stitched_ip_dir = cfg.output_dir + "/stitched_ip"
        model = model.transform(
            CreateStitchedIP(cfg._resolve_fpga_part(), cfg.synth_clk_period_ns)
        )
        # TODO copy all ip sources into output dir? as zip?
        copytree(model.get_metadata_prop("vivado_stitch_proj"), stitched_ip_dir)
        print("Vivado stitched IP written into " + stitched_ip_dir)
    if VerificationStepType.STITCHED_IP_RTLSIM in cfg._resolve_verification_steps():
        # prepare ip-stitched rtlsim
        verify_model = copy.deepcopy(model)
        # rtlsim only supports impl_style=rtl for StreamingFIFO, ensure that
        for fifo_layer in verify_model.get_nodes_by_op_type("StreamingFIFO"):
            getCustomOp(fifo_layer).set_nodeattr("impl_style", "rtl")
        # similarly for StreamingDataWidthConverter with impl_style=hls
        for dwc_layer in verify_model.get_nodes_by_op_type(
            "StreamingDataWidthConverter_Batch"
        ):
            getCustomOp(dwc_layer).set_nodeattr("impl_style", "hls")
        verify_model = verify_model.transform(PrepareRTLSim())
        verify_model.set_metadata_prop("exec_mode", "rtlsim")
        verify_step(verify_model, cfg, "stitched_ip_rtlsim", need_parent=True)
    return model


def step_make_pynq_driver(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Create a PYNQ Python driver that can be used to interface the generated
    accelerator."""

    if DataflowOutputType.PYNQ_DRIVER in cfg.generate_outputs:
        driver_dir = cfg.output_dir + "/driver"
        model = model.transform(MakePYNQDriver(cfg._resolve_driver_platform()))
        copytree(model.get_metadata_prop("pynq_driver_dir"), driver_dir)
        print("PYNQ Python driver written into " + driver_dir)
    return model


def step_out_of_context_synthesis(model: ModelWrapper, cfg: DataflowBuildConfig):
    "Run out-of-context synthesis and generate reports."
    if DataflowOutputType.OOC_SYNTH in cfg.generate_outputs:
        assert (
            DataflowOutputType.STITCHED_IP in cfg.generate_outputs
        ), "OOC needs stitched IP"
        model = model.transform(
            SynthOutOfContext(
                part=cfg._resolve_fpga_part(), clk_period_ns=cfg.synth_clk_period_ns
            )
        )
        report_dir = cfg.output_dir + "/report"
        os.makedirs(report_dir, exist_ok=True)
        ooc_res_dict = model.get_metadata_prop("res_total_ooc_synth")
        ooc_res_dict = eval(ooc_res_dict)

        estimate_network_performance = model.analysis(dataflow_performance)
        # add some more metrics to estimated performance
        n_clock_cycles_per_sec = float(ooc_res_dict["fmax_mhz"]) * (10 ** 6)
        est_fps = n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
        ooc_res_dict["estimated_throughput_fps"] = est_fps
        with open(report_dir + "/ooc_synth_and_timing.json", "w") as f:
            json.dump(ooc_res_dict, f, indent=2)
    return model


def step_synthesize_bitfile(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Synthesize a bitfile for the using the specified shell flow, using either
    Vivado or Vitis, to target the specified board."""

    if DataflowOutputType.BITFILE in cfg.generate_outputs:
        bitfile_dir = cfg.output_dir + "/bitfile"
        os.makedirs(bitfile_dir)
        report_dir = cfg.output_dir + "/report"
        os.makedirs(report_dir, exist_ok=True)
        if cfg.shell_flow_type == ShellFlowType.VIVADO_ZYNQ:
            model = model.transform(
                ZynqBuild(cfg.board, cfg.synth_clk_period_ns, cfg.enable_hw_debug)
            )
            copy(model.get_metadata_prop("bitfile"), bitfile_dir + "/finn-accel.bit")
            copy(model.get_metadata_prop("hw_handoff"), bitfile_dir + "/finn-accel.hwh")
            copy(
                model.get_metadata_prop("vivado_synth_rpt"),
                report_dir + "/post_synth_resources.xml",
            )
            vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
            timing_rpt = (
                "%s/finn_zynq_link.runs/impl_1/top_wrapper_timing_summary_routed.rpt"
                % vivado_pynq_proj_dir
            )
            copy(timing_rpt, report_dir + "/post_route_timing.rpt")

        elif cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
            model = model.transform(
                VitisBuild(
                    cfg._resolve_fpga_part(),
                    cfg.synth_clk_period_ns,
                    cfg.vitis_platform,
                    strategy=cfg._resolve_vitis_opt_strategy(),
                    enable_debug=cfg.enable_hw_debug,
                    floorplan_file=cfg.vitis_floorplan_file,
                )
            )
            copy(model.get_metadata_prop("bitfile"), bitfile_dir + "/finn-accel.xclbin")
            copy(
                model.get_metadata_prop("vivado_synth_rpt"),
                report_dir + "/post_synth_resources.xml",
            )
        else:
            raise Exception("Unrecognized shell_flow_type: " + str(cfg.shell_flow_type))
        print("Bitfile written into " + bitfile_dir)

    return model


def step_deployment_package(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Create a deployment package including the driver and bitfile."""

    if DataflowOutputType.DEPLOYMENT_PACKAGE in cfg.generate_outputs:
        deploy_dir = cfg.output_dir + "/deploy"
        bitfile_dir = cfg.output_dir + "/bitfile"
        driver_dir = cfg.output_dir + "/driver"
        os.makedirs(deploy_dir)
        copytree(bitfile_dir, deploy_dir + "/bitfile")
        copytree(driver_dir, deploy_dir + "/driver")
    return model


#: map step name strings to step functions
build_dataflow_step_lookup = {
    "step_tidy_up": step_tidy_up,
    "step_streamline": step_streamline,
    "step_convert_to_hls": step_convert_to_hls,
    "step_create_dataflow_partition": step_create_dataflow_partition,
    "step_target_fps_parallelization": step_target_fps_parallelization,
    "step_apply_folding_config": step_apply_folding_config,
    "step_generate_estimate_reports": step_generate_estimate_reports,
    "step_hls_ipgen": step_hls_ipgen,
    "step_set_fifo_depths": step_set_fifo_depths,
    "step_create_stitched_ip": step_create_stitched_ip,
    "step_make_pynq_driver": step_make_pynq_driver,
    "step_out_of_context_synthesis": step_out_of_context_synthesis,
    "step_synthesize_bitfile": step_synthesize_bitfile,
    "step_deployment_package": step_deployment_package,
}
