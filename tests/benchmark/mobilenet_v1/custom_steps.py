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
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps

import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    ShellFlowType,
    VerificationStepType,
)
from finn.builder.build_dataflow_steps import verify_step
from finn.transformation.general import ApplyConfig
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds


def step_mobilenet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(Streamline())
    additional_streamline_transformations = [
        DoubleToSingleFloat(),
        reorder.MoveMulPastDWConv(),
        absorb.AbsorbMulIntoMultiThreshold(),
        ChangeDataLayoutQuantAvgPool2d(),
        InferDataLayouts(),
        reorder.MoveTransposePastScalarMul(),
        absorb.AbsorbTransposeIntoFlatten(),
        reorder.MoveFlattenPastAffine(),
        reorder.MoveFlattenPastTopK(),
        reorder.MoveScalarMulPastMatMul(),
        CollapseRepeatedMul(),
        RemoveIdentityOps(),
        RoundAndClipThresholds(),
    ]
    for trn in additional_streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model


def step_mobilenet_lower_convs(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataLayouts())
    return model


def step_mobilenet_slr_floorplan(model: ModelWrapper, cfg: DataflowBuildConfig):
    if cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
        try:
            from finnexperimental.analysis.partitioning import partition

            # apply partitioning of the model, restricting the first and last layers
            # to SLR0
            default_slr = 0
            abs_anchors = [(0, [default_slr]), (-1, [default_slr])]
            floorplan = partition(
                model,
                cfg.synth_clk_period_ns,
                cfg.board,
                abs_anchors=abs_anchors,
                multivariant=False,
            )[0]
            # apply floorplan to model
            model = model.transform(ApplyConfig(floorplan))
            print("SLR floorplanning applied")
        except Exception:
            print("No SLR floorplanning applied")
    return model


def create_zcu104_rtlsim_prehook(model):
    """
    Create a pre-hook function for ZCU104 stitched IP RTLsim that initializes
    URAM weights via AXI-Lite for nodes with runtime_writeable_weights enabled.

    This function must be called AFTER the stitched IP is created, as it needs
    to access the final model structure and weight values.
    """
    import os
    from qonnx.custom_op.registry import getCustomOp

    # Find all nodes that need weight initialization via AXI-Lite
    nodes_to_init = []

    for node in model.graph.node:
        if node.op_type in ["MVAU_hls", "MVAU_rtl", "Thresholding_hls", "Thresholding_rtl"]:
            inst = getCustomOp(node)
            try:
                runtime_writeable = inst.get_nodeattr("runtime_writeable_weights")
                if runtime_writeable == 1:
                    nodes_to_init.append((node, inst))
            except:
                pass  # Node doesn't have runtime_writeable_weights attribute

    if not nodes_to_init:
        # No nodes need initialization
        return None

    # Pre-generate all weight streams
    weight_streams = {}
    axilite_names = {}

    for idx, (node, inst) in enumerate(nodes_to_init):
        # Get weights (input[1] for MVAU, input[1] for Thresholding)
        if "MVAU" in node.op_type:
            weight_idx = 1
        elif "Thresholding" in node.op_type:
            weight_idx = 1
        else:
            continue

        weights = model.get_initializer(node.input[weight_idx])
        dat_fname = f"weights_{node.name}.dat"

        # Generate weight file in runtime format
        inst.make_weight_file(weights, "decoupled_runtime", dat_fname)

        # Read the weight stream
        with open(dat_fname, "r") as f:
            weight_stream_hex = f.read().strip()
        os.remove(dat_fname)

        # Convert hex strings to integers
        weight_stream = [int(x, 16) for x in weight_stream_hex.split("\n")]
        weight_streams[node.name] = weight_stream

        # AXI-Lite interfaces are named s_axilite_0, s_axilite_1, etc.
        axilite_names[node.name] = f"s_axilite_{idx}"

    # Create the actual pre-hook function
    def write_all_weights_prehook(sim):
        """Pre-hook that writes weights to all nodes via AXI-Lite before simulation."""
        print(f"ZCU104 RTLsim prehook: Initializing {len(weight_streams)} node(s) via AXI-Lite...")

        for node_name in weight_streams:
            weight_stream = weight_streams[node_name]
            axilite_if = axilite_names[node_name]

            print(f"  Writing {len(weight_stream)} words to {node_name} via {axilite_if}")

            # Prepare AXI-Lite write transactions
            addr = 0
            writes = []
            for weight_val in weight_stream:
                hex_val = format(weight_val, "x")
                writes.append((addr, hex_val))
                addr += 4  # Word-aligned addresses

            # Execute AXI-Lite writes
            sim.write_axilite(axilite_if, iter(writes))

        # Run simulation to complete AXI-Lite transactions
        sim.run()

        # Reset before main test input
        import finn.core.onnx_exec as oxe

        oxe.rtlsim_exec.reset_rtlsim(sim)

        print("ZCU104 RTLsim prehook: Weight initialization complete")

    return write_all_weights_prehook


def step_create_stitched_ip_zcu104(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Custom stitched IP creation step for ZCU104 that handles URAM initialization.
    This step creates the prehook dynamically based on the model after IP creation.
    """
    import os
    from copy import deepcopy
    from qonnx.util.cleanup import cleanup_model

    from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
    from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
    from finn.util.basic import get_liveness_threshold_cycles

    # Create stitched IP
    model = model.transform(CreateStitchedIP(cfg.synth_clk_period_ns, cfg.fpga_part))
    model = cleanup_model(model)

    # Handle verification with ZCU104 prehook
    if VerificationStepType.STITCHED_IP_RTLSIM in cfg._resolve_verification_steps():
        prev_liveness = os.environ.get("LIVENESS_THRESHOLD", None)
        os.environ["LIVENESS_THRESHOLD"] = str(get_liveness_threshold_cycles(model))

        verify_model = deepcopy(model)
        verify_model = verify_model.transform(PrepareRTLSim())

        # Enable waveform tracing if requested
        if cfg.verify_save_rtlsim_waveforms:
            waveform_dir = cfg.output_dir + "/report/rtlsim_waveforms"
            os.makedirs(waveform_dir, exist_ok=True)
            abspath = os.path.abspath(waveform_dir)
            verify_model.set_metadata_prop("rtlsim_trace", abspath + "/stitched_ip_rtlsim.wdb")

        # Create prehook for ZCU104 (dynamically based on model)
        prehook = create_zcu104_rtlsim_prehook(verify_model)

        # Run verification with or without prehook
        if prehook is not None:
            print("Using ZCU104-specific RTLsim prehook for weight initialization")
            verify_step(
                verify_model, cfg, "stitched_ip_rtlsim", need_parent=True, rtlsim_pre_hook=prehook
            )
        else:
            # No nodes with runtime_writeable_weights, use standard verification
            verify_step(verify_model, cfg, "stitched_ip_rtlsim", need_parent=True)

        # Restore liveness threshold
        if prev_liveness is not None:
            os.environ["LIVENESS_THRESHOLD"] = prev_liveness
        elif "LIVENESS_THRESHOLD" in os.environ:
            del os.environ["LIVENESS_THRESHOLD"]

    return model
