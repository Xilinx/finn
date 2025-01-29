# Adapted from Christoph's attention-dummy repository

# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas wrapper around PyTorch tensors adding quantization information
from brevitas.quant_tensor import QuantTensor
# Brevitas: Quantized versions of PyTorch layers
from brevitas.nn import (
    QuantMultiheadAttention,
    QuantEltwiseAdd,
    QuantIdentity,
    QuantLinear,
    QuantReLU
)
from qonnx.core.modelwrapper import ModelWrapper
# Progressbar
from tqdm import trange
import numpy as np
from brevitas.export import export_qonnx
import random
import json
import subprocess
# FINN dataflow builder
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import AutoFIFOSizingMethod
from bench_base import bench, step_synth_harness
import os
from util import summarize_table, summarize_section, power_xml_to_dict, prepare_inputs, delete_dir_contents

# Custom build steps required to streamline and convert the attention operator
from dut.transformer_custom_steps import (
    step_tidy_up_pre_attention,
    step_tidy_up_post_attention,
    step_streamline_attention,
    step_streamline_residual,
    step_streamline_norms,
    step_streamline_positional,
    step_convert_attention_to_hw,
    step_convert_elementwise_binary_to_hw,
    step_convert_lookup_to_hw,
    step_replicate_streams,
    set_target_parallelization,
    set_fifo_depths,
    step_apply_folding_config,
    node_by_node_rtlsim, # noqa: Maybe unused, only for debugging
    node_by_node_python, # noqa: Maybe unused, only for debugging
    node_by_node_cppsim
)
from performance.platform_build_steps import(
     test_step_gen_vitis_xo,
     test_step_gen_instrumentation_wrapper,
     test_step_gen_instrwrap_sim,
     test_step_insert_tlastmarker,
     test_step_export_xo,
     test_step_build_platform,
     test_step_run_instrwrap_sim
)

### ADAPTED FROM utils.py
# Seeds all relevant random number generators to the same seed for
# reproducibility
def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

template_folding_yaml = """
# Per operator type default configurations
defaults:
    # Scaled dot-product attention head implemented via HLS
    ScaledDotProductAttention_hls:
        # Type of memory to be used for internal buffer storage
        #   Options: auto, block, distributed, ultra
        ram_style: block
        # Type of memory to be used for threshold storage
        #   Options: auto, block, distributed
        ram_style_thresholds: block
        # Type of memory to be used fo the attention mask (if present)
        #   Options: auto, block, distributed
        ram_style_mask: block
        # Resource type to be used for implementing multiplications/MACs
        #   Options: auto, lut or dsp
        mac_resource: lut
    # Addition of two inputs (constants or streamed) implemented via HLS
    ElementwiseAdd_hls:
        # Type of memory to be used for internal buffer storage and/or constant
        # parameter tensors
        #   Options: auto, block, distributed, ultra
        ram_style: distributed
    # Matrix vector activation unit implemented via HLS
    MVAU_hls:
        # Resource type to be used for implementing multiplications/MACs
        #   Options: auto, lut or dsp
        resType: dsp
        # Memory mode for weight storage
        #   Options: internal_embedded, internal_decoupled, external
        mem_mode: internal_decoupled
        # Type of memory to be used for weight storage if "internal_decoupled"
        #   Options: auto, block, distributed, ultra
        ram_style: block
        # Type of memory to be used for threshold storage
        #   Options: auto, block, distributed
        ram_style_thresholds: block
        # Makes weights writeable through AXI-lite interface at runtime
        runtime_writeable_weights: 0
    # Matrix vector activation unit implemented via RTL
    MVAU_rtl:
        # Resource type to be used for implementing multiplications/MACs
        #   Options: auto, lut or dsp
        # Note: RTL MVAU currently does not support LUT-based implementation
        resType: dsp
        # Memory mode for weight storage
        #   Options: internal_embedded, internal_decoupled, external
        mem_mode: internal_decoupled
        # Type of memory to be used for weight storage if "internal_decoupled"
        #   Options: auto, block, distributed, ultra
        ram_style: block
        # Makes weights writeable through AXI-lite interface at runtime
        runtime_writeable_weights: 0
    # Multi-thresholds implemented via HLS (applies to standalone thresholds)
    Thresholding_hls:
        # Memory mode for threshold storage
        #   Options: internal_embedded, internal_decoupled
        mem_mode: internal_decoupled
        # Type of memory to be used for threshold storage if "internal_decoupled"
        #   Options: distributed, block
        ram_style: distributed
        # Makes thresholds writeable through AXI-lite interface at runtime
        runtime_writeable_weights: 0
    # Multi-thresholds implemented via RTL (applies to standalone thresholds)
    Thresholding_rtl:
        # Decides to use BRAM, URAM or LUTs for threshold memory, depending on the
        # depth of the thresholds
        # Note: This combination forces "distributed" LUT implementation
        depth_trigger_uram: 2147483647  # "infinity"
        depth_trigger_bram: 2147483647  # "infinity"
    #    # Note: This combination forces "block" RAM implementation
    #    depth_trigger_uram: 0
    #    depth_trigger_bram: 1
    #    # Note: This combination forces "ultra" RAM implementation
    #    depth_trigger_uram: 1
    #    depth_trigger_bram: 0
    #    # Note: This combination is equivalent to "auto"
    #    depth_trigger_uram: 0
    #    depth_trigger_bram: 0
        # Makes thresholds writeable through AXI-lite interface at runtime
        runtime_writeable_weights: 0
    # FIFO implemented via RTL (there is no HLS FIFO implementation in FINN)
    StreamingFIFO_rtl:
        # RTL vs. IPI implementation of FIFOs
        #   Options: rtl, vivado
        impl_style: rtl
        # Resource type for FIFOs when impl_style is vivado
        #   Options: auto, block, distributed, ultra
        ram_style: distributed
    # Individual, named node-specific configurations here
    # ...
"""

class bench_transformer_gpt(bench):
    def step_build(self):
        #with open("params.yaml") as file:
        #    params = yaml.safe_load(file)
        # Seed all RNGs
        seed(self.params["seed"])

        # Extract sequence length and embedding dimension from the output of the
        # first quantizer in the model
        # Note: Embedding and Sequence dimension flip later
        model = ModelWrapper(self.build_inputs["onnx_path"])
        _, emb_dim, seq_len = model.get_tensor_shape(
            "/emb_add/input_quant/export_handler/Quant_output_0"
        )

        # Prepare config files
        # TODO: make configurable
        # TODO: log intermediate files such as inp.npy, folding.yaml, or specialize_layers.jon as artifacts, maybe create in unique temp dirs
        specialize_layers_dict = {
            "Defaults": {
                "preferred_impl_style": ["rtl", ["MVAU", "Thresholding"]]
            },
            "": {
                "preferred_impl_style": ""
            }
        }
        with open("specialize_layers.json", "w") as f:
                json.dump(specialize_layers_dict, f, indent=2)
        with open("folding.yaml", "w") as f:
                f.write(template_folding_yaml)

        #TODO: make configurable instead of hardcoding exception
        self.board = "U280"
        self.part = "xcu280-fsvh2892-2L-e"

        # Create a configuration for building the scaled dot-product attention
        # operator to a hardware accelerator
        cfg = build_cfg.DataflowBuildConfig(
            # Unpack the build configuration parameters
            #**params["build"],
            output_dir = self.build_inputs["build_dir"],
            stitched_ip_gen_dcp = True,
            synth_clk_period_ns = self.clock_period_ns,
            board = self.board,
            shell_flow_type = "vitis_alveo", #TODO: proper Alveo support instead of hardcoding
            folding_config_file = "folding.yaml",
            specialize_layers_config_file = "specialize_layers.json",
            standalone_thresholds = True,
            max_multithreshold_bit_width = 16,
            mvau_wwidth_max = 2048,
            split_large_fifos = True,

            verbose=False, # if True prints stdout and stderr to console instead of build_dataflow.log

            generate_outputs=[
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                build_cfg.DataflowOutputType.STITCHED_IP, # required for HarnessBuild, OOC_SYNTH, and RTLSIM
                #build_cfg.DataflowOutputType.PYNQ_DRIVER, #TODO: currently broken (assert i_consumer.op_type == "StreamingDataflowPartition"), might be useful for functional verification on hw later
                #build_cfg.DataflowOutputType.OOC_SYNTH, # requires stitched-ip, not needed because ZynqBuild/HarnessBuild is performed
                #build_cfg.DataflowOutputType.BITFILE, # does not require stitched-ip, not needed because HarnessBuild is performed
                #build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE, # not possible due to float components
                #build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE # not needed, just a copy operation
            ],

            verify_steps=[
                # Verify the model after converting to the FINN onnx dialect
                build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
                # Verify the model again using python mode after the default
                # streamlining step
                build_cfg.VerificationStepType.STREAMLINED_PYTHON,
                # Verify the model again after tidy up transformations, right before
                # converting to HLS
                build_cfg.VerificationStepType.TIDY_UP_PYTHON,
                # Verify the model after generating C++ HLS and applying folding
                build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            ],
            # File with test inputs for verification
            verify_input_npy=self.build_inputs["input_npy_path"],
            # File with expected test outputs for verification
            verify_expected_output_npy=self.build_inputs["output_npy_path"],
            # Save the intermediate model graphs
            save_intermediate_models=True,
            # Avoid RTL simulation for setting the FIFO sizes
            auto_fifo_strategy=AutoFIFOSizingMethod.CHARACTERIZE,
            # Do not automatically set FIFO sizes as this requires RTL simulation
            # not implemented for the attention operator
            auto_fifo_depths=False,
            # Build steps to execute
            steps=[
                # Need to apply some tidy-up transformations before converting to
                # the finn dialect of onnx
                step_tidy_up_pre_attention,
                # Convert all QONNX Quant nodes to Multithreshold nodes
                "step_qonnx_to_finn",
                # Tidy up the graph after converting from QONNX to FINN format
                # Note: Triggers a verification step
                "step_tidy_up",
                # Positional encoding needs to be streamlined first with slightly
                # different order of certain streamlining transformations to avoid
                # weird rounding issue of intermediate results
                step_streamline_positional,
                # Custom streamlining for models containing attention operators
                step_streamline_attention,
                # Streamlining of the residual branches
                step_streamline_residual,
                # Streamline the normalization layers, i.e., transposed batch norm
                step_streamline_norms,
                # Another round using the default streamlining steps
                # Note: Triggers a verification step
                "step_streamline",
                # New conversion of the scaled dot-product attention pattern
                step_convert_attention_to_hw,
                # Another tidy-up step to remove unnecessary dimensions and
                # operations after converting the attention operators to HLS
                step_tidy_up_post_attention,
                # Convert the elementwise binary operations to hardware operators.
                # These include for example adding residual branches and positional
                # encoding
                step_convert_elementwise_binary_to_hw,
                # Convert the Gather layer realizing the input token embedding to
                # the FINN hardware implementation, i.e., the Lookup layer
                step_convert_lookup_to_hw,
                # Properly replicate the stream feeding the query, key and value
                # projections
                step_replicate_streams,
                # Convert most other layers supported by FINN to HW operators
                "step_convert_to_hw",
                # Specialize HW layer implementations as either HLS or RTL
                "step_specialize_layers",
                "step_create_dataflow_partition",
                # Set the folding configuration to meet the cycles per sequence
                # target
                set_target_parallelization(seq_len, emb_dim),
                # Apply folding configuration, specifying hardware implementation
                # details
                # Note: This triggers a verification step
                step_apply_folding_config,
                "step_minimize_bit_width",
                # The ScaledDotProductAttention custom op does not define any
                # estimates
                "step_generate_estimate_reports",
                "step_hw_codegen",
                "step_hw_ipgen",
                # Set the attention- and residual-related FIFO depths insert FIFOs
                # and apply folding configuration once again
                # Note: Implement all FIFOs with a depth at least as deep as the
                # sequence length in URAM.
                set_fifo_depths(seq_len, emb_dim, uram_threshold=seq_len),
                # Run additional node-by-node verification in RTL simulation of the
                # model before creating the stitched IP
                # Note: end-to-end verification of the stitched IP in RTL simulation
                # is still not possible due to missing float IPs
                node_by_node_cppsim,
                # Only for debugging for now, does not work if "vivado" style
                # StreamingFIFOs are used
                # node_by_node_rtlsim,

                test_step_insert_tlastmarker, # required for instrumentation_wrapper

                "step_create_stitched_ip",

                # "step_measure_rtlsim_performance", # not possible due to float components

                step_synth_harness, #TODO: replace with instr wrapper (or port it into this step)
                
                #"step_out_of_context_synthesis", # for synthesis results (e.g. utilization)

                # normal deployment TODO: replace with instr wrapper (or port it into this step as an option) 
                #"step_synthesize_bitfile", 
                #"step_make_pynq_driver",
                #"step_deployment_package",

                #test_step_gen_vitis_xo, # preparation step for original instr wrapper integration
                #test_step_gen_instrumentation_wrapper, # preparation step for original instr wrapper integration

                #test_step_gen_instrwrap_sim, # preparation step for simulation of original instr wrapper integration
                #test_step_run_instrwrap_sim, # simulation with instr wrapper, disabled for now due to extreme runtime
                
                #test_step_export_xo, # preparation step for original instr wrapper integration
                #test_step_build_platform # synthesis with instr wrapper
            ]
        )
        # Run the build process on the dummy attention operator graph
        # TODO: maybe let this function return the cfg only, so it can be modified by bench context
        build.build_dataflow_cfg(self.build_inputs["onnx_path"], cfg)

    def run(self):
        self.steps_full_build_flow()
