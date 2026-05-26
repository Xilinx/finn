# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
Phases for FINN dataflow builder pipeline.

Phases group related fine-grained steps into logical build phases.
All phases internally call functions from build_dataflow_steps.py.

Users can:
- Use phases via default_phase_build_steps
- Still use fine-grained steps
- Mix phases and fine-grained steps in custom pipelines
- Replace individual phases with custom implementations
- Inject custom steps before/after phases using inject_steps_before/after config
"""

import os
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import (
    step_apply_folding_config,
    step_convert_to_hw,
    step_create_dataflow_partition,
    step_create_stitched_ip,
    step_deployment_package,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_loop_rolling,
    step_make_driver,
    step_measure_rtlsim_performance,
    step_minimize_bit_width,
    step_qonnx_to_finn,
    step_set_fifo_depths,
    step_specialize_layers,
    step_streamline,
    step_synthesize_bitfile,
    step_target_fps_parallelization,
    step_tidy_up,
    step_transpose_decomposition,
)


def _execute_step(step_fn, model: ModelWrapper, cfg: DataflowBuildConfig):
    """Execute a step and save intermediate model if configured.

    This helper allows phases to save intermediate models after each internal step,
    making fine-grained checkpoints available for inspection even when using phases.
    """
    model = step_fn(model, cfg)

    # Save intermediate model if requested
    if cfg.save_intermediate_models:
        step_name = step_fn.__name__
        chkpt_name = f"{step_name}.onnx"
        intermediate_model_dir = cfg.output_dir + "/intermediate_models"
        if not os.path.exists(intermediate_model_dir):
            os.makedirs(intermediate_model_dir)
        model.save(f"{intermediate_model_dir}/{chkpt_name}")

    return model


def phase_prepare_model(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Phase: Import and prepare model for FINN transformations.

    This phase handles the initial model import and cleanup, converting from
    QONNX dialect to FINN and performing basic tidying operations.

    Internal steps:
    - step_qonnx_to_finn: Convert QONNX dialect to FINN
    - step_tidy_up: Shape/dtype inference, constant folding, cleanup

    Args:
        model: Input ModelWrapper
        cfg: Build configuration

    Returns:
        Prepared ModelWrapper ready for optimization
    """
    model = _execute_step(step_qonnx_to_finn, model, cfg)
    model = _execute_step(step_tidy_up, model, cfg)
    return model


def phase_optimize_model(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Phase: Apply model-specific streamlining transformations.

    This phase applies streamlining to move and absorb operations for hardware
    efficiency. Streamlining is highly model-dependent and frequently customized.

    Internal steps:
    - step_streamline: Apply streamlining transformations

    Note: This phase can be easily replaced with a custom streamline function
    in the steps list for model-specific optimizations.

    Args:
        model: Input ModelWrapper
        cfg: Build configuration

    Returns:
        Streamlined ModelWrapper
    """
    model = _execute_step(step_streamline, model, cfg)
    return model


def phase_convert_to_hardware(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Phase: Convert model to hardware-eligible operations and specialize.

    This phase identifies hardware-eligible operations, creates the dataflow
    partition, specializes layers for the target backend (HLS/RTL), and handles
    loop rolling for FINNLoop nodes.

    Internal steps:
    - step_convert_to_hw: Infer hardware layer types
    - step_create_dataflow_partition: Create accelerator subgraph
    - step_specialize_layers: Convert to HLS or RTL variants
    - step_loop_rolling: Process FINNLoop nodes (auto-detects if needed)

    Args:
        model: Input ModelWrapper
        cfg: Build configuration

    Returns:
        ModelWrapper with hardware-specialized operations
    """
    model = _execute_step(step_convert_to_hw, model, cfg)
    model = _execute_step(step_create_dataflow_partition, model, cfg)
    model = _execute_step(step_specialize_layers, model, cfg)
    model = _execute_step(step_loop_rolling, model, cfg)
    return model


def phase_optimize_hardware(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Phase: Configure parallelism, apply folding, optimize bit widths, generate reports.

    This phase configures the hardware parallelism and resource usage. It applies
    folding configurations, minimizes bit widths (after folding), decomposes
    transpose/shuffle operations, and generates analytical performance/resource reports.

    Internal steps (each step checks its own config parameters):
    - step_target_fps_parallelization: Auto-parallelization (if target_fps set)
    - step_apply_folding_config: Apply folding configuration (if config provided)
    - step_minimize_bit_width: Minimize weight/accumulator bit widths (if enabled)
    - step_transpose_decomposition: Decompose Shuffle nodes
    - step_generate_estimate_reports: Generate analytical estimates (if requested)

    Note: This is the extension point for future analytical FIFO sizing.

    Args:
        model: Input ModelWrapper
        cfg: Build configuration

    Returns:
        ModelWrapper with optimized parallelism and resource configuration
    """
    model = _execute_step(step_target_fps_parallelization, model, cfg)
    model = _execute_step(step_apply_folding_config, model, cfg)
    model = _execute_step(step_minimize_bit_width, model, cfg)
    model = _execute_step(step_transpose_decomposition, model, cfg)
    model = _execute_step(step_generate_estimate_reports, model, cfg)
    return model


def phase_build_hardware(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Phase: Generate hardware code, synthesize IP blocks, size FIFOs.

    This phase generates hardware code for each layer (HLS C++ for HLS layers,
    RTL/SystemVerilog for RTL layers), synthesizes IP blocks via Vitis HLS, and
    sizes FIFOs. FIFO sizing is automatically skipped if FIFOs already exist in
    the model (e.g., from analytical sizing).

    Internal steps:
    - step_hw_codegen: Generate HLS C++ or RTL code via PrepareIP
    - step_hw_ipgen: Synthesize IP blocks via HLSSynthIP
    - step_set_fifo_depths: Auto or manual FIFO sizing (auto-skipped if FIFOs exist)

    Note: When analytical FIFO sizing is available (future), it would create
    StreamingFIFO nodes in phase_optimize_hardware, causing this phase to
    auto-skip hardware-based FIFO characterization.

    Args:
        model: Input ModelWrapper
        cfg: Build configuration

    Returns:
        ModelWrapper with generated and synthesized IP blocks
    """
    model = _execute_step(step_hw_codegen, model, cfg)
    model = _execute_step(step_hw_ipgen, model, cfg)

    # FIFO sizing - auto-detect if already done (e.g., analytically)
    fifo_nodes = model.get_nodes_by_op_type("StreamingFIFO")
    if len(fifo_nodes) == 0 and cfg.auto_fifo_depths:
        # No FIFOs yet, run characterization/rtlsim
        model = _execute_step(step_set_fifo_depths, model, cfg)
    elif len(fifo_nodes) > 0:
        # FIFOs already sized (analytical or manual), skip hardware characterization
        print("FIFOs already present in model, skipping step_set_fifo_depths")

    return model


def phase_synthesize_hardware(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Phase: Create final hardware artifacts (stitched IP or bitfile + deployment package).

    This phase creates the final hardware deliverables based on requested outputs.
    It can generate stitched IP (including optional OOC synthesis), measure RTL
    simulation performance, or create a complete bitfile with driver and deployment
    package.

    Internal steps (each step checks generate_outputs):
    - step_create_stitched_ip: Create stitched IP (includes OOC synth if requested)
    - step_measure_rtlsim_performance: Measure RTL sim performance (if requested)
    - step_synthesize_bitfile: Full bitfile synthesis (if BITFILE requested)
    - step_make_driver: Generate PYNQ or C++ driver (if BITFILE requested)
    - step_deployment_package: Package for deployment (if requested)

    Note: OOC (out-of-context) synthesis happens inside step_create_stitched_ip
    when DataflowOutputType.OOC_SYNTH is requested, not as a separate step.

    Args:
        model: Input ModelWrapper
        cfg: Build configuration

    Returns:
        ModelWrapper with final hardware artifacts generated
    """
    model = _execute_step(step_create_stitched_ip, model, cfg)
    model = _execute_step(step_measure_rtlsim_performance, model, cfg)
    model = _execute_step(step_synthesize_bitfile, model, cfg)
    model = _execute_step(step_make_driver, model, cfg)
    model = _execute_step(step_deployment_package, model, cfg)
    return model


#: Map phase name strings to phase functions
build_dataflow_phase_lookup = {
    "phase_prepare_model": phase_prepare_model,
    "phase_optimize_model": phase_optimize_model,
    "phase_convert_to_hardware": phase_convert_to_hardware,
    "phase_optimize_hardware": phase_optimize_hardware,
    "phase_build_hardware": phase_build_hardware,
    "phase_synthesize_hardware": phase_synthesize_hardware,
}
