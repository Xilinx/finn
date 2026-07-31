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
from qonnx.custom_op.registry import getCustomOp

from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.builder.build_dataflow_steps import (
    step_apply_folding_config,
    step_assign_ddr_weight_offsets,
    step_convert_to_hw,
    step_create_dataflow_partition,
    step_create_stitched_ip,
    step_deployment_package,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_loop_body_ipgen_and_stitch,
    step_loop_body_set_fifo_depths,
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
from finn.util.fpgadataflow import is_mlo


def _save_intermediate_model(model: ModelWrapper, step_name: str, cfg: DataflowBuildConfig):
    """Helper to save intermediate model checkpoint."""
    intermediate_model_dir = cfg.output_dir + "/intermediate_models"
    if not os.path.exists(intermediate_model_dir):
        os.makedirs(intermediate_model_dir)
    model.save(f"{intermediate_model_dir}/{step_name}.onnx")


def _execute_step(step_fn, model: ModelWrapper, cfg: DataflowBuildConfig):
    """Execute a step with injection support and save intermediate model if configured.

    This helper allows phases to:
    - Inject custom steps before/after any internal step using cfg.inject_steps_before/after
    - Save intermediate models after each internal step for inspection

    Step injection works at both phase and internal step level. For example:
    - inject_steps_after={"step_hw_codegen": [my_func]} will run my_func after
      step_hw_codegen, even when running phase_build_hardware.
    """
    step_name = step_fn.__name__

    # Inject steps BEFORE this step
    if step_name in cfg.inject_steps_before:
        for injected_step in cfg.inject_steps_before[step_name]:
            model = injected_step(model, cfg)
            if cfg.save_intermediate_models:
                _save_intermediate_model(model, injected_step.__name__, cfg)

    # Execute main step
    model = step_fn(model, cfg)

    # Save main step checkpoint
    if cfg.save_intermediate_models:
        _save_intermediate_model(model, step_name, cfg)

    # Inject steps AFTER this step
    if step_name in cfg.inject_steps_after:
        for injected_step in cfg.inject_steps_after[step_name]:
            model = injected_step(model, cfg)
            if cfg.save_intermediate_models:
                _save_intermediate_model(model, injected_step.__name__, cfg)

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
    loop rolling for FINNLoop nodes. step_convert_to_hw validates that all layers
    are fpgadataflow layers or form a contiguous dataflow block, so a failed
    conversion is reported before the dataflow partition is created.

    Internal steps:
    - step_convert_to_hw: Infer hardware layer types (validates conversion success)
    - step_create_dataflow_partition: Create accelerator subgraph
    - step_specialize_layers: Convert to HLS or RTL variants
    - step_loop_rolling: Process FINNLoop nodes (auto-detects if needed)

    Args:
        model: Input ModelWrapper
        cfg: Build configuration

    Returns:
        ModelWrapper with hardware-specialized operations

    Raises:
        AssertionError: If dataflow conversion validation fails
    """
    model = _execute_step(step_convert_to_hw, model, cfg)
    model = _execute_step(step_create_dataflow_partition, model, cfg)
    model = _execute_step(step_specialize_layers, model, cfg)
    model = _execute_step(step_loop_rolling, model, cfg)

    return model


def phase_optimize_hardware(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Phase: Configure parallelism, apply folding, optimize bit widths,
    FIFO sizing, generate reports.

    This phase configures the hardware parallelism and resource usage. It applies
    folding configurations, minimizes bit widths (after folding), decomposes
    transpose/shuffle operations, sizes FIFOs,
    and generates analytical performance/resource reports.

    Internal steps (each step checks its own config parameters):
    - step_target_fps_parallelization: Auto-parallelization (if target_fps set)
    - step_apply_folding_config: Apply folding configuration (if config provided)
    - step_minimize_bit_width: Minimize weight/accumulator bit widths (if enabled)
    - step_transpose_decomposition: Decompose Shuffle nodes
    - step_set_fifo_depths: FIFO sizing (see placement rules below)
    - step_generate_estimate_reports: Generate analytical estimates (if requested)

    Whether FIFO sizing runs here is an orchestration decision owned by the phase
    (the step itself just sizes FIFOs):

    - For MLO models it is deferred to phase_build_hardware, because the
      characterize strategy requires FINNLoop nodes to have stitched IP, which
      depends on loop body FIFO sizing and IP generation happening first.
    - For estimate-only builds whose configured sizing strategy needs synthesis it
      is skipped, so the estimate flow stays fast and synthesis-free. A future
      analytical (non-synthesis) sizing strategy would still run and feed the
      estimate reports.

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
    if is_mlo(model):
        # deferred to phase_build_hardware (needs loop body IPs first)
        pass
    else:
        # FIFO depths only affect synthesized hardware, so skip sizing for an
        # estimate-only build whose sizing strategy needs synthesis (the rtlsim-based
        # auto strategies or the folding-config path). This keeps the estimate flow
        # fast and synthesis-free. A future analytical (non-synthesis) strategy would
        # not be skipped, so it could still feed FIFO resources into the estimates.
        only_estimates = all(
            out == DataflowOutputType.ESTIMATE_REPORTS for out in cfg.generate_outputs
        )
        sizing_needs_synthesis = not cfg.auto_fifo_depths or cfg.auto_fifo_strategy in (
            "heuristic_rtlsim",
            "largefifo_rtlsim",
        )
        if only_estimates and sizing_needs_synthesis:
            print(
                "Skipping step_set_fifo_depths: only estimate reports requested and "
                "the configured FIFO sizing strategy requires synthesis. FIFO sizing "
                "is not needed for an estimate-only build."
            )
        else:
            model = _execute_step(step_set_fifo_depths, model, cfg)
    model = _execute_step(step_generate_estimate_reports, model, cfg)
    return model


def _apply_to_loop_bodies(model: ModelWrapper, cfg: DataflowBuildConfig, step_fn):
    """Apply a step function to all FINNLoop bodies recursively (depth-first).

    Args:
        model: ModelWrapper containing FINNLoop nodes
        cfg: Build configuration
        step_fn: Step function to apply to each loop body

    Returns:
        ModelWrapper with step applied to all loop bodies
    """
    for node in model.get_nodes_by_op_type("FINNLoop"):
        node_inst = getCustomOp(node)
        loop_model = node_inst.get_nodeattr("body")

        # Recursively process nested FINNLoop nodes first (depth-first)
        if loop_model.get_nodes_by_op_type("FINNLoop"):
            loop_model = _apply_to_loop_bodies(loop_model, cfg, step_fn)

        # Tag the loop body with its enclosing FINNLoop name so the step can
        # emit per-loop debug_fifo logs / waveforms without needing a special
        # signature. Steps read this via model.get_metadata_prop("loop_context").
        loop_model.set_metadata_prop("loop_context", node.name)
        print(f"Running {step_fn.__name__} for FINNLoop: {node.name}")
        loop_model = step_fn(loop_model, cfg)

        # Clear the transient tag so it doesn't leak into the saved body graph
        loop_model.set_metadata_prop("loop_context", "")

        node_inst.set_nodeattr("body", loop_model.graph)

    return model


def phase_build_hardware(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Phase: Generate hardware code, synthesize IP blocks.

    This phase generates hardware code for each layer (HLS C++ for HLS layers,
    RTL/SystemVerilog for RTL layers) and synthesizes IP blocks via Vitis HLS.

    For models with FINNLoop nodes, loop bodies are processed in a specific order:
    1. Assign DDR weight offsets (MLO & DDR only - baked into RTL by step_hw_codegen)
    2. FIFO sizing for loop bodies (creates FIFO nodes in subgraphs)
    3. step_hw_codegen for main model (applies to subgraphs, so loop body FIFOs get codegen)
    4. Create stitched IP for loop bodies (subgraph IP needed by FINNLoop wrapper)
    5. step_set_fifo_depths for main model (MLO only - needs loop body stitched IPs)
    6. step_hw_ipgen for main model (FINNLoop ipgen uses the subgraph IPs)

    For non-MLO models, step_set_fifo_depths already ran in phase_optimize_hardware.

    Internal steps:
    - step_assign_ddr_weight_offsets: Assign DDR weight offsets (MLO/DDR only)
    - step_loop_body_set_fifo_depths: FIFO sizing for loop bodies (MLO only)
    - step_hw_codegen: Generate HLS C++ or RTL code via PrepareIP
    - step_loop_body_ipgen_and_stitch: Synth IP and create stitchedIP for loop bodies (MLO only)
    - step_set_fifo_depths: FIFO sizing for main model (MLO only)
    - step_hw_ipgen: Synthesize IP blocks via HLSSynthIP

    Args:
        model: Input ModelWrapper
        cfg: Build configuration

    Returns:
        ModelWrapper with generated and synthesized IP blocks
    """
    # Step 1: MLO/DDR weight offset assignment (no-op unless mem_type DDR)
    # Must happen before step_hw_codegen so PrepareIP bakes the offsets into the RTL
    model = _execute_step(step_assign_ddr_weight_offsets, model, cfg)

    # Step 2: FIFO sizing for loop bodies (creates FIFO nodes in subgraphs)
    # Must happen before step_hw_codegen so the new FIFO nodes get code generated
    model = _apply_to_loop_bodies(model, cfg, step_loop_body_set_fifo_depths)

    # Step 3: HW codegen for main model (applies to subgraphs via apply_to_subgraphs=True)
    model = _execute_step(step_hw_codegen, model, cfg)

    # Step 4: Create stitched IP for loop bodies
    # Must happen before step_set_fifo_depths (MLO) so FINNLoop can be simulated
    model = _apply_to_loop_bodies(model, cfg, step_loop_body_ipgen_and_stitch)

    # Step 5: FIFO sizing for main model (MLO only)
    # Must happen after loop body stitched IPs so FINNLoop can be characterized
    if is_mlo(model):
        model = _execute_step(step_set_fifo_depths, model, cfg)
        # The FIFO nodes have no generated code yet: step_set_fifo_depths runs
        # its own PrepareIP before characterizing, but inserts the FIFOs after
        # it, and SplitLargeFIFOs and RemoveShallowFIFOs then add and remove
        # more. This has to come after all of that, which is why it is here
        # rather than at the end of the step. Everything else is already
        # generated and PrepareIP skips it -- though it still walks the graph
        # and every subgraph to find that out, warning per node as it goes.
        model = _execute_step(step_hw_codegen, model, cfg)

    # Step 6: HW ipgen for main model
    model = _execute_step(step_hw_ipgen, model, cfg)

    return model


def phase_generate_outputs(model: ModelWrapper, cfg: DataflowBuildConfig):
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
    "phase_generate_outputs": phase_generate_outputs,
}
