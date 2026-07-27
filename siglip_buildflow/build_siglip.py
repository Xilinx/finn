#!/usr/bin/env python3

"""
SigLIP (12-layer vision transformer, MLO) FINN stitched-IP build.
"""

import argparse
import os
import sys
from collections import namedtuple

import custom_steps

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import (
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(HERE, "siglip_w4a6_qat_w4a6_t196_clean.onnx")
OUTPUT_DIR = os.path.join(HERE, "output_siglip")

# Static FIFO depths for the OUTSIDE-loop FIFOs. With auto_fifo_depths=False,
# step_set_fifo_depths inserts top-level FIFOs and applies these depths via
# ApplyConfig -- no rtlsim characterization (which took days for this MLO design).
# The FINNLoop body FIFOs are NOT covered here: they were already sized and baked
# into the loop IP during step_hw_codegen (prepare_loop_ops_fifo_sizing), and the
# auto_fifo_depths=False path only touches top-level nodes.
STATIC_FIFO_CONFIG = os.path.join(HERE, "configs", "static_fifo_depths.json")

# Target board: VCK190 (Versal AI Core). ``board`` drives the VIVADO_VERSAL shell
# flow for bitfile generation; ``fpga_part`` (the same xcvc1902 part the board
# resolves to) is kept explicit so IP synthesis / resource estimation match.
BOARD = "VCK190"
FPGA_PART = "xcvc1902-vsva2197-2MP-e-S"
# The versal shell template now reconfigures the CIPS pl0_ref_clk from FREQ_MHZ
# (= 1 / (synth_clk_period_ns * 1e-3)); see the "Configure pl0_ref_clk based on
# FREQ_MHZ" change to templates.py. So synth_clk_period_ns is authoritative for
# the frequency the FINN kernels actually run at in the shell -- it no longer
# defaults to the golden 333 MHz pl0_ref_clk. Target 250 MHz for timing margin;
# FINN IP-gen / folding / OOC synthesis all use this same clock.
CLK_PERIOD_NS = 4.0  # 250 MHz (pl0_ref_clk = FREQ_MHZ, set from this period)
# MLO loops the single encoder-layer body once per layer (12 iterations/frame),
# and SetFolding budgets the body against target_cycles_per_frame WITHOUT dividing
# by iterations. So to hit ~50 FPS overall, fold the body to ~1/12 of the frame
# budget by targeting 50 * 12. The resulting per-node folding is written to
# output_siglip/auto_folding_config.json (the effective single-layer config).
LOOP_ITERATIONS = 12
TARGET_FPS = 50 * LOOP_ITERATIONS  # ~50 FPS end-to-end through the rolled loop

# ---------------------------------------------------------------------------
# MLO loop-body markers
# ---------------------------------------------------------------------------
# ``loop_body_hierarchy`` MUST be [["", "layers.0"]]: step_loop_rolling stamps
# the in-range nodes with a hardcoded ``name_scopes = "['', 'layers.0']"`` before
# LoopExtraction prefix-matches against it (see step_loop_rolling in
# build_dataflow_steps.py). This mirrors the BERT flow and the end2end MLO test.
LOOP_BODY_HIERARCHY = [["", "layers.0"]]

# SetLoopBoundary only reads ``.name`` off each endpoint, so lightweight name
# holders are enough. Set these to the first encoder layer's [start, end]
# FINN-HW-layer node names (the names that exist after step_specialize_layers).
NodeRef = namedtuple("NodeRef", ["name"])
LOOP_BODY_RANGE = (NodeRef("DuplicateStreams_hls_0"), NodeRef("ElementwiseAdd_rtl_9"))


# Prebuilt loop-body folding configs (see gen_folding_configs.py). Each targets a
# fraction of the VC1902 DSP budget (~25/50/75%) with the loop body balanced for
# equal per-node cycles. Selected via --folding; replaces target_fps autofolding.
FOLDING_CONFIGS = {
    "small": os.path.join(HERE, "configs", "folding_small.json"),
    "med": os.path.join(HERE, "configs", "folding_med.json"),
    "large": os.path.join(HERE, "configs", "folding_large.json"),
}


def select_build_steps():
    """Ordered pipeline (BERT-shaped, SigLIP-adapted) using FINN builder phases.

    We use the stock phases for everything except model optimization: SigLIP needs
    a custom streamline (see below), so ``phase_optimize_model`` -- which runs the
    stock ``step_streamline`` -- is replaced by SigLIP's custom callables inserted
    between ``phase_prepare_model`` and ``phase_convert_to_hardware``.

    ``phase_build_hardware`` owns the MLO ordering (loop-body FIFO sizing -> hw
    codegen -> loop-body ipgen+stitch -> main-model set_fifo_depths -> main ipgen),
    which the previous flat step list did not perform. ``step_target_fps_parallelization``
    self-guards on ``target_fps``, so no step-list surgery is needed for manual
    folding: passing ``target_fps=None`` makes it a no-op.
    """
    return [
        # --- pre-processing (custom, before the stock import phase) ---
        custom_steps.step_siglip_cleanup,
        custom_steps.step_extract_norm_scale_bias,  # LN gamma/beta -> Mul/Add so LN converts
        # Golden reference generated on the RAW QONNX graph (before qonnx_to_finn),
        # BERT-style. The reference is independent of ConvertQONNXtoFINN, so
        # QONNX_TO_FINN_PYTHON verification measures conversion fidelity and every
        # later (equivalence-preserving) step is checked against the true QONNX output.
        custom_steps.step_generate_reference_io,
        # --- import + tidy (stock) ---
        "phase_prepare_model",  # step_qonnx_to_finn + step_tidy_up
        # --- streamlining (custom; replaces phase_optimize_model) ---
        # Custom step_streamline: same as stock but threads preserve_thresh_shape=True
        # so channelwise LN gamma/beta don't expand per-tensor thresholds to per-channel
        # (which trips MultiThreshold's NCHW channel-axis assumption on [N,tokens,C]).
        custom_steps.step_streamline,  # lowers the Conv head to MatMul, streamlines
        custom_steps.step_siglip_streamlining,  # clear q/k/v fork Mul so MatMuls stay integer
        custom_steps.step_absorb_signed_ln_scale,  # uniform layers for loop rolling
        # --- convert to hw + specialize + loop rolling (stock) ---
        # phase_convert_to_hardware: step_convert_to_hw (infers Softmax / Gelu(PWPolyF)
        # / LayerNorm / MVAU; now hard-asserts an all-dataflow / contiguous block) +
        # step_create_dataflow_partition + step_specialize_layers + step_loop_rolling
        # (MLO: rolls the 12 encoder layers into one FINNLoop body).
        "phase_convert_to_hardware",
        # --- folding + bit width + transpose (stock) ---
        # phase_optimize_hardware: step_target_fps_parallelization (no-op if target_fps
        # is None) + step_apply_folding_config + step_minimize_bit_width +
        # step_transpose_decomposition. For MLO, set_fifo_depths is deferred to
        # phase_build_hardware (needs the loop-body stitched IPs first).
        "phase_optimize_hardware",
        # --- codegen + ipgen + MLO fifo sizing (stock) ---
        # phase_build_hardware owns the MLO order: loop-body FIFO sizing -> hw codegen
        # -> loop-body ipgen+stitch -> main set_fifo_depths -> main ipgen.
        "phase_build_hardware",
        # --- stitched IP + Versal (VCK190) shell -> bitfile + driver + deploy pkg ---
        "phase_generate_outputs",
    ]


def make_cfg(start_step=None, stop_step=None, folding=None) -> build_cfg.DataflowBuildConfig:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Folding source: a prebuilt config (balanced loop body, fixed DSP budget) OR
    # target_fps autofolding. The prebuilt config also carries the outside-loop
    # FIFO depth in its Defaults, so it doubles as the folding_config_file that
    # step_set_fifo_depths re-applies (auto_fifo_depths=False).
    # With target_fps=None, step_target_fps_parallelization (run inside
    # phase_optimize_hardware) self-guards and becomes a no-op.
    use_manual_folding = folding is not None
    if use_manual_folding:
        folding_config_file = FOLDING_CONFIGS[folding]
        target_fps = None
    else:
        folding_config_file = STATIC_FIFO_CONFIG
        target_fps = TARGET_FPS

    return build_cfg.DataflowBuildConfig(
        output_dir=OUTPUT_DIR,
        steps=select_build_steps(),
        start_step=start_step,
        stop_step=stop_step,
        synth_clk_period_ns=CLK_PERIOD_NS,
        fpga_part=FPGA_PART,
        # Versal shell: board drives the VIVADO_VERSAL flow, which integrates the
        # stitched IP into the golden VCK190 overlay (FINN_VERSAL_GOLDEN_DIR) and
        # emits a full PL PDI in step_synthesize_bitfile.
        board=BOARD,
        shell_flow_type=ShellFlowType.VIVADO_VERSAL,
        # MLO
        mlo=True,
        # Weights streamed from a single DDR address space (the merged mlo_ddr
        # flow): step_hw_codegen runs AssignMemoryOffset to byte-pack per-MVAU
        # weights + intermediate frames into DDR, and stitched-IP rtlsim drives
        # the m_axi_MVAU_* / m_axi_intermediate_frame ports via those offsets.
        mlo_weight_mem="DDR",
        loop_body_hierarchy=LOOP_BODY_HIERARCHY,
        loop_body_range=LOOP_BODY_RANGE,
        # folding: prebuilt config (target_fps=None) or target-throughput autofolding
        target_fps=target_fps,
        # thresholds / fifos
        preserve_thresh_shape=True,
        standalone_thresholds=True,
        split_large_fifos=True,
        # Static outside-loop FIFO sizing: skip the multi-day characterization
        # rtlsim. Loop-body FIFOs are already sized into the loop IP (step_hw_codegen).
        auto_fifo_depths=False,
        folding_config_file=folding_config_file,
        stitched_ip_gen_dcp=False,
        mute_config_assertions=True,
        save_intermediate_models=True,
        verification_atol=0.1,
        verify_save_rtlsim_waveforms=False,
        # STITCHED_IP_RTLSIM dropped: the MLO rtlsim verify is very slow, so we go
        # straight to synthesis. Conversion fidelity is still checked at
        # QONNX_TO_FINN_PYTHON, and every later step is equivalence-preserving.
        verify_steps=[
            VerificationStepType.TIDY_UP_PYTHON,
            VerificationStepType.QONNX_TO_FINN_PYTHON,
        ],
        verify_input_npy=os.path.join(OUTPUT_DIR, "input.npy"),
        verify_expected_output_npy=os.path.join(OUTPUT_DIR, "expected_output.npy"),
        generate_outputs=[
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL_PATH,
        help="Path to the input ONNX model (default: %(default)s)",
    )
    parser.add_argument(
        "--start-step",
        default=None,
        help=(
            "Resume from this build step, reloading the intermediate model saved "
            "before it (requires a prior run with save_intermediate_models). "
            "With phase-based builds, use a phase name, e.g. "
            "--start-step phase_build_hardware"
        ),
    )
    parser.add_argument(
        "--stop-step",
        default=None,
        help=(
            "Stop after this build step/phase (inclusive). Use a phase name, e.g. "
            "--stop-step phase_convert_to_hardware to dry-run up to and including "
            "the hardware-conversion validation."
        ),
    )
    parser.add_argument(
        "--folding",
        choices=sorted(FOLDING_CONFIGS),
        default=None,
        help=(
            "Use a prebuilt loop-body folding config (~25/50/75%% of the VC1902 "
            "DSP budget) instead of target_fps autofolding. Drops "
            "step_target_fps_parallelization."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    cfg = make_cfg(start_step=args.start_step, stop_step=args.stop_step, folding=args.folding)
    build.build_dataflow_cfg(args.model, cfg)


if __name__ == "__main__":
    main()
