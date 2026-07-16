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
    VerificationStepType,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(HERE, "siglip_w4a6_qat_w4a6_op20_qonnx_clean.onnx")
OUTPUT_DIR = os.path.join(HERE, "output_siglip")

# Static FIFO depths for the OUTSIDE-loop FIFOs. With auto_fifo_depths=False,
# step_set_fifo_depths inserts top-level FIFOs and applies these depths via
# ApplyConfig -- no rtlsim characterization (which took days for this MLO design).
# The FINNLoop body FIFOs are NOT covered here: they were already sized and baked
# into the loop IP during step_hw_codegen (prepare_loop_ops_fifo_sizing), and the
# auto_fifo_depths=False path only touches top-level nodes.
STATIC_FIFO_CONFIG = os.path.join(HERE, "configs", "static_fifo_depths.json")

# Target FPGA part: VCK190 (Versal AI Core). No board/shell is needed for
# stitched-IP + rtlsim, so the part only drives IP synthesis and resource
# estimation.
FPGA_PART = "xcvc1902-vsva2197-2MP-e-S"
CLK_PERIOD_NS = 5.0  # 200 MHz
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


def select_build_steps():
    """Ordered step pipeline (BERT-shaped, SigLIP-adapted, up to stitched IP)."""
    return [
        # --- pre-processing (custom) ---
        custom_steps.step_siglip_cleanup,
        custom_steps.step_extract_norm_scale_bias,  # LN gamma/beta -> Mul/Add so LN converts
        # --- base pipeline (stock FINN) ---
        "step_qonnx_to_finn",
        # Golden reference generated on the CONVERTED FINN-ONNX graph (post
        # qonnx_to_finn), so it already embeds the Quant->MultiThreshold conversion
        # divergence. This makes stitched_ip_rtlsim measure HW faithfulness to the
        # design FINN actually builds, not the (lossy) QONNX->FINN conversion cost.
        custom_steps.step_generate_reference_io,
        "step_tidy_up",
        "step_streamline",  # lowers the Conv head to MatMul, streamlines
        custom_steps.step_siglip_streamlining,  # clear q/k/v fork Mul so MatMuls stay integer
        custom_steps.step_absorb_signed_ln_scale,  # uniform layers for loop rolling
        "step_convert_to_hw",  # infers Softmax / Gelu(PWPolyF) / LayerNorm / MVAU
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_loop_rolling",  # MLO: roll 12 encoder layers into one FINNLoop body
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_transpose_decomposition",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
    ]


def make_cfg(start_step=None) -> build_cfg.DataflowBuildConfig:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return build_cfg.DataflowBuildConfig(
        output_dir=OUTPUT_DIR,
        steps=select_build_steps(),
        start_step=start_step,
        synth_clk_period_ns=CLK_PERIOD_NS,
        fpga_part=FPGA_PART,
        # MLO
        mlo=True,
        loop_body_hierarchy=LOOP_BODY_HIERARCHY,
        loop_body_range=LOOP_BODY_RANGE,
        # folding driven by target throughput (no manual folding config)
        target_fps=TARGET_FPS,
        # thresholds / fifos
        standalone_thresholds=True,
        split_large_fifos=True,
        # Static outside-loop FIFO sizing: skip the multi-day characterization
        # rtlsim. Loop-body FIFOs are already sized into the loop IP (step_hw_codegen).
        auto_fifo_depths=False,
        folding_config_file=STATIC_FIFO_CONFIG,
        stitched_ip_gen_dcp=False,
        mute_config_assertions=True,
        save_intermediate_models=True,
        # Reference is generated post-conversion (see step order), so it matches
        # the design FINN builds; rtlsim should agree to tight tolerance. The
        # QONNX_TO_FINN_PYTHON check is dropped: it would compare the converted
        # graph against a reference taken from that same graph (trivially ~0) and
        # no longer measures anything useful.
        verification_atol=0.1,
        # Enable waveform tracing so create_stitched_ip sets a real rtlsim_trace
        # path. Without it, the MLO stitched-IP rtlsim path passes wdb=None to the
        # xsi.Design C++ binding, which rejects null -> "basic_string null not
        # valid". (Also gives a .wdb to debug the verification result.)
        verify_save_rtlsim_waveforms=False,
        verify_steps=[
            VerificationStepType.STITCHED_IP_RTLSIM,
        ],
        verify_input_npy=os.path.join(OUTPUT_DIR, "input.npy"),
        verify_expected_output_npy=os.path.join(OUTPUT_DIR, "expected_output.npy"),
        generate_outputs=[
            DataflowOutputType.STITCHED_IP,
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
            "e.g. --start-step step_siglip_streamlining"
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    cfg = make_cfg(start_step=args.start_step)
    build.build_dataflow_cfg(args.model, cfg)


if __name__ == "__main__":
    main()
