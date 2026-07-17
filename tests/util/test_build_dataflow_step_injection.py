# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for custom step injection into the phase-based dataflow build flow.

These tests exercise ``DataflowBuildConfig.inject_steps_before`` /
``inject_steps_after`` for both phase-level and internal-step-level injection.
The build is restricted to the first two phases (model preparation and
streamlining), which run on CPU only, so the test needs no Vivado/synthesis.
"""

import pytest

import os
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import step_set_fifo_depths
from finn.util.basic import make_build_dir

# module-level log that injected steps append to, so the test can assert the
# order in which they ran relative to each other
CALL_ORDER = []


def _make_recording_step(tag):
    """Create a custom build step that records its invocation and passes the
    model through unchanged. The returned function is given a unique __name__ so
    that its intermediate model checkpoint is saved as ``<tag>.onnx``."""

    def step(model, cfg):
        CALL_ORDER.append(tag)
        return model

    step.__name__ = tag
    return step


@pytest.mark.util
def test_build_dataflow_step_injection():
    CALL_ORDER.clear()
    output_dir = make_build_dir("test_step_injection_")
    model_file = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/build_dataflow/model.onnx"

    # injected steps: two at phase granularity, two at internal-step granularity,
    # covering both the before and after hooks
    before_prepare_phase = _make_recording_step("inj_before_prepare_phase")
    after_qonnx = _make_recording_step("inj_after_qonnx")
    before_tidy = _make_recording_step("inj_before_tidy")
    after_prepare_phase = _make_recording_step("inj_after_prepare_phase")
    before_streamline = _make_recording_step("inj_before_streamline")
    after_optimize_phase = _make_recording_step("inj_after_optimize_phase")

    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        # restrict to the two CPU-only phases that our injections target
        steps=["phase_prepare_model", "phase_optimize_model"],
        inject_steps_before={
            # phase-level: resolved into the top-level step list
            "phase_prepare_model": [before_prepare_phase],
            # internal-step-level: resolved inside the phase
            "step_tidy_up": [before_tidy],
            "step_streamline": [before_streamline],
        },
        inject_steps_after={
            "step_qonnx_to_finn": [after_qonnx],
            "phase_prepare_model": [after_prepare_phase],
            "phase_optimize_model": [after_optimize_phase],
        },
        generate_outputs=[],
    )
    ret = build.build_dataflow_cfg(model_file, cfg)
    assert ret == 0, "build with injected steps failed"

    # all six injected steps must have run, in this exact order. This verifies:
    # - phase before/after hooks fire around the whole phase (prepare/optimize)
    # - internal-step before/after hooks fire around the correct internal step
    # - the two granularities interleave correctly: phase_prepare_model runs
    #   step_qonnx_to_finn then step_tidy_up, so the after-qonnx and before-tidy
    #   hooks land between the phase-level before/after hooks
    assert CALL_ORDER == [
        "inj_before_prepare_phase",
        "inj_after_qonnx",
        "inj_before_tidy",
        "inj_after_prepare_phase",
        "inj_before_streamline",
        "inj_after_optimize_phase",
    ]

    # injected steps save intermediate models like regular steps (default
    # save_intermediate_models=True), keyed by their __name__
    im_dir = output_dir + "/intermediate_models"
    for tag in CALL_ORDER:
        assert os.path.isfile(im_dir + f"/{tag}.onnx"), f"no checkpoint for {tag}"


@pytest.mark.util
def test_build_dataflow_no_injection_is_noop():
    """Without injection config, only the phase's own steps run and no extra
    checkpoints are produced."""
    output_dir = make_build_dir("test_no_injection_")
    model_file = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/build_dataflow/model.onnx"

    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        steps=["phase_prepare_model"],
        generate_outputs=[],
    )
    ret = build.build_dataflow_cfg(model_file, cfg)
    assert ret == 0

    # the phase's internal steps still checkpoint, but no injected steps exist
    im_dir = output_dir + "/intermediate_models"
    assert os.path.isfile(im_dir + "/step_qonnx_to_finn.onnx")
    assert os.path.isfile(im_dir + "/step_tidy_up.onnx")
    injected = [f for f in os.listdir(im_dir) if f.startswith("inj_")]
    assert injected == []


def _trivial_model():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["inp"], ["out"])
    graph = helper.make_graph([node], "test", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    return ModelWrapper(model)


@pytest.mark.util
def test_step_set_fifo_depths_skipped_for_estimate_only():
    """FIFO sizing uses an rtlsim-based strategy by default, which needs Vivado.
    For an estimate-only build (only ESTIMATE_REPORTS requested) it must be
    skipped and return the model untouched, so the flow stays synthesis-free."""
    output_dir = make_build_dir("test_fifo_skip_")
    model = _trivial_model()
    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    )
    # default auto_fifo_strategy is rtlsim-based, so the step should skip and
    # return the exact same model object without running any synthesis
    out = step_set_fifo_depths(model, cfg)
    assert out is model
    assert not os.path.isfile(output_dir + "/final_hw_config.json")
