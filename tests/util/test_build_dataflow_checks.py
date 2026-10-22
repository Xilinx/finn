# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for build configuration checks."""

import pytest

import json
import numpy as np
import os
from onnx import TensorProto, helper
from unittest.mock import patch

from finn.builder.build_dataflow import (
    build_dataflow_cfg,
    build_makes_driver,
    build_reads_part,
)
from finn.builder.build_dataflow_checks import _build_runs_tools
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
    estimate_only_dataflow_steps,
    verify_step_prereqs,
)
from finn.builder.build_dataflow_phases import tool_running_phases
from finn.builder.build_dataflow_steps import tool_running_steps
from finn.util.basic import make_build_dir


def make_test_model(build_dir):
    """Create minimal ONNX model for testing."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["inp"], ["out"])
    graph = helper.make_graph([node], "test", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model_path = os.path.join(build_dir, "model.onnx")
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    return model_path


def cfg(output_dir, **kw):
    """Create config that stops immediately after first phase."""
    return DataflowBuildConfig(
        output_dir=output_dir,
        enable_build_pdb_debug=False,
        synth_clk_period_ns=5.0,
        stop_step=kw.pop("stop_step", "phase_prepare_model"),
        generate_outputs=kw.pop("generate_outputs", [DataflowOutputType.ESTIMATE_REPORTS]),
        **kw
    )


@pytest.mark.util
class TestConfigCheckIntegration:
    def test_report_files_created(self):
        """Config check report should be saved to output_dir."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            build_dataflow_cfg(
                model_path,
                cfg(output_dir, board="AUP-ZU3_8GB", shell_flow_type=ShellFlowType.VIVADO_ZYNQ),
            )

        assert os.path.exists(os.path.join(output_dir, "config_check_report.txt"))
        assert os.path.exists(os.path.join(output_dir, "config_check_report.json"))

    def test_invalid_config_raises(self):
        """Invalid config should raise AssertionError."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2025.1"}):
            with pytest.raises(AssertionError, match="Configuration check failed"):
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        board="V80",
                        shell_flow_type=ShellFlowType.VITIS_ALVEO,
                        generate_outputs=[DataflowOutputType.BITFILE],
                    ),
                )

    def test_muted_config_proceeds(self):
        """Invalid config with mute_config_assertions=True should not raise."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        # This would normally fail (V80 needs SLASH_ALVEO), but muting allows it to proceed
        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2025.1"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        board="V80",
                        shell_flow_type=ShellFlowType.VITIS_ALVEO,
                        generate_outputs=[DataflowOutputType.BITFILE],
                        mute_config_assertions=True,
                    ),
                )
            except AssertionError as e:
                assert "Configuration check failed" not in str(e)
            except Exception:
                pass  # Other errors are fine

        assert os.path.exists(os.path.join(output_dir, "config_check_report.txt"))

    def test_report_contains_errors(self):
        """Report JSON should contain the detected errors."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2025.1"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        board="V80",
                        shell_flow_type=ShellFlowType.VITIS_ALVEO,
                        generate_outputs=[DataflowOutputType.BITFILE],
                        mute_config_assertions=True,
                    ),
                )
            except Exception:
                pass

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)

        assert report["summary"]["errors"] > 0
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert "board-shell-mismatch" in error_names

    @pytest.mark.parametrize(
        "board,flow,should_error",
        [
            ("AUP-ZU3_8GB", ShellFlowType.VIVADO_ZYNQ, False),
            ("U55C", ShellFlowType.VITIS_ALVEO, False),
            ("AUP-ZU3_8GB", ShellFlowType.VITIS_ALVEO, True),
            ("U55C", ShellFlowType.VIVADO_ZYNQ, True),
        ],
    )
    def test_board_shell_compatibility(self, board, flow, should_error):
        """Test various board/shell flow combinations with BITFILE generation."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        env = {
            "XILINX_VIVADO": "/tools/Vivado/2024.2",
            "VITIS_PATH": "/tools/Vitis/2024.2",
            "PLATFORM_REPO_PATHS": "/opt/platforms",
            "XILINX_XRT": "/opt/xilinx/xrt",
        }
        with patch.dict("os.environ", env):
            if should_error:
                with pytest.raises(AssertionError, match="Configuration check failed"):
                    build_dataflow_cfg(
                        model_path,
                        cfg(
                            output_dir,
                            board=board,
                            shell_flow_type=flow,
                            generate_outputs=[DataflowOutputType.BITFILE],
                            target_fps=1000,
                        ),
                    )
            else:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        board=board,
                        shell_flow_type=flow,
                        generate_outputs=[DataflowOutputType.BITFILE],
                        target_fps=1000,
                    ),
                )
                assert os.path.exists(os.path.join(output_dir, "config_check_report.txt"))

    def test_shell_without_bitfile_warning(self):
        """Setting shell_flow_type without BITFILE should produce a warning."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            build_dataflow_cfg(
                model_path, cfg(output_dir, shell_flow_type=ShellFlowType.VIVADO_ZYNQ)
            )

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        assert "shell_no_bitfile" in [c["name"] for c in report["checks"] if not c["passed"]]

    def test_aupzu3_vivado_version(self):
        """AUP-ZU3_8GB requires Vivado 2024.1+ for a build that runs the tools: here the
        default steps, which run HLS even when only estimate reports are requested."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2022.2"}):
            with pytest.raises(AssertionError, match="Configuration check failed"):
                build_dataflow_cfg(
                    model_path,
                    cfg(output_dir, board="AUP-ZU3_8GB", stop_step=None),
                )

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert error_names == ["vivado-too-old"]

    @pytest.mark.parametrize("board", ["Pynq-Z1", "Pynq-Z2"])
    def test_zynq7000_retired(self, board):
        """Retired Zynq-7000 boards are blocked at config-check time."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        board=board,
                        shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
                        mute_config_assertions=True,
                    ),
                )
            except Exception:
                pass

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert "retired-board" in error_names

    def test_verify_step_prereq_missing_phase(self):
        """A verify_steps entry whose phase is excluded by stop_step should warn, since
        that verification would otherwise silently never run.

        This is a warning rather than an error: a custom step/flow can legitimately
        handle verification itself, so this shouldn't block the build."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        stop_step="phase_build_hardware",
                        verify_steps=[VerificationStepType.STITCHED_IP_RTLSIM],
                        mute_config_assertions=True,
                    ),
                )
            except Exception:
                pass

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        warning_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "WARNING"
        ]
        assert "verify_step_prereq" in warning_names

    def test_verify_step_prereq_present_phase_ok(self):
        """A verify_steps entry whose phase is included should not error."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        stop_step="phase_optimize_model",
                        verify_steps=[VerificationStepType.TIDY_UP_PYTHON],
                        mute_config_assertions=True,
                    ),
                )
            except Exception:
                pass

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert "verify_step_prereq" not in error_names

    def test_verify_step_prereq_fine_grained_step_ok(self):
        """A custom steps list naming the fine-grained step directly (instead of
        its enclosing phase) should also satisfy the check."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        steps=["phase_prepare_model", "step_streamline"],
                        stop_step="step_streamline",
                        verify_steps=[VerificationStepType.STREAMLINED_PYTHON],
                        mute_config_assertions=True,
                    ),
                )
            except Exception:
                pass

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert "verify_step_prereq" not in error_names

    def test_folding_missing_errors(self):
        """A non-estimate output with neither target_fps nor folding_config_file
        should error, instead of silently building at PE=1/SIMD=1."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            with pytest.raises(AssertionError, match="Configuration check failed"):
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        board="AUP-ZU3_8GB",
                        shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
                        generate_outputs=[DataflowOutputType.BITFILE],
                    ),
                )

    def test_folding_missing_estimate_only_ok(self):
        """ESTIMATE_REPORTS alone doesn't require target_fps or folding_config_file."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            build_dataflow_cfg(model_path, cfg(output_dir))

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert "folding_missing" not in error_names

    def test_folding_target_fps_satisfies_check(self):
        """Setting target_fps clears the folding_missing check."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            build_dataflow_cfg(
                model_path,
                cfg(
                    output_dir,
                    board="AUP-ZU3_8GB",
                    shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
                    generate_outputs=[DataflowOutputType.BITFILE],
                    target_fps=1000,
                ),
            )

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert "folding_missing" not in error_names

    @pytest.mark.parametrize(
        "kw, code",
        [
            # a board and a part that is not its part: kernels for one part, a shell
            # for the other
            (dict(board="AUP-ZU3_8GB", fpga_part="xczu7ev-ffvc1156-2-e"), "board-part-mismatch"),
            # a part FINN's platform data does not hold
            (dict(fpga_part="xcmy100tffg1761-2"), "unknown-part"),
            # a Vitis platform for a board of another shell
            (
                dict(board="AUP-ZU3_8GB", vitis_platform="xilinx_u55c_gen3x16_xdma_3_202210_1"),
                "vitis-platform-unused",
            ),
            # a bitfile for a board without a shell flow
            (
                dict(board="VEK280", generate_outputs=[DataflowOutputType.BITFILE]),
                "no-shell",
            ),
            # a bitfile with the Zynq shell flow but no board
            (
                dict(
                    fpga_part="xczu7ev-ffvc1156-2-e",
                    shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
                    generate_outputs=[DataflowOutputType.BITFILE],
                ),
                "board-required",
            ),
            # an unknown board
            (dict(board="ZCU999"), "unknown-board"),
            # a V80 part built with the Vitis shell flow, which builds for UltraScale+
            (
                dict(
                    fpga_part="xcv80-lsva4737-2MHP-e-s",
                    shell_flow_type=ShellFlowType.VITIS_ALVEO,
                    vitis_platform="xilinx_u55c_gen3x16_xdma_3_202210_1",
                    generate_outputs=[DataflowOutputType.BITFILE],
                ),
                "shell-architecture-mismatch",
            ),
        ],
    )
    def test_target_refusals_are_errors_named_by_code(self, kw, code):
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(output_dir, mute_config_assertions=True, target_fps=1000, **kw),
                )
            except Exception:
                pass

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert code in error_names

    @pytest.mark.parametrize("board", ["AUP-ZU3_8GB", "V80"])
    @pytest.mark.parametrize(
        "kw, errors",
        [
            # estimate reports from steps that stop before IP generation run no AMD tool:
            # no Vivado needed
            (dict(steps=estimate_only_dataflow_steps, stop_step=None), []),
            (dict(stop_step="phase_optimize_hardware"), []),
            # the default steps run HLS even for estimate reports alone, so the board's
            # Vivado minimum is checked, and cannot be without Vivado
            (dict(stop_step=None), ["vivado-unknown"]),
            # so does a bitfile
            (
                dict(generate_outputs=[DataflowOutputType.BITFILE], target_fps=1000),
                ["vivado-unknown"],
            ),
        ],
    )
    def test_vivado_minimum_without_vivado(self, board, kw, errors):
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ"):
            os.environ.pop("XILINX_VIVADO", None)
            try:
                build_dataflow_cfg(model_path, cfg(output_dir, board=board, **kw))
            except Exception:
                pass  # the checks raise on an error; a passing build may fail later

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert error_names == errors

    def test_tools_are_decided_by_the_steps_the_build_runs(self):
        estimates = [DataflowOutputType.ESTIMATE_REPORTS]
        assert _build_runs_tools(cfg("/tmp/x", stop_step=None, generate_outputs=estimates))
        assert not _build_runs_tools(
            cfg("/tmp/x", stop_step="phase_optimize_hardware", generate_outputs=estimates)
        )
        assert not _build_runs_tools(
            cfg("/tmp/x", steps=estimate_only_dataflow_steps, stop_step=None)
        )
        # FIFO sizing runs HLS once more than estimates are requested
        stitched = [DataflowOutputType.STITCHED_IP]
        assert _build_runs_tools(
            cfg(
                "/tmp/x",
                steps=estimate_only_dataflow_steps,
                stop_step=None,
                generate_outputs=stitched,
            )
        )
        # a custom step counts as tool-free
        assert not _build_runs_tools(
            cfg(
                "/tmp/x",
                steps=[lambda model, cfg: model],
                stop_step=None,
                generate_outputs=estimates,
            )
        )
        # steps that cannot be resolved are not guessed at
        with pytest.raises(ValueError, match="Unknown step or phase: step_nonexistent"):
            _build_runs_tools(cfg("/tmp/x", steps=["step_nonexistent"]))

    @pytest.mark.parametrize(
        "kw, message",
        [
            (dict(steps=["step_nonexistent"]), "Unknown step or phase: step_nonexistent"),
            (dict(stop_step="step_nonexistent"), "step_nonexistent"),
        ],
    )
    def test_unresolvable_steps_are_an_error(self, kw, message):
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ"):
            os.environ.pop("XILINX_VIVADO", None)
            with pytest.raises(AssertionError, match="Configuration check failed"):
                build_dataflow_cfg(model_path, cfg(output_dir, board="AUP-ZU3_8GB", **kw))

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        errors = {
            c["name"]: c["message"]
            for c in report["checks"]
            if not c["passed"] and c["severity"] == "ERROR"
        }
        assert list(errors) == ["steps-unresolved"]
        assert message in errors["steps-unresolved"]

    def test_no_outputs_resolves_the_target(self):
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            build_dataflow_cfg(model_path, cfg(output_dir, board="ZCU104", generate_outputs=[]))

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        assert report["summary"]["errors"] == 0

    @pytest.mark.parametrize(
        "kw",
        [
            # a resumed run of the stitched-IP step that only verifies an existing
            # stitched IP: the verification runs Vivado's simulator
            dict(
                steps=["step_create_stitched_ip"],
                verify_steps=[VerificationStepType.STITCHED_IP_RTLSIM],
            ),
            # C++ simulation of HLS layers compiles against Vitis HLS
            dict(
                steps=["step_minimize_bit_width"],
                verify_steps=[VerificationStepType.FOLDED_HLS_CPPSIM],
            ),
            dict(
                steps=estimate_only_dataflow_steps,
                verify_steps=[VerificationStepType.FOLDED_HLS_CPPSIM],
            ),
        ],
    )
    def test_verification_that_runs_tools_needs_vivado(self, kw):
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        npy = {}
        for name in ("verify_input_npy", "verify_expected_output_npy"):
            npy[name] = os.path.join(build_dir, name + ".npy")
            np.save(npy[name], np.zeros((1, 4), dtype=np.float32))

        with patch.dict("os.environ"):
            os.environ.pop("XILINX_VIVADO", None)
            with pytest.raises(AssertionError, match="Configuration check failed"):
                build_dataflow_cfg(
                    model_path,
                    cfg(output_dir, board="AUP-ZU3_8GB", stop_step=None, **npy, **kw),
                )

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert error_names == ["vivado-unknown"]

    @pytest.mark.parametrize(
        "vstep",
        [
            VerificationStepType.FOLDED_HLS_CPPSIM,
            VerificationStepType.NODE_BY_NODE_RTLSIM,
            VerificationStepType.STITCHED_IP_RTLSIM,
        ],
    )
    def test_a_verification_that_runs_tools_marks_its_step_and_phase(self, vstep):
        """The step (and phase) that runs a tool-running verification counts as running
        tools whenever that verification is requested, whatever the outputs."""
        phase, step = verify_step_prereqs[vstep]
        config = cfg("/tmp/x", stop_step=None, verify_steps=[vstep])
        assert tool_running_steps[step](config)
        assert tool_running_phases[phase](config)

    @pytest.mark.parametrize(
        "kw, code",
        [
            # a driver for a board without a shell flow, from a resumed run of the
            # driver step alone
            (
                dict(
                    board="VEK280",
                    steps=["step_make_driver"],
                    generate_outputs=[DataflowOutputType.PYNQ_DRIVER],
                    target_fps=100,
                ),
                "no-shell",
            ),
            # estimate reports need a part, from the specialization onwards
            (dict(steps=estimate_only_dataflow_steps), "target-unstated"),
        ],
    )
    def test_the_target_the_steps_need_is_checked_before_they_run(self, kw, code):
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            with pytest.raises(AssertionError, match="Configuration check failed"):
                build_dataflow_cfg(model_path, cfg(output_dir, stop_step=None, **kw))

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert error_names == [code]

    @pytest.mark.parametrize(
        "kw",
        [
            # model preparation reads no part
            dict(stop_step="phase_optimize_model"),
            dict(steps=["step_qonnx_to_finn", "step_tidy_up", "step_streamline"]),
            # a custom step counts as reading none
            dict(steps=[lambda model, cfg: model]),
        ],
    )
    def test_a_build_that_reads_no_part_needs_no_target(self, kw):
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            build_dataflow_cfg(model_path, cfg(output_dir, **{"stop_step": None, **kw}))

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)
        assert report["summary"]["errors"] == 0

    def test_part_and_driver_are_decided_by_the_steps_the_build_runs(self):
        estimates = [DataflowOutputType.ESTIMATE_REPORTS]
        assert build_reads_part(cfg("/tmp/x", stop_step=None))
        assert build_reads_part(cfg("/tmp/x", steps=["step_specialize_layers"], stop_step=None))
        assert not build_reads_part(cfg("/tmp/x", stop_step="phase_optimize_model"))
        # estimate reports read the part; a step that makes no estimates does not
        reports = cfg("/tmp/x", steps=["step_generate_estimate_reports"], stop_step=None)
        assert build_reads_part(reports)
        reports.generate_outputs = []
        assert not build_reads_part(reports)
        # a driver needs a shell flow only from a step that makes one
        driver = [DataflowOutputType.PYNQ_DRIVER]
        assert build_makes_driver(cfg("/tmp/x", stop_step=None, generate_outputs=driver))
        assert not build_makes_driver(cfg("/tmp/x", stop_step=None, generate_outputs=estimates))
        assert not build_makes_driver(
            cfg("/tmp/x", stop_step="phase_build_hardware", generate_outputs=driver)
        )
        assert build_makes_driver(
            cfg(
                "/tmp/x",
                steps=["step_make_driver"],
                stop_step=None,
                generate_outputs=[DataflowOutputType.CPP_DRIVER],
            )
        )
