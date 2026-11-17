# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import logging
import os
import subprocess
import sys
import warnings
from onnx import TensorProto, helper

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.util.basic import launch_process_helper, make_build_dir
from finn.util.build_log import log_run


def make_test_model(build_dir, op_type="Relu", domain=""):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(op_type, ["inp"], ["out"], domain=domain)
    graph = helper.make_graph([node], "test", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model_path = os.path.join(build_dir, "model.onnx")
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    return model_path


def make_cfg(output_dir, steps, **kw):
    return DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        steps=steps,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        **kw,
    )


def passing_step(model, cfg):
    return model


def raising(error):
    def failing_step(model, cfg):
        raise error

    return failing_step


@pytest.mark.util
@pytest.mark.parametrize(
    "op_type, domain, exit_code, last_line",
    [
        ("Relu", "", 0, "Completed successfully"),
        # no step supports this op, so the first step fails
        ("NoSuchOp", "qonnx.custom_op.general", 1, "Build failed"),
    ],
)
def test_the_command_exits_nonzero_only_if_the_build_failed(op_type, domain, exit_code, last_line):
    build_dir = make_build_dir("test_runner_command_")
    make_test_model(build_dir, op_type, domain)
    cfg = make_cfg(
        os.path.join(build_dir, "output"), ["step_tidy_up"], enable_build_pdb_debug=False
    )
    with open(os.path.join(build_dir, "dataflow_build_config.json"), "w") as f:
        f.write(cfg.to_json())
    result = subprocess.run(
        [sys.executable, "-m", "finn.builder.build_dataflow", build_dir],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stderr
    assert result.stdout.splitlines()[-1] == last_line


@pytest.mark.util
def test_a_failed_step_returns_minus_one_and_starts_the_debugger(monkeypatch):
    debugged = []
    monkeypatch.setattr(build.pdb, "post_mortem", debugged.append)
    build_dir = make_build_dir("test_runner_failed_step_")
    output_dir = os.path.join(build_dir, "output")
    cfg = make_cfg(output_dir, [raising(RuntimeError("step broke"))], enable_build_pdb_debug=True)
    assert build.build_dataflow_cfg(make_test_model(build_dir), cfg) == -1
    assert len(debugged) == 1
    assert not os.path.exists(os.path.join(output_dir, "time_per_step.json"))


@pytest.mark.util
@pytest.mark.parametrize(
    "error, code",
    [(KeyboardInterrupt(), None), (SystemExit(3), 3), (SystemExit(0), 0)],
    ids=["KeyboardInterrupt", "SystemExit(3)", "SystemExit(0)"],
)
def test_an_interrupt_or_exit_in_a_step_passes_through(monkeypatch, error, code):
    debugged = []
    monkeypatch.setattr(build.pdb, "post_mortem", debugged.append)
    stdout, stderr = sys.stdout, sys.stderr
    build_dir = make_build_dir("test_runner_interrupt_")
    output_dir = os.path.join(build_dir, "output")
    cfg = make_cfg(output_dir, [raising(error)], enable_build_pdb_debug=True, verbose=False)
    with pytest.raises((KeyboardInterrupt, SystemExit)) as exc:
        build.build_dataflow_cfg(make_test_model(build_dir), cfg)
    assert exc.type is type(error)
    assert getattr(exc.value, "code", None) == code
    assert debugged == []
    assert sys.stdout is stdout and sys.stderr is stderr
    assert not os.path.exists(os.path.join(output_dir, "time_per_step.json"))


@pytest.mark.util
def test_a_successful_build_writes_time_per_step():
    build_dir = make_build_dir("test_runner_success_")
    output_dir = os.path.join(build_dir, "output")
    cfg = make_cfg(output_dir, [passing_step])
    assert build.build_dataflow_cfg(make_test_model(build_dir), cfg) == 0
    assert os.path.isfile(os.path.join(output_dir, "time_per_step.json"))


@pytest.mark.util
def test_build_dataflow_directory_restores_the_working_directory_on_error():
    build_dir = make_build_dir("test_runner_cwd_")
    make_test_model(build_dir)
    cfg = make_cfg(os.path.join(build_dir, "output"), None, board="nope")
    with open(os.path.join(build_dir, "dataflow_build_config.json"), "w") as f:
        f.write(cfg.to_json())
    cwd = os.getcwd()
    try:
        with pytest.raises(AssertionError, match="Configuration check failed"):
            build.build_dataflow_directory(build_dir)
        assert os.getcwd() == cwd
    finally:
        os.chdir(cwd)


@pytest.mark.util
@pytest.mark.parametrize("option", ["start_step", "stop_step"])
def test_an_unknown_step_name_is_reported_with_the_valid_names(option):
    build_dir = make_build_dir("test_runner_unknown_step_")
    output_dir = os.path.join(build_dir, "output")
    cfg = make_cfg(output_dir, [passing_step], **{option: "step_foo"})
    with pytest.raises(ValueError) as exc:
        build.build_dataflow_cfg(make_test_model(build_dir), cfg)
    message = str(exc.value)
    assert option in message and "'step_foo'" in message
    assert "passing_step" in message


def talking_step(model, cfg):
    sys.stdout.write("(Pdb) from inside the step\n")
    logging.getLogger("finn.transformation.test").info("a record from the step")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("a warning from the step", UserWarning)
    launch_process_helper(["echo", "a line of tool output"])
    return model


@pytest.mark.util
def test_a_step_writes_to_the_real_stdout_and_its_records_reach_the_build_log(capfd):
    finn = logging.getLogger("finn")
    state = finn.level, list(finn.handlers), finn.propagate
    build_dir = make_build_dir("test_runner_step_output_")
    output_dir = os.path.join(build_dir, "output")
    # and a built-in step's message, which it used to print
    cfg = make_cfg(output_dir, [talking_step, "step_apply_folding_config"])
    assert build.build_dataflow_cfg(make_test_model(build_dir), cfg) == 0
    printed = capfd.readouterr()
    log_path = os.path.join(output_dir, "build_dataflow.log")
    with open(log_path) as f:
        log = f.read()
    assert "(Pdb) from inside the step" in printed.out
    assert "Running step: talking_step [1/2]" in printed.out
    assert "1 Python warnings, in " + log_path in printed.out
    assert "a record from the step" not in printed.out + printed.err
    assert "a line of tool output" not in printed.out + printed.err
    assert "INFO talking_step finn.transformation.test: a record from the step" in log
    assert "WARNING talking_step py.warnings: " in log
    assert "UserWarning: a warning from the step" in log
    assert "INFO talking_step finn.util.basic: a line of tool output" in log
    assert "(Pdb)" not in log
    assert "No folding config json provided" in log
    assert "No folding config json provided" not in printed.out + printed.err
    assert (finn.level, finn.handlers, finn.propagate) == state


TOOL = ["sh", "-c", "printf 'stdout tail'; printf 'stderr tail' >&2"]


@pytest.mark.util
def test_tool_output_is_logged_per_stream_in_a_run(tmp_path):
    path = tmp_path / "run.log"
    with log_run(str(path), runner="finn.builder.build_dataflow"):
        launch_process_helper(TOOL)
    log = path.read_text()
    assert "finn.util.basic: stdout tail\n" in log
    assert "finn.util.basic: stderr tail\n" in log


@pytest.mark.util
def test_tool_output_outside_a_run_is_written_through(capfd, monkeypatch):
    # a handler that takes no INFO record must not swallow the tool's output
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.INFO)
    handler = logging.NullHandler()
    root.addHandler(handler)
    try:
        launch_process_helper(TOOL)
    finally:
        root.removeHandler(handler)
    printed = capfd.readouterr()
    assert (printed.out, printed.err) == ("stdout tail", "stderr tail")
