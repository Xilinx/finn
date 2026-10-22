# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
import subprocess
import sys
from onnx import TensorProto, helper

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.util.basic import make_build_dir


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
        **kw
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
