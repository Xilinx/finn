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
