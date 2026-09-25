# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for estimate-only flow that run WITHOUT Vivado.

These tests verify that all fpgadataflow HLS/RTL ops support the estimation
infrastructure and that the estimate-only build flow works correctly.
Designed to run in GitHub Actions quicktest (no synthesis tools required).
"""

import pytest

import os
from qonnx.custom_op.registry import get_ops_in_domain

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import make_build_dir


def _discover_all_hwcustomops():
    """Discover all HWCustomOp classes from HLS and RTL domains.

    Returns list of (op_name, op_class, domain) tuples for parametrization.
    Only scans specialized (HLS/RTL) ops since base ops are converted
    to these before estimation runs.
    """
    domains = [
        "finn.custom_op.fpgadataflow.hls",
        "finn.custom_op.fpgadataflow.rtl",
    ]
    all_ops = []
    for domain in domains:
        try:
            ops = get_ops_in_domain(domain)
            for op_name, op_class in ops:
                if issubclass(op_class, HWCustomOp):
                    all_ops.append((op_name, op_class, domain))
        except (ModuleNotFoundError, KeyError):
            pass
    return all_ops


# Get ops once at module load time for parametrization
_ALL_OPS = _discover_all_hwcustomops()


@pytest.mark.util
def test_all_fpgadataflow_ops_discovered():
    """Verify that op discovery finds a reasonable number of ops."""
    all_ops = _discover_all_hwcustomops()
    # Expect at least 20 ops across HLS and RTL domains
    assert (
        len(all_ops) >= 20
    ), f"Expected >=20 ops, found {len(all_ops)}: {[op[0] for op in all_ops]}"


@pytest.mark.util
@pytest.mark.parametrize("op_name,op_class,domain", _ALL_OPS)
def test_op_has_estimation_methods(op_name, op_class, domain):
    """Verify each HWCustomOp has required estimation methods."""
    # Check node_res_estimation exists and is callable
    assert hasattr(op_class, "node_res_estimation"), f"{op_name} missing node_res_estimation"
    assert callable(
        getattr(op_class, "node_res_estimation")
    ), f"{op_name}.node_res_estimation not callable"

    # Check get_exp_cycles exists and is callable
    assert hasattr(op_class, "get_exp_cycles"), f"{op_name} missing get_exp_cycles"
    assert callable(getattr(op_class, "get_exp_cycles")), f"{op_name}.get_exp_cycles not callable"

    # Check it's a proper subclass
    assert issubclass(op_class, HWCustomOp), f"{op_name} should be HWCustomOp subclass"


@pytest.mark.util
def test_analysis_functions_importable():
    """Verify analysis functions are callable (imports succeed without Vivado)."""
    assert callable(res_estimation)
    assert callable(res_estimation_complete)
    assert callable(exp_cycles_per_layer)
    assert callable(dataflow_performance)


@pytest.mark.util
def test_estimate_only_steps_defined():
    """Verify estimate_only_dataflow_steps is properly defined."""
    # Should contain the 4 phases for estimate-only flow
    expected_phases = [
        "phase_prepare_model",
        "phase_optimize_model",
        "phase_convert_to_hardware",
        "phase_optimize_hardware",
    ]
    assert build_cfg.estimate_only_dataflow_steps == expected_phases


@pytest.mark.util
def test_estimate_only_flow_produces_reports():
    """Test estimate-only flow runs and produces estimate reports without Vivado."""
    output_dir = make_build_dir("test_estimate_flow_")
    model_file = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/build_dataflow/model.onnx"

    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        steps=build_cfg.estimate_only_dataflow_steps,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    )
    ret = build.build_dataflow_cfg(model_file, cfg)
    assert ret == 0, "estimate-only build failed"

    # Verify estimate reports were generated
    assert os.path.isfile(
        output_dir + "/report/estimate_network_performance.json"
    ), "estimate_network_performance.json not generated"
