# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration check system for FINN builds - catches incompatibilities early."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
)
from finn.util.basic import get_vivado_version, part_map, pynq_part_map, vitis_part_map


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Check:
    name: str
    severity: Severity
    passed: bool
    message: str
    suggestion: Optional[str] = None


@dataclass
class Report:
    timestamp: str
    vivado_version: Optional[Tuple[int, int]]
    checks: List[Check] = field(default_factory=list)

    def has_errors(self) -> bool:
        return any(c.severity == Severity.ERROR and not c.passed for c in self.checks)


def _check(name, severity, condition, msg_fail, suggestion=None):
    """Helper to create a Check result."""
    if condition:
        return Check(name, severity, True, "OK")
    return Check(name, severity, False, msg_fail, suggestion)


def run_all_config_checks(cfg: DataflowBuildConfig) -> Report:
    """Run all configuration checks and return report."""
    v = get_vivado_version()
    checks = []

    has_bitfile = cfg.generate_outputs and DataflowOutputType.BITFILE in cfg.generate_outputs
    alveo_boards = set(vitis_part_map.keys())
    pynq_boards = set(pynq_part_map.keys())

    # === Vivado Version Checks ===
    if cfg.board == "V80":
        checks.append(
            _check(
                "v80_vivado",
                Severity.ERROR,
                v and v >= (2024, 2),
                "V80 board requires Vivado 2024.2 or later, "
                f"found {v[0]}.{v[1] if v else 'unknown'}",
                "Upgrade to Vivado 2024.2 or later to use V80",
            )
        )

    if cfg.shell_flow_type == ShellFlowType.SLASH_ALVEO:
        checks.append(
            _check(
                "slash_vivado",
                Severity.ERROR,
                v and v >= (2025, 1),
                "SLASH_ALVEO shell flow requires Vivado 2025.1, "
                f"found {v[0]}.{v[1] if v else 'unknown'}",
                "Upgrade to Vivado 2025.1 for SLASH support",
            )
        )

    if cfg.mlo:
        checks.append(
            _check(
                "mlo_vivado",
                Severity.ERROR,
                v and v >= (2024, 2),
                "MLO requires Vivado 2024.2 or later, " f"found {v[0]}.{v[1] if v else 'unknown'}",
                "Upgrade to Vivado 2024.2 or later for MLO support",
            )
        )

    if cfg.board == "AUP-ZU3_8GB":
        checks.append(
            _check(
                "aupzu3_vivado",
                Severity.ERROR,
                v and v >= (2024, 1),
                f"AUP-ZU3_8GB board requires Vivado 2024.1 or later, "
                f"found {v[0]}.{v[1] if v else 'unknown'}",
                "Upgrade to Vivado 2024.1 or later to use AUP-ZU3_8GB",
            )
        )

    if v and v not in [(2022, 2), (2024, 2)]:
        checks.append(
            _check(
                "vivado_stable",
                Severity.INFO,
                False,
                f"Vivado {v[0]}.{v[1]} is used. Recommended versions are "
                "2022.2 and 2024.2, other versions can be used but may have unexpected issues.",
            )
        )

    # === Board + Shell Flow Compatibility ===
    if cfg.board and cfg.board not in part_map:
        checks.append(
            _check(
                "board_exists",
                Severity.ERROR,
                False,
                f"Board '{cfg.board}' not found in supported boards",
                f"Valid boards: {', '.join(sorted(part_map.keys()))}",
            )
        )

    if has_bitfile:
        is_v80 = cfg.board == "V80" or (cfg.fpga_part and cfg.fpga_part.startswith("xcv80"))
        is_slash = cfg.shell_flow_type == ShellFlowType.SLASH_ALVEO
        if is_v80 or is_slash:  # Only check if V80 or SLASH is involved
            checks.append(
                _check(
                    "v80_slash",
                    Severity.ERROR,
                    is_v80 == is_slash,  # Pass if both match (both true or both false)
                    "V80 board and SLASH_ALVEO shell flow must be used together. "
                    f"Got board={cfg.board}, shell_flow_type={cfg.shell_flow_type}",
                    "Use board='V80' with shell_flow_type=SLASH_ALVEO, or use a different "
                    "board with VITIS_ALVEO/VIVADO_ZYNQ",
                )
            )

    if (
        has_bitfile
        and cfg.board in alveo_boards
        and cfg.shell_flow_type != ShellFlowType.VITIS_ALVEO
    ):
        checks.append(
            _check(
                "alveo_shell",
                Severity.ERROR,
                False,
                f"Alveo board '{cfg.board}' requires VITIS_ALVEO shell flow, "
                f"but {cfg.shell_flow_type} was specified",
                "Set shell_flow_type=ShellFlowType.VITIS_ALVEO for Alveo U* boards",
            )
        )

    if (
        has_bitfile
        and cfg.board in pynq_boards
        and cfg.shell_flow_type != ShellFlowType.VIVADO_ZYNQ
    ):
        checks.append(
            _check(
                "pynq_shell",
                Severity.ERROR,
                False,
                f"Zynq board '{cfg.board}' requires VIVADO_ZYNQ shell flow, "
                f"but {cfg.shell_flow_type} was specified",
                "Set shell_flow_type=ShellFlowType.VIVADO_ZYNQ for Zynq/PYNQ boards",
            )
        )

    if cfg.board in ("VEK280", "VCK190") and has_bitfile:
        checks.append(
            _check(
                "versal_deploy",
                Severity.ERROR,
                False,
                f"Versal board '{cfg.board}' does not yet support bitfile generation. "
                "System deployment is not available for Versal devices",
                "Remove BITFILE from generate_outputs, or use a non-Versal board",
            )
        )

    # === Required Field Dependencies ===
    if cfg.mlo and (cfg.loop_body_hierarchy is None or cfg.loop_body_range is None):
        missing = [f for f in ["loop_body_hierarchy", "loop_body_range"] if getattr(cfg, f) is None]
        checks.append(
            _check(
                "mlo_fields",
                Severity.ERROR,
                False,
                f"MLO requires {', '.join(missing)} to identify the repeating layer sequence",
                "Provide loop body metadata from PyTorch model hierarchy",
            )
        )

    if has_bitfile and cfg.shell_flow_type == ShellFlowType.VIVADO_ZYNQ and cfg.board is None:
        checks.append(
            _check(
                "zynq_board",
                Severity.ERROR,
                False,
                "BITFILE generation with VIVADO_ZYNQ requires 'board' to be set. "
                "ZynqBuild needs the board name and will fail in step_synthesize_bitfile "
                "if missing",
                "Set board to a valid Zynq board name (e.g., 'Pynq-Z1', 'ZCU104')",
            )
        )

    if has_bitfile and cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
        if cfg.board is None and cfg.vitis_platform is None:
            checks.append(
                _check(
                    "vitis_platform",
                    Severity.ERROR,
                    False,
                    "BITFILE generation with VITIS_ALVEO requires 'board' or 'vitis_platform' "
                    "to be set. VitisLink needs platform resolution and will fail in "
                    "step_synthesize_bitfile if missing",
                    "Set board (e.g., 'U250') or vitis_platform explicitly",
                )
            )

    # === Environment Variables ===
    if has_bitfile and cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
        missing = [
            v for v in ["VITIS_PATH", "PLATFORM_REPO_PATHS", "XILINX_XRT"] if v not in os.environ
        ]
        if missing:
            checks.append(
                _check(
                    "vitis_envvars",
                    Severity.ERROR,
                    False,
                    f"VITIS_ALVEO flow requires environment variables: {', '.join(missing)}",
                    "VITIS_PATH and XILINX_XRT are set by the Docker container (ensure "
                    "SKIP_XRT is not set). PLATFORM_REPO_PATHS must be set by the user",
                )
            )

    # === File Existence ===
    if cfg.folding_config_file and not os.path.isfile(cfg.folding_config_file):
        checks.append(
            _check(
                "folding_config",
                Severity.ERROR,
                False,
                f"folding_config_file not found: {cfg.folding_config_file}",
                "Check the path or remove folding_config_file and set target_fps "
                "to use automatic folding",
            )
        )

    if cfg.specialize_layers_config_file and not os.path.isfile(cfg.specialize_layers_config_file):
        checks.append(
            _check(
                "specialize_config",
                Severity.ERROR,
                False,
                f"specialize_layers_config_file not found: {cfg.specialize_layers_config_file}",
                "Check the path or remove specialize_layers_config_file "
                "to use default layer specialization",
            )
        )

    if cfg.verify_steps:
        for f, name in [
            (cfg.verify_input_npy, "verify_input_npy"),
            (cfg.verify_expected_output_npy, "verify_expected_output_npy"),
        ]:
            if f and not os.path.isfile(f):
                checks.append(
                    _check(
                        "verify_files",
                        Severity.ERROR,
                        False,
                        f"{name} not found: {f}",
                        "Provide valid verification input/output .npy files "
                        "or disable verification",
                    )
                )

    # === Warnings ===
    if cfg.shell_flow_type is not None and not has_bitfile:
        checks.append(
            _check(
                "shell_no_bitfile",
                Severity.WARNING,
                False,
                f"shell_flow_type={cfg.shell_flow_type.name} is set but BITFILE is not in "
                "generate_outputs. Shell flow type is only used for bitfile generation",
                "Add BITFILE to generate_outputs or remove shell_flow_type",
            )
        )

    if not cfg.standalone_thresholds:
        checks.append(
            _check(
                "standalone_thresholds",
                Severity.WARNING,
                False,
                "standalone_thresholds=False: MatMul+MultiThreshold patterns will be fused "
                "into MVAU with output activation, which cannot use RTL MVAU. In those cases "
                "HLS MVAU will be used instead",
                "Set standalone_thresholds=True if you want to make sure to always get the "
                "RTL MVAU implementation",
            )
        )

    if cfg.target_fps and cfg.folding_config_file:
        checks.append(
            _check(
                "fps_folding",
                Severity.WARNING,
                False,
                "Both target_fps and folding_config_file are set. target_fps drives automatic "
                "folding, but folding_config_file overrides it. Setting both is redundant",
                "Use target_fps for automatic folding OR folding_config_file for manual "
                "folding, not both",
            )
        )

    if cfg.mlo and cfg.generate_outputs:
        if DataflowOutputType.ESTIMATE_REPORTS in cfg.generate_outputs:
            checks.append(
                _check(
                    "mlo_estimates",
                    Severity.WARNING,
                    False,
                    "MLO enabled with ESTIMATE_REPORTS: Network performance cannot be "
                    "estimated for MLO models. Individual layer estimates will still be "
                    "generated",
                )
            )
        if DataflowOutputType.RTLSIM_PERFORMANCE in cfg.generate_outputs:
            checks.append(
                _check(
                    "mlo_rtlsim",
                    Severity.WARNING,
                    False,
                    "MLO enabled with RTLSIM_PERFORMANCE: RTL simulation performance "
                    "measurement is skipped for MLO models",
                    "Remove RTLSIM_PERFORMANCE from generate_outputs or disable MLO",
                )
            )

    if cfg.mlo and cfg.verify_save_full_context:
        checks.append(
            _check(
                "mlo_context",
                Severity.WARNING,
                False,
                "MLO with verify_save_full_context=True: After step_hw_ip_gen, MLO becomes "
                "a single IP block. Node-by-node rtlsim cannot access intermediate tensors "
                "inside the MLO",
            )
        )

    if cfg.minimize_bit_width is False:
        checks.append(
            _check(
                "minimize_bit_width",
                Severity.WARNING,
                False,
                "minimize_bit_width=False: Bit width minimization is recommended for all "
                "builds. Disabling it may result in suboptimal resource utilization",
                "Set minimize_bit_width=True for better resource usage",
            )
        )

    # === Info ===
    if cfg.verify_save_full_context:
        checks.append(
            _check(
                "full_context",
                Severity.INFO,
                False,
                "verify_save_full_context=True: Full execution context is not available for "
                "stitched IP rtlsim. The simulation sees the entire design as one IP block, "
                "so only top-level inputs and outputs are visible",
            )
        )

    if cfg.verify_save_rtlsim_waveforms or cfg.fifosim_save_waveform:
        checks.append(
            _check(
                "waveform_size",
                Severity.INFO,
                False,
                "Waveform saving enabled: .wdb files can become very large, especially for "
                "longer simulations. Ensure sufficient disk space is available",
            )
        )

    if cfg.generate_outputs and DataflowOutputType.CPP_DRIVER in cfg.generate_outputs:
        checks.append(
            _check(
                "cpp_driver",
                Severity.INFO,
                False,
                "C++ driver requested: The C++ driver is a community-maintained project",
            )
        )

    if cfg.shell_flow_type == ShellFlowType.SLASH_ALVEO:
        checks.append(
            _check(
                "slash_external",
                Severity.INFO,
                False,
                "SLASH_ALVEO shell flow is maintained externally. For issues specific to "
                "SLASH, please consult the SLASH documentation",
            )
        )

    return Report(timestamp=datetime.now().isoformat(), vivado_version=v, checks=checks)


def format_report(report: Report) -> str:
    """Format report for human-readable output."""
    lines = ["=" * 70, "FINN Build Configuration Check Report", "=" * 70]
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(
        f"Vivado: {report.vivado_version[0]}.{report.vivado_version[1]}"
        if report.vivado_version
        else "Vivado: Not detected"
    )
    lines.append("-" * 70)

    errors = [c for c in report.checks if c.severity == Severity.ERROR and not c.passed]
    warnings = [c for c in report.checks if c.severity == Severity.WARNING and not c.passed]
    infos = [c for c in report.checks if c.severity == Severity.INFO and not c.passed]

    for label, symbol, items in [
        ("ERRORS", "X", errors),
        ("WARNINGS", "!", warnings),
        ("INFO", "i", infos),
    ]:
        if items:
            lines.append(f"\n{label}:")
            for c in items:
                lines.append(f"  [{symbol}] {c.name}: {c.message}")
                if c.suggestion:
                    lines.append(f"      -> {c.suggestion}")

    lines.append(f"\nSUMMARY: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info")
    lines.append("=" * 70)
    return "\n".join(lines)


def save_report(report: Report, output_dir: str) -> str:
    """Save report to output_dir as .txt and .json. Returns path to text report."""
    txt_path = os.path.join(output_dir, "config_check_report.txt")
    json_path = os.path.join(output_dir, "config_check_report.json")

    with open(txt_path, "w") as f:
        f.write(format_report(report))

    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": report.timestamp,
                "vivado_version": list(report.vivado_version) if report.vivado_version else None,
                "checks": [
                    {
                        "name": c.name,
                        "severity": c.severity.value,
                        "passed": c.passed,
                        "message": c.message,
                        "suggestion": c.suggestion,
                    }
                    for c in report.checks
                ],
                "summary": {
                    "errors": sum(
                        1 for c in report.checks if c.severity == Severity.ERROR and not c.passed
                    ),
                    "warnings": sum(
                        1 for c in report.checks if c.severity == Severity.WARNING and not c.passed
                    ),
                    "info": sum(
                        1 for c in report.checks if c.severity == Severity.INFO and not c.passed
                    ),
                },
            },
            f,
            indent=2,
        )

    return txt_path
