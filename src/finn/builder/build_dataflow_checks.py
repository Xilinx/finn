# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration check system for FINN builds - catches incompatibilities early."""

import importlib
import json
import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    verify_step_prereqs,
)
from finn.platform import PlatformError
from finn.util.basic import get_vivado_version


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


def _cfg_default(field_name):
    """The declared default of a DataflowBuildConfig field, so checks that compare
    against "unset" track the dataclass rather than a hard-coded copy of it."""
    for f in fields(DataflowBuildConfig):
        if f.name == field_name:
            return f.default
    raise AttributeError("DataflowBuildConfig has no field %r" % field_name)


def _build_dataflow():
    # imported lazily via importlib, as below: build_dataflow imports this module
    return importlib.import_module("finn.builder.build_dataflow")


def _build_runs_tools(cfg: DataflowBuildConfig) -> bool:
    return _build_dataflow().build_runs_tools(cfg)


def run_all_config_checks(cfg: DataflowBuildConfig) -> Report:
    """Run all configuration checks and return report."""
    v = get_vivado_version()
    checks = []

    has_bitfile = DataflowOutputType.BITFILE in (cfg.generate_outputs or [])

    # === Build Steps ===
    # What the steps the build will run need: AMD tools, a part, a shell flow (for a
    # driver). Steps that cannot be resolved are an error of their own; what they need
    # is then unknown (None).
    try:
        needs_tools = _build_runs_tools(cfg)
        needs_part = _build_dataflow().build_reads_part(cfg)
        needs_shell = _build_dataflow().build_makes_driver(cfg)
    except (ValueError, AttributeError, ImportError) as error:
        needs_tools = needs_part = needs_shell = None
        checks.append(
            _check(
                "steps-unresolved",
                Severity.ERROR,
                False,
                f"The build steps cannot be resolved: {error}",
                "Check steps, start_step and stop_step against the build's steps and phases",
            )
        )

    # === Vivado Version Checks ===
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

    # === Target: board, part and shell flow ===
    # Resolved for what the build states, and for what its steps need. With
    # steps-unresolved, only what the build states is checked, without the Vivado
    # minimums.
    stated = cfg.board is not None or cfg.fpga_part is not None
    if stated or has_bitfile or needs_part or needs_shell:
        try:
            cfg._resolve_target(
                need_shell=bool(needs_shell),
                need_bitfile=has_bitfile,
                need_tools=bool(needs_tools),
            )
        except PlatformError as refusal:
            checks.append(_check(refusal.code, Severity.ERROR, False, str(refusal)))

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

    needs_folding = cfg.generate_outputs and any(
        o != DataflowOutputType.ESTIMATE_REPORTS for o in cfg.generate_outputs
    )
    if needs_folding and cfg.target_fps is None and cfg.folding_config_file is None:
        checks.append(
            _check(
                "folding_missing",
                Severity.ERROR,
                False,
                "Neither target_fps nor folding_config_file is set. Nodes stay at "
                "their creation-time PE=1/SIMD=1, the most folded and slowest "
                "implementation, and the build will silently run to completion "
                "with it instead of failing fast",
                "Set target_fps for automatic folding, or provide folding_config_file "
                "for manual PE/SIMD settings, so a slow build isn't mistaken for "
                "a deliberate choice",
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

        # each verify_steps entry is only ever triggered from inside one specific
        # step, itself part of one specific phase; if steps/start_step/stop_step
        # leaves that step out, the entry is silently never verified
        #
        # The (phase, step) prerequisite for each VerificationStepType is defined
        # next to the enum itself, in build_dataflow_config.verify_step_prereqs.
        try:
            # imported lazily via importlib (rather than a top-level import) since
            # build_dataflow imports this module, and a top-level import back
            # would be circular
            build_dataflow_mod = importlib.import_module("finn.builder.build_dataflow")
            resolved_names = {
                fn.__name__ for fn in build_dataflow_mod.resolve_build_steps(cfg, partial=True)
            }
        except (ValueError, AttributeError):
            # steps/start_step/stop_step can't be resolved; that will fail on its
            # own once the build actually starts, nothing more to check here
            resolved_names = None

        if resolved_names is not None:
            for vstep in cfg._resolve_verification_steps():
                phase_name, step_name = verify_step_prereqs.get(vstep, (None, None))
                phase_missing = phase_name and phase_name not in resolved_names
                step_missing = step_name not in resolved_names
                if phase_missing and step_missing:
                    # verify_steps entries may be VerificationStepType enum members
                    # or plain strings; normalize to the string form for the message
                    vstep_name = getattr(vstep, "value", vstep)
                    checks.append(
                        _check(
                            "verify_step_prereq",
                            Severity.WARNING,
                            False,
                            f"verify_steps includes {vstep_name}, but neither "
                            f"{phase_name} nor {step_name} is in the resolved "
                            "build steps (steps/start_step/stop_step). That "
                            "verification would silently never run",
                            f"Include {phase_name} (or {step_name}) in steps, "
                            "adjust start_step/stop_step so it runs, or remove "
                            "this entry from verify_steps",
                        )
                    )

    # === FIFO Debug Logging ===
    # fifo_log_flush_cycles reaches the RTL via int() in _attach_fifo_debug_logs and
    # InsertAndSetFIFODepths; None or a negative value only fails there, partway into
    # the build, so catch it here instead. 0 is legal: it disables periodic flushing.
    if cfg.debug_fifo and not isinstance(cfg.fifo_log_flush_cycles, int):
        checks.append(
            _check(
                "fifo_log_flush_cycles",
                Severity.ERROR,
                False,
                "debug_fifo=True but fifo_log_flush_cycles="
                f"{cfg.fifo_log_flush_cycles!r} is not an integer",
                "Set fifo_log_flush_cycles to a non-negative integer number of cycles, "
                "or 0 to only flush the logs when the simulation ends",
            )
        )
    elif cfg.debug_fifo and cfg.fifo_log_flush_cycles < 0:
        checks.append(
            _check(
                "fifo_log_flush_cycles",
                Severity.ERROR,
                False,
                f"fifo_log_flush_cycles={cfg.fifo_log_flush_cycles} is negative",
                "Set fifo_log_flush_cycles to a non-negative integer number of cycles, "
                "or 0 to only flush the logs when the simulation ends",
            )
        )

    if not cfg.debug_fifo:
        fifo_log_opts = [
            (name, getattr(cfg, name), _cfg_default(name))
            for name in ["fifo_log_verbose", "fifo_log_flush_cycles"]
        ]
        customized = [(n, val) for n, val, default in fifo_log_opts if val != default]
        if customized:
            checks.append(
                _check(
                    "fifo_log_without_debug",
                    Severity.WARNING,
                    False,
                    ", ".join(f"{n}={v!r}" for n, v in customized)
                    + " differs from the default but debug_fifo=False, so no FIFO logs "
                    "are written and the setting has no effect",
                    "Set debug_fifo=True to enable FIFO logging, or drop these settings",
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
