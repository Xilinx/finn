# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import finn.builder.build_dataflow_config as build_cfg


def get_verify_steps(steps, env_var="VERIFICATION_EN"):
    """Return ``steps`` when verification is enabled via the environment, else None.

    Verification (cppsim/rtlsim numeric checks) is slow, so the benchmark suite
    keeps it OFF by default and only runs it when ``VERIFICATION_EN`` is set to a
    truthy value (``1``/``true``/``yes``/``on``). This reuses the env-var
    convention from ``run-docker.sh``. Passing the result straight to
    ``DataflowBuildConfig(verify_steps=...)`` disables verification cleanly when
    ``None``. Whether the verification actually passed is assessed later by the
    aggregation harness (Phase 4), not by the per-test output check.
    """
    enabled = os.environ.get(env_var, "0").strip().lower() in ("1", "true", "yes", "on")
    return list(steps) if enabled else None


def bitfile_output_files(shell_flow_type):
    """Bitfile + synthesis-report artifacts produced by ``step_synthesize_bitfile``.

    The artifacts differ by shell flow: the Vivado/Zynq flow emits a ``.bit``
    plus a ``.hwh`` hand-off and a post-route timing report, whereas the
    Vitis/Alveo flow emits a ``.xclbin`` and no ``.hwh``/timing report.
    """
    common = [
        "report/post_synth_resources.xml",
        "report/post_synth_resources.json",
    ]
    if shell_flow_type == build_cfg.ShellFlowType.VIVADO_ZYNQ:
        return [
            "bitfile/finn-accel.bit",
            "bitfile/finn-accel.hwh",
            "report/post_route_timing.rpt",
        ] + common
    elif shell_flow_type == build_cfg.ShellFlowType.VITIS_ALVEO:
        return [
            "bitfile/finn-accel.xclbin",
        ] + common
    else:
        raise ValueError("Unsupported shell flow type: %s" % shell_flow_type)


def check_build_outputs(output_dir, expected_files, write_report=True):
    """Check that all expected build output *products* exist.

    Rather than asserting one file at a time (which aborts at the first missing
    artifact and hides the status of the rest), this checks every expected
    build-product artifact, records whether each is present, writes an
    ``output_products_check.json`` report into ``<output_dir>/report/``, and
    finally asserts that nothing is missing, listing every absent file at once.

    Verification results are intentionally *not* checked here -- verification is
    toggled via :func:`get_verify_steps` and its per-step pass/fail is assessed by
    the aggregation harness (Phase 4).

    Args:
        output_dir: build output directory returned by ``make_build_dir``.
        expected_files: build-product paths, relative to ``output_dir``.
        write_report: also persist the per-file status as JSON under ``report/``.
    """
    expected = list(expected_files)
    present = {rel: os.path.isfile(os.path.join(output_dir, rel)) for rel in expected}
    missing = sorted(rel for rel, ok in present.items() if not ok)

    if write_report:
        report_dir = os.path.join(output_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        report = {
            "output_dir": output_dir,
            "num_expected": len(expected),
            "num_present": len(expected) - len(missing),
            "num_missing": len(missing),
            "missing": missing,
            "files": present,
        }
        report_path = os.path.join(report_dir, "output_products_check.json")
        with open(report_path, "w") as report_file:
            json.dump(report, report_file, indent=2)

    assert not missing, "Missing %d/%d build outputs: %s" % (
        len(missing),
        len(expected),
        missing,
    )
