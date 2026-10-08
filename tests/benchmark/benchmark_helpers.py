# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import csv
import datetime
import json
import os
import subprocess

import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import pynq_part_map, slash_part_map, vitis_part_map


def benchmark_root():
    """Absolute path to the ``tests/benchmark`` suite root.

    Anchored on ``FINN_ROOT`` so the suite resolves its models/configs/goldens
    independently of the process working directory (several tests historically
    used a bare ``"tests/benchmark/"`` relative path, which only worked when
    pytest happened to run from the repo root).
    """
    return os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark")


def verification_io_dir():
    """Directory holding the verification golden I/O (``.npy``) files.

    The goldens are not committed to git. ``VERIFICATION_IO`` overrides the
    location (e.g. an external mount provided by CI); otherwise it defaults to
    the in-repo ``tests/benchmark/verification_io/``.
    """
    override = os.environ.get("VERIFICATION_IO", "").strip()
    return override if override else os.path.join(benchmark_root(), "verification_io")


def get_verify_steps(steps, env_var="VERIFICATION_EN", board_enabled=True):
    """Return ``steps`` when verification is enabled, else None.

    Verification (cppsim/rtlsim numeric checks) is slow, so the benchmark suite
    keeps it OFF by default and only runs it when ``VERIFICATION_EN`` is set to a
    truthy value (``1``/``true``/``yes``/``on``). This reuses the env-var
    convention from ``run-docker.sh``. Passing the result straight to
    ``DataflowBuildConfig(verify_steps=...)`` disables verification cleanly when
    ``None``. Whether the verification actually passed is assessed later by the
    aggregation harness (Phase 4), not by the per-test output check.

    ``board_enabled`` additionally scopes verification to a single baseline
    board. Numeric correctness is board/part-independent across the suite's
    UltraScale+ targets, so re-verifying the same (model, datatype) on a second
    board is pure redundancy. Multi-board models pass
    ``board_enabled=(board == BASELINE_BOARD)`` so only the baseline build
    verifies; single-board models leave it at the default ``True``.
    """
    enabled = os.environ.get(env_var, "0").strip().lower() in ("1", "true", "yes", "on")
    return list(steps) if (enabled and board_enabled) else None


def bitfile_output_files(board):
    """Bitfile + synthesis-report artifacts produced by ``step_synthesize_bitfile``.

    The artifacts differ by shell flow (resolved from the ``board`` via FINN's
    part maps): Vivado/Zynq emits a ``.bit`` plus a ``.hwh`` and a post-route
    timing report, Vitis/Alveo emits a ``.xclbin``, and SLASH (V80) emits a
    ``.vbin`` plus a ``slash_report.xml``.
    """
    common = [
        "report/post_synth_resources.xml",
        "report/post_synth_resources.json",
    ]
    if board in pynq_part_map:
        return [
            "bitfile/finn-accel.bit",
            "bitfile/finn-accel.hwh",
            "report/post_route_timing.rpt",
        ] + common
    elif board in vitis_part_map:
        return [
            "bitfile/finn-accel.xclbin",
        ] + common
    elif board in slash_part_map:
        # The SLASH link writes the report straight into bitfile/ and only emits
        # the JSON resource report (no post_synth_resources.xml).
        return [
            "bitfile/finn-accel.vbin",
            "bitfile/slash_report.xml",
            "report/post_synth_resources.json",
        ]
    else:
        raise ValueError("Unknown board, can't determine bitfile outputs: %s" % board)


# --- shared build configuration -------------------------------------------

# Full-build output products; reduced-flow models pass their own list.
DEFAULT_BUILD_OUTPUTS = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.PYNQ_DRIVER,
    build_cfg.DataflowOutputType.BITFILE,
    build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
]

# Shell-independent build-product artifacts every full benchmark build emits.
# Tests append the model-specific extras plus bitfile_output_files(shell).
CORE_BUILD_OUTPUT_FILES = [
    "time_per_step.json",
    "final_hw_config.json",
    "template_specialize_layers_config.json",
    "stitched_ip/ip/component.xml",
    "driver/driver.py",
    "report/estimate_layer_cycles.json",
    "report/estimate_layer_resources.json",
    "report/estimate_network_performance.json",
    "report/rtlsim_performance.json",
]


def benchmark_config_paths(subdir, folding_name, specialize_name):
    """Resolve a model's folding/specialize JSON paths under its benchmark dir."""
    base = os.path.join(benchmark_root(), subdir)
    folding = os.path.join(base, "folding_config", folding_name + ".json")
    specialize = os.path.join(base, "specialize_layers_config", specialize_name + ".json")
    return folding, specialize


def make_benchmark_cfg(board, output_dir, **overrides):
    """Build a ``DataflowBuildConfig`` from the shared benchmark skeleton.

    Fills the fields every benchmark build shares (``enable_build_pdb_debug``,
    ``output_dir``, ``board``, ``generate_outputs=DEFAULT_BUILD_OUTPUTS``) and
    forwards ``**overrides`` straight to ``DataflowBuildConfig``, so any field can
    be set or a default overridden without this factory knowing the signature.
    The fpga part, vitis platform, and shell flow are all left for the builder to
    resolve from ``board``.

    A key passed explicitly as ``None`` is dropped so the ``DataflowBuildConfig``
    default stands.
    """
    cfg_kwargs = dict(
        enable_build_pdb_debug=False,
        output_dir=output_dir,
        board=board,
        generate_outputs=list(DEFAULT_BUILD_OUTPUTS),
    )
    cfg_kwargs.update(overrides)
    cfg_kwargs = {key: value for key, value in cfg_kwargs.items() if value is not None}
    return build_cfg.DataflowBuildConfig(**cfg_kwargs)


def find_cached_build(prefix, build_dir=None):
    """Return the first ``build_dir`` child whose name starts with ``prefix``.

    Generic caching hook for flows that reuse a previously built/synthesized model
    (e.g. re-running only a later step such as FIFO sizing) instead of rebuilding
    from scratch: scan ``FINN_BUILD_DIR`` (or ``build_dir``) for a matching build
    directory. Unused by the current benchmark suite. Returns the directory path,
    or ``None`` if no match exists.
    """
    if build_dir is None:
        build_dir = os.environ.get("FINN_BUILD_DIR", ".")
    if not os.path.isdir(build_dir):
        return None
    for entry in sorted(os.listdir(build_dir)):
        if entry.startswith(prefix):
            candidate = os.path.join(build_dir, entry)
            if os.path.isdir(candidate):
                return candidate
    return None


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


# --- benchmark aggregation -------------------------------------------------
#
# Each build leaves a set of report JSONs under ``<output_dir>/report/``. The
# aggregation harness mines the few metrics worth tracking over time (estimated
# vs. rtlsim throughput/latency and estimated/post-synth resource usage) and
# collapses every (model, board) build of a run into one timestamped JSON + a
# flat CSV, so later runs can be diffed. The functions are kept here (not in the
# conftest) so they can be imported and exercised standalone against existing
# build directories.

BENCH_RESULTS_SUBDIR = "benchmark_results"


def _load_json(path):
    """Load a JSON file, returning None if it is missing or unreadable."""
    try:
        with open(path) as json_file:
            return json.load(json_file)
    except (OSError, ValueError):
        return None


def _git_commit():
    """Return the FINN git commit hash, or None if it can't be determined."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.environ.get("FINN_ROOT"),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_build_metrics(model, board, output_dir):
    """Flatten a single build's report JSONs into one metrics row.

    Missing reports are skipped (not every run produces post-synth resources),
    so the returned row always carries the (model, board, output_dir) identity
    plus whatever metrics were present.
    """
    report_dir = os.path.join(output_dir, "report")
    row = {"model": model, "board": board, "output_dir": output_dir}

    est_perf = _load_json(os.path.join(report_dir, "estimate_network_performance.json"))
    if est_perf:
        row["est_max_cycles"] = est_perf.get("max_cycles")
        row["est_critical_path_cycles"] = est_perf.get("critical_path_cycles")
        row["est_throughput_fps"] = est_perf.get("estimated_throughput_fps")
        row["est_latency_ns"] = est_perf.get("estimated_latency_ns")

    est_res = _load_json(os.path.join(report_dir, "estimate_layer_resources.json"))
    if isinstance(est_res, dict) and isinstance(est_res.get("total"), dict):
        total = est_res["total"]
        for key in ("LUT", "BRAM_18K", "URAM", "DSP"):
            row["est_" + key] = total.get(key)

    rtlsim = _load_json(os.path.join(report_dir, "rtlsim_performance.json"))
    if rtlsim:
        row["rtlsim_throughput_fps"] = rtlsim.get("throughput[images/s]")
        row["rtlsim_stable_throughput_fps"] = rtlsim.get("stable_throughput[images/s]")
        row["rtlsim_fclk_mhz"] = rtlsim.get("fclk[mhz]")
        row["rtlsim_latency_cycles"] = rtlsim.get("latency_cycles")

    synth = _load_json(os.path.join(report_dir, "post_synth_resources.json"))
    if isinstance(synth, dict) and isinstance(synth.get("(top)"), dict):
        top = synth["(top)"]
        for key in ("LUT", "FF", "SRL", "BRAM_36K", "BRAM_18K", "URAM", "DSP"):
            if key in top:
                row["synth_" + key] = top[key]

    return row


def aggregate_benchmark_results(entries, out_root=None):
    """Aggregate per-build report metrics into one timestamped JSON + CSV.

    Args:
        entries: iterable of ``(model, board, output_dir)`` tuples.
        out_root: directory to write ``benchmark_results/`` into; defaults to
            ``FINN_BUILD_DIR`` (falling back to the current directory).

    Returns:
        ``(json_path, csv_path)``, or ``(None, None)`` when there is nothing to
        aggregate.
    """
    rows = [collect_build_metrics(model, board, output_dir) for model, board, output_dir in entries]
    if not rows:
        return None, None

    if out_root is None:
        out_root = os.environ.get("FINN_BUILD_DIR", ".")
    results_dir = os.path.join(out_root, BENCH_RESULTS_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp_utc": timestamp,
        "git_commit": _git_commit(),
        "num_builds": len(rows),
        "results": rows,
    }

    json_path = os.path.join(results_dir, "results_%s.json" % timestamp)
    with open(json_path, "w") as json_file:
        json.dump(payload, json_file, indent=2)

    # Flat CSV: identity columns first, then the union of every metric key seen
    # (sorted) so the header is stable regardless of which reports were present.
    lead = ["model", "board", "output_dir"]
    extra = sorted({key for row in rows for key in row} - set(lead))
    fieldnames = lead + extra
    csv_path = os.path.join(results_dir, "results_%s.csv" % timestamp)
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def _parse_entry(arg):
    """Parse one CLI argument into a ``(model, board, output_dir)`` entry.

    Accepts either ``model:board:/path/to/build_dir`` (explicit labels) or a bare
    ``/path/to/build_dir`` (labelled ``model=<dir basename>``, ``board=unknown``),
    so existing build dirs can be aggregated without per-build metadata.
    """
    parts = arg.split(":", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    output_dir = arg
    return os.path.basename(os.path.normpath(output_dir)), "unknown", output_dir


if __name__ == "__main__":
    # Standalone aggregation over existing build dirs, e.g.:
    #   python benchmark_helpers.py /path/build_a /path/build_b
    #   python benchmark_helpers.py tfc-w1a1:AUP-ZU3_8GB:/path/build_a
    import sys

    cli_entries = [_parse_entry(a) for a in sys.argv[1:]]
    if not cli_entries:
        print("usage: python benchmark_helpers.py [model:board:]BUILD_DIR ...")
        sys.exit(1)
    out_json, out_csv = aggregate_benchmark_results(cli_entries)
    print(out_json)
    print(out_csv)
