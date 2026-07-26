# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Assemble the e2e benchmark comparison table from the per-(flow, model) JSONs
the test_e2e_* harnesses drop into the results directory.

For every model it prints one block comparing all recorded flows against the
baseline flow: rtlsim throughput, estimate-level resources (LUT/BRAM/URAM/DSP),
total FIFO storage, the FIFO-sizing / folding step runtimes and the total build
wall time. Delta columns are relative to baseline where a baseline exists.

Usage:
    python tests/benchmark/e2e_report.py                 # markdown to stdout
    python tests/benchmark/e2e_report.py --csv out.csv   # additionally as CSV
    FINN_E2E_RESULTS=/path python tests/benchmark/e2e_report.py
"""

import argparse
import glob
import json
import os

FLOW_ORDER = ["baseline", "fifo", "folding", "dwc", "combined", "aligner"]

# time_per_step.json keys that represent the transformations under test; the
# first present key wins (phase-based builds report phase names)
SIZING_KEYS = ["step_set_fifo_depths", "phase_optimize_hardware"]
FOLDING_KEYS = ["step_target_fps_parallelization", "step_apply_folding_config"]


def _results_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    return os.environ.get("FINN_E2E_RESULTS", os.path.join(repo, "e2e_results"))


def load_results(results_dir):
    by_model = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*__*.json"))):
        with open(path) as f:
            r = json.load(f)
        by_model.setdefault(r["model"], {})[r["flow"]] = r
    return by_model


def _tp(r):
    rtl = (r or {}).get("rtlsim") or {}
    return rtl.get("stable_throughput[images/s]") or rtl.get("throughput[images/s]")


def _res(r, key):
    res = (r or {}).get("resources_estimate") or {}
    v = res.get(key)
    return float(v) if v is not None else None


def _step_time(r, keys):
    times = (r or {}).get("step_times_s") or {}
    for k in keys:
        if k in times:
            return times[k]
    return None


def _fmt(v, unit=""):
    if v is None:
        return "-"
    if isinstance(v, float):
        v = f"{v:,.1f}" if abs(v) < 1e5 else f"{v:,.0f}"
    return f"{v}{unit}"


def _delta(v, base):
    if v is None or not base:
        return ""
    return f" ({(v - base) / base:+.1%})"


COLUMNS = [
    ("throughput img/s", _tp, True),
    ("LUT est", lambda r: _res(r, "LUT"), True),
    ("BRAM_18K est", lambda r: _res(r, "BRAM_18K"), True),
    ("URAM est", lambda r: _res(r, "URAM"), True),
    ("DSP est", lambda r: _res(r, "DSP"), True),
    ("FIFO KiB", lambda r: r.get("fifo_kb"), True),
    ("sizing step s", lambda r: _step_time(r, SIZING_KEYS), True),
    ("folding step s", lambda r: _step_time(r, FOLDING_KEYS), False),
    ("build wall s", lambda r: r.get("build_wall_s"), False),
]


def model_table(model, flows):
    base = flows.get("baseline")
    lines = [f"### {model}", ""]
    header = "| flow | " + " | ".join(name for name, _, _ in COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    lines += [header, sep]
    for flow in FLOW_ORDER:
        if flow not in flows:
            continue
        r = flows[flow]
        cells = []
        for _, get, with_delta in COLUMNS:
            v = get(r)
            cell = _fmt(v)
            if with_delta and flow != "baseline" and base is not None:
                cell += _delta(v, get(base))
            cells.append(cell)
        lines.append(f"| {flow} | " + " | ".join(cells) + " |")
    if base is None:
        lines.append("\n_(no baseline recorded for this model yet)_")
    lines.append("")
    return "\n".join(lines)


def to_csv(by_model, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "flow", "tree"] + [name for name, _, _ in COLUMNS])
        for model in sorted(by_model):
            for flow in FLOW_ORDER:
                r = by_model[model].get(flow)
                if r is None:
                    continue
                w.writerow([model, flow, r.get("tree", "")] + [get(r) for _, get, _ in COLUMNS])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=_results_dir())
    ap.add_argument("--csv", help="also write the table as CSV to this path")
    args = ap.parse_args()

    by_model = load_results(args.results)
    if not by_model:
        print(f"no results in {args.results}; run the test_e2e_* harnesses first")
        return

    print(f"# FINN e2e benchmark comparison\n\nresults: {args.results}\n")
    for model in sorted(by_model):
        print(model_table(model, by_model[model]))
    if args.csv:
        to_csv(by_model, args.csv)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
