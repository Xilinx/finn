# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tabulate the feature-flow results (FINN_BUILD_DIR/feature_flow_results/*.jsonl).

    python tests/benchmark/feature_flow_report.py [--all]

Shows the latest run per (flow, model, board) unless --all is given.
"""

import glob
import json
import os
import sys

COLUMNS = [
    ("flow", "flow", "%s"),
    ("model", "model", "%s"),
    ("board", "board", "%s"),
    ("ok", None, "%s"),
    ("fold s", "step_target_fps_parallelization", "%.1f"),
    ("fifo s", "step_set_fifo_depths", "%.1f"),
    ("total s", "total_time_s", "%.0f"),
    ("est cyc", "est_max_cycles", "%s"),
    ("target", "target_cycles", "%s"),
    ("est fps", "est_throughput_fps", "%.0f"),
    ("sim fps", "rtlsim_throughput_fps", "%.0f"),
    ("fifo kbit", "fifo_total_kbit", "%.1f"),
    ("dwc pad/all", None, "%s"),
    ("ip hits", "hls_ip_cache_hits", "%s"),
]


def cell(row, title, key, fmt):
    if title == "ok":
        verified = all(row["verification"].get(s) for s in row["verify_steps"])
        return "yes" if row["build_ok"] and verified else "NO"
    if title == "dwc pad/all":
        return "%s/%s" % (row.get("num_pad_crop_dwcs", "-"), row.get("num_dwcs", "-"))
    value = row.get(key, row.get("step_time_s", {}).get(key))
    return "-" if value is None else fmt % value


def main():
    results_dir = os.path.join(os.environ["FINN_BUILD_DIR"], "feature_flow_results")
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    # other result files (e.g. padding sweeps) have their own layout
    rows = [row for row in rows if "flow" in row]
    if "--all" not in sys.argv:
        latest = {}
        for row in rows:
            latest[(row["flow"], row["model"], row["board"])] = row
        rows = list(latest.values())
    rows.sort(key=lambda r: (r["model"], r["board"], r["flow"]))
    table = [[t for t, _, _ in COLUMNS]] + [[cell(r, *c) for c in COLUMNS] for r in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(COLUMNS))]
    for line in table:
        print("  ".join(v.ljust(w) for v, w in zip(line, widths)))


if __name__ == "__main__":
    main()
