# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Put this directory on sys.path so the benchmark tests in per-model subfolders
# can import the shared benchmark_helpers module by bare name.
import pytest

import os
import sys

_BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARK_DIR)

# Builds registered via the ``bench_recorder`` fixture during the session; drained
# by ``pytest_sessionfinish`` into one timestamped aggregated artifact.
_BENCH_ENTRIES = []


@pytest.fixture
def bench_recorder():
    """Return a callback each benchmark test calls after a successful build.

    Registering ``(model, board, output_dir)`` lets ``pytest_sessionfinish``
    locate the per-build report JSONs and fold their metrics into a single
    aggregated benchmark artifact for the whole session.
    """

    def _record(model, board, output_dir):
        _BENCH_ENTRIES.append((model, board, output_dir))

    return _record


def pytest_sessionfinish(session, exitstatus):
    """Aggregate every registered build's reports into one timestamped artifact."""
    if not _BENCH_ENTRIES:
        return
    # Imported here (not at module top) because benchmark_helpers only becomes
    # importable once _BENCHMARK_DIR is inserted on sys.path above.
    from benchmark_helpers import aggregate_benchmark_results  # noqa: PLC0415

    json_path, csv_path = aggregate_benchmark_results(_BENCH_ENTRIES)
    if json_path:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("benchmark results aggregated to %s" % json_path)
            reporter.write_line("benchmark results aggregated to %s" % csv_path)
