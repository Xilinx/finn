# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import logging
import multiprocessing
import subprocess
import sys
import warnings

from finn.util.build_log import FILE_ONLY, log_run, running_step

RUNNER = "finn.util.test_build_log"


@pytest.mark.util
def test_a_run_logs_each_record_with_its_step_and_restores_the_logging(tmp_path, capsys):
    runner = logging.getLogger(RUNNER)
    step = logging.getLogger(f"{RUNNER}.step")
    other = logging.getLogger("finn.transformation.fpgadataflow")
    finn = logging.getLogger("finn")
    state = finn.level, list(finn.handlers), finn.propagate
    path = tmp_path / "logs" / "run.log"
    with log_run(str(path), runner=runner.name):
        runner.info("before")
        with running_step("probe"):
            runner.info("inside")
            step.info("a step's summary")
            other.info("a module's line")
            other.info("a tool's output", extra=FILE_ONLY)
            other.warning("a module's warning")
            runner.error("the traceback", extra=FILE_ONLY)
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("a library's warning", UserWarning)
        runner.info("after")
    printed = capsys.readouterr()
    log = path.read_text()
    assert f"INFO - {RUNNER}: before" in log
    assert f"INFO probe {RUNNER}: inside" in log
    assert f"INFO probe {RUNNER}.step: a step's summary" in log
    assert "INFO probe finn.transformation.fpgadataflow: a module's line" in log
    assert "INFO probe finn.transformation.fpgadataflow: a tool's output" in log
    assert "WARNING probe finn.transformation.fpgadataflow: a module's warning" in log
    assert f"ERROR probe {RUNNER}: the traceback" in log
    assert "WARNING probe py.warnings: " in log and "UserWarning: a library's warning" in log
    assert f"INFO - {RUNNER}: after" in log
    assert printed.out.splitlines() == [
        "before",
        "inside",
        "a step's summary",
        "after",
        f"1 Python warnings, in {path}",
    ]
    assert printed.err.splitlines() == ["WARNING: a module's warning"]
    assert (finn.level, finn.handlers, finn.propagate) == state


@pytest.mark.util
def test_verbose_shows_each_python_warning_and_logs_debug_to_the_file_only(tmp_path, capsys):
    other = logging.getLogger("finn.transformation.fpgadataflow")
    finn = logging.getLogger("finn")
    level = finn.level
    path = tmp_path / "run.log"
    with log_run(str(path), runner=RUNNER, verbose=True):
        other.debug("one item's detail")
        other.info("a module's line")
        other.info("a tool's output", extra=FILE_ONLY)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("a library's warning", UserWarning)
    printed = capsys.readouterr()
    log = path.read_text()
    assert "DEBUG - finn.transformation.fpgadataflow: one item's detail" in log
    assert printed.out.splitlines() == ["a module's line", "a tool's output"]
    assert "UserWarning: a library's warning" in printed.err
    assert "Python warnings" not in printed.out + log
    assert finn.level == level


def warn_in_worker(index):
    warnings.simplefilter("always")
    warnings.warn(f"worker warning {index}", UserWarning)


@pytest.mark.util
def test_warnings_of_forked_workers_are_counted(tmp_path, capsys):
    # as in a NodeLocalTransformation, whose pool is made inside the step
    path = tmp_path / "run.log"
    with log_run(str(path), runner=RUNNER):
        with multiprocessing.get_context("fork").Pool(2) as pool:
            pool.map(warn_in_worker, range(4))
    assert f"4 Python warnings, in {path}" in capsys.readouterr().out
    assert path.read_text().count("UserWarning: worker warning") == 4


@pytest.mark.util
def test_warnings_reach_the_file_under_a_strict_root_and_the_hook_is_restored(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(logging.getLogger(), "level", logging.ERROR)
    seen = []
    hook = lambda *args, **kwargs: seen.append(args[0])  # noqa: E731
    monkeypatch.setattr(warnings, "showwarning", hook)
    path = tmp_path / "run.log"
    with log_run(str(path), runner=RUNNER):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("a library's warning", UserWarning)
    assert "UserWarning: a library's warning" in path.read_text()
    assert warnings.showwarning is hook and seen == []


@pytest.mark.util
def test_runs_do_not_nest(tmp_path):
    with log_run(str(tmp_path / "outer.log"), runner=RUNNER):
        with pytest.raises(RuntimeError):
            with log_run(str(tmp_path / "inner.log"), runner=RUNNER):
                pass
    assert not (tmp_path / "inner.log").exists()


INTERRUPTED_POOL = """
import logging
import multiprocessing
import os
import signal
import sys
import warnings

from finn.util.build_log import in_run, log_run


def hold_the_counter(ready):
    # as a worker terminated inside the counter's update does: keep its lock
    (counted,) = [h for h in logging.getLogger("py.warnings").handlers if hasattr(h, "count")]
    counted.count.get_lock().acquire()
    os.write(ready, b"x")
    signal.pause()


warnings.simplefilter("always")
reader, writer = os.pipe()
try:
    with log_run(sys.argv[1], runner="finn.test"):
        try:
            with multiprocessing.get_context("fork").Pool(1) as pool:
                pool.apply_async(hold_the_counter, (writer,))
                os.read(reader, 1)
                raise KeyboardInterrupt
        finally:
            warnings.warn("after the pool is terminated", UserWarning)
except KeyboardInterrupt:
    print("interrupted")
print("in_run", in_run())
loggers = logging.getLogger("py.warnings"), logging.getLogger("finn")
print("handlers", *(len(logger.handlers) for logger in loggers))
"""


@pytest.mark.util
def test_an_interrupted_pool_that_held_the_warning_counter_does_not_block_the_run(tmp_path):
    path = tmp_path / "run.log"
    done = subprocess.run(
        [sys.executable, "-c", INTERRUPTED_POOL, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-3:] == ["interrupted", "in_run False", "handlers 0 0"]
    assert "UserWarning: after the pool is terminated" in path.read_text()
