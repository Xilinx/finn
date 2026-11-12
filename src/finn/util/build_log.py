# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One run's log: FINN's records and Python's warnings, to a log file and the console.

Without ``verbose``, the console shows WARNING and above, and INFO only from the
``runner`` logger and the loggers under it (in a build, the step banners and the
outcome). Python's warnings go to the file, and the console shows their count at the
end. With ``verbose``, the console shows everything from INFO, and the file also
takes DEBUG.

The log holds the records of the ``finn`` loggers and Python's warnings, not what a
step prints or what other libraries log. Handlers that the caller attached to
``finn`` stay attached and receive every record of the run.
"""

import contextlib
import logging
import multiprocessing
import os
import sys
import warnings
from collections.abc import Iterator

#: Pass as ``extra`` to send a record to the log file only, unless ``verbose``.
FILE_ONLY = {"finn_console": False}

#: The step stamped on every record, ``-`` outside a step. A module global, not a context
#: variable, so that a step's threads and forked pool workers (HLSSynthIP) see it.
_step = "-"
_active = False


def in_run() -> bool:
    """Whether a ``log_run`` is active in this process, or in the process it was
    forked from."""
    return _active


@contextlib.contextmanager
def running_step(name: str) -> Iterator[None]:
    """Stamp the records of the ``with`` block with the step ``name``."""
    global _step
    outer, _step = _step, name
    try:
        yield
    finally:
        _step = outer


class _StepStamp(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.step = _step
        return True


class _Console(logging.Filter):
    """The records one console stream shows: stdout takes INFO, stderr (``errors``)
    WARNING and above."""

    def __init__(self, runner: str, verbose: bool, errors: bool):
        super().__init__()
        self.runner = runner
        self.verbose = verbose
        self.errors = errors

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.INFO or (record.levelno >= logging.WARNING) != self.errors:
            return False
        if self.verbose:
            return True
        if record.name == "py.warnings" or not getattr(record, "finn_console", True):
            return False
        own = record.name == self.runner or record.name.startswith(self.runner + ".")
        return record.levelno >= logging.WARNING or own


class _Counted(logging.Handler):
    """Counts records, also those of forked pool workers, through shared memory."""

    def __init__(self) -> None:
        super().__init__()
        self.count = multiprocessing.Value("i", 0)

    def emit(self, record: logging.LogRecord) -> None:
        # A pool worker terminated inside this block never releases the lock, so wait
        # for it only briefly and then leave the record uncounted.
        lock = self.count.get_lock()
        if lock.acquire(timeout=1):
            try:
                self.count.value += 1
            finally:
                lock.release()

    def total(self) -> int:
        """The count so far; read without the lock, which a terminated worker may hold."""
        return self.count.get_obj().value


class _ConsoleFormat(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return text if record.levelno < logging.WARNING else f"{record.levelname}: {text}"


@contextlib.contextmanager
def log_run(path: str, *, runner: str, verbose: bool = False) -> Iterator[None]:
    """Log the ``with`` block to the file ``path`` (appended to) and to the console, then
    restore the logging found. ``finn`` logs from INFO, or DEBUG with ``verbose``, and
    ``py.warnings`` from WARNING, unless the caller set their levels. Runs do not nest."""
    global _active
    if _active:
        raise RuntimeError("log_run is already active in this process")
    stamp = _StepStamp()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_handler = logging.FileHandler(path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(step)s %(name)s: %(message)s")
    )
    handlers: list[logging.Handler] = [file_handler]
    for stream, errors in ((sys.stdout, False), (sys.stderr, True)):
        console = logging.StreamHandler(stream)
        console.addFilter(_Console(runner, verbose, errors))
        console.setFormatter(_ConsoleFormat("%(message)s"))
        handlers.append(console)
    finn, python = logging.getLogger("finn"), logging.getLogger("py.warnings")
    saved = [(logger, logger.level, logger.propagate) for logger in (finn, python)]
    for handler in handlers:
        handler.addFilter(stamp)
        for logger in (finn, python):
            logger.addHandler(handler)
    counted = _Counted()
    python.addHandler(counted)
    for logger in (finn, python):
        # The root's handlers (a notebook's) would print each line twice.
        logger.propagate = False
    shown = finn.level or (logging.DEBUG if verbose else logging.INFO)
    if shown > logging.INFO:
        for handler in handlers:
            handler.addFilter(
                lambda record: record.levelno >= shown or record.name == "py.warnings"
            )
    finn.setLevel(shown)
    # Else the root's level, for example ERROR, would drop the warnings.
    python.setLevel(python.level or logging.WARNING)
    saved_hook = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if file is not None:
            saved_hook(message, category, filename, lineno, file, line)
        else:
            text = warnings.formatwarning(message, category, filename, lineno, line)
            python.warning(text.rstrip("\n"))

    warnings.showwarning = showwarning
    _active = True
    try:
        yield
    finally:
        _active = False
        warnings.showwarning = saved_hook
        try:
            warned = counted.total()
            if warned and not verbose:
                logging.getLogger(runner).info("%d Python warnings, in %s", warned, path)
        finally:
            python.removeHandler(counted)
            for logger, level, propagate in saved:
                logger.setLevel(level)
                logger.propagate = propagate
                for handler in handlers:
                    logger.removeHandler(handler)
            for handler in handlers:
                handler.close()
