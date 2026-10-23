# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Running external tools: check the exit status, keep a transcript, quote the errors.

``run_tool`` runs a tool's command line itself, not through a shell script, so FINN
sees the exit status of the tool. The tool writes its standard output and standard
error into two transcript files while it runs (``transcript_files``). If the tool
fails, ``ToolFailed`` names the transcript and quotes the first error lines.
``ToolUnavailable`` is a tool that is not installed or not set up.

The tool runs in its own session and process group, so that FINN can kill the tool
with the descendants that stay in its process group (``run_process``).
"""

import contextlib
import errno
import logging
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

from finn.util import vivado

StrPath = Union[str, os.PathLike]

log = logging.getLogger(__name__)

# How many error lines a ToolFailed quotes.
ERROR_LINES = 5
# An error line of a tool that does not print Vivado's messages (a compiler, a script).
_ERROR_LINE = re.compile(r"\berror:|^Error\b")
# Vivado echoes each line of a sourced Tcl script with this prefix.
_ECHOED_TCL = "# "
# Tcl prints the message of an uncaught error, then this line, then the trace.
_TCL_TRACE = "while executing"


def transcript_files(stem: StrPath) -> Tuple[Path, Path]:
    """Return the two files of the transcript ``stem``: ``<stem>.stdout.log`` and
    ``<stem>.stderr.log``."""
    stem = os.fspath(stem)
    return Path(stem + ".stdout.log"), Path(stem + ".stderr.log")


class ToolRun(subprocess.CompletedProcess):
    """A finished run. Like ``subprocess.CompletedProcess``, with ``transcript``: the
    file that received the standard output (the standard error file is beside it), or
    None if the output was kept in memory only."""

    def __init__(self, args, returncode, stdout, stderr, transcript=None):
        super().__init__(args, returncode, stdout, stderr)
        self.transcript = transcript


def resolve_executable(
    name: StrPath, cwd: Optional[StrPath] = None, env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Return the path of the executable that a child process started in ``cwd`` with
    the environment ``env`` runs for ``name``, as ``subprocess.Popen`` finds it. A name
    with a slash is relative to ``cwd``. Otherwise, the directories of PATH are
    searched, and an empty or relative directory is relative to ``cwd``. Returns None
    if there is no such executable."""
    name = os.fspath(name)
    base = os.path.abspath(cwd if cwd is not None else os.curdir)
    if os.sep in name:
        return os.path.join(base, name)
    for directory in os.get_exec_path(env):
        candidate = os.path.join(base, directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _popen(command, **kwargs):
    try:
        return subprocess.Popen(command, **kwargs)
    except OSError as exc:
        if exc.errno != errno.ENOEXEC:
            raise
    # ENOEXEC: a script without a "#!" line, such as a wrapper in FINN_TOOL_DIR_OVERRIDE.
    log.debug("%s has no #! line, running it with bash", command[0])
    return subprocess.Popen(["bash", *command], **kwargs)


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# Signals that end this process, for example when a multiprocessing pool terminates
# its workers, or when the terminal closes.
_TERMINATING_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


@contextlib.contextmanager
def _killed_on_termination(started):
    """While the body runs, SIGTERM and SIGHUP first kill the process group of the
    process in ``started`` (a list that holds it once it runs), then have the effect
    of the handler that was installed before. A signal that is ignored (SIGHUP under
    nohup) stays ignored and kills nothing. Signal handlers can only be changed in
    the main thread, so other threads get no such cleanup."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}

    def terminating(signum, frame):
        for proc in started:
            _kill_group(proc)
        handler = previous[signum]
        if callable(handler):
            handler(signum, frame)
        else:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for signum in _TERMINATING_SIGNALS:
        if signal.getsignal(signum) != signal.SIG_IGN:
            previous[signum] = signal.signal(signum, terminating)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)


@contextlib.contextmanager
def _signals_held():
    """Hold SIGINT, SIGTERM and SIGHUP while the body runs, and deliver them after it.
    Thus a process that the body starts and records is recorded before a signal can
    stop the body, and the cleanup finds it. Main thread only, as
    ``_killed_on_termination``."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    held = []
    previous = {}
    for signum in (signal.SIGINT, *_TERMINATING_SIGNALS):
        if signal.getsignal(signum) not in (signal.SIG_IGN, None):
            previous[signum] = signal.signal(signum, lambda signum, frame: held.append(signum))
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        for signum in dict.fromkeys(held):
            signal.raise_signal(signum)


def run_process(
    argv: Sequence[StrPath],
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[StrPath] = None,
    timeout: Optional[float] = None,
    check: bool = True,
    transcript: Optional[StrPath] = None,
) -> ToolRun:
    """Run ``argv`` in a new session and wait for it.

    The executable is ``resolve_executable(argv[0], cwd, env)``. If it is a script
    without a ``#!`` line, it runs with bash. With a ``transcript`` stem, the process
    writes its output into the two transcript files while it runs. Without one, the
    output is kept in memory. The result holds the output in both cases. With
    ``check``, a nonzero exit status raises ``subprocess.CalledProcessError``.

    FINN kills the process group of the tool (SIGKILL) on a timeout
    (``subprocess.TimeoutExpired``), on any exception while it waits, such as
    KeyboardInterrupt, and, in the main thread, on SIGTERM and SIGHUP unless this
    process ignores them. A signal that arrives while the tool starts waits until
    FINN holds the tool. The exception carries the output so far in ``output`` and
    ``stderr``. This does not cover a SIGKILL of this process, or descendants that
    leave the process group of the tool (a new process group or session).
    """
    command = [os.fspath(arg) for arg in argv]
    if not command:
        raise ValueError("Empty command")
    command[0] = resolve_executable(command[0], cwd, env) or command[0]
    files = None if transcript is None else transcript_files(transcript)
    log.info(
        "Running %s in %s, transcript %s",
        shlex.join(command),
        os.fspath(cwd) if cwd is not None else os.getcwd(),
        "none" if files is None else files[0],
    )
    started = time.monotonic()
    running = []

    def printed():
        return files[0].read_bytes(), files[1].read_bytes()

    with _killed_on_termination(running):
        try:
            if files is None:
                with _signals_held():
                    running.append(
                        _popen(
                            command,
                            cwd=cwd,
                            env=env,
                            start_new_session=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                    )
                out, err = running[0].communicate(timeout=timeout)
            else:
                files[0].parent.mkdir(parents=True, exist_ok=True)
                # The child writes the files itself, so they grow while it runs.
                with open(files[0], "wb") as stdout, open(files[1], "wb") as stderr:
                    with _signals_held():
                        running.append(
                            _popen(
                                command,
                                cwd=cwd,
                                env=env,
                                start_new_session=True,
                                stdout=stdout,
                                stderr=stderr,
                            )
                        )
                running[0].wait(timeout=timeout)
                out, err = printed()
        except BaseException as exc:
            if not running:
                raise
            proc = running[0]
            # Kill the whole group: a tool's descendants (Vivado's synthesis runs, a
            # JVM behind vitis-run) can outlive its leader.
            _kill_group(proc)
            if files is None:
                out, err = proc.communicate()
            else:
                proc.wait()
                out, err = printed()
            if isinstance(exc, (subprocess.TimeoutExpired, KeyboardInterrupt)):
                exc.output = out
                exc.stderr = err
            log.warning(
                "Stopped %s after %.1f s (%s)",
                command[0],
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise
    proc = running[0]
    log.info(
        "Finished %s after %.1f s with exit status %d",
        command[0],
        time.monotonic() - started,
        proc.returncode,
    )
    result = ToolRun(command, proc.returncode, out, err, None if files is None else files[0])
    if check:
        result.check_returncode()
    return result


def _text(stream):
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream or ""


class ToolUnavailable(LookupError):
    """A tool that FINN cannot run, because it is not installed or not set up."""

    def __init__(self, tool: str, reason: str):
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason

    def __reduce__(self):
        return (type(self), (self.tool, self.reason))


class ToolFailed(RuntimeError):
    """A tool that ran and failed.

    Attributes:
        tool: the name of the tool.
        command: its command line, if known.
        status: its exit status, or None if it did not exit with one.
        transcript: the file that holds its output, if there is one.
        errors: the first error lines it printed (see ``ToolFailed.of``).
        code: ``vivado.LICENSE_UNAVAILABLE`` if its output says that no license was
            found, else ``vivado.TOOL_ERROR``.
    """

    def __init__(
        self,
        tool: str,
        status: Optional[int] = None,
        transcript: Optional[StrPath] = None,
        errors: Sequence[str] = (),
        code: str = vivado.TOOL_ERROR,
        command: Sequence[str] = (),
    ):
        self.tool = tool
        self.command = tuple(command)
        self.status = status
        self.transcript = None if transcript is None else Path(transcript)
        self.errors = tuple(errors)
        self.code = code
        message = f"{tool} failed"
        if status is not None:
            message += f" with exit status {status}"
        if code != vivado.TOOL_ERROR:
            message += f" ({code})"
        if self.transcript is not None:
            message += f"; transcript {self.transcript}"
        super().__init__("\n  ".join((message, *self.errors)))

    def __reduce__(self):
        # Keep all fields across a multiprocessing pool (NodeLocalTransformation).
        return (
            type(self),
            (self.tool, self.status, self.transcript, self.errors, self.code, self.command),
        )

    @classmethod
    def of(
        cls,
        tool: str,
        printed: Union[str, bytes],
        *,
        status: Optional[int] = None,
        transcript: Optional[StrPath] = None,
        command: Sequence[str] = (),
    ) -> "ToolFailed":
        """The failure of ``tool``, read from what it ``printed``.

        The quoted lines are its errors, in the order printed: Vivado-style errors
        (``vivado.messages``), the message of an uncaught Tcl error, and other
        ``error:`` lines (a compiler's), then its critical warnings. If it printed no
        errors, its last lines.
        """
        if isinstance(printed, bytes):
            printed = printed.decode(errors="replace")
        lines = [line.strip() for line in printed.splitlines() if line.strip()]
        read = vivado.messages(lines)
        structured = set(read.errors)
        critical = set(read.critical_warnings)
        errors = [
            line
            for line, after in zip(lines, lines[1:] + [""])
            if line in structured
            or after == _TCL_TRACE
            or (
                line not in critical
                and not line.startswith(_ECHOED_TCL)
                and _ERROR_LINE.search(line)
            )
        ]
        if errors:
            quoted = list(dict.fromkeys(errors + list(read.critical_warnings)))[:ERROR_LINES]
        else:
            quoted = lines[-ERROR_LINES:]
        return cls(
            tool,
            status=status,
            transcript=transcript,
            errors=quoted,
            code=read.code,
            command=command,
        )

    @classmethod
    def of_run(cls, tool: str, result: subprocess.CompletedProcess) -> "ToolFailed":
        """The failure of a run of ``tool`` that exited with a nonzero status."""
        return cls.of(
            tool,
            # One line break between the streams, so that their lines stay apart.
            "\n".join(_text(stream) for stream in (result.stdout, result.stderr)),
            status=result.returncode,
            transcript=getattr(result, "transcript", None),
            command=result.args,
        )


def run_tool(
    command: Sequence[StrPath],
    *,
    cwd: Optional[StrPath] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    replay: Optional[StrPath] = None,
    transcript: Optional[StrPath] = None,
) -> ToolRun:
    """Run an external tool with ``run_process``, and raise ``ToolFailed`` if it exits
    with a nonzero status.

    Args:
        command: the command line, the tool first (see
            ``finn.util.basic.resolve_xilinx_tool``).
        cwd: the working directory of the tool.
        env: the whole environment of the tool. None means this process's
            environment.
        timeout: the seconds after which the tool is killed.
        replay: a bash script to write before the run, which repeats the run by hand.
            It runs the same executable as the run. It does not record ``env``: run it
            in the same environment.
        transcript: the stem of the transcript files. By default, the replay path
            without its suffix. Without either, the output is kept in memory only.
    """
    command = [os.fspath(arg) for arg in command]
    if not command:
        raise ValueError("Empty command")
    command[0] = resolve_executable(command[0], cwd, env) or command[0]
    if replay is not None:
        lines = ["#!/bin/bash", "set -e"]
        if cwd is not None:
            lines.append("cd -- " + shlex.quote(os.fspath(cwd)))
        lines.append("exec " + shlex.join(command))
        Path(replay).write_text("\n".join(lines) + "\n")
        if transcript is None:
            transcript = os.path.splitext(os.fspath(replay))[0]
    result = run_process(
        command, env=env, cwd=cwd, timeout=timeout, check=False, transcript=transcript
    )
    if result.returncode != 0:
        raise ToolFailed.of_run(os.path.basename(command[0]), result)
    return result
