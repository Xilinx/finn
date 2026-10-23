# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import os
import pickle
import signal
import subprocess
import sys
import time

from finn.util import launcher, vivado
from finn.util.basic import resolve_xilinx_tool
from finn.util.launcher import ToolFailed, run_process, run_tool, transcript_files


def make_tool(directory, name, body):
    """Write an executable ``name`` in ``directory`` that runs the Python ``body``."""
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)
    return path


def make_script(path, text):
    """Write an executable shell script without a "#!" line, as some site wrappers are."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def gone(pid, timeout=10):
    """Wait until process ``pid`` has exited (a zombie counts as exited)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as stat:
                if stat.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
        except (FileNotFoundError, ProcessLookupError):
            # Gone before open() (ENOENT) or reaped between open() and read() (ESRCH).
            return True
        time.sleep(0.05)
    return False


def wait_for_file(path, timeout=30):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} was not written"
        time.sleep(0.02)


# Starts a child that would outlive it, prints "started", writes both process IDs, then hangs.
HANGS = """
import json, os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print("started", flush=True)
with open(sys.argv[1] + ".tmp", "w") as f:
    json.dump({"tool": os.getpid(), "child": child.pid}, f)
os.rename(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(60)
"""

# Runs HANGS with run_process.
RUNS_HANGS = """
import sys
from finn.util.launcher import run_process
run_process([sys.executable, "-c", sys.argv[1], sys.argv[2]])
"""

# Writes argv[1] when it runs, waits for argv[1] + ".release", and exits 0.
WAITS_FOR_RELEASE = """
import os, sys, time
open(sys.argv[1], "w").close()
deadline = time.monotonic() + 60
while not os.path.exists(sys.argv[1] + ".release"):
    assert time.monotonic() < deadline, "not released"
    time.sleep(0.02)
"""

# Runs WAITS_FOR_RELEASE with run_process while SIGHUP is ignored, as under nohup.
RUNS_UNDER_NOHUP = """
import signal, sys
from finn.util.launcher import run_process
signal.signal(signal.SIGHUP, signal.SIG_IGN)
run_process([sys.executable, "-c", sys.argv[1], sys.argv[2]])
"""

# Vivado's output for a synthesis without a license.
UNLICENSED = """\
****** Vivado v2025.2 (64-bit)
INFO: [Device 21-403] Loading part xcvc1902-vsva2197-2MP-e-S
WARNING: [Vivado 12-13651] The IP file has been moved
ERROR: [Common 17-345] A valid license was not found for feature 'Synthesis' and/or \
device 'xcvc1902'. Please run the Vivado License Manager for assistance.
ERROR: [Common 17-69] Command failed: Synthesis failed - please see the console
INFO: [Common 17-206] Exiting Vivado
"""


@pytest.mark.util
def test_transcript_holds_both_streams(tmp_path):
    body = "import sys\nprint('on stdout')\nprint('on stderr', file=sys.stderr)\n"
    tool = make_tool(tmp_path / "tools", "xsim", body)
    result = run_tool([tool], transcript=tmp_path / "logs" / "xsim")
    stdout, stderr = transcript_files(tmp_path / "logs" / "xsim")
    assert (stdout.read_text(), stderr.read_text()) == ("on stdout\n", "on stderr\n")
    assert (result.returncode, result.stdout, result.stderr, result.transcript) == (
        0,
        b"on stdout\n",
        b"on stderr\n",
        stdout,
    )


@pytest.mark.util
def test_replay_and_default_transcript(tmp_path):
    tool = make_tool(tmp_path / "tools", "vitis_hls", "import os; print(os.getcwd())\n")
    work = tmp_path / "work dir"
    work.mkdir()
    result = run_tool([tool, "script.tcl"], cwd=work, replay=work / "ipgen.sh")
    assert result.transcript == work / "ipgen.stdout.log"
    assert result.stdout.decode().strip() == str(work)
    # The replay repeats the run, in the same directory, and passes the exit status on.
    replay = subprocess.run(["bash", work / "ipgen.sh"], capture_output=True)
    assert replay.stdout == result.stdout
    make_tool(tmp_path / "tools", "vitis_hls", "raise SystemExit(3)\n")
    assert subprocess.run(["bash", work / "ipgen.sh"]).returncode == 3
    # Without a transcript or a replay, the output is kept in memory only.
    assert run_tool([sys.executable, "-c", "print('x')"]).transcript is None


@pytest.mark.util
def test_script_without_shebang_runs_with_bash(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    make_script(
        tools / "vivado", 'echo "wrapped $*"\necho "ERROR: [Common 17-69] failed"\nexit 4\n'
    )
    with pytest.raises(ToolFailed) as failed:
        run_tool([tools / "vivado", "-mode", "batch"], cwd=tmp_path, replay=tmp_path / "run.sh")
    assert (failed.value.status, failed.value.errors) == (4, ("ERROR: [Common 17-69] failed",))
    assert (tmp_path / "run.stdout.log").read_text().startswith("wrapped -mode batch\n")
    # Found on PATH by its bare name, as resolve_xilinx_tool returns it.
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    result = run_process(["vivado", "-version"], check=False)
    assert (result.returncode, result.stdout.splitlines()[0]) == (4, b"wrapped -version")
    # The replay script runs it too.
    assert subprocess.run(["bash", tmp_path / "run.sh"]).returncode == 4


@pytest.mark.util
def test_failure_names_its_transcript_and_quotes_its_errors(tmp_path):
    tool = make_tool(
        tmp_path / "tools",
        "vivado",
        "print('INFO: [Common 17-206] Exiting')\n"
        "print('CRITICAL WARNING: [Vivado 12-1411] cannot place')\n"
        "print('ERROR: [Synth 8-439] module foo not found')\n"
        "raise SystemExit(1)\n",
    )
    with pytest.raises(ToolFailed) as failed:
        run_tool([tool, "-mode", "batch"], cwd=tmp_path, replay=tmp_path / "synth.sh")
    transcript = tmp_path / "synth.stdout.log"
    assert (failed.value.tool, failed.value.status) == ("vivado", 1)
    assert failed.value.command == (str(tool), "-mode", "batch")
    assert failed.value.transcript == transcript
    assert failed.value.code == vivado.TOOL_ERROR
    # Errors first, then critical warnings.
    assert failed.value.errors == (
        "ERROR: [Synth 8-439] module foo not found",
        "CRITICAL WARNING: [Vivado 12-1411] cannot place",
    )
    assert str(transcript) in str(failed.value)
    assert failed.value.errors[0] in str(failed.value)
    # A NodeLocalTransformation's worker sends the failure through pickle.
    copy = pickle.loads(pickle.dumps(failed.value))
    assert (str(copy), copy.tool, copy.command, copy.transcript, copy.errors, copy.code) == (
        str(failed.value),
        "vivado",
        failed.value.command,
        transcript,
        failed.value.errors,
        vivado.TOOL_ERROR,
    )


@pytest.mark.util
def test_license_failure_is_classified(tmp_path):
    body = f"import sys\nsys.stdout.write({UNLICENSED!r})\nraise SystemExit(1)\n"
    tool = make_tool(tmp_path / "tools", "vivado", body)
    with pytest.raises(ToolFailed) as failed:
        run_tool([tool], transcript=tmp_path / "synth")
    assert failed.value.code == vivado.LICENSE_UNAVAILABLE
    assert failed.value.errors[0].startswith("ERROR: [Common 17-345] A valid license")
    assert vivado.LICENSE_UNAVAILABLE in str(failed.value)


@pytest.mark.util
def test_compiler_errors_come_before_critical_warnings():
    mixed = ToolFailed.of(
        "vitis_hls",
        "CRITICAL WARNING: [HLS 200-123] example warning\n"
        "a.cpp:1:1: error: undeclared identifier 'x'\n"
        "ERROR: [HLS 207-3776] use of undeclared identifier 'x'\n",
        status=1,
    )
    assert mixed.errors == (
        "a.cpp:1:1: error: undeclared identifier 'x'",
        "ERROR: [HLS 207-3776] use of undeclared identifier 'x'",
        "CRITICAL WARNING: [HLS 200-123] example warning",
    )
    quiet = ToolFailed.of("tool", b"one\ntwo\n", status=2)
    assert (quiet.errors, quiet.code, quiet.transcript) == (("one", "two"), vivado.TOOL_ERROR, None)
    # stdout without a final newline does not merge with the first line of stderr.
    result = subprocess.CompletedProcess(
        ["vitis_hls"], 2, b"progress...", b"ERROR: [HLS 207-3776] undeclared x\n"
    )
    assert ToolFailed.of_run("vitis_hls", result).errors == ("ERROR: [HLS 207-3776] undeclared x",)


@pytest.mark.util
def test_tcl_errors_are_quoted():
    # Vivado 2025.2 prints a plain Tcl error on stdout, with no ERROR: line
    warning = "CRITICAL WARNING: [BD 41-1356] Slave segment is not assigned into address space"
    plain = (
        '# puts "error: in a branch that did not run"\n'
        'invalid command name "no_such_command"\n'
        "    while executing\n"
        '"no_such_command"\n'
        '    (file "bad.tcl" line 2)\n'
        "INFO: [Common 17-206] Exiting Vivado\n"
    )
    assert ToolFailed.of("vivado", plain).errors == ('invalid command name "no_such_command"',)
    assert ToolFailed.of("vivado", warning + "\n" + plain).errors == (
        'invalid command name "no_such_command"',
        warning,
    )
    nested = (
        "root cause: missing project metadata\n"
        "    while executing\n"
        '"error "root cause: missing project metadata""\n'
        '    (procedure "inner" line 2)\n'
        "    invoked from within\n"
        '"inner"\n'
        '    (procedure "outer" line 2)\n'
        "    invoked from within\n"
        '"outer"\n'
        '    (file "bad.tcl" line 7)\n'
        "INFO: [Common 17-206] Exiting Vivado\n"
    )
    assert ToolFailed.of("vivado", nested).errors == ("root cause: missing project metadata",)
    # Without an error, the last lines are quoted, not earlier critical warnings.
    tail = ("step 1", "step 2", "step 3", "step 4", "failed somehow")
    unknown = ToolFailed.of("vivado", "\n".join((warning, *tail)), status=1)
    assert unknown.errors == tail


@pytest.mark.util
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("in_files", [False, True])
def test_a_signal_while_the_tool_starts_stops_it(tmp_path, monkeypatch, signum, in_files):
    # The signal arrives after the tool has started, before run_process has its handle
    started = []
    real_popen = launcher._popen

    def popen(*args, **kwargs):
        started.append(real_popen(*args, **kwargs))
        signal.raise_signal(signum)
        return started[-1]

    monkeypatch.setattr(launcher, "_popen", popen)
    received = []
    if signum == signal.SIGINT:
        previous = signal.signal(signum, signal.default_int_handler)
    else:
        previous = signal.signal(signum, lambda signum, frame: received.append(signum))
    hang = [sys.executable, "-c", "import time; time.sleep(60)"]
    transcript = tmp_path / "hang" if in_files else None
    try:
        if signum == signal.SIGINT:
            with pytest.raises(KeyboardInterrupt):
                run_process(hang, timeout=30, transcript=transcript)
        else:
            run_process(hang, timeout=30, check=False, transcript=transcript)
            assert received == [signum]
    finally:
        signal.signal(signum, previous)
        status = started[0].poll()
        started[0].kill()
        started[0].wait()
    assert status == -signal.SIGKILL


@pytest.mark.util
@pytest.mark.vivado
def test_a_failing_vivado_run_quotes_its_error(tmp_path):
    # Vivado prints a failed command's ERROR message on stderr
    (tmp_path / "bad.tcl").write_text("read_verilog missing.v\n")
    with pytest.raises(ToolFailed) as failed:
        run_tool(
            [resolve_xilinx_tool("vivado"), "-mode", "batch", "-source", "bad.tcl"],
            cwd=tmp_path,
            replay=tmp_path / "bad.sh",
        )
    assert failed.value.status != 0
    assert failed.value.errors[0].startswith("ERROR: [")
    assert "missing.v" in failed.value.errors[0]
    assert failed.value.errors[0] in (tmp_path / "bad.stderr.log").read_text()


@pytest.mark.util
def test_messages_of_vivado_tools():
    # The simulator's messages have the same form, some without an ID.
    sim = vivado.messages(
        [
            "ERROR: [VRFC 10-2989] 'x' is not declared",
            "FATAL_ERROR: Iteration",
            "CRITICAL WARNING: [XSIM 43-4099] module not found",
        ]
    )
    assert sim.errors == ("ERROR: [VRFC 10-2989] 'x' is not declared", "FATAL_ERROR: Iteration")
    assert sim.critical_warnings == ("CRITICAL WARNING: [XSIM 43-4099] module not found",)
    assert sim.code == vivado.TOOL_ERROR
    assert vivado.messages("FLEXnet Licensing error:-5,357").license_unavailable


@pytest.mark.util
def test_timeout_kills_the_process_group(tmp_path, monkeypatch):
    pids = tmp_path / "pids.json"
    real_popen = launcher._popen

    def popen(*args, **kwargs):
        # The timeout counts from here: wait until the tool has started its child
        proc = real_popen(*args, **kwargs)
        wait_for_file(pids, timeout=60)
        return proc

    monkeypatch.setattr(launcher, "_popen", popen)
    with pytest.raises(subprocess.TimeoutExpired) as stopped:
        run_process([sys.executable, "-c", HANGS, pids], timeout=1, transcript=tmp_path / "slow")
    assert stopped.value.output == b"started\n"
    assert (tmp_path / "slow.stdout.log").read_text() == "started\n"
    assert gone(json.loads(pids.read_text())["child"])


@pytest.mark.util
def test_sigterm_kills_the_process_group(tmp_path):
    # As a multiprocessing pool terminates a worker that runs a tool.
    pids = tmp_path / "pids.json"
    runner = subprocess.Popen([sys.executable, "-c", RUNS_HANGS, HANGS, pids])
    started = {}
    try:
        wait_for_file(pids)
        started = json.loads(pids.read_text())
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=10) == -signal.SIGTERM
        assert gone(started["tool"]) and gone(started["child"])
    finally:
        runner.kill()
        runner.wait()
        for pid in started.values():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.util
def test_ignored_sighup_does_not_stop_the_tool(tmp_path):
    # A build started with nohup ignores SIGHUP. A hangup during a tool run must not
    # kill the tool.
    ready = tmp_path / "ready"
    runner = subprocess.Popen([sys.executable, "-c", RUNS_UNDER_NOHUP, WAITS_FOR_RELEASE, ready])
    try:
        wait_for_file(ready)
        runner.send_signal(signal.SIGHUP)
        (tmp_path / "ready.release").touch()
        assert runner.wait(timeout=60) == 0
    finally:
        runner.kill()
        runner.wait()
