# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import subprocess
import sys

from finn.util.hls import CallHLS
from finn.util.launcher import ToolFailed

# A stand-in for vitis_hls and vitis-run: prints its name, where it runs and its
# arguments.
FAKE_HLS = """
import json, os, sys
print(json.dumps({"tool": os.path.basename(sys.argv[0]), "cwd": os.getcwd(), "argv": sys.argv[1:]}))
"""

# A stand-in that fails as HLS does on a design error.
FAILING_HLS = """
print("INFO: [HLS 200-10] Analyzing design file")
print("ERROR: [HLS 207-3776] use of undeclared identifier 'foo'")
raise SystemExit(1)
"""


@pytest.fixture
def hls_tools(tmp_path, monkeypatch):
    """Point FINN_TOOL_DIR_OVERRIDE at a directory of fake HLS frontends running
    ``body``, and return a code generation directory with a tcl script."""

    def make(body, release="2024.2"):
        tools = tmp_path / "tools"
        tools.mkdir(exist_ok=True)
        for name in ("vitis_hls", "vitis-run"):
            (tools / name).write_text("#!" + sys.executable + "\n" + body)
            (tools / name).chmod(0o755)
        monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(tools))
        monkeypatch.setenv("XILINX_VIVADO", f"/tools/Xilinx/Vivado/{release}")
        code_gen_dir = tmp_path / "code_gen_ipgen"
        code_gen_dir.mkdir(exist_ok=True)
        (code_gen_dir / "hls_syn_node.tcl").write_text("exit\n")
        return code_gen_dir

    return make


def call_hls(code_gen_dir):
    builder = CallHLS()
    builder.append_tcl(str(code_gen_dir / "hls_syn_node.tcl"))
    builder.set_ipgen_path(str(code_gen_dir / "project_node"))
    builder.build(str(code_gen_dir))
    return builder


@pytest.mark.util
@pytest.mark.parametrize(
    "release, frontend, args",
    [("2024.2", "vitis_hls", []), ("2025.1", "vitis-run", ["--mode", "hls", "--tcl"])],
)
def test_call_hls_runs_the_frontend_in_code_gen_dir(hls_tools, release, frontend, args):
    code_gen_dir = hls_tools(FAKE_HLS, release)
    builder = call_hls(code_gen_dir)
    tcl = str(code_gen_dir / "hls_syn_node.tcl")
    ran = json.loads((code_gen_dir / "ipgen.stdout.log").read_text())
    assert ran == {"tool": frontend, "cwd": str(code_gen_dir), "argv": args + [tcl]}
    assert builder.ipgen_script == str(code_gen_dir / "ipgen.sh")
    replay = subprocess.run(["bash", builder.ipgen_script], capture_output=True, check=True)
    assert json.loads(replay.stdout) == ran


@pytest.mark.util
def test_call_hls_raises_when_hls_fails(hls_tools):
    code_gen_dir = hls_tools(FAILING_HLS)
    with pytest.raises(ToolFailed) as failed:
        call_hls(code_gen_dir)
    assert (failed.value.tool, failed.value.status) == ("vitis_hls", 1)
    assert failed.value.transcript == code_gen_dir / "ipgen.stdout.log"
    assert failed.value.errors == ("ERROR: [HLS 207-3776] use of undeclared identifier 'foo'",)
