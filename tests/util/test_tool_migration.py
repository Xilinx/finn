# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage of the actual tool adapters, without FPGA tools."""

import pytest

import contextlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.core import rtlsim_exec
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.transformation.fpgadataflow import alveo_build, make_driver
from finn.transformation.fpgadataflow.make_driver import MakeCPPDriver
from finn.util.basic import CppBuilder, launch_process_helper
from finn.util.build_log import log_run
from finn.util.launcher import ToolFailed, ToolUnavailable, run_process, run_tool
from finn.xsi import setup

FAKE_TOOL = """
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(),
                  "ld_path": os.environ.get("LD_LIBRARY_PATH")}))
print("diagnostic from " + name, file=sys.stderr)
for output in json.loads(os.environ.get("FAKE_OUTPUTS", "{}" )).get(name, []):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("generated")
if os.environ.get("FAKE_FAIL") == name:
    print("error: simulated failure", file=sys.stderr)
    sys.exit(7)
"""


@pytest.fixture
def tool_dir(tmp_path, monkeypatch):
    tools = tmp_path / "tool wrappers"
    tools.mkdir()
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(tools))

    def make(name, body=FAKE_TOOL):
        path = tools / name
        path.write_text("#!" + sys.executable + "\n" + body)
        path.chmod(0o755)
        return path

    return make


def printed_call(path):
    return json.loads(path.read_text().splitlines()[0])


@pytest.mark.util
@pytest.mark.parametrize("check", [False, True])
@pytest.mark.parametrize("as_string", [False, True])
def test_helper_preserves_decoding_output_and_checked_status(tool_dir, capfd, check, as_string):
    tool = tool_dir(
        "utility with spaces",
        "import os; os.write(1, b'out\\xff\\r\\nline\\rend\\n'); "
        "os.write(2, b'err\\xfe\\r\\n'); raise SystemExit(9)",
    )
    args = str(tool) if as_string else [str(tool)]
    if check:
        with pytest.raises(subprocess.CalledProcessError) as failed:
            launch_process_helper(args, check=True)
        assert failed.value.cmd == args
        assert (failed.value.output, failed.value.stderr) == (
            "out\ufffd\nline\nend\n",
            "err\ufffd\n",
        )
    else:
        assert launch_process_helper(args) == ("out\ufffd\nline\nend\n", "err\ufffd\n")
    assert tuple(capfd.readouterr()) == ("out\ufffd\nline\nend\n", "err\ufffd\n")


@pytest.mark.util
def test_tool_failure_logs_both_streams_before_raising(tmp_path):
    log_path = tmp_path / "build.log"
    with log_run(str(log_path), runner="finn.builder.build_dataflow"):
        with pytest.raises(ToolFailed):
            run_tool(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('error: bad', file=sys.stderr); sys.exit(3)",
                ],
                transcript=tmp_path / "failed",
                log_output=True,
            )
    assert "finn.util.launcher: out" in log_path.read_text()
    assert "finn.util.launcher: error: bad" in log_path.read_text()


@pytest.mark.util
def test_missing_tools_are_distinct_from_bad_working_and_transcript_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ToolUnavailable, match="not found in PATH"):
        run_tool(["missing-compiler"])
    not_executable = tmp_path / "compiler"
    not_executable.write_text("exit 0\n")
    with pytest.raises(ToolUnavailable, match="no executable"):
        run_tool([not_executable])
    with pytest.raises(FileNotFoundError):
        run_tool([sys.executable], cwd=tmp_path / "missing-directory")
    with pytest.raises(NotADirectoryError):
        run_tool([sys.executable], cwd=not_executable)
    with pytest.raises(OSError) as failed:
        run_tool([sys.executable], transcript=not_executable / "transcript")
    assert not isinstance(failed.value, ToolUnavailable)


@pytest.mark.util
def test_environment_overrides_reach_execution_and_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLCHAIN_PRIVATE_ENV", "do not record this")
    replay = tmp_path / "repeat.sh"
    value = "/library path/with 'quotes'"
    cmd = [sys.executable, "-c", "import os; print(os.environ['LD_LIBRARY_PATH'])"]
    result = run_tool(cmd, env_overrides={"LD_LIBRARY_PATH": value}, replay=replay)
    assert result.stdout.decode().strip() == value
    assert "TOOLCHAIN_PRIVATE_ENV" not in replay.read_text()
    repeated = run_process(["bash", replay], env={"PATH": os.environ["PATH"]})
    assert repeated.stdout == result.stdout
    with pytest.raises(ValueError, match="Invalid environment variable"):
        run_tool(cmd, env_overrides={"BAD;NAME": "value"}, replay=replay)


@pytest.mark.util
def test_cppbuilder_expands_inputs_without_splitting_paths_or_accumulating_flags(
    tmp_path, monkeypatch
):
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler unavailable")
    includes = tmp_path / "include files"
    includes.mkdir()
    (includes / "answer.hpp").write_text("int answer();\n")
    sources = tmp_path / "source files"
    sources.mkdir()
    (sources / "main.cpp").write_text(
        "#include <answer.hpp>\n#include <iostream>\nint main() { std::cout << answer(); }\n"
    )
    (sources / "answer.cpp").write_text("int answer() { return VALUE; }\n")
    monkeypatch.setenv("TOOLCHAIN_INCLUDE", str(includes))
    builder = CppBuilder()
    builder.append_includes("-I$TOOLCHAIN_INCLUDE -DVALUE=42")
    builder.append_includes(["-std=c++17"])
    builder.append_sources(sources / "*.cpp")
    builder.set_executable_path(sources / "node model")
    for _ in range(2):
        builder.build(sources)
        assert run_tool([builder.executable_path]).stdout == b"42"
        assert builder.compile_components.count("-DVALUE=42") == 1
    assert (sources / "compile.stdout.log").is_file()
    assert (sources / "compile.stderr.log").is_file()
    run_process(["bash", builder.compile_script])


@pytest.mark.util
def test_cppbuilder_raises_on_failed_compilation_with_an_old_executable(
    tmp_path, tool_dir, monkeypatch
):
    tool_dir("g++")
    monkeypatch.setenv("FAKE_FAIL", "g++")
    (tmp_path / "source.cpp").write_text("int main() {}")
    executable = tmp_path / "node_model"
    executable.write_text("old executable")
    builder = CppBuilder()
    builder.append_sources(tmp_path / "*.cpp")
    builder.set_executable_path(executable)
    with pytest.raises(ToolFailed) as failed:
        builder.build(tmp_path)
    assert failed.value.status == 7
    assert failed.value.transcript == tmp_path / "compile.stdout.log"
    assert "error: simulated failure" in str(failed.value)
    assert executable.read_text() == "old executable"
    builder.cpp_files = [str(tmp_path / "missing*.cpp")]
    with pytest.raises(FileNotFoundError, match="No C\\+\\+ sources match"):
        builder.build(tmp_path)


@pytest.mark.util
def test_cppsim_rejects_a_failure_before_reading_stale_outputs(tmp_path, tool_dir, monkeypatch):
    executable = tool_dir("node_model")
    monkeypatch.setenv("FAKE_FAIL", "node_model")
    (tmp_path / "output_0.npy").write_bytes(b"stale")
    attrs = {"executable_path": str(executable), "code_gen_dir_cppsim": str(tmp_path)}
    node = types.SimpleNamespace(get_nodeattr=attrs.__getitem__)
    with pytest.raises(ToolFailed):
        HLSBackend.exec_precompiled_singlenode_model(node)
    assert printed_call(tmp_path / "execute.stdout.log")["cwd"] == os.getcwd()
    assert (tmp_path / "execute.stderr.log").is_file()


def make_model(nodes=None):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])
    nodes = nodes if nodes is not None else [helper.make_node("Identity", ["inp"], ["out"])]
    return ModelWrapper(qonnx_make_model(helper.make_graph(nodes, "test", [inp], [out])))


def kernel_model(directory):
    directory.mkdir()
    (directory / "ip").mkdir()
    (directory / "ip" / "component.xml").write_text("<component/>")
    model = make_model()
    model.set_metadata_prop("vivado_stitch_proj", str(directory))
    model.set_metadata_prop("vitis_xo", str(directory / "kernel.xo"))
    model.set_metadata_prop(
        "vivado_stitch_ifnames",
        json.dumps({"axilite": [], "aximm": [], "s_axis": [["in", 32]], "m_axis": [["out", 32]]}),
    )
    return model


@pytest.fixture
def link_model(tmp_path, monkeypatch, tool_dir):
    for tool in ("vivado", "v++", "slashkit"):
        tool_dir(tool)
    for name in ("VITIS_PATH", "PLATFORM_REPO_PATHS", "XILINX_XRT"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.delenv("PWD", raising=False)
    nodes = []
    for index in range(2):
        kernel = kernel_model(tmp_path / f"kernel {index}")
        path = tmp_path / f"kernel {index}.onnx"
        kernel.save(path)
        nodes.append(
            helper.make_node(
                "StreamingDataflowPartition",
                ["inp" if index == 0 else "mid"],
                ["mid" if index == 0 else "out"],
                name=f"kernel{index}",
                domain="finn.custom_op.fpgadataflow",
                model=str(path),
            )
        )
    directory = tmp_path / "link with spaces"
    directory.mkdir()
    monkeypatch.setattr(alveo_build, "make_build_dir", lambda prefix: str(directory))
    return make_model(nodes), directory


@pytest.mark.util
@pytest.mark.parametrize("outcome", ["success", "failure", "missing"])
def test_xo_packaging_checks_status_and_artifacts(tmp_path, tool_dir, monkeypatch, outcome):
    tool_dir("vivado")
    for name in ("VITIS_PATH", "PLATFORM_REPO_PATHS", "XILINX_XRT"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.delenv("PWD", raising=False)
    directory = tmp_path / "xo with spaces"
    model = kernel_model(directory)
    if outcome != "missing":
        monkeypatch.setenv("FAKE_OUTPUTS", json.dumps({"vivado": ["design.xo"]}))
    if outcome == "failure":
        monkeypatch.setenv("FAKE_FAIL", "vivado")
        (directory / "design.xo").write_text("stale")
    transform = alveo_build.CreateVitisXO("design")
    if outcome == "success":
        transform.apply(model)
        assert model.get_metadata_prop("vitis_xo") == str(directory / "design.xo")
    else:
        with pytest.raises(ToolFailed if outcome == "failure" else AssertionError):
            transform.apply(model)
    call = printed_call(directory / "gen_xo.stdout.log")
    assert call["cwd"] == str(directory)
    assert call["argv"] == ["-mode", "batch", "-source", "gen_xo.tcl"]


@pytest.mark.util
@pytest.mark.parametrize(
    "outcome", ["success", "failure", "missing", "report_failure", "report_missing"]
)
def test_vitis_link_preserves_arguments_and_rejects_failed_tools(link_model, monkeypatch, outcome):
    model, directory = link_model
    if outcome != "missing":
        monkeypatch.setenv(
            "FAKE_OUTPUTS", json.dumps({"v++": ["a.xclbin"], "vivado": ["synth_report.xml"]})
        )
    if outcome == "report_missing":
        monkeypatch.setenv("FAKE_OUTPUTS", json.dumps({"v++": ["a.xclbin"]}))
    if outcome in ("failure", "report_failure"):
        monkeypatch.setenv("FAKE_FAIL", "v++" if outcome == "failure" else "vivado")
        (directory / "a.xclbin").write_text("stale")
    transform = alveo_build.VitisLink("/platform path/u55c.xpfm", 5, enable_debug=True)
    if outcome == "success":
        transform.apply(model)
        assert model.get_metadata_prop("bitfile") == str(directory / "a.xclbin")
        assert model.get_metadata_prop("vivado_synth_rpt") == str(directory / "synth_report.xml")
    else:
        with pytest.raises(
            AssertionError if outcome in ("missing", "report_missing") else ToolFailed
        ):
            transform.apply(model)
    call = printed_call(directory / "run_vitis_link.stdout.log")
    assert call["cwd"] == str(directory)
    args = call["argv"]
    assert args[args.index("--platform") + 1] == "/platform path/u55c.xpfm"
    assert len([arg for arg in args if arg.endswith("kernel.xo")]) == 2
    assert args.count("--dk") == 2
    assert args[args.index("--kernel_frequency") + 1] == "200"
    if outcome in ("success", "report_failure", "report_missing"):
        report = printed_call(directory / "gen_report_xml.stdout.log")
        assert report["argv"][-1] == "gen_report_xml.tcl"
    else:
        assert not (directory / "gen_report_xml.stdout.log").exists()


@pytest.mark.util
@pytest.mark.parametrize("hardware", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "missing"])
def test_slash_link_checks_status_and_mode_specific_outputs(
    link_model, monkeypatch, hardware, outcome
):
    model, directory = link_model
    outputs = ["finn.vbin", "finn.vbin.prj/report_utilization_finn.xml"]
    if outcome != "missing":
        monkeypatch.setenv("FAKE_OUTPUTS", json.dumps({"slashkit": outputs}))
    if outcome == "failure":
        monkeypatch.setenv("FAKE_FAIL", "slashkit")
        (directory / "finn.vbin").write_text("stale")
    transform = alveo_build.SlashLink(build_hardware=hardware)
    if outcome == "success":
        transform.apply(model)
        assert model.get_metadata_prop("bitfile") == str(directory / "finn.vbin")
        assert (model.get_metadata_prop("slash_report") is not None) == hardware
    else:
        with pytest.raises(ToolFailed if outcome == "failure" else AssertionError):
            transform.apply(model)
    call = printed_call(directory / "run_slash_link.stdout.log")
    assert call["cwd"] == str(directory)
    args = call["argv"]
    assert args[args.index("--platform") + 1] == ("hw" if hardware else "sim")
    assert len(args[args.index("--kernels") + 1 :]) == 2


@pytest.mark.util
def test_slash_hardware_requires_report_but_simulation_does_not(link_model, monkeypatch):
    model, _ = link_model
    monkeypatch.setenv("FAKE_OUTPUTS", json.dumps({"slashkit": ["finn.vbin"]}))
    with pytest.raises(AssertionError, match="no report generated"):
        alveo_build.SlashLink(build_hardware=True).apply(model)
    alveo_build.SlashLink(build_hardware=False).apply(model)


@pytest.mark.util
def test_driver_utilities_keep_distinct_transcripts_and_fail_at_the_command(
    tmp_path, tool_dir, monkeypatch
):
    tool_dir("git")
    tool_dir("xclbinutil")
    logs = tmp_path / "separate logs"
    logs.mkdir()
    monkeypatch.setattr(make_driver, "make_build_dir", lambda prefix: str(logs))
    driver = MakeCPPDriver("vitis-xrt", "latest")
    driver._run_command(["git", "init"], cwd=str(tmp_path))
    driver._run_command(
        ["git", "remote", "add", "origin", "repo path with spaces"], cwd=str(tmp_path)
    )
    driver._run_command(["xclbinutil", "-i", "bitfile path with spaces"], cwd=str(tmp_path))
    assert printed_call(logs / "git-init.stdout.log")["argv"] == ["init"]
    assert printed_call(logs / "git-remote.stdout.log")["argv"][-1] == "repo path with spaces"
    assert (
        printed_call(tmp_path / "xclbinutil.stdout.log")["argv"][-1] == "bitfile path with spaces"
    )
    monkeypatch.setenv("FAKE_FAIL", "git")
    with pytest.raises(ToolFailed, match="simulated failure"):
        driver._run_command("git fetch origin HEAD", cwd=str(tmp_path))
    assert driver.command_log_dir == str(logs)
    assert not list(tmp_path.glob("git-*"))


@pytest.mark.util
@pytest.mark.parametrize("fail", [False, True])
def test_xsi_setup_can_compile_before_the_extension_exists(tmp_path, tool_dir, monkeypatch, fail):
    tool_dir("g++")
    source = tmp_path / "finn_xsi"
    source.mkdir()
    monkeypatch.setenv("FINN_ROOT", str(tmp_path))
    if fail:
        monkeypatch.setenv("FAKE_FAIL", "g++")
    else:
        monkeypatch.setenv("FAKE_OUTPUTS", json.dumps({"g++": ["xsi.so"]}))
    assert setup.build_xsi(force=True, verbose=False) == (not fail)
    call = printed_call(source / "compile_xsi.stdout.log")
    assert call["cwd"] == str(source)
    assert call["argv"][-2:] == ["-ldl", "-lrt"]


@pytest.mark.util
def test_vitis_driver_config_returns_shapes_and_extracted_kernel_names(
    tmp_path, tool_dir, monkeypatch
):
    layout = {
        "ip_layout": {
            "m_ip_data": [
                {"m_type": "IP_KERNEL", "m_base_address": "0x0", "m_name": "wrong:idma10"},
                {"m_type": "IP_KERNEL", "m_base_address": "0x0", "m_name": "wrong:odma10"},
                {"m_type": "IP_KERNEL", "m_base_address": "0x0", "m_name": "input_kernel:idma1"},
                {
                    "m_type": "IP_KERNEL",
                    "m_base_address": "0x1000",
                    "m_name": "output_kernel:odma1",
                },
            ]
        }
    }
    tool_dir(
        "xclbinutil",
        "from pathlib import Path\n"
        + "Path('ip_layout.json').write_text("
        + repr(json.dumps(layout))
        + ")\n",
    )
    shapes = {
        "idma_names": ["idma1"],
        "odma_names": ["odma1"],
        "ishape_normal": [[1, 4]],
        "oshape_normal": [[1, 4]],
        "ishape_folded": [[1, 1, 4]],
        "oshape_folded": [[1, 1, 4]],
        "ishape_packed": [[1, 1, 16]],
        "oshape_packed": [[1, 1, 16]],
    }
    monkeypatch.setattr(make_driver, "get_driver_shapes", lambda model: shapes)
    model = make_model()
    bitfile = tmp_path / "bitfile with spaces.xclbin"
    model.set_metadata_prop("bitfile", str(bitfile))
    config = MakeCPPDriver("vitis-xrt", "latest")._build_vitis_config(model)
    assert config[0]["xclbinPath"] == str(bitfile)
    assert config[0]["idmas"][0]["kernelName"] == "input_kernel:{idma1}"
    assert config[0]["odmas"][0]["kernelName"] == "output_kernel:{odma1}"
    assert config[0]["idmas"][0]["packedShape"] == [1, 1, 16]
    assert config[0]["odmas"][0]["normalShape"] == [1, 4]


@pytest.mark.util
def test_generated_cpp_driver_excludes_git_diagnostics(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("Git unavailable")
    source = tmp_path / "local driver repository"
    source.mkdir()
    header = source / "src/FINNCppDriver/config/FinnDriverUsedDatatypes.h.in"
    header.parent.mkdir(parents=True)
    header.write_text("using Input = $inputDatatype;\nusing Output = $outputDatatype;\n")
    run_process(["git", "init"], cwd=source)
    run_process(["git", "add", "."], cwd=source)
    run_process(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "Fixture",
        ],
        cwd=source,
    )
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setenv("FINN_BUILD_DIR", str(build))
    monkeypatch.setattr(
        make_driver,
        "get_driver_shapes",
        lambda model: {
            "idt": ["DataType[INT8]"],
            "odt": ["DataType[INT8]"],
            "idma_names": [],
            "odma_names": [],
            "ishape_normal": [],
            "ishape_folded": [],
            "ishape_packed": [],
            "oshape_normal": [],
            "oshape_folded": [],
            "oshape_packed": [],
        },
    )
    driver = MakeCPPDriver("slash-vrt", "latest")
    driver.repository_url = str(source)
    model = make_model()
    model.set_metadata_prop("bitfile", str(tmp_path / "design.vbin"))
    driver.apply(model)
    generated = Path(model.get_metadata_prop("cpp_driver_dir"))
    logs = Path(model.get_metadata_prop("cpp_driver_log_dir"))
    assert logs.parent == generated.parent and logs != generated
    assert len(list(logs.glob("git-*.sh"))) == 5
    assert len(list(logs.glob("git-*.stdout.log"))) == 5
    assert len(list(logs.glob("git-*.stderr.log"))) == 5
    package = tmp_path / "output/driver"
    shutil.copytree(generated, package)
    assert not list(package.glob("git-*"))
    assert json.loads((package / "acceleratorconfig.json").read_text())[0]["bitfile"].endswith(
        "design.vbin"
    )


@pytest.mark.util
def test_cppsim_cancellation_stops_the_tool_and_its_child(tmp_path, tool_dir):
    pids_file = tmp_path / "pids.json"
    tool = tool_dir(
        "node_model",
        f"""
import json, os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open({str(pids_file)!r} + ".tmp", "w") as stream:
    json.dump({{"tool": os.getpid(), "child": child.pid}}, stream)
os.rename({str(pids_file)!r} + ".tmp", {str(pids_file)!r})
time.sleep(120)
""",
    )
    runner = """
import sys
from types import SimpleNamespace
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
attrs = {"executable_path": sys.argv[1], "code_gen_dir_cppsim": sys.argv[2]}
HLSBackend.exec_precompiled_singlenode_model(SimpleNamespace(get_nodeattr=attrs.__getitem__))
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", runner, str(tool), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    pids = None
    try:
        deadline = time.monotonic() + 60
        while not pids_file.exists():
            assert proc.poll() is None, "cppsim runner exited before readiness"
            assert time.monotonic() < deadline, "cppsim tool did not become ready"
            time.sleep(0.02)
        pids = json.loads(pids_file.read_text())
        proc.terminate()
        proc.communicate(timeout=30)
        assert proc.returncode == -signal.SIGTERM
        deadline = time.monotonic() + 10
        for pid in pids.values():
            while True:
                try:
                    state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                    if state == "Z":
                        break
                except (FileNotFoundError, ProcessLookupError):
                    break
                assert time.monotonic() < deadline, f"cppsim process {pid} survived cancellation"
                time.sleep(0.02)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.communicate(timeout=30)
        if pids is not None:
            try:
                os.killpg(pids["tool"], signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.util
@pytest.mark.parametrize("outcome", ["success", "failure", "missing"])
def test_xelab_adapter_checks_status_and_preserves_its_arguments(
    tmp_path, tool_dir, monkeypatch, outcome
):
    tool_dir("xelab")
    # Load the actual adapter source without publishing a fake adapter in the
    # package. Compilation does not need the C extension or simulation engine.
    root = Path(__file__).resolve().parents[2] / "finn_xsi"
    monkeypatch.syspath_prepend(str(root))
    with monkeypatch.context() as modules:
        modules.setitem(sys.modules, "xsi", types.ModuleType("xsi"))
        engine = types.ModuleType("finn_xsi.sim_engine")
        engine.SimEngine = object
        modules.setitem(sys.modules, "finn_xsi.sim_engine", engine)
        spec = importlib.util.spec_from_file_location(
            "toolchain_adapter", root / "finn_xsi" / "adapter.py"
        )
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
    if outcome != "missing":
        monkeypatch.setenv("FAKE_OUTPUTS", json.dumps({"xelab": ["xsim.dir/top/xsimk.so"]}))
    if outcome == "failure":
        monkeypatch.setenv("FAKE_FAIL", "xelab")
    monkeypatch.setenv("FINN_XELAB_MT", "1")
    sources = [str(tmp_path / "mod.sv"), str(tmp_path / "mod_pkg.sv")]
    if outcome == "success":
        assert (
            adapter.compile_sim_obj("top", sources, str(tmp_path), debug=True)[1]
            == "xsim.dir/top/xsimk.so"
        )
    else:
        with pytest.raises(ToolFailed if outcome == "failure" else FileNotFoundError):
            adapter.compile_sim_obj("top", sources, str(tmp_path), debug=True)
    args = printed_call(tmp_path / "compile_xelab.stdout.log")["argv"]
    assert args[args.index("--mt") + 1] == "off"
    assert args[args.index("-debug") + 1] == "all"
    project = (tmp_path / "rtlsim.prj").read_text()
    assert project.index("mod_pkg.sv") < project.index("mod.sv")


@pytest.mark.util
@pytest.mark.parametrize("failure", [None, "compiler", "simulation"])
@pytest.mark.parametrize("in_build", [False, True])
def test_cpp_rtlsim_passes_library_environment_and_rejects_stale_results(
    tmp_path, tool_dir, monkeypatch, capfd, failure, in_build
):
    directory = tmp_path / "simulation with spaces"
    directory.mkdir()
    snapshot = directory / "xsim.dir" / "top" / "xsimk.so"
    snapshot.parent.mkdir(parents=True)
    snapshot.touch()
    wrapper = directory / "top.v"
    wrapper.touch()
    (directory / "results.txt").write_text("invalid stale results")
    runtime = tool_dir(
        "rtlsim_xsi", FAKE_TOOL + '\nPath("results.txt").write_text("CYCLES\\t42\\nTIMEOUT\\t0")\n'
    )
    compiler = """
import json, os, shutil, sys
print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}))
if os.environ.get("FAKE_FAIL") == "g++":
    print("error: compilation failed", file=sys.stderr)
    sys.exit(7)
shutil.copyfile(os.environ["FAKE_RTLEXEC"], "rtlsim_xsi")
os.chmod("rtlsim_xsi", 0o755)
"""
    tool_dir("g++", compiler)
    monkeypatch.setenv("FAKE_RTLEXEC", str(runtime))
    if failure:
        monkeypatch.setenv("FAKE_FAIL", "g++" if failure == "compiler" else "rtlsim_xsi")
    monkeypatch.setenv("XILINX_VIVADO", "/Vivado with spaces/2025.2")
    monkeypatch.setattr(
        rtlsim_exec, "finnxsi", types.SimpleNamespace(get_simkernel_so=lambda: "kernel.so")
    )
    node = helper.make_node(
        "StreamingFIFO_rtl",
        ["inp"],
        ["out"],
        name="fifo",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        depth=2,
        folded_shape=[1, 1, 4],
        normal_shape=[1, 4],
        dataType="INT4",
    )
    model = make_model([node])
    model.set_metadata_prop("wrapper_filename", str(wrapper))
    model.set_metadata_prop("vivado_stitch_proj", str(directory))
    model.set_metadata_prop("rtlsim_so", str(snapshot))
    model.set_metadata_prop(
        "vivado_stitch_ifnames",
        json.dumps({"s_axis": [["in", 16]], "m_axis": [["out", 16]], "aximm": []}),
    )
    build_log = tmp_path / "build.log"
    context = (
        log_run(str(build_log), runner="finn.builder.build_dataflow")
        if in_build
        else contextlib.nullcontext()
    )
    with context:
        if failure:
            with pytest.raises(ToolFailed):
                rtlsim_exec.rtlsim_exec_cppxsi(model, {"inp": 1}, dummy_data_mode=True)
            assert (directory / "results.txt").read_text() == "invalid stale results"
        else:
            assert rtlsim_exec.rtlsim_exec_cppxsi(model, {"inp": 1}, dummy_data_mode=True) == {
                "CYCLES": 42,
                "TIMEOUT": 0,
            }
    console = capfd.readouterr()
    assert "diagnostic from rtlsim_xsi" not in console.out + console.err
    if in_build:
        assert "diagnostic from rtlsim_xsi" not in build_log.read_text()
    if failure != "compiler":
        call = printed_call(directory / "rtlsim_xsi.stdout.log")
        assert call["cwd"] == str(directory)
        assert call["ld_path"] == "/Vivado with spaces/2025.2/lib/lnx64.o"
        assert "LD_LIBRARY_PATH=" in (directory / "run_rtlsim.sh").read_text()
