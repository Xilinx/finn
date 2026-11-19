# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage of the actual tool adapters, without FPGA tools."""

import pytest

import json
import os
import sys
import types
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.core import rtlsim_exec
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.transformation.fpgadataflow import alveo_build, make_driver
from finn.transformation.fpgadataflow.make_driver import MakeCPPDriver
from finn.util.basic import CppBuilder
from finn.util.launcher import ToolFailed

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
def test_xo_packaging_rejects_a_failed_vivado_run_with_a_stale_xo(tmp_path, tool_dir, monkeypatch):
    tool_dir("vivado")
    for name in ("VITIS_PATH", "PLATFORM_REPO_PATHS", "XILINX_XRT"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.delenv("PWD", raising=False)  # the scripts used to read it
    directory = tmp_path / "xo with spaces"
    model = kernel_model(directory)
    (directory / "design.xo").write_text("stale")
    monkeypatch.setenv("FAKE_FAIL", "vivado")
    with pytest.raises(ToolFailed):
        alveo_build.CreateVitisXO("design").apply(model)


@pytest.mark.util
@pytest.mark.parametrize("failing", ["v++", "vivado"])
def test_vitis_link_rejects_failed_tools_and_keeps_its_arguments(link_model, monkeypatch, failing):
    model, directory = link_model
    monkeypatch.setenv(
        "FAKE_OUTPUTS", json.dumps({"v++": ["a.xclbin"], "vivado": ["synth_report.xml"]})
    )
    monkeypatch.setenv("FAKE_FAIL", failing)
    (directory / "a.xclbin").write_text("stale")
    transform = alveo_build.VitisLink("/platform path/u55c.xpfm", 5, enable_debug=True)
    with pytest.raises(ToolFailed):
        transform.apply(model)
    # A failed link leaves no bitfile; a failed report leaves no report path.
    failed_prop = "bitfile" if failing == "v++" else "vivado_synth_rpt"
    assert model.get_metadata_prop(failed_prop) is None
    # The command was a shell string before: paths with spaces and lists must stay intact.
    args = printed_call(directory / "run_vitis_link.stdout.log")["argv"]
    assert args[args.index("--platform") + 1] == "/platform path/u55c.xpfm"
    assert len([arg for arg in args if arg.endswith("kernel.xo")]) == 2
    assert args.count("--dk") == 2


@pytest.mark.util
def test_slash_link_rejects_a_failed_link_with_a_stale_vbin(link_model, monkeypatch):
    model, directory = link_model
    monkeypatch.setenv("FAKE_FAIL", "slashkit")
    (directory / "finn.vbin").write_text("stale")
    with pytest.raises(ToolFailed):
        alveo_build.SlashLink(build_hardware=True).apply(model)
    assert model.get_metadata_prop("bitfile") is None


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
@pytest.mark.parametrize("failure", [None, "compiler", "simulation"])
def test_cpp_rtlsim_passes_library_environment_and_rejects_stale_results(
    tmp_path, tool_dir, monkeypatch, failure
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
    if failure:
        with pytest.raises(ToolFailed):
            rtlsim_exec.rtlsim_exec_cppxsi(model, {"inp": 1}, dummy_data_mode=True)
        assert (directory / "results.txt").read_text() == "invalid stale results"
    else:
        assert rtlsim_exec.rtlsim_exec_cppxsi(model, {"inp": 1}, dummy_data_mode=True) == {
            "CYCLES": 42,
            "TIMEOUT": 0,
        }
    if failure != "compiler":
        call = printed_call(directory / "rtlsim_xsi.stdout.log")
        assert call["cwd"] == str(directory)
        assert call["ld_path"] == "/Vivado with spaces/2025.2/lib/lnx64.o"
        assert "LD_LIBRARY_PATH=" in (directory / "run_rtlsim.sh").read_text()
