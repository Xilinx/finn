import pytest

import json
import numpy as np
import sys
from pathlib import Path
from types import SimpleNamespace

onnx = pytest.importorskip("onnx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_examples.tinydeit.common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    EXPORTED_PWPOLYF_SEQUENCE,
    collapse_exported_pwpolyf,
    exported_pwpolyf_match_indices,
    find_mlo_loop_body_ranges,
    find_transformer_blocks,
    mark_attention_multithreshold_layouts_unknown,
    move_forked_scalar_mul_past_matmul,
)


def _tinydeit_build_module():
    pytest.importorskip("qonnx")
    pytest.importorskip("finn")
    from transformer_examples.tinydeit import build as tinydeit_build  # noqa: PLC0415

    return tinydeit_build


def _tinydeit_build_args(mode):
    return SimpleNamespace(
        mode=mode,
        stitched_rtlsim=False,
        clock_ns=5.0,
        board="VCK190",
        target_fps=1000,
        mvau_wwidth_max=10000,
        folding_two_pass_relaxation=False,
        folding_config_file=None,
        atol=1e-1,
        fifosim_n_inferences=1,
        rtlsim_batch_size=2,
        stitched_rtlsim_liveness_threshold=None,
        stitched_ip_dcp=False,
        post_transpose_folding=True,
    )


@pytest.mark.transform
def test_tinydeit_checkpoint_structure():
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("TinyDeiT checkpoint is not present")
    model = onnx.load(str(DEFAULT_CHECKPOINT), load_external_data=False)
    blocks = find_transformer_blocks(model)
    assert len(blocks) == 12
    assert all(len(block["softmax_nodes"]) == 1 for block in blocks)
    matches = exported_pwpolyf_match_indices(model)
    assert len(matches) == 12
    assert matches[0][1] - matches[0][0] + 1 == len(EXPORTED_PWPOLYF_SEQUENCE)


def _make_erf_gelu_model(*, complete=True):
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    shape = [1, 4]
    inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, shape)
    outp = onnx.helper.make_tensor_value_info("outp", onnx.TensorProto.FLOAT, shape)
    sqrt2 = onnx.helper.make_tensor("sqrt2", onnx.TensorProto.FLOAT, [], [np.float32(np.sqrt(2))])
    one = onnx.helper.make_tensor("one", onnx.TensorProto.FLOAT, [], [np.float32(1.0)])
    half = onnx.helper.make_tensor("half", onnx.TensorProto.FLOAT, [], [np.float32(0.5)])
    nodes = [
        onnx.helper.make_node("Div", ["inp", "sqrt2"], ["div_out"], name="Div_0"),
        onnx.helper.make_node("Erf", ["div_out"], ["erf_out"], name="Erf_0"),
    ]
    if complete:
        nodes.extend(
            [
                onnx.helper.make_node("Add", ["erf_out", "one"], ["add_out"], name="Add_0"),
                onnx.helper.make_node("Mul", ["add_out", "half"], ["mul_half_out"], name="Mul_0"),
                onnx.helper.make_node("Mul", ["inp", "mul_half_out"], ["outp"], name="Mul_1"),
            ]
        )
    else:
        nodes.append(onnx.helper.make_node("Identity", ["erf_out"], ["outp"], name="Output"))
    graph = onnx.helper.make_graph(
        nodes,
        "erf_gelu_graph",
        [inp],
        [outp],
        initializer=[sqrt2, one, half],
    )
    return ModelWrapper(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    )


def _make_exported_pwpolyf_model():
    """Build a valid graph containing the legacy exported operator sequence."""
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415

    shape = [1, 4]
    inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, shape)
    outp = onnx.helper.make_tensor_value_info("outp", onnx.TensorProto.FLOAT, shape)
    initializers = [
        onnx.numpy_helper.from_array(np.array(shape, dtype=np.int64), name="shape"),
        onnx.numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name="floats"),
        onnx.numpy_helper.from_array(np.ones(shape, dtype=np.int32), name="ints"),
        onnx.numpy_helper.from_array(np.ones(shape, dtype=np.bool_), name="bools"),
        onnx.numpy_helper.from_array(np.ones(shape, dtype=np.int32), name="shifts"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="gather_indices"),
        onnx.numpy_helper.from_array(np.array([[0]], dtype=np.int64), name="gathernd_indices"),
        onnx.numpy_helper.from_array(np.zeros((81, 3), dtype=np.float32), name="coeffs"),
        onnx.numpy_helper.from_array(np.array(-10.0, dtype=np.float32), name="clip_min"),
        onnx.numpy_helper.from_array(np.array(10.0, dtype=np.float32), name="clip_max"),
    ]
    nodes = []
    for idx, op_type in enumerate(EXPORTED_PWPOLYF_SEQUENCE):
        output_name = "outp" if idx == len(EXPORTED_PWPOLYF_SEQUENCE) - 1 else f"tmp_{idx}"
        kwargs = {}
        if op_type == "Reshape":
            inputs = ["inp", "shape"]
        elif op_type == "Cast":
            inputs = ["inp"]
            kwargs["to"] = onnx.TensorProto.FLOAT
        elif op_type in {"Less", "Equal", "GreaterOrEqual"}:
            inputs = ["inp", "floats"]
        elif op_type == "BitShift":
            inputs = ["ints", "shifts"]
            kwargs["direction"] = "RIGHT"
        elif op_type in {"BitwiseOr", "BitwiseAnd"}:
            inputs = ["ints", "ints"]
        elif op_type == "Where":
            inputs = ["bools", "inp", "floats"]
        elif op_type in {"Sub", "Max", "Mul", "Add"}:
            inputs = ["inp", "floats"]
        elif op_type == "And":
            inputs = ["bools", "bools"]
        elif op_type == "Clip":
            inputs = ["inp", "clip_min", "clip_max"]
        elif op_type == "Unsqueeze":
            inputs = ["inp", "axes"]
        elif op_type == "GatherND":
            inputs = ["coeffs", "gathernd_indices"]
        elif op_type == "Gather":
            inputs = ["inp", "gather_indices"]
            kwargs["axis"] = 1
        else:
            raise AssertionError(f"Unhandled exported PWPolyF operator {op_type}")
        nodes.append(
            onnx.helper.make_node(
                op_type,
                inputs,
                [output_name],
                name=f"{op_type}_{idx}",
                **kwargs,
            )
        )
    graph = onnx.helper.make_graph(
        nodes,
        "exported_pwpolyf_graph",
        [inp],
        [outp],
        initializer=initializers,
    )
    return ModelWrapper(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    )


@pytest.mark.transform
def test_tinydeit_collapse_accepts_exported_polynomial():
    pytest.importorskip("qonnx")
    model = _make_exported_pwpolyf_model()

    model, count = collapse_exported_pwpolyf(model, expected_count=1)

    assert count == 1
    assert [node.op_type for node in model.graph.node] == ["PWPolyF"]


@pytest.mark.transform
def test_tinydeit_collapse_accepts_erf_gelu_export():
    pytest.importorskip("qonnx")
    model = _make_erf_gelu_model()

    model, count = collapse_exported_pwpolyf(model, expected_count=1)

    assert count == 1
    assert [node.op_type for node in model.graph.node] == ["PWPolyF"]


@pytest.mark.transform
def test_tinydeit_collapse_rejects_partial_erf_gelu_export():
    pytest.importorskip("qonnx")
    model = _make_erf_gelu_model(complete=False)

    with pytest.raises(RuntimeError, match="converted only 0 complete GELU"):
        collapse_exported_pwpolyf(model, expected_count=1)


@pytest.mark.transform
def test_tinydeit_collapse_rejects_mixed_exports(monkeypatch):
    pytest.importorskip("qonnx")
    from transformer_examples.tinydeit import common  # noqa: PLC0415

    model = _make_erf_gelu_model()
    monkeypatch.setattr(common, "exported_pwpolyf_match_indices", lambda _: [(0, 50)])

    with pytest.raises(RuntimeError, match="mixes polynomial PWPolyF and Erf GELU"):
        collapse_exported_pwpolyf(model, expected_count=1)


@pytest.mark.transform
def test_tinydeit_moves_forked_scalar_mul_past_matmul():
    pytest.importorskip("qonnx")
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
    from qonnx.core.onnx_exec import execute_onnx  # noqa: PLC0415
    from qonnx.transformation.infer_shapes import InferShapes  # noqa: PLC0415

    inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 4])
    out0 = onnx.helper.make_tensor_value_info("out0", onnx.TensorProto.FLOAT, [1, 4])
    out1 = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [1, 4])
    scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [1], [np.float32(0.25)])
    w0 = onnx.numpy_helper.from_array(np.arange(16, dtype=np.float32).reshape(4, 4), name="w0")
    w1 = onnx.numpy_helper.from_array(
        np.flip(np.arange(16, dtype=np.float32).reshape(4, 4), axis=1).copy(), name="w1"
    )
    nodes = [
        onnx.helper.make_node("Mul", ["inp", "scale"], ["scaled"], name="scale_mul"),
        onnx.helper.make_node("MatMul", ["scaled", "w0"], ["out0"], name="mm0"),
        onnx.helper.make_node("MatMul", ["scaled", "w1"], ["out1"], name="mm1"),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "forked_scalar_mul_graph",
        [inp],
        [out0, out1],
        initializer=[scale, w0, w1],
    )
    model = ModelWrapper(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    ).transform(InferShapes())
    input_dict = {"inp": np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)}
    expected = execute_onnx(model, input_dict)

    model, moved = move_forked_scalar_mul_past_matmul(model)
    actual = execute_onnx(model, input_dict)

    assert moved == 1
    assert all(node.name != "scale_mul" for node in model.graph.node)
    matmuls = [node for node in model.graph.node if node.op_type == "MatMul"]
    assert len(matmuls) == 2
    assert all(node.input[0] == "inp" for node in matmuls)
    assert model.find_producer("out0").op_type == "Mul"
    assert model.find_producer("out1").op_type == "Mul"
    np.testing.assert_allclose(actual["out0"], expected["out0"])
    np.testing.assert_allclose(actual["out1"], expected["out1"])


@pytest.mark.transform
def test_tinydeit_marks_only_attention_multithreshold_layouts_unknown():
    pytest.importorskip("qonnx")
    from qonnx.core.modelwrapper import ModelWrapper  # noqa: PLC0415
    from qonnx.custom_op.registry import getCustomOp  # noqa: PLC0415

    shape = [1, 3, 7, 4]
    nodes = [
        onnx.helper.make_node(
            "MultiThreshold",
            ["global_in", "thresholds"],
            ["image_quantized"],
            name="image_threshold",
            domain="qonnx.custom_op.general",
            out_dtype="INT3",
            data_layout="NCHW",
        )
    ]
    tensor_name = "image_quantized"
    attention_nodes = []
    for block in range(2):
        nodes.append(
            onnx.helper.make_node(
                "LayerNormalization", [tensor_name], [f"b{block}_ln0"], name=f"b{block}_ln0"
            )
        )
        tensor_name = f"b{block}_ln0"
        for threshold in range(5):
            output_name = f"b{block}_attention_threshold_{threshold}_out"
            node = onnx.helper.make_node(
                "MultiThreshold",
                [tensor_name, "thresholds"],
                [output_name],
                name=f"b{block}_attention_threshold_{threshold}",
                domain="qonnx.custom_op.general",
                out_dtype="INT3",
                data_layout="NCHW",
            )
            nodes.append(node)
            attention_nodes.append(node.name)
            tensor_name = output_name
        nodes.append(
            onnx.helper.make_node(
                "LayerNormalization", [tensor_name], [f"b{block}_ln1"], name=f"b{block}_ln1"
            )
        )
        tensor_name = f"b{block}_ln1"
    nodes.append(
        onnx.helper.make_node(
            "LayerNormalization", [tensor_name], ["global_out"], name="post_stack_ln"
        )
    )
    graph = onnx.helper.make_graph(
        nodes,
        "tinydeit_attention_layouts",
        [onnx.helper.make_tensor_value_info("global_in", onnx.TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("global_out", onnx.TensorProto.FLOAT, shape)],
        initializer=[
            onnx.numpy_helper.from_array(np.zeros((1, 7), dtype=np.float32), name="thresholds")
        ],
    )
    model = ModelWrapper(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    )
    for node in nodes:
        for output_name in node.output:
            model.set_tensor_shape(output_name, shape)

    model, normalized = mark_attention_multithreshold_layouts_unknown(model, depth=2)

    assert normalized == 10
    assert getCustomOp(model.graph.node[0]).get_nodeattr("data_layout") == "NCHW"
    by_name = {node.name: node for node in model.graph.node}
    assert all(
        getCustomOp(by_name[node_name]).get_nodeattr("data_layout") == "UNKNOWN"
        for node_name in attention_nodes
    )


@pytest.mark.transform
def test_tinydeit_loop_body_range_excludes_trailing_duplicate():
    nodes = [
        onnx.helper.make_node("InputPrep", ["global_in"], ["prep"], name="prep"),
        onnx.helper.make_node(
            "DuplicateStreams_hls", ["prep"], ["b0_ln0_in", "b0_skip"], name="dup0"
        ),
        onnx.helper.make_node("LayerNorm_rtl", ["b0_ln0_in"], ["b0_ln0"], name="ln0"),
        onnx.helper.make_node("ElementwiseAdd_rtl", ["b0_ln0", "p0"], ["b0_mid"], name="mid0"),
        onnx.helper.make_node("LayerNorm_rtl", ["b0_mid"], ["b0_ln1"], name="ln1"),
        onnx.helper.make_node("ElementwiseAdd_rtl", ["b0_ln1", "b0_skip"], ["b0_out"], name="add0"),
        onnx.helper.make_node(
            "DuplicateStreams_hls", ["b0_out"], ["b1_ln0_in", "b1_skip"], name="dup1"
        ),
        onnx.helper.make_node("LayerNorm_rtl", ["b1_ln0_in"], ["b1_ln0"], name="ln2"),
        onnx.helper.make_node("ElementwiseAdd_rtl", ["b1_ln0", "p1"], ["b1_mid"], name="mid1"),
        onnx.helper.make_node("LayerNorm_rtl", ["b1_mid"], ["b1_ln1"], name="ln3"),
        onnx.helper.make_node("ElementwiseAdd_rtl", ["b1_ln1", "b1_skip"], ["b1_out"], name="add1"),
        onnx.helper.make_node("LayerNorm_rtl", ["b1_out"], ["global_out"], name="post_ln"),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "tinydeit_loop_ranges",
        [onnx.helper.make_tensor_value_info("global_in", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("global_out", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(graph)

    ranges = find_mlo_loop_body_ranges(model, depth=2)

    assert [(item["loop_start_node"], item["loop_end_node"]) for item in ranges] == [
        ("dup0", "add0"),
        ("dup1", "add1"),
    ]
    assert ranges[0]["loop_op_types"] == ranges[1]["loop_op_types"]


@pytest.mark.transform
def test_tinydeit_rtlsim_results_file_validation(tmp_path):
    tinydeit_build = _tinydeit_build_module()

    missing_interval = tmp_path / "missing_interval.txt"
    missing_interval.write_text("cycles 100\n")
    with pytest.raises(ValueError, match="interval_cycles"):
        tinydeit_build._resolve_existing_rtlsim_results_file(str(missing_interval))

    results = tmp_path / "results.txt"
    results.write_text(
        "\n".join(
            [
                "cycles 125",
                "latency_cycles 124",
                "interval_cycles 100",
                "interval_valid 1",
                "completed_output_frames 2",
                "steady_state_frames 1",
                "steady_state_cycles 100",
                "N 2",
                "TIMEOUT 0",
                "UNFINISHED_INS 0",
                "UNFINISHED_OUTS 0",
                "RUNTIME_S 7",
                "",
            ]
        )
    )
    resolved = tinydeit_build._resolve_existing_rtlsim_results_file(str(results))
    summary = tinydeit_build._rtlsim_summary_from_path(resolved, clock_ns=5.0)

    assert resolved == results
    assert summary["rtlsim_interval_cycles"] == 100
    assert summary["rtlsim_timeout"] == 0
    assert summary["rtlsim_throughput_fps"] == 2_000_000.0

    single_frame = tmp_path / "single_frame.txt"
    single_frame.write_text(
        "\n".join(
            [
                "cycles 125",
                "latency_cycles 124",
                "interval_cycles 100",
                "N 1",
                "TIMEOUT 0",
                "UNFINISHED_INS 0",
                "UNFINISHED_OUTS 0",
                "",
            ]
        )
    )
    single_summary = tinydeit_build._rtlsim_summary_from_path(single_frame, clock_ns=5.0)
    assert single_summary["rtlsim_interval_valid"] is False
    assert single_summary["rtlsim_stable_throughput_valid"] is False
    assert "rtlsim_throughput_fps" not in single_summary


@pytest.mark.transform
def test_tinydeit_uses_only_valid_stitched_performance_report(tmp_path):
    tinydeit_build = _tinydeit_build_module()
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    report_path = report_dir / "rtlsim_performance.json"
    report_path.write_text(
        json.dumps(
            {
                "cycles": 500,
                "latency_cycles": 300,
                "interval_cycles": 200,
                "completed_output_frames": 2,
                "steady_state_frames": 1,
                "steady_state_cycles": 200,
                "stable_throughput_valid": True,
                "stable_throughput[images/s]": 1_000_000.0,
                "measurement_scope": "stitched_mlo",
                "external_memory_model": "ideal_axi_mm",
                "performance_interpretation": "ideal_memory_upper_bound",
            }
        )
    )

    summary = tinydeit_build._rtlsim_performance_summary(tmp_path)
    assert summary["rtlsim_stable_throughput_valid"] is True
    assert summary["rtlsim_throughput_fps"] == 1_000_000.0
    assert summary["rtlsim_completed_output_frames"] == 2
    assert summary["rtlsim_measurement_scope"] == "stitched_mlo"
    assert summary["rtlsim_external_memory_model"] == "ideal_axi_mm"

    invalid_report = json.loads(report_path.read_text())
    invalid_report["stable_throughput_valid"] = False
    report_path.write_text(json.dumps(invalid_report))
    invalid_summary = tinydeit_build._rtlsim_performance_summary(tmp_path)
    assert invalid_summary["rtlsim_stable_throughput_valid"] is False
    assert "rtlsim_throughput_fps" not in invalid_summary


@pytest.mark.transform
def test_tinydeit_preserves_fresh_rtlsim_artifacts(tmp_path, monkeypatch):
    tinydeit_build = _tinydeit_build_module()

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    finn_build_dir = tmp_path / "finn-build"
    rtlsim_dir = finn_build_dir / "rtlsim_finn_design_wrapper_abc"
    rtlsim_dir.mkdir(parents=True)
    (rtlsim_dir / "results.txt").write_text(
        "\n".join(
            [
                "cycles 100",
                "latency_cycles 50",
                "interval_cycles 50",
                "interval_valid 1",
                "completed_output_frames 2",
                "steady_state_frames 1",
                "steady_state_cycles 50",
                "N 2",
                "TIMEOUT 0",
                "UNFINISHED_INS 0",
                "UNFINISHED_OUTS 0",
                "",
            ]
        )
    )
    (rtlsim_dir / "rtlsim_xsi_log.txt").write_text("completed\n")
    monkeypatch.setenv("FINN_BUILD_DIR", str(finn_build_dir))

    preserved = tinydeit_build._preserve_rtlsim_artifacts(output_dir)

    assert preserved == output_dir / "rtlsim_fifo_sizing" / "results.txt"
    assert "interval_cycles 50" in preserved.read_text()
    assert (output_dir / "rtlsim_fifo_sizing" / "rtlsim_xsi_log.txt").read_text() == "completed\n"
    summary = tinydeit_build._rtlsim_summary(output_dir, clock_ns=4.0, search_build_dir=False)
    assert summary["rtlsim_results_path"] == str(preserved.resolve())
    assert summary["rtlsim_throughput_fps"] == 5_000_000.0


@pytest.mark.transform
def test_tinydeit_dcp_validation_requires_fresh_clean_artifacts(tmp_path):
    tinydeit_build = _tinydeit_build_module()

    missing = tinydeit_build._dcp_validation_summary(tmp_path)
    assert missing["dcp_validation_status"] == "failed"
    assert set(missing["dcp_validation_errors"]) == {
        "missing_stitched_ip_dir",
        "missing_top_level_synth_dcp",
        "missing_top_level_routed_dcp",
        "missing_timing_report",
    }

    stitched_ip = tmp_path / "stitched_ip"
    stitched_ip.mkdir()
    (stitched_ip / "finn_design.dcp").write_text("synth dcp\n")
    (stitched_ip / "finn_design_routed.dcp").write_text("routed dcp\n")
    (stitched_ip / "ooc_timing.rpt").write_text("timing\n")
    (stitched_ip / "vivado.log").write_text("INFO: clean\n")

    clean = tinydeit_build._dcp_validation_summary(tmp_path)
    assert clean["dcp_validation_status"] == "passed"
    assert clean["dcp_synth_dcp_count"] == 1
    assert clean["dcp_routed_dcp_count"] == 1
    assert clean["dcp_timing_report_count"] == 1

    error_dir = stitched_ip / "runs" / "synth_1"
    error_dir.mkdir(parents=True)
    (error_dir / ".vivado.error.rst").write_text("")
    (error_dir / "runme.log").write_text(
        "ERROR: [Common 17-345] A valid license was not found for feature "
        "'Synthesis' and/or device 'xcvc1902'\n"
    )

    failed = tinydeit_build._dcp_validation_summary(tmp_path)
    assert failed["dcp_validation_status"] == "failed"
    assert "vivado_error_rst" in failed["dcp_validation_errors"]
    assert "vivado_license_error" in failed["dcp_validation_errors"]
    assert failed["vivado_license_error"] is True


@pytest.mark.transform
def test_tinydeit_vivado_license_preflight_records_license_failure(tmp_path, monkeypatch):
    tinydeit_build = _tinydeit_build_module()

    run_calls = []

    class FakeCompletedProcess:
        returncode = 1
        stdout = (
            "ERROR: [Common 17-345] A valid license was not found for feature "
            "'Synthesis' and/or device 'xcvc1902'\n"
        )

    def fake_run(cmd, **kwargs):
        run_calls.append((cmd, kwargs))
        return FakeCompletedProcess()

    monkeypatch.setenv("XILINXD_LICENSE_FILE", "2100@aiengine-eng")
    monkeypatch.setattr(tinydeit_build, "_vivado_executable", lambda: "/bin/true")
    monkeypatch.setattr(tinydeit_build.subprocess, "run", fake_run)

    args = SimpleNamespace(
        board="VCK190",
        vivado_license_preflight_timeout_s=42,
    )
    summary = tinydeit_build._run_vivado_license_preflight(args, tmp_path)

    assert len(run_calls) == 1
    assert "-source" in run_calls[0][0]
    assert run_calls[0][1]["timeout"] == 42
    assert summary["vivado_license_preflight_status"] == "failed"
    assert summary["vivado_license_preflight_failure"] == "license_error"
    assert summary["vivado_license_preflight_license_error"] is True
    assert summary["vivado_license_preflight_vivado_error"] is True
    assert summary["vivado_license_preflight_part"] == "xcvc1902-vsva2197-2MP-e-S"
    assert summary["vivado_license_preflight_xilinxd_license_file"] == "2100@aiengine-eng"

    log_path = Path(summary["vivado_license_preflight_log_path"])
    assert "A valid license was not found" in log_path.read_text()
    summary_path = tmp_path / "vivado_license_preflight.json"
    assert json.loads(summary_path.read_text()) == summary


@pytest.mark.transform
@pytest.mark.parametrize("mode", ["estimate", "rtl", "dcp", "full-rtlsim", "bitfile"])
def test_tinydeit_build_config_uses_phases_and_injections(tmp_path, mode):
    tinydeit_build = _tinydeit_build_module()

    cfg = tinydeit_build.build_config(_tinydeit_build_args(mode), tmp_path)

    if mode == "estimate":
        assert cfg.steps == ["phase_optimize_hardware"]
    else:
        assert cfg.steps == [
            "phase_optimize_hardware",
            "phase_build_hardware",
            "phase_generate_outputs",
        ]
    assert cfg.inject_steps_after == {
        "phase_optimize_hardware": [tinydeit_build.step_tinydeit_snapshot_folding_config],
        "step_minimize_bit_width": [tinydeit_build.step_round_mlo_threshold_params],
        "step_transpose_decomposition": [
            tinydeit_build.step_tinydeit_post_transpose_parallelization,
            tinydeit_build.step_tinydeit_hls_lut_mvaus,
        ],
    }
    assert cfg.fifosim_n_inferences == 1
    assert cfg.rtlsim_use_vivado_comps is (mode != "full-rtlsim")
    assert cfg.mlo is False


@pytest.mark.transform
def test_tinydeit_converts_lut_mvaus_to_hls(tmp_path, monkeypatch):
    tinydeit_build = _tinydeit_build_module()
    extracted_config = {}

    def record_extracted_config(model, path, attrs):
        extracted_config.update(model=model, path=path, attrs=attrs)

    monkeypatch.setattr(tinydeit_build, "extract_model_config_to_json", record_extracted_config)

    nodes = [
        onnx.helper.make_node(
            "MVAU_rtl",
            ["inp", "weights"],
            ["lut_out"],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            name="lut_mvau",
            resType="lut",
            pumpedCompute=1,
            mem_mode="external",
            mlo_max_iter=12,
            code_gen_dir_ipgen="/stale/lut/codegen",
            ipgen_path="/stale/lut/ipgen",
            ip_path="/stale/lut/ip",
            gen_top_module="stale_lut_top",
        ),
        onnx.helper.make_node(
            "MVAU_rtl",
            ["lut_out", "weights"],
            ["out"],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            name="dsp_mvau",
            resType="dsp",
            pumpedCompute=1,
            mem_mode="dynamic",
            mlo_max_iter=12,
            code_gen_dir_ipgen="/stale/dsp/codegen",
            ipgen_path="/stale/dsp/ipgen",
            ip_path="/stale/dsp/ip",
            gen_top_module="stale_dsp_top",
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "lut_mvau_conversion",
        [onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1])],
    )
    model = tinydeit_build.ModelWrapper(onnx.helper.make_model(graph))

    cfg = SimpleNamespace(
        output_dir=str(tmp_path), folding_config_file="original_folding_config.json"
    )
    model = tinydeit_build.step_tinydeit_hls_lut_mvaus(model, cfg)
    model = tinydeit_build.step_tinydeit_snapshot_folding_config(model, cfg)

    assert [node.op_type for node in model.graph.node] == ["MVAU_hls", "MVAU_rtl"]
    assert [node.name for node in model.graph.node] == ["MVAU_hls_0", "MVAU_rtl_0"]
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow.hls"
    assert (
        tinydeit_build.getHWCustomOp(model.graph.node[0], model).get_nodeattr("mem_mode")
        == "external_mem"
    )
    assert (
        tinydeit_build.getHWCustomOp(model.graph.node[1], model).get_nodeattr("mem_mode")
        == "external_mem"
    )
    assert {attr.name for attr in model.graph.node[0].attribute}.isdisjoint(
        {
            "code_gen_dir_ipgen",
            "gen_top_module",
            "ip_path",
            "ipgen_path",
            "pumpedCompute",
        }
    )
    assert {attr.name for attr in model.graph.node[1].attribute}.isdisjoint(
        {"code_gen_dir_ipgen", "gen_top_module", "ip_path", "ipgen_path"}
    )
    normalized_config = str(tmp_path / "auto_folding_config.json")
    assert cfg.folding_config_file == normalized_config
    assert extracted_config == {
        "model": model,
        "path": normalized_config,
        "attrs": tinydeit_build.FOLDING_HW_ATTRS,
    }


@pytest.mark.transform
def test_tinydeit_rejects_incompatible_folding_before_build(tmp_path):
    tinydeit_build = _tinydeit_build_module()

    inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 197, 197, 3])
    outp = onnx.helper.make_tensor_value_info("outp", onnx.TensorProto.FLOAT, [1, 197, 197, 3])
    thresholds = onnx.numpy_helper.from_array(np.zeros((3, 7), dtype=np.float32), name="thresholds")
    node = onnx.helper.make_node(
        "Thresholding_rtl",
        ["inp", "thresholds"],
        ["outp"],
        name="Thresholding_rtl_0",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=3,
        PE=1,
        inputDataType="INT3",
        weightDataType="INT4",
        outputDataType="UINT3",
        numInputVectors=[1, 197, 197],
        numSteps=7,
    )
    graph = onnx.helper.make_graph(
        [node], "folding_compatibility", [inp], [outp], initializer=[thresholds]
    )
    model = tinydeit_build.ModelWrapper(onnx.helper.make_model(graph))
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps({"Thresholding_rtl_0": {"PE": 197}}))

    with pytest.raises(ValueError, match="NumChannels=3 is not divisible by PE=197"):
        tinydeit_build.validate_folding_config_compatibility(model, incompatible)

    compatible = tmp_path / "compatible.json"
    compatible.write_text(json.dumps({"Thresholding_rtl_0": {"PE": 3}}))
    tinydeit_build.validate_folding_config_compatibility(model, compatible)

    shuffle_inp = onnx.helper.make_tensor_value_info(
        "shuffle_inp", onnx.TensorProto.FLOAT, [1, 14, 192, 14]
    )
    shuffle_out = onnx.helper.make_tensor_value_info(
        "shuffle_out", onnx.TensorProto.FLOAT, [1, 192, 14, 14]
    )
    shuffle = onnx.helper.make_node(
        "OuterShuffle_hls",
        ["shuffle_inp"],
        ["shuffle_out"],
        name="OuterShuffle_hls_0",
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        data_type="INT3",
        in_shape=[1, 14, 192, 14],
        out_shape=[1, 192, 14, 14],
        transpose_in_shape=[1, 14, 192, 14],
        transpose_out_shape=[1, 192, 14, 14],
        loop_coeffs=[37632, 14, 2688, 1],
        perm=[0, 2, 1, 3],
        SIMD=1,
        NumChannels=14,
    )
    shuffle_graph = onnx.helper.make_graph(
        [shuffle], "shuffle_folding_compatibility", [shuffle_inp], [shuffle_out]
    )
    shuffle_model = tinydeit_build.ModelWrapper(onnx.helper.make_model(shuffle_graph))
    incompatible_shuffle = tmp_path / "incompatible_shuffle.json"
    incompatible_shuffle.write_text(json.dumps({"OuterShuffle_hls_0": {"SIMD": 28}}))

    with pytest.raises(ValueError, match="Unable to determine a new SIMD value"):
        tinydeit_build.validate_folding_config_compatibility(shuffle_model, incompatible_shuffle)

    compatible_shuffle = tmp_path / "compatible_shuffle.json"
    compatible_shuffle.write_text(json.dumps({"OuterShuffle_hls_0": {"SIMD": 7}}))
    tinydeit_build.validate_folding_config_compatibility(shuffle_model, compatible_shuffle)


@pytest.mark.transform
def test_tinydeit_vck190_folding_configs():
    example_dir = Path(__file__).resolve().parents[1] / "transformer_examples" / "tinydeit"
    config_dir = example_dir / "configs"
    config_paths = sorted(config_dir.glob("*.json"))

    assert [path.name for path in config_paths] == [
        "w3a3_vck190_250mhz.json",
        "w3a3_vck190_performance.json",
        "w4a4_vck190_250mhz.json",
        "w4a4_vck190_performance.json",
    ]
    assert not (example_dir / "results").exists()

    for config_path in config_paths:
        config = json.loads(config_path.read_text())
        assert "AddCLSToken_rtl_0" not in config
        assert len(config) > 1
        assert all(isinstance(node_config, dict) for node_config in config.values())
        if "performance" in config_path.stem:
            assert config["Defaults"] == {"pumpedCompute": [1, ["MVAU_rtl"]]}
            assert config["Pad1D_rtl_0"] == {"SIMD": 3}
            loop_prefix = "FINNLoop_0_body_FINNLoop_0"
            assert config[f"{loop_prefix}_InnerShuffle_rtl_0"] == {"SIMD": 197}
            assert config[f"{loop_prefix}_HWSoftmax_rtl_0"] == {"SIMD": 197}

            for index in range(4):
                assert config[f"{loop_prefix}_OuterShuffle_hls_{index}"] == {"SIMD": 8}
            for index in range(2):
                assert config[f"{loop_prefix}_LayerNorm_rtl_{index}"] == {"SIMD": 4}

            if config_path.stem.startswith("w3a3"):
                assert config[f"{loop_prefix}_MVAU_rtl_7"] == {
                    "PE": 3,
                    "SIMD": 768,
                    "mem_mode": "external",
                    "resType": "lut",
                }
            else:
                assert config[f"{loop_prefix}_MVAU_rtl_3"] == {
                    "PE": 197,
                    "SIMD": 8,
                    "mem_mode": "dynamic",
                    "resType": "lut",
                }
                assert config[f"{loop_prefix}_MVAU_rtl_4"] == {
                    "PE": 4,
                    "SIMD": 197,
                    "mem_mode": "dynamic",
                    "resType": "lut",
                }
                assert config[f"{loop_prefix}_MVAU_rtl_5"] == {
                    "PE": 3,
                    "SIMD": 192,
                    "mem_mode": "internal_decoupled",
                    "resType": "lut",
                }
                assert config[f"{loop_prefix}_MVAU_rtl_7"] == {
                    "PE": 3,
                    "SIMD": 768,
                    "mem_mode": "external",
                    "resType": "dsp",
                }
        else:
            assert config["Defaults"] == {}
            assert config["Pad1D_rtl_0"] == {"SIMD": 1}
