import pytest

import json
import sys
from pathlib import Path
from types import SimpleNamespace

onnx = pytest.importorskip("onnx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_examples.tinydeit.common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    EXPORTED_PWPOLYF_SEQUENCE,
    exported_pwpolyf_match_indices,
    find_mlo_loop_body_ranges,
    find_transformer_blocks,
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
def test_tinydeit_vck190_configs_and_signoff_evidence():
    example_dir = Path(__file__).resolve().parents[1] / "transformer_examples" / "tinydeit"
    evidence = json.loads((example_dir / "results" / "vck190_signoff.json").read_text())

    assert evidence["schema_version"] == 1
    measurements = {item["name"]: item for item in evidence["measurements"]}
    assert set(measurements) == {
        "w3a3_vck190_300mhz_10k",
        "w4a4_qat_smoke_vck190_200mhz_7k",
    }

    for measurement in measurements.values():
        config = json.loads((example_dir / measurement["configuration"]).read_text())
        assert config["Defaults"]["pumpedCompute"] == [1, ["MVAU_rtl"]]
        assert measurement["target"]["board"] == "VCK190"
        assert measurement["target"]["base_clock_period_ns"] == 8.334
        assert measurement["target"]["double_pumped_clock_period_ns"] == 4.167
        assert measurement["timing"]["no_clock_endpoints"] == 0
        assert measurement["timing"]["unconstrained_internal_endpoints"] == 0
        assert measurement["timing"]["constraints_met"] is True
        assert (
            measurement["routing"]["fully_routed_nets"] == measurement["routing"]["routable_nets"]
        )
        assert measurement["routing"]["routing_error_nets"] == 0
        assert measurement["routing"]["passed"] is True
        assert measurement["rtlsim"] == {
            "status": "not_available",
            "measurement_requirement": "completed_multi_frame_stitched_mlo",
            "external_memory_model": "ideal_axi_mm",
            "performance_interpretation": "ideal_memory_upper_bound",
            "historical_single_frame_result_rejected": True,
            "interval_cycles": None,
            "throughput_fps": None,
            "passed": None,
        }
        rejected_hash = measurement["artifact_sha256"]["rejected_single_frame_rtlsim_results"]
        assert len(rejected_hash) == 64

    w3a3 = measurements["w3a3_vck190_300mhz_10k"]
    assert w3a3["quantization"]["audit_passed"] is True
    assert w3a3["quantization"]["bitwidth_counts"] == {"3": 170, "8": 2}
    assert w3a3["accuracy"]["status"] == "not_available"

    w4a4 = measurements["w4a4_qat_smoke_vck190_200mhz_7k"]
    assert w4a4["accuracy"] == {
        "status": "smoke_checkpoint_only",
        "validation_subset": "tiny",
        "top1_percent": 0.0,
        "top5_percent": 12.5,
        "quality_claim": False,
    }
