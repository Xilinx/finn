# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from types import SimpleNamespace

import finn.builder.build_dataflow_steps as steps
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.transformation.general import ApplyConfig


class _FakeMLOModel:
    def transform(self, _transformation):
        return self

    def analysis(self, _analysis):
        return {"critical_path_cycles": 100}

    def get_nodes_by_op_type(self, op_type):
        assert op_type == "FINNLoop"
        return [object()]


class _FakeFIFOModel:
    def __init__(self):
        self.fifo = object()
        self.transforms = []

    def get_nodes_by_op_type(self, op_type):
        assert op_type == "StreamingFIFO_rtl"
        return [self.fifo]

    def transform(self, transformation):
        self.transforms.append(type(transformation).__name__)
        return self


class _FakeFIFOInstance:
    def __init__(self):
        self.attrs = {
            "impl_style": "vivado",
            "code_gen_dir_ipgen": "old_codegen",
            "ipgen_path": "old_ip",
        }

    def get_nodeattr(self, name):
        return self.attrs[name]

    def set_nodeattr(self, name, value):
        self.attrs[name] = value


class _FakeMLOFoldingModel:
    def __init__(self):
        self.graph = SimpleNamespace(node=[SimpleNamespace(op_type="FINNLoop")])
        self.transforms = []

    def transform(self, transformation, **_kwargs):
        self.transforms.append(transformation)
        return self

    def get_nodes_by_op_type(self, _op_type):
        return []


class _FakeLoopBodyModel:
    def __init__(self, loop_context):
        self.loop_context = loop_context
        self.transforms = []

    def get_metadata_prop(self, name):
        assert name == "loop_context"
        return self.loop_context

    def transform(self, transformation):
        self.transforms.append(transformation)
        return self


def test_mlo_fifo_step_preserves_generated_loop_body(tmp_path, monkeypatch):
    """Late top-level FIFO configuration must not recurse into FINNLoop IP."""
    model = _FakeMLOFoldingModel()
    folding_config = tmp_path / "folding.json"
    folding_config.write_text("{}")
    monkeypatch.setattr(steps, "extract_model_config_to_json", lambda *_args: None)
    monkeypatch.setattr(steps, "extract_model_config_consolidate_shuffles", lambda *_args: None)
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path),
        synth_clk_period_ns=5.0,
        board="VCK190",
        auto_fifo_depths=False,
        folding_config_file=str(folding_config),
        generate_outputs=[],
    )

    returned = steps.step_set_fifo_depths(model, cfg)

    assert returned is model
    applied_configs = [x for x in model.transforms if isinstance(x, ApplyConfig)]
    assert len(applied_configs) == 1
    assert applied_configs[0].recurse_subgraphs is False


def test_mlo_performance_step_uses_two_frames_and_ideal_memory(tmp_path, monkeypatch):
    model = _FakeMLOModel()
    prehook = object()
    call = {}

    monkeypatch.setattr(steps, "is_mlo", lambda _model: True)
    monkeypatch.setattr(steps, "deepcopy", lambda original: original)
    monkeypatch.setattr(steps, "prepare_for_stitched_ip_rtlsim", lambda original, _cfg: original)
    monkeypatch.setattr(steps, "mlo_prehook_func_factory", lambda _node: prehook)
    monkeypatch.setattr(steps, "get_liveness_threshold_cycles", lambda: 123)

    def fake_throughput_test(model_arg, clk_ns, **kwargs):
        call.update(
            model=model_arg,
            clk_ns=clk_ns,
            liveness_threshold=os.environ["LIVENESS_THRESHOLD"],
            **kwargs,
        )
        return {
            "N": kwargs["batchsize"],
            "completed_output_frames": 2,
            "interval_valid": 1,
            "steady_state_frames": 1,
            "steady_state_cycles": 50,
            "stable_throughput_valid": True,
        }

    monkeypatch.setattr(steps, "throughput_test_rtlsim", fake_throughput_test)

    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path),
        synth_clk_period_ns=5.0,
        rtlsim_batch_size=1,
        stitched_rtlsim_liveness_threshold=777,
        mlo=True,
        generate_outputs=[
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )

    returned = steps.step_measure_rtlsim_performance(model, cfg)

    assert returned is model
    assert call["model"] is model
    assert call["clk_ns"] == 5.0
    assert call["batchsize"] == 2
    assert call["pre_hook"] is prehook
    assert call["collect_performance"] is True
    assert call["liveness_threshold"] == "777"

    with open(tmp_path / "report" / "rtlsim_performance.json") as report_file:
        report = json.load(report_file)
    assert report["measurement_scope"] == "stitched_mlo"
    assert report["external_memory_model"] == "ideal_axi_mm"
    assert report["external_memory_model_is_ideal"] is True
    assert report["performance_interpretation"] == "ideal_memory_upper_bound"
    assert report["io_bandwidth_scope"] == "top_level_axi_stream_only"
    assert report["N"] == 2


def test_mlo_loop_body_replaces_vivado_fifos_for_rtlsim(tmp_path, monkeypatch):
    model = _FakeFIFOModel()
    fifo_instance = _FakeFIFOInstance()
    monkeypatch.setattr(steps, "getHWCustomOp", lambda _node, _model: fifo_instance)
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path),
        synth_clk_period_ns=5.0,
        board="VCK190",
        rtlsim_use_vivado_comps=False,
        generate_outputs=[],
    )

    returned = steps.step_loop_body_ipgen_and_stitch(model, cfg)

    assert returned is model
    assert fifo_instance.attrs == {
        "impl_style": "rtl",
        "code_gen_dir_ipgen": "",
        "ipgen_path": "",
    }
    assert model.transforms == ["PrepareIP", "HLSSynthIP", "CreateStitchedIP"]


def test_mlo_loop_body_names_hls_before_fifo_characterization(tmp_path, monkeypatch):
    """Nested HLS modules must be namespaced before their RTL is generated."""
    model = _FakeLoopBodyModel("FINNLoop_0")
    insert_call = {}

    class _FakeInsertAndSetFIFODepths:
        pass

    def fake_insert_and_set_fifo_depths(*args, **kwargs):
        insert_call["args"] = args
        insert_call["kwargs"] = kwargs
        return _FakeInsertAndSetFIFODepths()

    monkeypatch.setattr(
        steps,
        "InsertAndSetFIFODepths",
        fake_insert_and_set_fifo_depths,
    )
    monkeypatch.setattr(steps, "snapshot_fifo_logs", lambda *_args, **_kwargs: None)
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path),
        synth_clk_period_ns=5.0,
        board="VCK190",
        generate_outputs=[],
    )

    returned = steps.step_loop_body_set_fifo_depths(model, cfg)

    assert returned is model
    assert model.transforms[0].prefix == "FINNLoop_0_"
    assert insert_call["kwargs"]["node_name_prefix"] == "FINNLoop_0_"
