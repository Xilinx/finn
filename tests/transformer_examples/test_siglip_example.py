# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import numpy as np
import onnx
import qonnx.core.onnx_exec as oxe
import sys
from onnx import TensorProto, helper, numpy_helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_examples.siglip.build import (  # noqa: E402
    _verification_steps,
    build_siglip,
)
from transformer_examples.siglip.config import (  # noqa: E402
    DEFAULT_PROFILE,
    load_profile,
)
from transformer_examples.siglip.mlo import (  # noqa: E402
    find_vision_loop_body_ranges,
    make_mlo_boundary_step,
    step_round_siglip_thresholds_before_mlo,
    step_size_siglip_loop_residual_fifos,
    step_size_siglip_top_residual_fifo,
)
from transformer_examples.siglip.phases import (  # noqa: E402
    _absorb_pre_matmul_dequant,
    _DuplicateSafeModelWrapper,
    _fold_quantized_matmul_into_multithreshold,
    _integerize_static_lhs_matmul,
    _select_graph_output,
    make_siglip_folding_step,
)


def _node(op_type, index):
    return SimpleNamespace(op_type=op_type, name=f"{op_type}_{index}", input=[], output=[])


def _repeated_encoder(depth=12, mismatch=None):
    nodes = []
    for block in range(depth):
        nodes.extend(
            [
                _node("LayerNorm_rtl", 4 * block),
                _node("MVAU_rtl", 4 * block + 1),
                _node("LayerNorm_rtl", 4 * block + 2),
                _node("PWPolyF_rtl" if block == mismatch else "MVAU_rtl", 4 * block + 3),
            ]
        )
    nodes.append(_node("LayerNorm_rtl", 4 * depth))
    return SimpleNamespace(graph=SimpleNamespace(node=nodes))


def test_default_profile_records_verified_w6a7_contract():
    profile = load_profile(DEFAULT_PROFILE)
    assert profile.model["model_id"] == "google/siglip2-base-patch16-224"
    assert profile.model["vision_depth"] == 12
    assert profile.quantization == {
        "scheme": "qv_lsq",
        "weight_bits": 6,
        "activation_bits": 7,
        "edge_bits": 8,
    }
    assert profile.model["output_name"] == "image_embeds"
    assert profile.reference_metrics["accuracy"]["images"] == 50_000
    assert profile.reference_metrics["finn_latency_model"] is None
    ooc = profile.reference_metrics["ooc_implementation"]
    assert ooc["part"] == "xcvc1902-vsva2197-2MP-e-S"
    assert ooc["vivado_version"] == "2024.2"
    assert ooc["clock_period_ns"] == 3.999
    assert ooc["wns_ns"] == 0.037
    assert ooc["fmax_mhz"] == 252.39777889954567
    assert ooc["resources"] == {
        "LUT": 163595,
        "FF": 285187,
        "DSP": 911,
        "BRAM_36K": 796,
        "BRAM_18K": 89,
        "URAM": 459,
        "SRL": 31019,
    }
    assert profile.reference_metrics["board_runtime_throughput_fps"] is None
    assert profile.reference_metrics["ideal_memory_rtlsim_throughput_fps"] is None
    assert profile.reference_metrics.get("stitched_rtlsim_max_absolute_error") is None
    assert profile.build["verification_atol"] == 0.27
    assert profile.build["fifo_depth_cap"] == 32
    assert profile.resolve_file(profile.build["folding_config"]).is_file()

    specialization_path = profile.resolve_file(profile.build["specialization_config"])
    folding_path = profile.resolve_file(profile.build["folding_config"])
    specialization = json.loads(specialization_path.read_text())
    folding = json.loads(folding_path.read_text())
    assert "DuplicateStreams" in specialization["Defaults"]["preferred_impl_style"][1]
    assert any("DuplicateStreams_rtl" in key for key in folding)
    assert not any("DuplicateStreams_hls" in key for key in folding)
    assert folding["ElementwiseAdd_rtl_1"]["ram_style"] == "block"
    assert folding["MVAU_rtl_0"]["PE"] == 28
    assert folding["MVAU_rtl_0"]["SIMD"] == 8
    assert folding["MVAU_rtl_1"]["PE"] == 32
    assert folding["MVAU_rtl_1"]["SIMD"] == 7
    top_outer_shuffles = {key for key in folding if key.startswith("OuterShuffle_hls_")}
    top_inner_shuffles = {key for key in folding if key.startswith("InnerShuffle_rtl_")}
    loop_outer_prefix = "FINNLoop_0_body_FINNLoop_0_OuterShuffle_hls_"
    loop_inner_prefix = "FINNLoop_0_body_FINNLoop_0_InnerShuffle_rtl_"
    assert top_outer_shuffles == {f"OuterShuffle_hls_{index}" for index in range(5)}
    assert top_inner_shuffles == {f"InnerShuffle_rtl_{index}" for index in range(4)}
    assert [folding[f"OuterShuffle_hls_{index}"]["SIMD"] for index in range(5)] == [
        8,
        2,
        2,
        4,
        1,
    ]
    assert [folding[f"InnerShuffle_rtl_{index}"]["SIMD"] for index in range(4)] == [
        3,
        2,
        2,
        2,
    ]
    assert {key for key in folding if key.startswith(loop_outer_prefix)} == {
        f"{loop_outer_prefix}{index}" for index in range(8)
    }
    assert {key for key in folding if key.startswith(loop_inner_prefix)} == {
        f"{loop_inner_prefix}{index}" for index in range(5)
    }
    assert [folding[f"{loop_outer_prefix}{index}"]["SIMD"] for index in range(8)] == [
        16,
        16,
        16,
        4,
        4,
        4,
        1,
        1,
    ]
    assert [folding[f"{loop_inner_prefix}{index}"]["SIMD"] for index in range(5)] == [
        4,
        4,
        4,
        4,
        1,
    ]
    for name in ("MVAU_hls_0", "MVAU_hls_1", "MVAU_hls_2"):
        assert folding[name]["PE"] == 1
        assert folding[name]["weight_buffer_count"] == 1
    for index in (0, 1, 2, 5, 6, 7):
        settings = folding[f"FINNLoop_0_body_FINNLoop_0_MVAU_rtl_{index}"]
        assert settings["PE"] == 16
        assert settings["SIMD"] == 12
        assert settings["TH"] == 4
        assert settings["mem_mode"] == "external_mem"
        assert settings["pumpedCompute"] == 0
    assert folding["MVAU_rtl_3"]["ram_style"] == "distributed"
    for index in (6, 7):
        assert folding[f"FINNLoop_0_body_FINNLoop_0_MVAU_rtl_{index}"]["ram_style"] == "block"
    for index in (0, 1):
        key = f"FINNLoop_0_body_FINNLoop_0_LayerNorm_rtl_{index}"
        assert folding[key]["numRsqrtRefinements"] == 2
    assert folding["FINNLoop_0_body_FINNLoop_0_PWPolyF_rtl_0"]["K"] == 2
    for op_type in ("ElementwiseAdd", "ElementwiseMul"):
        key = f"FINNLoop_0_body_FINNLoop_0_{op_type}_rtl_7"
        assert folding[key]["ram_style"] == "distributed"
    for index in (4, 7):
        key = f"FINNLoop_0_body_FINNLoop_0_Thresholding_rtl_{index}"
        assert folding[key]["depth_trigger_bram"] == 1_000_000


def test_profile_rejects_unvalidated_board(tmp_path):
    text = DEFAULT_PROFILE.read_text().replace('"board": "VCK190"', '"board": "U250"')
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(text)
    with pytest.raises(ValueError, match="board=VCK190"):
        load_profile(profile_path)


def test_final_folding_stage_rejects_stale_node_names(tmp_path):
    node = SimpleNamespace(
        name="MVAU_rtl_0",
        op_type="MVAU_rtl",
        domain="finn.custom_op.fpgadataflow.rtl",
        attribute=[],
    )
    model = SimpleNamespace(graph=SimpleNamespace(node=[node]))
    config_path = tmp_path / "folding.json"
    config_path.write_text('{"Defaults": {}, "old_MVAU_rtl_0": {"PE": 1}}')

    folding_step = make_siglip_folding_step(config_path, require_complete=True)
    with pytest.raises(RuntimeError, match="folding profile does not match"):
        folding_step(model, SimpleNamespace())


def test_readable_names_preserve_repeated_siglip_gelu_input():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("Relu", ["x"], ["hidden"]),
        helper.make_node("Mul", ["hidden", "hidden"], ["y"]),
    ]
    model = _DuplicateSafeModelWrapper(
        helper.make_model(helper.make_graph(nodes, "gelu", [x], [y]))
    )
    model = (
        model.transform(InferShapes())
        .transform(GiveUniqueNodeNames())
        .transform(GiveReadableTensorNames())
    )

    mul = model.get_nodes_by_op_type("Mul")[0]
    assert mul.input[0] == mul.input[1]
    assert model.find_producer(mul.input[0]).op_type == "Relu"
    onnx.checker.check_model(model.model)


def test_estimate_mode_rejects_simulators_it_does_not_build():
    for level in ("cppsim", "rtlsim"):
        with pytest.raises(ValueError, match="requires stitched_ip or ooc_synth"):
            _verification_steps(level, "estimate")


def test_build_uses_explicit_top_level_fifo_sizing(monkeypatch, tmp_path):
    captured = {}

    def capture_config(model_path, cfg):
        captured["model_path"] = model_path
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(
        "transformer_examples.siglip.build.build.build_dataflow_cfg",
        capture_config,
    )
    model_path = tmp_path / "siglip.onnx"
    status = build_siglip(
        model_path,
        tmp_path / "output",
        load_profile(DEFAULT_PROFILE),
        "stitched_ip",
        "none",
        None,
        None,
        False,
    )

    assert status == 0
    assert captured["model_path"] == str(model_path)
    cfg = captured["cfg"]
    assert cfg.auto_fifo_depths is False
    assert cfg.verify_rtlsim_behavioral is False
    assert cfg.inject_steps_before["step_set_fifo_depths"] == [step_size_siglip_top_residual_fifo]
    assert cfg.inject_steps_before["step_hw_codegen"] == [step_size_siglip_loop_residual_fifos]


def test_selects_embedding_output_and_removes_comparison_branch():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    class_scores = helper.make_tensor_value_info("class_scores", TensorProto.FLOAT, [1, 4])
    image_embeds = helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("Identity", ["x"], ["image_embeds"]),
        helper.make_node("Relu", ["image_embeds"], ["class_scores"]),
    ]
    model = _DuplicateSafeModelWrapper(
        helper.make_model(helper.make_graph(nodes, "outputs", [x], [class_scores, image_embeds]))
    )

    model = _select_graph_output(model, "image_embeds")
    assert [value_info.name for value_info in model.graph.output] == ["image_embeds"]
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Identity"


def test_moves_scalar_dequant_after_static_weight_matmul():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    x_scaled = helper.make_tensor_value_info("x_scaled", TensorProto.FLOAT, [1, 4])
    mm_out = helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["x", "act_scale"], ["x_scaled"], name="dequant"),
            helper.make_node("MatMul", ["x_scaled", "weight"], ["mm_out"], name="proj"),
            helper.make_node("Mul", ["mm_out", "weight_scale"], ["y"]),
        ],
        "pre-matmul-dequant",
        [x],
        [y],
        value_info=[x_scaled, mm_out],
        initializer=[
            numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), "act_scale"),
            numpy_helper.from_array(
                np.asarray(
                    [[1, 0, -1], [0, 1, 1], [1, -1, 0], [-1, 0, 1]],
                    dtype=np.float32,
                ),
                "weight",
            ),
            numpy_helper.from_array(np.asarray([0.5, 0.75, 1.0], dtype=np.float32), "weight_scale"),
        ],
    )
    model = ModelWrapper(helper.make_model(graph))
    model.set_tensor_datatype("x", DataType["INT4"])
    model.set_tensor_datatype("x_scaled", DataType["FLOAT32"])
    model.set_tensor_datatype("weight", DataType["INT4"])
    model.set_tensor_datatype("mm_out", DataType["FLOAT32"])

    model = _absorb_pre_matmul_dequant(model)

    matmul = [node for node in model.graph.node if node.name == "proj"][0]
    assert list(matmul.input) == ["x", "weight"]
    assert model.get_tensor_datatype("mm_out") == DataType["INT32"]
    np.testing.assert_array_equal(
        model.get_initializer("weight_scale"),
        np.asarray([0.125, 0.1875, 0.25], dtype=np.float32),
    )
    assert "dequant" not in [node.name for node in model.graph.node]


@pytest.mark.parametrize("head_scale", [np.float32(0.5), None])
def test_folds_quantized_projection_into_integer_thresholds(head_scale):
    codes = np.arange(-5, 6, dtype=np.float32).reshape(-1, 1)
    codes = np.concatenate([codes, -codes], axis=1)
    weights = np.eye(2, dtype=np.float32)
    activation_scale = np.float32(0.5)
    weight_scale = np.asarray([0.5, 0.25], dtype=np.float32)
    bias = np.asarray([0.25, -0.125], dtype=np.float32)
    effective_head_scale = np.float32(1.0) if head_scale is None else head_scale
    output_scale = np.float32(0.25)
    output_bias = np.float32(-4.0)

    lower_levels = np.arange(-4, 3, dtype=np.int64)
    thresholds = ((lower_levels.astype(np.float64) + 0.5) * output_scale).astype(np.float32)
    lower_even = (lower_levels % 2) == 0
    thresholds[lower_even] = np.nextafter(thresholds[lower_even], np.float32(np.inf))
    thresholds = thresholds.reshape(1, -1)

    raw = helper.make_tensor_value_info("raw", TensorProto.FLOAT, list(codes.shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [codes.shape[0], 2, 1])
    nodes = [
        helper.make_node(
            "MultiThreshold",
            ["raw", "input_thresholds"],
            ["input_count"],
            name="input_quantize",
            domain="qonnx.custom_op.general",
            out_dtype="UINT4",
            data_layout="NHWC",
        ),
        helper.make_node("Add", ["input_count", "input_bias"], ["x"], name="input_bias"),
        helper.make_node("Mul", ["x", "activation_scale"], ["x_scaled"], name="dequant"),
        helper.make_node("MatMul", ["x_scaled", "weights"], ["acc"], name="projection"),
        helper.make_node("Mul", ["acc", "weight_scale"], ["weighted"], name="weight_scale"),
        helper.make_node("Add", ["weighted", "bias"], ["biased"], name="bias"),
        helper.make_node("Reshape", ["biased", "shape"], ["reshaped"], name="reshape"),
        helper.make_node(
            "Transpose", ["reshaped"], ["transposed"], name="transpose", perm=[0, 2, 1]
        ),
    ]
    multithreshold_input = "transposed"
    if head_scale is not None:
        nodes.append(
            helper.make_node("Mul", ["transposed", "head_scale"], ["scaled"], name="head_scale")
        )
        multithreshold_input = "scaled"
    nodes.extend(
        [
            helper.make_node(
                "MultiThreshold",
                [multithreshold_input, "thresholds"],
                ["quant_count"],
                name="quantize",
                domain="qonnx.custom_op.general",
                out_dtype="UINT3",
            ),
            helper.make_node(
                "Add", ["quant_count", "output_bias"], ["quant_code"], name="quant_bias"
            ),
            helper.make_node("Mul", ["quant_code", "output_scale"], ["y"], name="quant_scale"),
        ]
    )
    initializers = [
        numpy_helper.from_array(
            np.arange(-4.5, 5.0, dtype=np.float32).reshape(1, -1), "input_thresholds"
        ),
        numpy_helper.from_array(np.asarray(-5.0, dtype=np.float32), "input_bias"),
        numpy_helper.from_array(np.asarray(activation_scale), "activation_scale"),
        numpy_helper.from_array(weights, "weights"),
        numpy_helper.from_array(weight_scale.reshape(1, -1), "weight_scale"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(np.asarray([codes.shape[0], 1, 2], dtype=np.int64), "shape"),
        numpy_helper.from_array(thresholds, "thresholds"),
        numpy_helper.from_array(np.asarray(output_bias), "output_bias"),
        numpy_helper.from_array(np.asarray(output_scale), "output_scale"),
    ]
    if head_scale is not None:
        initializers.append(numpy_helper.from_array(np.asarray(head_scale), "head_scale"))
    graph = helper.make_graph(nodes, "quantized-projection", [raw], [y], initializer=initializers)
    model = ModelWrapper(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    model.set_tensor_datatype("weights", DataType["INT2"])
    model = model.transform(InferShapes()).transform(InferDataTypes())

    model = _fold_quantized_matmul_into_multithreshold(model)

    projection = next(node for node in model.graph.node if node.name == "projection")
    quantize = next(node for node in model.graph.node if node.name == "quantize")
    reshape = next(node for node in model.graph.node if node.name == "reshape")
    transpose = next(node for node in model.graph.node if node.name == "transpose")
    assert list(projection.input) == ["x", "weights"]
    assert quantize.input[0] == "acc"
    assert reshape.input[0] == quantize.output[0]
    assert transpose.output[0] == "quant_count"
    assert getCustomOp(quantize).get_nodeattr("data_layout") == "NHWC"
    assert not {"dequant", "weight_scale", "bias", "head_scale"}.intersection(
        node.name for node in model.graph.node
    )

    accumulator_boundaries = (
        (lower_levels.astype(np.float64) + 0.5) * output_scale
        - bias.reshape(-1, 1) * effective_head_scale
    ) / (activation_scale * weight_scale.reshape(-1, 1) * effective_head_scale)
    expected_thresholds = np.ceil(accumulator_boundaries)
    expected_thresholds[:, lower_even] = np.floor(accumulator_boundaries[:, lower_even]) + 1
    np.testing.assert_array_equal(
        model.get_initializer(quantize.input[1]), expected_thresholds.astype(np.float32)
    )

    accumulator = codes.astype(np.int64) @ weights.astype(np.int64)
    real = (
        accumulator * activation_scale * weight_scale.reshape(1, -1) + bias.reshape(1, -1)
    ) * effective_head_scale
    expected = np.clip(np.rint(real / output_scale), -4, 3) * output_scale
    expected = expected.reshape(codes.shape[0], 1, 2).transpose(0, 2, 1)
    actual = oxe.execute_onnx(model, {"raw": codes})["y"]
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_integerizes_exact_static_attention_query_once():
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [1, 2, 4, 3])
    mm_out = helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 2, 1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1, 3])
    lhs = np.asarray(
        [[[[0.5, -0.5, 0.0, 0.75]], [[-0.5, 0.5, 0.75, 0.0]]]],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["lhs", "rhs"], ["mm_out"], name="pool_scores"),
            helper.make_node("Mul", ["mm_out", "score_scale"], ["y"]),
        ],
        "static-lhs-matmul",
        [rhs],
        [y],
        value_info=[mm_out],
        initializer=[
            numpy_helper.from_array(lhs, "lhs"),
            numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), "score_scale"),
        ],
    )
    model = ModelWrapper(helper.make_model(graph))
    model.set_tensor_datatype("lhs", DataType["FLOAT32"])
    model.set_tensor_datatype("rhs", DataType["INT4"])
    model.set_tensor_datatype("mm_out", DataType["FLOAT32"])

    model = _integerize_static_lhs_matmul(model)
    integer_lhs = model.get_initializer("lhs").copy()
    integer_scale = model.get_initializer("score_scale").copy()

    np.testing.assert_array_equal(integer_lhs * integer_scale, lhs * 0.5)
    assert np.min(integer_lhs) >= -128
    assert np.max(integer_lhs) <= 127
    assert model.get_tensor_datatype("lhs") == DataType["INT8"]
    assert model.get_tensor_datatype("mm_out") == DataType["INT32"]

    model = _integerize_static_lhs_matmul(model)
    np.testing.assert_array_equal(model.get_initializer("lhs"), integer_lhs)
    np.testing.assert_array_equal(model.get_initializer("score_scale"), integer_scale)


def test_detects_all_repeated_vision_blocks():
    ranges = find_vision_loop_body_ranges(_repeated_encoder(), depth=12)
    assert len(ranges) == 12
    assert ranges[0]["op_types"] == ranges[-1]["op_types"]
    assert ranges[0]["start_node"] == "LayerNorm_rtl_0"
    assert ranges[-1]["end_node"] == "MVAU_rtl_47"


def test_exposes_topology_mismatch_to_mlo_step(tmp_path):
    model = _repeated_encoder(mismatch=7)
    cfg = SimpleNamespace(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match=r"blocks \[7\]"):
        make_mlo_boundary_step(12)(model, cfg)


def test_rounds_integer_thresholds_before_mlo_parameter_extraction():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])
    thresholds = np.asarray([[-1.2, 0.2, 1.8]], dtype=np.float32)
    node = helper.make_node(
        "Thresholding",
        ["inp", "thresholds"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="FLOAT32",
        outputDataType="UINT2",
        numInputVectors=[1],
        numSteps=3,
    )
    graph = helper.make_graph(
        [node],
        "round-before-mlo",
        [inp],
        [out],
        initializer=[numpy_helper.from_array(thresholds, "thresholds")],
    )
    model = ModelWrapper(helper.make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["FLOAT32"])
    model.set_tensor_datatype("out", DataType["UINT2"])

    model = step_round_siglip_thresholds_before_mlo(model, SimpleNamespace())

    np.testing.assert_array_equal(
        model.get_initializer("thresholds"),
        np.asarray([[-1.0, 1.0, 2.0]], dtype=np.float32),
    )
    assert model.get_tensor_datatype("thresholds").is_integer()


def test_sizes_post_loop_layernorm_residual_for_a_complete_frame(monkeypatch):
    fork = _node("DuplicateStreams_rtl", 0)
    fork.input = ["x"]
    fork.output = ["norm_input", "bypass"]
    norm = _node("LayerNorm_rtl", 0)
    norm.input = ["norm_input"]
    norm.output = ["normalized"]
    compute = _node("MVAU_rtl", 0)
    compute.input = ["normalized"]
    compute.output = ["branch"]
    join = _node("ElementwiseAdd_rtl", 0)
    join.input = ["bypass", "branch"]
    join.output = ["y"]

    class FakeInstance:
        def __init__(self, attrs, folded_input_shape=None):
            self.attrs = attrs
            self.folded_input_shape = folded_input_shape

        def get_nodeattr(self, name):
            return self.attrs[name]

        def set_nodeattr(self, name, value):
            self.attrs[name] = value

        def get_folded_input_shape(self, _index):
            return self.folded_input_shape

    instances = {
        fork.name: FakeInstance({"outFIFODepths": [2, 2]}),
        norm.name: FakeInstance({}, (1, 1, 768, 1)),
        join.name: FakeInstance(
            {"lhs_style": "input", "rhs_style": "input", "inFIFODepths": [2, 2]}
        ),
    }

    class FakeModel:
        def __init__(self, nodes):
            self.nodes = nodes

        def get_nodes_by_op_type(self, op_type):
            return [node for node in self.nodes if node.op_type == op_type]

        def find_consumer(self, tensor_name):
            return next(
                (node for node in self.nodes if tensor_name in node.input),
                None,
            )

        def find_consumers(self, tensor_name):
            return [node for node in self.nodes if tensor_name in node.input]

    monkeypatch.setattr(
        "transformer_examples.siglip.mlo.getCustomOp",
        lambda node: instances[node.name],
    )
    cfg = SimpleNamespace(auto_fifo_depths=False, fifo_depth_cap=32)

    model = FakeModel([fork, norm, compute, join])
    assert step_size_siglip_top_residual_fifo(model, cfg) is model
    assert instances[fork.name].attrs["outFIFODepths"] == [2, 1024]
    assert instances[join.name].attrs["inFIFODepths"] == [1024, 2]
    assert cfg.fifo_depth_cap == 1024


def test_restores_full_frame_depth_for_loop_body_residual_fifos(monkeypatch):
    nodes = []
    instances = {}

    class FakeInstance:
        def __init__(self, attrs, folded_input_shape=None):
            self.attrs = attrs
            self.folded_input_shape = folded_input_shape

        def get_nodeattr(self, name):
            return self.attrs[name]

        def set_nodeattr(self, name, value):
            self.attrs[name] = value

        def get_folded_input_shape(self, _index):
            return self.folded_input_shape

    for index in range(2):
        fork = _node("DuplicateStreams_rtl", index)
        fork.input = [f"x_{index}"]
        fork.output = [f"norm_fifo_in_{index}", f"bypass_fifo_in_{index}"]
        norm_fifo = _node("StreamingFIFO_rtl", 2 * index)
        norm_fifo.input = [fork.output[0]]
        norm_fifo.output = [f"norm_input_{index}"]
        norm = _node("LayerNorm_rtl", index)
        norm.input = [norm_fifo.output[0]]
        norm.output = [f"normalized_{index}"]
        compute = _node("MVAU_rtl", index)
        compute.input = [norm.output[0]]
        compute.output = [f"branch_{index}"]
        bypass_fifo = _node("StreamingFIFO_rtl", 2 * index + 1)
        bypass_fifo.input = [fork.output[1]]
        bypass_fifo.output = [f"bypass_{index}"]
        join = _node("ElementwiseAdd_rtl", index)
        join.input = [bypass_fifo.output[0], compute.output[0]]
        join.output = [f"y_{index}"]
        nodes.extend([fork, norm_fifo, norm, compute, bypass_fifo, join])
        instances[fork.name] = FakeInstance({"outFIFODepths": [32, 32]})
        instances[bypass_fifo.name] = FakeInstance({"depth": 32}, (1, 196, 768, 1))
        instances[join.name] = FakeInstance(
            {"lhs_style": "input", "rhs_style": "input", "inFIFODepths": [32, 2]}
        )

    class FakeModel:
        def __init__(self, graph_nodes):
            self.nodes = graph_nodes
            self.graph = SimpleNamespace(node=graph_nodes)

        def get_nodes_by_op_type(self, op_type):
            return [node for node in self.nodes if node.op_type == op_type]

        def find_consumer(self, tensor_name):
            return next(
                (node for node in self.nodes if tensor_name in node.input),
                None,
            )

        def find_producer(self, tensor_name):
            return next(
                (node for node in self.nodes if tensor_name in node.output),
                None,
            )

        def find_consumers(self, tensor_name):
            return [node for node in self.nodes if tensor_name in node.input]

    body = FakeModel(nodes)
    loop = _node("FINNLoop", 0)
    top = FakeModel([loop])
    instances[loop.name] = FakeInstance({"body": body})
    monkeypatch.setattr(
        "transformer_examples.siglip.mlo.getCustomOp",
        lambda node: instances[node.name],
    )

    assert step_size_siglip_loop_residual_fifos(top, SimpleNamespace()) is top
    for index in range(2):
        bypass_fifo = instances[f"StreamingFIFO_rtl_{2 * index + 1}"]
        assert bypass_fifo.attrs["depth"] == 150_528
        assert instances[f"DuplicateStreams_rtl_{index}"].attrs["outFIFODepths"] == [
            32,
            150_528,
        ]
        assert instances[f"ElementwiseAdd_rtl_{index}"].attrs["inFIFODepths"] == [
            150_528,
            2,
        ]
