import pytest

import sys
from pathlib import Path

onnx = pytest.importorskip("onnx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinydeit.common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    EXPORTED_PWPOLYF_SEQUENCE,
    exported_pwpolyf_match_indices,
    find_mlo_loop_body_ranges,
    find_transformer_blocks,
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
