# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferSelectTokenLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.test import tree_model_test

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10
NUM_TOKENS = 4
NUM_CHANNELS = 4


def make_selecttoken_modelwrapper(
    token_index, idt, num_tokens=NUM_TOKENS, num_channels=NUM_CHANNELS
):
    indices = np.asarray(token_index, dtype=np.int64)
    output_shape = [1, num_channels] if indices.ndim == 0 else [1, len(indices), num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, num_tokens, num_channels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, output_shape)
    gather = helper.make_node("Gather", ["inp", "indices"], ["outp"], axis=1)
    graph = helper.make_graph(
        [gather],
        "selecttoken-model",
        [inp],
        [outp],
        [numpy_helper.from_array(indices, name="indices")],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="selecttoken-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    return model


def prepare_selecttoken_stitched_ip_model(model):
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    return model.transform(CreateStitchedIP(FPGA_PART, CLK_NS))


# Token selection and SIMD are coupled to cover positive/negative indices and
# all channel-folding extremes without an unnecessarily large Cartesian matrix.
@pytest.mark.parametrize(
    "config",
    [
        pytest.param((2, 1), id="token2-simd1"),
        pytest.param((-1, 2), id="token-last-simd2"),
        pytest.param((0, 4), id="token0-simd4"),
    ],
)
@pytest.mark.parametrize(
    "idt",
    [
        pytest.param(DataType["INT8"], id="INT8"),
        pytest.param(DataType["UINT4"], id="UINT4"),
        pytest.param(DataType["INT6"], id="INT6"),
    ],
)
@pytest.mark.parametrize("exec_mode", ["rtlsim", "stitched_rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_selecttoken(config, idt, exec_mode):
    token_index, simd = config
    input_tensor = gen_finn_dt_tensor(idt, (1, NUM_TOKENS, NUM_CHANNELS))
    input_dict = {"inp": input_tensor}
    y_expected = input_tensor[:, token_index, :]
    model = make_selecttoken_modelwrapper(token_index, idt)

    # Golden reference from the original Gather graph.
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all(), "Execution of Gather model failed"

    model = model.transform(InferSelectTokenLayer())
    selecttoken_nodes = model.get_nodes_by_op_type("SelectToken")
    assert len(selecttoken_nodes) == 1
    selecttoken = getCustomOp(selecttoken_nodes[0])
    assert selecttoken.get_nodeattr("NumTokens") == NUM_TOKENS
    assert selecttoken.get_nodeattr("NumChannels") == NUM_CHANNELS
    assert selecttoken.get_nodeattr("TokenIndex") == token_index

    # Check the inferred hardware-independent node before specialization.
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all(), "Execution of inferred SelectToken failed"

    selecttoken.set_nodeattr("SIMD", simd)
    model = model.transform(SpecializeLayers(FPGA_PART))
    assert len(model.get_nodes_by_op_type("SelectToken_rtl")) == 1

    if exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    elif exec_mode == "stitched_rtlsim":
        model = prepare_selecttoken_stitched_ip_model(model)
        model.set_metadata_prop("exec_mode", "rtlsim")
    else:
        raise ValueError("Unknown exec_mode")

    y_produced = oxe.execute_onnx(model, input_dict)["outp"].reshape(y_expected.shape)
    assert (y_produced == y_expected).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("SelectToken_rtl")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0


@pytest.mark.transform
def test_infer_selecttoken_layer_rejects_nonscalar_gather():
    indices = [1, 2]
    input_tensor = np.arange(16, dtype=np.float32).reshape(1, NUM_TOKENS, NUM_CHANNELS)
    input_dict = {"inp": input_tensor}
    model = make_selecttoken_modelwrapper(indices, DataType["INT8"])
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    model = model.transform(InferSelectTokenLayer())
    assert model.graph.node[0].op_type == "Gather"

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert np.array_equal(y_produced, y_expected)


# (NumTokens, NumChannels, TokenIndex, SIMD, idt): first, middle and last token,
# CF from 1 to 8, and a single token, where the core's width counter degenerates.
SELECTTOKEN_TREE_MODEL_CONFIGS = [
    pytest.param((4, 4, 2, 1, "INT8"), id="t4-mid-cf4"),
    pytest.param((4, 4, -1, 2, "UINT4"), id="t4-last-cf2"),
    pytest.param((4, 4, 0, 4, "INT8"), id="t4-first-cf1"),
    pytest.param((16, 32, 7, 4, "INT8"), id="t16-mid-cf8"),
    pytest.param((16, 8, 15, 8, "INT8"), id="t16-last-cf1"),
    pytest.param((1, 8, 0, 2, "INT8"), id="t1-cf4"),
]


def make_selecttoken_rtl_tree_model(config, part):
    """The single-node SelectToken_rtl model for ``config``, specialized and folded."""
    num_tokens, num_channels, token_index, simd, idt = config
    model = make_selecttoken_modelwrapper(token_index, DataType[idt], num_tokens, num_channels)
    model = model.transform(InferSelectTokenLayer())
    assert model.graph.node[0].op_type == "SelectToken"
    model = model.transform(SpecializeLayers(part))
    assert model.graph.node[0].op_type == "SelectToken_rtl"
    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", simd)
    return model.transform(GiveUniqueNodeNames())


@pytest.mark.parametrize("config", SELECTTOKEN_TREE_MODEL_CONFIGS)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_selecttoken(config):
    part = "xczu7ev-ffvc1156-2-e"
    target_clk_ns = 5
    model = make_selecttoken_rtl_tree_model(config, part)
    node_details = ("SelectToken_rtl", config)

    # Exact, as for Crop_rtl, whose core this is: a solid read row and the
    # selected token's folds two cycles into rtlsim's window after their reads.
    # Every delta measured is zero. The floor of one is for the reference, not
    # the model: rtlsim's window is cut at cycles_rtlsim // 5, which lands on
    # the frame only while the simulator's reset and drain overhead (two or
    # three cycles here) stays under five cycles.
    max_allowed_volume_frac = 0.0
    volume_const = 1
    max_allowed_length_frac = 0.0
    length_const = 1

    assert tree_model_test(
        model,
        node_details,
        part,
        target_clk_ns,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    ), "characterized TAV does not match RTLsim'd one!"


@pytest.mark.fpgadataflow
def test_selecttoken_rtl_tree_model_token_counts():
    """One period of the SelectToken_rtl schedule moves exactly one sequence.

    Reads every token's folds and writes the selected token's, from the node's
    own geometry. Before this node had a model of its own it inherited Crop's,
    which read Crop attributes nothing sets on a SelectToken; the last
    assertion pins the writes to the selected token itself.
    """
    part = "xczu7ev-ffvc1156-2-e"
    for param in SELECTTOKEN_TREE_MODEL_CONFIGS:
        config = param.values[0]
        num_tokens, num_channels, token_index, simd, _ = config
        model = make_selecttoken_rtl_tree_model(config, part)
        node = getCustomOp(model.graph.node[0])
        tree = node.get_tree_model()
        assert tree is not None, config
        cum = tree.cumulative(periods=1)
        cf = num_channels // simd
        assert cum.shape[0] == node.get_exp_cycles(), f"{config}: period {cum.shape[0]}"
        assert cum[-1, 0] == num_tokens * cf, f"{config}: reads {cum[-1, 0]}"
        assert cum[-1, 1] == cf, f"{config}: writes {cum[-1, 1]}"
        # the writes are the selected token's folds, in order, one per cycle
        selected = token_index % num_tokens
        wr = np.flatnonzero(np.diff(np.concatenate(([0], cum[:, 1]))))
        # the core's 2-cycle read-to-write latency; see Crop.get_tree_model
        first = (selected * cf + 2) % cum.shape[0]
        assert np.array_equal(np.sort(wr), np.sort((first + np.arange(cf)) % cum.shape[0]))
