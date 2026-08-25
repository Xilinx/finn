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

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10
NUM_TOKENS = 4
NUM_CHANNELS = 4


def make_selecttoken_modelwrapper(token_index, idt):
    indices = np.asarray(token_index, dtype=np.int64)
    output_shape = [1, NUM_CHANNELS] if indices.ndim == 0 else [1, len(indices), NUM_CHANNELS]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, NUM_TOKENS, NUM_CHANNELS])
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
