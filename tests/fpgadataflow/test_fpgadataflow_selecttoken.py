############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10
NUM_TOKENS = 4
NUM_CHANNELS = 4


def make_selecttoken_modelwrapper(token_index, idt):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, NUM_TOKENS, NUM_CHANNELS])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, NUM_CHANNELS])
    selecttoken_node = helper.make_node(
        "SelectToken",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumTokens=NUM_TOKENS,
        NumChannels=NUM_CHANNELS,
        TokenIndex=token_index,
        SIMD=1,
        inputDataType=idt.name,
        outputDataType=idt.name,
    )
    graph = helper.make_graph([selecttoken_node], "selecttoken-model", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="selecttoken-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# data types
@pytest.mark.parametrize("idt", [DataType["INT8"], DataType["UINT4"]])
# token index
@pytest.mark.parametrize("token_index", [2, -1])
# folding
@pytest.mark.parametrize("fold", [-1, 2, 1])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_selecttoken(idt, token_index, fold, exec_mode):
    simd = 1 if fold == -1 else max(1, NUM_CHANNELS // fold)
    assert NUM_CHANNELS % simd == 0

    input_tensor = gen_finn_dt_tensor(idt, (1, NUM_TOKENS, NUM_CHANNELS))
    input_dict = prepare_inputs(input_tensor)
    y_expected = input_tensor[:, token_index, :]
    model = make_selecttoken_modelwrapper(token_index, idt)

    # golden reference before specializing
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all(), "Execution of hw layer failed"

    node = getCustomOp(model.get_nodes_by_op_type("SelectToken")[0])
    node.set_nodeattr("SIMD", simd)
    model = model.transform(SpecializeLayers(FPGA_PART))
    assert model.graph.node[0].op_type == "SelectToken_rtl"

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("SelectToken_rtl")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0
