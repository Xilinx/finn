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


def make_crop_modelwrapper(idt):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [3, 4, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [2, 2, 4])
    crop_node = helper.make_node(
        "Crop",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        DataType=idt.name,
        ImgDim=[3, 4],
        NumChannels=4,
        CropNorth=1,
        CropEast=1,
        CropSouth=0,
        CropWest=1,
        SIMD=1,
        numInputVectors=[0],
    )
    graph = helper.make_graph([crop_node], "crop-model", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="crop-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# data types
@pytest.mark.parametrize("idt", [DataType["INT8"], DataType["UINT4"]])
# folding
@pytest.mark.parametrize("fold", [-1, 2, 1])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_crop_rtl(idt, fold, exec_mode):
    simd = 1 if fold == -1 else max(1, 4 // fold)
    assert 4 % simd == 0

    input_tensor = gen_finn_dt_tensor(idt, (3, 4, 4))
    input_dict = prepare_inputs(input_tensor)
    y_expected = input_tensor[1:, 1:3, :]
    model = make_crop_modelwrapper(idt)

    # golden reference before specializing
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all(), "Execution of hw layer failed"

    node = getCustomOp(model.get_nodes_by_op_type("Crop")[0])
    node.set_nodeattr("SIMD", simd)
    model = model.transform(SpecializeLayers(FPGA_PART))
    assert model.graph.node[0].op_type == "Crop_rtl"

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
        node = model.get_nodes_by_op_type("Crop_rtl")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0
