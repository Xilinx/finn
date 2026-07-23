############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

import pytest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferCrop,
    InferSelectTokenLayer,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10


def _make_gather_model(indices, output_shape):
    tokens = helper.make_tensor_value_info("tokens", TensorProto.FLOAT, [1, 4, 4])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)
    gather = helper.make_node("Gather", ["tokens", "idx"], ["out"], axis=1, name="gather_tokens")
    graph = helper.make_graph(
        [gather],
        "crop_test",
        [tokens],
        [output],
        [numpy_helper.from_array(indices, name="idx")],
    )
    model = ModelWrapper(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)]))
    model.set_tensor_datatype("tokens", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])
    return model


def _make_crop_model():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [3, 4, 4])
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 2, 4])
    crop = helper.make_node(
        "Crop",
        ["inp"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="Crop_0",
        DataType="INT8",
        ImgDim=[3, 4],
        NumChannels=4,
        CropNorth=1,
        CropEast=1,
        CropSouth=0,
        CropWest=1,
        SIMD=2,
        numInputVectors=[0],
    )
    graph = helper.make_graph([crop], "crop_rtlsim_test", [inp], [output])
    model = ModelWrapper(helper.make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])
    return model


def test_crop_rtl_default_and_hls_fallback():
    for preferred_style, expected_op in [("", "Crop_rtl"), ("hls", "Crop_hls")]:
        model = _make_gather_model(np.asarray([1, 2], dtype=np.int64), [1, 2, 4])
        model = model.transform(InferCrop())
        crop = getCustomOp(model.graph.node[0])
        crop.set_nodeattr("preferred_impl_style", preferred_style)
        model = model.transform(SpecializeLayers(FPGA_PART))
        model = model.transform(GiveUniqueNodeNames())
        assert model.graph.node[0].op_type == expected_op


def test_selecttoken_uses_crop_rtl(tmp_path):
    model = _make_gather_model(np.asarray(2, dtype=np.int64), [1, 4])
    tokens = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    expected = tokens[:, 2, :]

    model = model.transform(InferSelectTokenLayer())
    assert model.graph.node[0].op_type == "SelectToken"
    assert np.array_equal(execute_onnx(model, {"tokens": tokens})["out"], expected)

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    node = model.graph.node[0]
    assert node.op_type == "SelectToken_rtl"
    inst = getCustomOp(node)
    inst.set_nodeattr("SIMD", 2)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, FPGA_PART, CLK_NS)

    wrapper = (tmp_path / (inst.get_nodeattr("gen_top_module") + ".v")).read_text()
    assert (tmp_path / "crop.sv").is_file()
    assert "crop #(" in wrapper
    assert ".H(1)" in wrapper
    assert ".W(4)" in wrapper
    assert ".CF(2)" in wrapper
    assert ".CROP_W(2)" in wrapper
    assert ".CROP_E(1)" in wrapper


def test_crop_rtl_codegen(tmp_path):
    model = _make_gather_model(np.asarray([1, 2], dtype=np.int64), [1, 2, 4])
    model = model.transform(InferCrop())
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", 2)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, FPGA_PART, CLK_NS)

    wrapper = (tmp_path / (inst.get_nodeattr("gen_top_module") + ".v")).read_text()
    assert (tmp_path / "crop.sv").is_file()
    assert "crop #(" in wrapper
    assert ".H(1)" in wrapper
    assert ".W(4)" in wrapper
    assert ".CF(2)" in wrapper
    assert ".CROP_W(1)" in wrapper
    assert ".CROP_E(1)" in wrapper


@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_crop_rtl_rtlsim():
    model = _make_crop_model()
    inp = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    expected = inp[1:, 1:3, :]
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    assert np.array_equal(execute_onnx(model, {"inp": inp})["out"], expected)


@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_selecttoken_crop_rtl_rtlsim():
    model = _make_gather_model(np.asarray(3, dtype=np.int64), [1, 4])
    tokens = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    model = model.transform(InferSelectTokenLayer())
    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", 2)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    assert np.array_equal(execute_onnx(model, {"tokens": tokens})["out"], tokens[:, 3, :])
