############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import pytest

import json
import numpy as np
import os
import tempfile
import torch
import torch.onnx
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from torch import nn

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferShuffle
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.transpose_decomposition import (
    InferInnerOuterShuffles,
    ShuffleDecomposition,
)
from finn.util.config import extract_model_config_consolidate_shuffles

test_fpga_part: str = "xcvc1902-vsva2197-2MP-e-S"
test_synth_clk_period_ns: int = 10


class PytorchShuffle(nn.Module):
    """From pytorch create a reshape and transpose combination
    that can be used for testing"""

    def __init__(
        self,
        transpose_perm: tuple[int],
        reshape1_shape: tuple[int] = None,
        reshape2_shape: tuple[int] = None,
    ) -> None:
        super(PytorchShuffle, self).__init__()
        self.transpose_perm = transpose_perm
        self.reshape1_shape = reshape1_shape
        self.reshape2_shape = reshape2_shape

    def forward(self, x):
        if self.reshape1_shape is not None:
            x = x.reshape(*self.reshape1_shape)
        x = x.permute(*self.transpose_perm)
        if self.reshape2_shape is not None:
            x = x.reshape(*self.reshape2_shape)
        return x


def construct_onnx_model(
    input_shape: tuple[int],
    transpose_perm: tuple[int],
    reshape1_shape: tuple[int],
    reshape2_shape: tuple[int],
    dt: DataType,
) -> ModelWrapper:
    """Creates an ONNX model that can be used for testing
    the shuffle operation compiler integration. Uses the
    pytorch methods in PytorchShuffle to generate the model."""

    model = PytorchShuffle(
        transpose_perm=transpose_perm, reshape1_shape=reshape1_shape, reshape2_shape=reshape2_shape
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
        model_input = torch.rand(input_shape)
        export_qonnx(model, model_input, temp_file.name, opset_version=17)
        qonnx_cleanup(temp_file.name, out_file=temp_file.name)

        new_model = ModelWrapper(temp_file.name)
        new_model.set_tensor_datatype(new_model.graph.input[0].name, dt)
        new_model.set_tensor_datatype(new_model.graph.output[0].name, dt)
        new_model.transform(InferShapes())
        new_model.transform(InferDataTypes())
        return new_model
    raise RuntimeError("Error unable to export the ONNX file to the temporary location")


class SetShuffleSIMD(Transformation):
    """Set SIMD parameter and enable waveform generation for all Inner and Outer shuffle nodes."""

    def __init__(self, simd_value, enable_waveforms=False):
        super().__init__()
        self.simd_value = simd_value
        self.enable_waveforms = enable_waveforms

    def apply(self, model):
        for node in model.graph.node:
            if node.op_type in ["Shuffle"] and "finn.custom_op.fpgadataflow" in node.domain:
                inst = getCustomOp(node)
                inst.set_nodeattr("SIMD", self.simd_value)

                # Enable waveform generation for debugging
                if self.enable_waveforms:
                    inst.set_nodeattr("rtlsim_trace", "debug.wdb")
        return model, False


@pytest.mark.parametrize(
    "cpp_shuffle_param",
    [
        {
            "in_shape": (1, 128, 384),  # Shuffle A
            "transpose_in_shape": (1, 128, 12, 32),
            "out_shape": (1, 12, 128, 32),
            "transpose_out_shape": None,
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (1, 128, 384),  # Shuffle B
            "transpose_in_shape": (1, 128, 12, 32),
            "out_shape": (1, 12, 32, 128),
            "transpose_out_shape": None,
            "perm": (0, 2, 3, 1),
        },
        {
            "in_shape": (4, 8, 4),  # Brute Force cannot be simplified into 2D case
            "transpose_in_shape": None,
            "out_shape": (4, 8, 4),
            "transpose_out_shape": None,
            "perm": (2, 1, 0),
        },
        {
            "in_shape": (2, 4, 3),  # Brute Force cannot be simplified into 2D case
            "transpose_in_shape": None,
            "out_shape": (2, 3, 4),
            "transpose_out_shape": None,
            "perm": (0, 2, 1),
        },
        {
            "in_shape": (1, 12, 128, 32),  # Shuffle C
            "transpose_in_shape": None,
            "out_shape": (1, 128, 12, 32),
            "transpose_out_shape": (1, 128, 384),
            "perm": (0, 2, 1, 3),
        },
    ],
)
@pytest.mark.parametrize("datatype", ["INT8", "INT4"])
@pytest.mark.parametrize("simd", ["simd1", "simd2", "simd4"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_cppsim_shuffle_layer(cpp_shuffle_param, datatype, simd):
    """Checks cppsim of the shuffle_hls layer"""
    dt = DataType[datatype]
    simd = int(simd[-1])
    in_shape = cpp_shuffle_param["in_shape"]

    model = construct_onnx_model(
        input_shape=in_shape,
        transpose_perm=cpp_shuffle_param["perm"],
        reshape1_shape=cpp_shuffle_param["transpose_in_shape"],
        reshape2_shape=cpp_shuffle_param["transpose_out_shape"],
        dt=dt,
    )

    input = gen_finn_dt_tensor(dt, in_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Get a reference for the shuffle
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Attempt to build the HLS for this
    model = model.transform(InferShuffle())
    model = model.transform(SpecializeLayers(test_fpga_part))

    model = model.transform(SetShuffleSIMD(simd))
    model = model.transform(ShuffleDecomposition())
    model = model.transform(InferInnerOuterShuffles())
    model = model.transform(SpecializeLayers(test_fpga_part))

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    y_hw = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_hw), "Model output does not match expected output"


@pytest.mark.parametrize(
    "shuffle_param",
    [
        {
            "in_shape": (1, 128, 384),  # Shuffle A
            "transpose_in_shape": (1, 128, 12, 32),
            "out_shape": (1, 12, 128, 32),
            "transpose_out_shape": None,
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (1, 12, 128, 32),  # Shuffle C
            "transpose_in_shape": None,
            "out_shape": (1, 128, 12, 32),
            "transpose_out_shape": (1, 128, 384),
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (128, 384),  # pTranspose Test
            "transpose_in_shape": None,
            "out_shape": (384, 128),
            "transpose_out_shape": None,
            "perm": (1, 0),
        },
        {
            "in_shape": (32, 16, 8, 12),  # Mixed Transpose test
            "transpose_in_shape": None,
            "out_shape": (8, 12, 32, 16),
            "transpose_out_shape": None,
            "perm": (2, 3, 0, 1),
        },
        {
            "in_shape": (2, 2, 12, 8),
            "transpose_in_shape": None,
            "out_shape": (2, 2, 8, 12),
            "transpose_out_shape": None,
            "perm": (0, 1, 3, 2),
        },
        {
            "in_shape": (32, 16, 12, 8),  # Mixed Transpose test
            "transpose_in_shape": None,
            "out_shape": (8, 12, 16, 32),
            "transpose_out_shape": None,
            "perm": (3, 2, 1, 0),
        },
        {
            "in_shape": (64, 256),
            "transpose_in_shape": None,
            "out_shape": (256, 64),
            "transpose_out_shape": None,
            "perm": (1, 0),
        },
        {
            "in_shape": (512, 128),
            "transpose_in_shape": None,
            "out_shape": (128, 512),
            "transpose_out_shape": None,
            "perm": (1, 0),
        },
        {
            "in_shape": (256, 512),
            "transpose_in_shape": None,
            "out_shape": (512, 256),
            "transpose_out_shape": None,
            "perm": (1, 0),
        },
        {
            "in_shape": (8, 16, 32),
            "transpose_in_shape": None,
            "out_shape": (32, 16, 8),
            "transpose_out_shape": None,
            "perm": (2, 1, 0),
        },
        {
            "in_shape": (4, 64, 128),
            "transpose_in_shape": None,
            "out_shape": (64, 4, 128),
            "transpose_out_shape": None,
            "perm": (1, 0, 2),
        },
        {
            "in_shape": (16, 8, 64),
            "transpose_in_shape": None,
            "out_shape": (64, 16, 8),
            "transpose_out_shape": None,
            "perm": (2, 0, 1),
        },
        {
            "in_shape": (8, 8, 8, 8),
            "transpose_in_shape": None,
            "out_shape": (8, 8, 8, 8),
            "transpose_out_shape": None,
            "perm": (3, 1, 0, 2),
        },
        {
            "in_shape": (4, 8, 16, 32),
            "transpose_in_shape": None,
            "out_shape": (16, 32, 4, 8),
            "transpose_out_shape": None,
            "perm": (2, 3, 0, 1),
        },
        {
            "in_shape": (1, 256, 192),
            "transpose_in_shape": (1, 256, 6, 32),
            "out_shape": (1, 6, 256, 32),
            "transpose_out_shape": (1, 6, 8192),
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (1, 64, 512),
            "transpose_in_shape": (1, 64, 16, 32),
            "out_shape": (1, 16, 64, 32),
            "transpose_out_shape": None,
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (2, 32, 128),
            "transpose_in_shape": (2, 32, 4, 32),
            "out_shape": (2, 4, 32, 32),
            "transpose_out_shape": (2, 4, 1024),
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (4, 4),
            "transpose_in_shape": None,
            "out_shape": (4, 4),
            "transpose_out_shape": None,
            "perm": (1, 0),
        },
        {
            "in_shape": (1, 8, 8),
            "transpose_in_shape": None,
            "out_shape": (8, 1, 8),
            "transpose_out_shape": None,
            "perm": (1, 0, 2),
        },
        {
            "in_shape": (1, 1024, 768),
            "transpose_in_shape": (1, 1024, 24, 32),
            "out_shape": (1, 24, 1024, 32),
            "transpose_out_shape": None,
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (8, 128, 256),
            "transpose_in_shape": None,
            "out_shape": (256, 128, 8),
            "transpose_out_shape": None,
            "perm": (2, 1, 0),
        },
        {
            "in_shape": (6, 12, 18, 24),
            "transpose_in_shape": None,
            "out_shape": (18, 6, 24, 12),
            "transpose_out_shape": None,
            "perm": (2, 0, 3, 1),
        },
        {
            "in_shape": (7, 12, 16),
            "transpose_in_shape": None,
            "out_shape": (16, 7, 12),
            "transpose_out_shape": None,
            "perm": (2, 0, 1),
        },
        {
            "in_shape": (5, 10, 15, 20),
            "transpose_in_shape": None,
            "out_shape": (15, 20, 5, 10),
            "transpose_out_shape": None,
            "perm": (2, 3, 0, 1),
        },
        {
            "in_shape": (256, 128),
            "transpose_in_shape": None,
            "out_shape": (128, 256),
            "transpose_out_shape": None,
            "perm": (1, 0),
        },
        {
            "in_shape": (64, 96),
            "transpose_in_shape": None,
            "out_shape": (96, 64),
            "transpose_out_shape": None,
            "perm": (1, 0),
        },
        {
            "in_shape": (1, 96, 128),
            "transpose_in_shape": (1, 96, 4, 32),
            "out_shape": (1, 4, 96, 32),
            "transpose_out_shape": (1, 4, 3072),
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (4, 48, 64),
            "transpose_in_shape": (4, 48, 4, 16),
            "out_shape": (4, 4, 48, 16),
            "transpose_out_shape": (4, 4, 768),
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (8, 32, 64, 16),
            "transpose_in_shape": None,
            "out_shape": (64, 8, 16, 32),
            "transpose_out_shape": None,
            "perm": (2, 0, 3, 1),
        },
        {
            "in_shape": (3, 6, 9, 12),
            "transpose_in_shape": None,
            "out_shape": (9, 12, 3, 6),
            "transpose_out_shape": None,
            "perm": (2, 3, 0, 1),
        },
    ],
)
@pytest.mark.parametrize("datatype", ["INT8"])
@pytest.mark.parametrize("simd", ["simd2", "simd4"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_rtlsim_shuffle_layer(shuffle_param, datatype, simd):
    """Checks rtlsim of the shuffle_hls layer"""
    os.environ["LIVENESS_THRESHOLD"] = "10000000"  # Need to bump this up for these RTL sims
    dt = DataType[datatype]
    simd = int(simd[-1])
    in_shape = shuffle_param["in_shape"]

    model = construct_onnx_model(
        input_shape=in_shape,
        transpose_perm=shuffle_param["perm"],
        reshape1_shape=shuffle_param["transpose_in_shape"],
        reshape2_shape=shuffle_param["transpose_out_shape"],
        dt=dt,
    )

    input = gen_finn_dt_tensor(dt, in_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Get a reference for the shuffle
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Attempt to build the HLS/RTL for this
    model = model.transform(InferShuffle())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(SetShuffleSIMD(simd, enable_waveforms=True))

    model = model.transform(ShuffleDecomposition())
    model = model.transform(InferInnerOuterShuffles())
    model = model.transform(SpecializeLayers(test_fpga_part))

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    y_hw = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_hw), "Model output does not match expected output"


@pytest.mark.parametrize(
    "shuffle_sip_param",
    [
        {
            "in_shape": (1, 128, 384),  # Shuffle A
            "transpose_in_shape": (1, 128, 12, 32),
            "out_shape": (1, 12, 128, 32),
            "transpose_out_shape": None,
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (1, 12, 128, 32),  # Shuffle C
            "transpose_in_shape": None,
            "out_shape": (1, 128, 12, 32),
            "transpose_out_shape": (1, 128, 384),
            "perm": (0, 2, 1, 3),
        },
        {
            "in_shape": (128, 384),  # pTranspose Test
            "transpose_in_shape": None,
            "out_shape": (384, 128),
            "transpose_out_shape": None,
            "perm": (1, 0),
        },
        {
            "in_shape": (32, 16, 8, 12),  # Mixed Transpose test
            "transpose_in_shape": None,
            "out_shape": (8, 12, 32, 16),
            "transpose_out_shape": None,
            "perm": (2, 3, 0, 1),
        },
        {
            "in_shape": (2, 2, 12, 8),
            "transpose_in_shape": None,
            "out_shape": (2, 2, 8, 12),
            "transpose_out_shape": None,
            "perm": (0, 1, 3, 2),
        },
        {
            "in_shape": (32, 16, 12, 8),  # Mixed Transpose test
            "transpose_in_shape": None,
            "out_shape": (8, 12, 16, 32),
            "transpose_out_shape": None,
            "perm": (3, 2, 1, 0),
        },
    ],
)
@pytest.mark.parametrize("datatype", ["INT8"])
@pytest.mark.parametrize("simd", ["simd2", "simd4"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_stitched_ip_shuffle_layer(shuffle_sip_param, datatype, simd):
    """Build stitched IP for shuffle layer tests and save results for buffer analysis"""
    dt = DataType[datatype]
    simd = int(simd[-1])
    in_shape = shuffle_sip_param["in_shape"]

    model = construct_onnx_model(
        input_shape=in_shape,
        transpose_perm=shuffle_sip_param["perm"],
        reshape1_shape=shuffle_sip_param["transpose_in_shape"],
        reshape2_shape=shuffle_sip_param["transpose_out_shape"],
        dt=dt,
    )

    input = gen_finn_dt_tensor(dt, in_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Get a reference for the shuffle
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    model = model.transform(InferShuffle())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(SetShuffleSIMD(simd))

    model = model.transform(ShuffleDecomposition())
    model = model.transform(InferInnerOuterShuffles())
    model = model.transform(SpecializeLayers(test_fpga_part))

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())

    model = model.transform(CreateStitchedIP(test_fpga_part, test_synth_clk_period_ns))

    model.set_metadata_prop("exec_mode", "rtlsim")
    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    assert np.allclose(y_ref, y_hw), "Model output does not match expected output"


def test_shuffle_config_consolidation():
    dt = DataType["INT8"]
    model = construct_onnx_model(
        input_shape=(32, 16, 8, 12),
        transpose_perm=(2, 3, 0, 1),
        reshape1_shape=None,
        reshape2_shape=None,
        dt=dt,
    )

    model = model.transform(InferShuffle())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(SetShuffleSIMD(4))

    original_shuffle_name = None
    for node in model.graph.node:
        if node.op_type == "Shuffle" and "finn.custom_op.fpgadataflow" in node.domain:
            original_shuffle_name = node.name
            break

    model = model.transform(ShuffleDecomposition())
    model = model.transform(InferInnerOuterShuffles())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())

    decomposed_nodes = []
    for node in model.graph.node:
        if node.op_type in ["InnerShuffle_rtl", "OuterShuffle_hls"]:
            decomposed_nodes.append(node.name)
            orig_name = getCustomOp(node).get_nodeattr("original_node_name")
            assert orig_name == original_shuffle_name

    assert len(decomposed_nodes) > 0

    consolidated_file = os.environ["FINN_BUILD_DIR"] + "/consolidated.json"
    extract_model_config_consolidate_shuffles(model, consolidated_file, ["SIMD"])

    with open(consolidated_file, "r") as f:
        consolidated_config = json.load(f)

    assert original_shuffle_name in consolidated_config
    assert consolidated_config[original_shuffle_name]["SIMD"] == 4
    for decomposed_name in decomposed_nodes:
        assert decomposed_name not in consolidated_config
