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
from finn import xsi as finnxsi
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
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
from finn.util.basic import get_watchdog_timeout_cycles, make_build_dir, robust_rmtree
from finn.util.config import extract_model_config_consolidate_shuffles
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

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
        new_model.set_tensor_datatype(new_model.get_first_global_in(), dt)
        new_model.set_tensor_datatype(new_model.get_first_global_out(), dt)
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
    in_name = model.get_first_global_in()
    out_name = model.get_first_global_out()
    input_t = {in_name: input}

    # Get a reference for the shuffle
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Attempt to build the HLS for this
    model = model.transform(InferShuffle(_filter=lambda *_: True))
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
    "reshape_transpose_param",
    [
        {
            "in_shape": (1, 768, 14, 14),  # exact SigLIP head dims
            "transpose_in_shape": (1, 768, 196),
            "out_shape": (1, 196, 768),
            "transpose_out_shape": None,
            "perm": (0, 2, 1),
        },
    ],
    ids=["siglip_1x768x14x14"],
)
@pytest.mark.parametrize("datatype", ["INT8"])
@pytest.mark.parametrize("simd", ["simd1"])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fused_reshape_inner_transpose(
    reshape_transpose_param, datatype, simd, exec_mode, monkeypatch
):
    """cppsim/rtlsim of a single Reshape+Transpose that fuses into one Shuffle and
    must decompose to a single InnerShuffle operating on the flattened view.

    Guards against the regression where the InnerShuffle was built from the
    physical in_shape (dropping the fused reshape) instead of transpose_in_shape.
    """
    monkeypatch.setenv("LIVENESS_THRESHOLD", "10000000")
    dt = DataType[datatype]
    simd = int(simd[-1])
    in_shape = reshape_transpose_param["in_shape"]
    transpose_in_shape = reshape_transpose_param["transpose_in_shape"]

    model = construct_onnx_model(
        input_shape=in_shape,
        transpose_perm=reshape_transpose_param["perm"],
        reshape1_shape=transpose_in_shape,
        reshape2_shape=reshape_transpose_param["transpose_out_shape"],
        dt=dt,
    )

    input = gen_finn_dt_tensor(dt, in_shape)
    in_name = model.get_first_global_in()
    out_name = model.get_first_global_out()
    input_t = {in_name: input}

    # Reference: the plain Reshape+Transpose ONNX
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Fuse Reshape+Transpose into one Shuffle and confirm the reshape was captured
    # (physical in_shape differs from the shape the transpose acts on).
    model = model.transform(InferShuffle(_filter=lambda *_: True))
    model = model.transform(SpecializeLayers(test_fpga_part))
    shuffle_nodes = [
        n
        for n in model.graph.node
        if n.op_type == "Shuffle" and "finn.custom_op.fpgadataflow" in n.domain
    ]
    assert len(shuffle_nodes) == 1, "expected a single fused Shuffle node"
    shuffle_inst = getCustomOp(shuffle_nodes[0])
    assert list(shuffle_inst.get_nodeattr("in_shape")) == list(in_shape)
    assert list(shuffle_inst.get_nodeattr("transpose_in_shape")) == list(transpose_in_shape)
    assert list(shuffle_inst.get_nodeattr("in_shape")) != list(
        transpose_in_shape
    ), "this test must exercise a FUSED reshape (in_shape != transpose_in_shape)"

    model = model.transform(SetShuffleSIMD(simd))
    model = model.transform(ShuffleDecomposition())
    model = model.transform(InferInnerOuterShuffles())
    model = model.transform(SpecializeLayers(test_fpga_part))

    # A swap-last-two transpose must map to exactly one InnerShuffle (no
    # OuterShuffle), and that InnerShuffle must operate on the flattened
    # transpose_in_shape rather than the physical in_shape.
    inner = [n for n in model.graph.node if n.op_type == "InnerShuffle_rtl"]
    outer = [n for n in model.graph.node if n.op_type == "OuterShuffle_hls"]
    assert len(inner) == 1, "fused reshape + swap-last-two must yield one InnerShuffle"
    assert len(outer) == 0, "no OuterShuffle expected for a pure inner transpose"
    inner_inst = getCustomOp(inner[0])
    # in_shape matches the physical input tensor; the flattened view the
    # transpose acts on is carried separately in transpose_in_shape.
    assert list(inner_inst.get_nodeattr("in_shape")) == list(
        in_shape
    ), "InnerShuffle in_shape must match the physical input tensor"
    assert list(inner_inst.get_nodeattr("transpose_in_shape")) == list(
        transpose_in_shape
    ), "InnerShuffle must carry the flattened transpose_in_shape"
    assert list(inner_inst.get_normal_output_shape()) == list(reshape_transpose_param["out_shape"])

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(SetExecMode(exec_mode))
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise ValueError(f"unknown exec_mode {exec_mode}")

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
def test_rtlsim_shuffle_layer(shuffle_param, datatype, simd, monkeypatch):
    """Checks rtlsim of the shuffle_hls layer"""
    monkeypatch.setenv("LIVENESS_THRESHOLD", "10000000")
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
    in_name = model.get_first_global_in()
    out_name = model.get_first_global_out()
    input_t = {in_name: input}

    # Get a reference for the shuffle
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Attempt to build the HLS/RTL for this
    model = model.transform(InferShuffle(_filter=lambda *_: True))
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

    for node in model.graph.node:
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


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
def test_stitched_ip_shuffle_layer(shuffle_sip_param, datatype, simd, monkeypatch):
    """Build stitched IP for shuffle layer tests and save results for buffer analysis"""
    monkeypatch.setenv("LIVENESS_THRESHOLD", "10000000")
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
    in_name = model.get_first_global_in()
    out_name = model.get_first_global_out()
    input_t = {in_name: input}

    # Get a reference for the shuffle
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    model = model.transform(InferShuffle(_filter=lambda *_: True))
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

    model = model.transform(InferShuffle(_filter=lambda *_: True))
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

    test_dir = make_build_dir("test_shuffle_config_")
    consolidated_file = os.path.join(test_dir, "consolidated.json")
    extract_model_config_consolidate_shuffles(model, consolidated_file, ["SIMD"])

    with open(consolidated_file, "r") as f:
        consolidated_config = json.load(f)

    assert original_shuffle_name in consolidated_config
    assert consolidated_config[original_shuffle_name]["SIMD"] == 4
    for decomposed_name in decomposed_nodes:
        assert decomposed_name not in consolidated_config
    robust_rmtree(test_dir)


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_inner_shuffle_rtlsim_stalled_final_write_is_quiescent(monkeypatch):
    """An incomplete page must not be announced while its final write is stalled."""
    monkeypatch.setenv("LIVENESS_THRESHOLD", "10000000")
    if not finnxsi.is_available():
        pytest.skip("finn_xsi (XSI rtlsim) not available")

    simd = 2
    dt = DataType["UINT8"]
    in_shape = (1, 4, 4)
    model = construct_onnx_model(
        input_shape=in_shape,
        transpose_perm=(0, 2, 1),
        reshape1_shape=None,
        reshape2_shape=None,
        dt=dt,
    )
    model = model.transform(InferShuffle(_filter=lambda *_: True))
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(SetShuffleSIMD(simd))
    model = model.transform(ShuffleDecomposition())
    model = model.transform(InferInnerOuterShuffles())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    inst = getCustomOp(model.get_nodes_by_op_type("InnerShuffle_rtl")[0])
    in_folded = inst.get_folded_input_shape(0)
    frame = np.arange(16, dtype=np.float32).reshape(in_folded)
    packed_frame = npy_to_rtlsim_input(
        frame,
        inst.get_input_datatype(0),
        inst.get_instream_width(0),
    )

    sim = inst.get_rtlsim()
    try:
        inst.reset_rtlsim(sim)
        sim.stream_input("in0_V", (f"{value:x}" for value in packed_frame[:-1]))
        assert not sim.run()

        output_valid = sim.get_bus_port("out0_V", "tvalid")

        class IdleProbe:
            def __init__(self, cycles):
                self.remaining = cycles
                self.saw_valid = False

            def __bool__(self):
                return True

            def __call__(self, _sim):
                self.saw_valid |= output_valid.read().as_bool()
                self.remaining -= 1
                return None if self.remaining == 0 else {}

        probe = IdleProbe(32)
        sim.enlist(probe)
        assert not sim.run()
        assert not probe.saw_valid
    finally:
        inst.close_rtlsim(sim)


@pytest.mark.parametrize("throttle", [(float("inf"), 0), (1, 15)], ids=["backtoback", "bursty"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_inner_shuffle_rtl_bursty(throttle, monkeypatch):
    monkeypatch.setenv("LIVENESS_THRESHOLD", "10000000")
    if not finnxsi.is_available():
        pytest.skip("finn_xsi (XSI rtlsim) not available")

    SIMD = 4
    dt = DataType["INT8"]
    in_shape = (4, 8, 8)  # (N frames, rows, cols); transpose swaps rows/cols

    model = construct_onnx_model(
        input_shape=in_shape,
        transpose_perm=(0, 2, 1),
        reshape1_shape=None,
        reshape2_shape=None,
        dt=dt,
    )

    x = gen_finn_dt_tensor(dt, in_shape)
    in_name = model.get_first_global_in()
    out_name = model.get_first_global_out()
    y_ref = oxe.execute_onnx(model, {in_name: x})[out_name]

    model = model.transform(InferShuffle(_filter=lambda *_: True))
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(SetShuffleSIMD(SIMD))
    model = model.transform(ShuffleDecomposition())
    model = model.transform(InferInnerOuterShuffles())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    inst = getCustomOp(model.get_nodes_by_op_type("InnerShuffle_rtl")[0])

    in_dt, in_w, in_folded = (
        inst.get_input_datatype(0),
        inst.get_instream_width(0),
        inst.get_folded_input_shape(0),
    )
    out_dt, out_w, out_folded = (
        inst.get_output_datatype(0),
        inst.get_outstream_width(0),
        inst.get_folded_output_shape(0),
    )
    out_normal = tuple(inst.get_normal_output_shape(0))
    num_out = inst.get_number_output_values()

    packed_in = npy_to_rtlsim_input(np.asarray(x, dtype=np.float32).reshape(in_folded), in_dt, in_w)
    hex_in = map(lambda v: f"{v:0x}", packed_in)

    sim = inst.get_rtlsim()
    liveness = get_watchdog_timeout_cycles(inst.get_exp_cycles())
    try:
        inst.reset_rtlsim(sim)
        sim.stream_input("in0_V", hex_in, throttle=throttle)
        out_buf = sim.collect_output(
            "out0_V", num_out, watchdog=sim.create_watchdog("out0_V timeout", liveness)
        )
        assert not sim.run(), "rtlsim watchdog timed out"
        packed_out = [int(v, base=16) for v in out_buf]
    finally:
        inst.close_rtlsim(sim)

    got = rtlsim_output_to_npy(packed_out, None, out_dt, out_folded, out_w, out_dt.bitwidth())
    got = np.asarray(got, dtype=np.float32).reshape(out_normal)

    assert got.shape == y_ref.shape, "shape %s != ref %s" % (got.shape, y_ref.shape)
    assert np.allclose(got, y_ref), (
        "InnerShuffle_rtl output does not match reference transpose (throttle=%s): "
        "a read overtook its write on the ping-pong page" % (throttle,)
    )
