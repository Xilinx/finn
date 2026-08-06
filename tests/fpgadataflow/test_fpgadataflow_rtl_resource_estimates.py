# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from onnx import helper

from finn.custom_op.fpgadataflow.rtl.elementwise_binary_rtl import (
    ElementwiseAdd_rtl,
    ElementwiseMul_rtl,
)
from finn.custom_op.fpgadataflow.rtl.hwsoftmax_rtl import HWSoftmax_rtl
from finn.custom_op.fpgadataflow.rtl.layernorm_rtl import LayerNorm_rtl
from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl

VERSAL_PART = "xcvc1902-vsva2197-2MP-e-S"


def test_pumped_low_bit_versal_mvau_dsp_estimation():
    # These are the two pumped DSP MVAUs in the VCK190 W3A3 performance fold.
    # Pumping halves physical SIMD, while low-bit output-lane packing keeps
    # each configuration at 384 DSP58s.
    for mw, mh, simd, pe in [(768, 192, 768, 4), (192, 768, 192, 16)]:
        node = helper.make_node(
            "MVAU_rtl",
            ["inp", "weights"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow.rtl",
            MW=mw,
            MH=mh,
            SIMD=simd,
            PE=pe,
            inputDataType="INT3",
            weightDataType="INT3",
            outputDataType="INT32",
            noActivation=1,
            resType="dsp",
            pumpedCompute=1,
        )

        assert MVAU_rtl(node).dsp_estimation(VERSAL_PART) == 384


def test_float_elementwise_dsp_estimation():
    attrs = {
        "lhs_dtype": "FLOAT32",
        "rhs_dtype": "FLOAT32",
        "out_dtype": "FLOAT32",
        "lhs_shape": [1, 8],
        "rhs_shape": [1, 8],
        "out_shape": [1, 8],
        "PE": 8,
    }

    for op_type, op_cls in [
        ("ElementwiseAdd_rtl", ElementwiseAdd_rtl),
        ("ElementwiseMul_rtl", ElementwiseMul_rtl),
    ]:
        node = helper.make_node(
            op_type,
            ["lhs", "rhs"],
            ["out"],
            domain="finn.custom_op.fpgadataflow.rtl",
            **attrs,
        )
        assert op_cls(node).dsp_estimation(VERSAL_PART) == 8


def test_integer_elementwise_dsp_estimation():
    attrs = {
        "lhs_dtype": "INT8",
        "rhs_dtype": "INT8",
        "lhs_shape": [1, 8],
        "rhs_shape": [1, 8],
        "out_shape": [1, 8],
        "PE": 8,
    }
    add_node = helper.make_node(
        "ElementwiseAdd_rtl",
        ["lhs", "rhs"],
        ["out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        out_dtype="INT9",
        **attrs,
    )
    mul_node = helper.make_node(
        "ElementwiseMul_rtl",
        ["lhs", "rhs"],
        ["out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        out_dtype="INT16",
        **attrs,
    )

    assert ElementwiseAdd_rtl(add_node).dsp_estimation(VERSAL_PART) == 0
    assert ElementwiseMul_rtl(mul_node).dsp_estimation(VERSAL_PART) == 8


def test_layernorm_dsp_estimation():
    node = helper.make_node(
        "LayerNorm_rtl",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        SIMD=4,
        ifm_dim=[1, 197, 192],
        epsilon=1e-5,
        inputDataType="FLOAT32",
        outputDataType="FLOAT32",
    )

    assert LayerNorm_rtl(node).dsp_estimation(VERSAL_PART) == 21

    for width, simd, expected in [(4, 4, 23), (10, 5, 27), (12, 4, 21)]:
        node = helper.make_node(
            "LayerNorm_rtl",
            ["inp"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow.rtl",
            SIMD=simd,
            ifm_dim=[1, width],
            epsilon=1e-5,
            inputDataType="FLOAT32",
            outputDataType="FLOAT32",
        )
        assert LayerNorm_rtl(node).dsp_estimation(VERSAL_PART) == expected


def test_hwsoftmax_dsp_estimation():
    node = helper.make_node(
        "HWSoftmax_rtl",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        SIMD=197,
        ifm_dim=[1, 3, 197, 197],
        input_data_type="FLOAT32",
        NumChannels=197,
    )

    assert HWSoftmax_rtl(node).dsp_estimation(VERSAL_PART) == 794
