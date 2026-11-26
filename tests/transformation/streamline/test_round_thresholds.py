# Copyright (c) 2020-2022, Xilinx, Inc.
# Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.streamline.round_thresholds import (
    RoundAndClipThresholds,
    widened_datatype,
)


def expected_threshold_datatype(model, i_dtype):
    """The input type, or the widened one if a threshold is never met."""
    step = 1 if i_dtype.is_integer() else i_dtype.scale_factor()
    never = model.get_initializer("thresholds") == np.float32(i_dtype.max() + step)
    return widened_datatype(i_dtype) if np.any(never) else i_dtype


def float32_floor(x):
    """The largest float32 not above x: beyond 2^24 the float32 container of a
    tensor cannot hold every integer, and rounding up would leave the range."""
    f = np.float32(x)
    return f if f <= x else np.nextafter(f, np.float32(-np.inf))


# Tests the RoundAndClipThresholds transformation under various input, output
# data type combinations with purely integer inputs. Without proper rounding,
# this tests only the clipping, range and type-casting behavior of the
# transformation.
@pytest.mark.parametrize("op_type", ["MultiThreshold", "Thresholding"])
@pytest.mark.parametrize(
    "i_dtype",
    [
        # Explanation for selecting these test configurations:
        # 1. Below 24-bit thresholds we will not observe any interesting rounding
        #    behavior, as all integers < 2^24 can be exactly represented in 32-bit
        #    floating-point. Thus, we test thresholds at 25-bit signed integers and
        #    generate test inputs slightly above and below this.
        # 2. We want to test out-of-range clipping of thresholds, in particular
        #    clipping of the negative portion of signed thresholds. Thus, we only
        #    generate signed thresholds, but test with signed and unsigned
        #    inputs of smaller, larger and equal range.
        # 3. Testing proper floating-point thresholds requires a separate test-case
        "INT23",
        "UINT23",
        "INT24",
        "UINT24",
        "INT25",
        "UINT25",
        "INT26",
        "UINT26",
    ],
)
@pytest.mark.parametrize(
    "o_dtype",
    [
        # Explanation for selecting these test configurations:
        # 1. Outputs of MultiThreshold are typically much smaller bit-width than the
        #    inputs and thresholds.
        # 2. However, with randomly samples thresholds from a rather large range due
        #    to the selected input bit-widths (see above), we risk not adequately
        #    covering the input range if we sample too few thresholds. The number of
        #    thresholds sampled depends on the bit-width of the output, thus we use
        #    rather high bit-width for testing.
        # 3. For a "real" model, the quantization procedure *should* take care of
        #    adequately covering the true input range.
        "INT8",
        "UINT8",
    ],
)
@pytest.mark.parametrize(
    "n_elems",
    [
        # Explanation for selecting these test configurations:
        # 1. Small edge cases and quickly running through tests: 1, 2, 3, 4
        # 2. Large test case 256, hopefully amplifying any rarely occurring errors
        1,
        2,
        3,
        4,
        256,
    ],
)
@pytest.mark.streamline
def test_round_and_clip_thresholds_ints(op_type, i_dtype, o_dtype, n_elems):
    i_dtype = DataType[i_dtype]
    t_dtype = DataType["INT25"]  # Note: Matches configuration above
    o_dtype = DataType[o_dtype]
    n_thresholds = o_dtype.get_num_possible_values() - 1

    if op_type == "MultiThreshold":
        node = helper.make_node(
            "MultiThreshold",
            domain="qonnx.custom_op.general",
            inputs=["inp", "thresholds"],
            outputs=["out"],
            out_dtype=str(o_dtype),
            out_bias=float(o_dtype.min()),
        )
    else:  # Thresholding
        node = helper.make_node(
            "Thresholding",
            ["inp", "thresholds"],
            ["out"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=n_elems,
            PE=1,
            inputDataType=i_dtype.name,
            weightDataType="INT32",
            outputDataType=o_dtype.name,
            numInputVectors=[1],
            numSteps=n_thresholds,
            ActVal=int(o_dtype.min()),
        )

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, n_elems])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, n_elems])
    thresholds = helper.make_tensor_value_info(
        "thresholds", TensorProto.FLOAT, [n_elems, n_thresholds]
    )
    graph = helper.make_graph([node], "thresholds", [inp, thresholds], [out])
    model = ModelWrapper(helper.make_model(graph))

    inp_vals = gen_finn_dt_tensor(i_dtype, [1, n_elems])
    inp_vals[0][0] = float32_floor(i_dtype.max())
    thresholds = np.sort(gen_finn_dt_tensor(t_dtype, [n_elems, n_thresholds]))
    model.set_tensor_datatype("inp", i_dtype)
    model.set_tensor_datatype("thresholds", t_dtype)
    model.set_tensor_datatype("out", o_dtype)
    model.set_initializer("thresholds", thresholds)

    # Execute the model before running the RoundAndClipThresholds transformation
    out_expected = oxe.execute_onnx(model, {"inp": inp_vals})["out"]
    assert model.get_tensor_datatype("thresholds") == t_dtype

    model = model.transform(RoundAndClipThresholds())

    # After this transformation, the thresholds and output data type should be
    # inferred correctly: the input type, widened only for never-met thresholds
    new_tdt = expected_threshold_datatype(model, i_dtype)
    assert model.get_tensor_datatype("thresholds") == new_tdt
    assert model.get_tensor_datatype("out") == o_dtype

    # For Thresholding, verify weightDataType attribute is also set
    if op_type == "Thresholding":
        inst = getCustomOp(model.graph.node[0])
        assert DataType[inst.get_nodeattr("weightDataType")] == new_tdt

    # After this transformation, the container type used to store the thresholds
    # values must be float32. No other type-cast or type promotion may happen.
    assert model.get_initializer("thresholds").dtype == np.float32

    # After rounding, all thresholds must be integers represented as float32
    assert all(x.is_integer() for x in model.get_initializer("thresholds").flatten())

    # Execute the model after running the RoundAndClipThresholds transformation
    out_produced = oxe.execute_onnx(model, {"inp": inp_vals})["out"]

    assert np.all(out_produced == out_expected)


# Tests the RoundAndClipThresholds transformation under various input, output
# data type combinations with purely integer inputs. This test case tests actual
# rounding of floating-point thresholds.
@pytest.mark.parametrize("op_type", ["MultiThreshold", "Thresholding"])
@pytest.mark.parametrize(
    "i_dtype",
    [
        # Explanation for selecting these test configurations:
        # 1. Below 24-bit thresholds we will not observe any interesting rounding
        #    behavior, as all integers < 2^24 can be exactly represented in 32-bit
        #    floating-point. Thus, we test thresholds at 25-bit signed integers and
        #    generate test inputs slightly above and below this.
        # 2. We want to test out-of-range clipping of thresholds, in particular
        #    clipping of the negative portion of signed thresholds. Thus, we only
        #    generate signed thresholds, but test with signed and unsigned
        #    inputs of smaller, larger and equal range.
        # 3. Testing proper floating-point thresholds requires a separate test-case
        "INT23",
        "UINT23",
        "INT24",
        "UINT24",
        "INT25",
        "UINT25",
        "INT26",
        "UINT26",
    ],
)
@pytest.mark.parametrize(
    "o_dtype",
    [
        # Explanation for selecting these test configurations:
        # 1. Outputs of MultiThreshold are typically much smaller bit-width than the
        #    inputs and thresholds.
        # 2. However, with randomly samples thresholds from a rather large range due
        #    to the selected input bit-widths (see above), we risk not adequately
        #    covering the input range if we sample too few thresholds. The number of
        #    thresholds sampled depends on the bit-width of the output, thus we use
        #    rather high bit-width for testing.
        # 3. For a "real" model, the quantization procedure *should* take care of
        #    adequately covering the true input range.
        "INT8",
        "UINT8",
    ],
)
@pytest.mark.parametrize(
    "n_elems",
    [
        # Explanation for selecting these test configurations:
        # 1. Small edge cases and quickly running through tests: 1, 2, 3, 4
        # 2. Large test case 256, hopefully amplifying any rarely occurring errors
        1,
        2,
        3,
        4,
        256,
    ],
)
@pytest.mark.streamline
def test_round_and_clip_thresholds_floats(op_type, i_dtype, o_dtype, n_elems):
    i_dtype = DataType[i_dtype]
    t_dtype = DataType["FLOAT32"]
    o_dtype = DataType[o_dtype]
    n_thresholds = o_dtype.get_num_possible_values() - 1

    if op_type == "MultiThreshold":
        node = helper.make_node(
            "MultiThreshold",
            domain="qonnx.custom_op.general",
            inputs=["inp", "thresholds"],
            outputs=["out"],
            out_dtype=str(o_dtype),
            out_bias=float(o_dtype.min()),
        )
    else:  # Thresholding
        node = helper.make_node(
            "Thresholding",
            ["inp", "thresholds"],
            ["out"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=n_elems,
            PE=1,
            inputDataType=i_dtype.name,
            weightDataType="INT32",
            outputDataType=o_dtype.name,
            numInputVectors=[1],
            numSteps=n_thresholds,
            ActVal=int(o_dtype.min()),
        )

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, n_elems])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, n_elems])
    thresholds = helper.make_tensor_value_info(
        "thresholds", TensorProto.FLOAT, [n_elems, n_thresholds]
    )
    graph = helper.make_graph([node], "thresholds", [inp, thresholds], [out])
    model = ModelWrapper(helper.make_model(graph))

    inp_vals = gen_finn_dt_tensor(i_dtype, [1, n_elems])
    # Draw uniformly random prototype thresholds in [0,+1] range
    thresholds = np.random.rand(n_elems, n_thresholds)
    # Type alias to 25-bit signed integer type used to set the range of the
    # thresholds
    INT25 = DataType["INT25"]
    # Map the prototype thresholds into the test integer range and sort
    thresholds = np.sort((INT25.max() - INT25.min()) * thresholds + INT25.min())
    # Set data type annotations for the input and thresholds tensor
    model.set_tensor_datatype("inp", i_dtype)
    model.set_tensor_datatype("thresholds", t_dtype)
    model.set_tensor_datatype("out", o_dtype)
    model.set_initializer("thresholds", thresholds)

    # Execute the model before running the RoundAndClipThresholds transformation
    out_expected = oxe.execute_onnx(model, {"inp": inp_vals})["out"]
    # Before rounding the threshold data type must be as annotated
    assert model.get_tensor_datatype("thresholds") == t_dtype

    model = model.transform(RoundAndClipThresholds())

    new_tdt = expected_threshold_datatype(model, i_dtype)
    assert model.get_tensor_datatype("thresholds") == new_tdt
    assert model.get_tensor_datatype("out") == o_dtype

    # For Thresholding, verify weightDataType attribute is also set
    if op_type == "Thresholding":
        inst = getCustomOp(model.graph.node[0])
        assert DataType[inst.get_nodeattr("weightDataType")] == new_tdt

    # After this transformation, the container type used to store the thresholds
    # values must be float32. No other type-cast or type promotion may happen.
    assert model.get_initializer("thresholds").dtype == np.float32
    # After rounding, all thresholds must be integers represented as float32
    assert all(x.is_integer() for x in model.get_initializer("thresholds").flatten())

    out_produced = oxe.execute_onnx(model, {"inp": inp_vals})["out"]

    assert np.allclose(out_produced, out_expected, atol=1.0e-3)


# Tests the RoundAndClipThresholds transformation under various input, output
# data type combinations with fixed point inputs. Without proper rounding,
# this tests only the clipping, range and type-casting behavior of the
# transformation.
@pytest.mark.parametrize("op_type", ["MultiThreshold", "Thresholding"])
@pytest.mark.parametrize(
    "i_dtype",
    [
        "FIXED<13,3>",
        "FIXED<14,3>",
        "FIXED<15,3>",
        "FIXED<16,3>",
    ],
)
@pytest.mark.parametrize(
    "o_dtype",
    [
        # Explanation for selecting these test configurations:
        # 1. Outputs of MultiThreshold are typically much smaller bit-width than the
        #    inputs and thresholds.
        # 2. However, with randomly samples thresholds from a rather large range due
        #    to the selected input bit-widths (see above), we risk not adequately
        #    covering the input range if we sample too few thresholds. The number of
        #    thresholds sampled depends on the bit-width of the output, thus we use
        #    rather high bit-width for testing.
        # 3. For a "real" model, the quantization procedure *should* take care of
        #    adequately covering the true input range.
        "INT8",
        "UINT8",
    ],
)
@pytest.mark.parametrize(
    "n_elems",
    [
        # Explanation for selecting these test configurations:
        # 1. Small edge cases and quickly running through tests: 1, 2, 3, 4
        # 2. Large test case 256, hopefully amplifying any rarely occurring errors
        1,
        2,
        3,
        4,
        256,
    ],
)
@pytest.mark.streamline
def test_round_and_clip_thresholds_fxp(op_type, i_dtype, o_dtype, n_elems):
    i_dtype = DataType[i_dtype]
    t_dtype = DataType["FIXED<15,3>"]
    o_dtype = DataType[o_dtype]
    n_thresholds = o_dtype.get_num_possible_values() - 1

    if op_type == "MultiThreshold":
        node = helper.make_node(
            "MultiThreshold",
            domain="qonnx.custom_op.general",
            inputs=["inp", "thresholds"],
            outputs=["out"],
            out_dtype=str(o_dtype),
            out_bias=float(o_dtype.min()),
        )
    else:  # Thresholding
        node = helper.make_node(
            "Thresholding",
            ["inp", "thresholds"],
            ["out"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=n_elems,
            PE=1,
            inputDataType=i_dtype.name,
            weightDataType="INT32",
            outputDataType=o_dtype.name,
            numInputVectors=[1],
            numSteps=n_thresholds,
            ActVal=int(o_dtype.min()),
        )

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, n_elems])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, n_elems])
    thresholds = helper.make_tensor_value_info(
        "thresholds", TensorProto.FLOAT, [n_elems, n_thresholds]
    )
    graph = helper.make_graph([node], "thresholds", [inp, thresholds], [out])
    model = ModelWrapper(helper.make_model(graph))

    inp_vals = gen_finn_dt_tensor(i_dtype, [1, n_elems])
    # Draw uniformly random prototype thresholds in [0,+1] range
    thresholds = np.random.rand(n_elems, n_thresholds)
    # Type alias to 25-bit signed integer type used to set the range of the
    # thresholds
    FXP25 = DataType["FIXED<15,3>"]
    # Map the prototype thresholds into the test integer range and sort
    thresholds = np.sort((FXP25.max() - FXP25.min()) * thresholds + FXP25.min())
    # Set data type annotations for the input and thresholds tensor
    model.set_tensor_datatype("inp", i_dtype)
    model.set_tensor_datatype("thresholds", t_dtype)
    model.set_tensor_datatype("out", o_dtype)
    model.set_initializer("thresholds", thresholds)

    # Execute the model before running the RoundAndClipThresholds transformation
    out_expected = oxe.execute_onnx(model, {"inp": inp_vals})["out"]
    # Before rounding the threshold data type must be as annotated
    assert model.get_tensor_datatype("thresholds") == t_dtype

    model = model.transform(RoundAndClipThresholds())

    new_tdt = expected_threshold_datatype(model, i_dtype)
    assert model.get_tensor_datatype("thresholds") == new_tdt
    assert model.get_tensor_datatype("out") == o_dtype

    # For Thresholding, verify weightDataType attribute is also set
    if op_type == "Thresholding":
        inst = getCustomOp(model.graph.node[0])
        assert DataType[inst.get_nodeattr("weightDataType")] == new_tdt

    # After this transformation, the container type used to store the thresholds
    # values must be float32. No other type-cast or type promotion may happen.
    assert model.get_initializer("thresholds").dtype == np.float32
    # After rounding, all thresholds must be integers represented as float32
    assert all(
        x.is_integer()
        for x in model.get_initializer("thresholds").flatten() / new_tdt.scale_factor()
    )

    out_produced = oxe.execute_onnx(model, {"inp": inp_vals})["out"]

    assert np.allclose(out_produced, out_expected, atol=1.0e-3)


# Tests the RoundAndClipThresholds transformation under various input, output
# data type combinations with fixed point inputs. This test case tests actual
# rounding of floating-point thresholds.
@pytest.mark.parametrize("op_type", ["MultiThreshold", "Thresholding"])
@pytest.mark.parametrize(
    "i_dtype",
    [
        "FIXED<13,3>",
        "FIXED<14,3>",
        "FIXED<15,3>",
        "FIXED<16,3>",
    ],
)
@pytest.mark.parametrize(
    "o_dtype",
    [
        # Explanation for selecting these test configurations:
        # 1. Outputs of MultiThreshold are typically much smaller bit-width than the
        #    inputs and thresholds.
        # 2. However, with randomly samples thresholds from a rather large range due
        #    to the selected input bit-widths (see above), we risk not adequately
        #    covering the input range if we sample too few thresholds. The number of
        #    thresholds sampled depends on the bit-width of the output, thus we use
        #    rather high bit-width for testing.
        # 3. For a "real" model, the quantization procedure *should* take care of
        #    adequately covering the true input range.
        "INT8",
        "UINT8",
    ],
)
@pytest.mark.parametrize(
    "n_elems",
    [
        # Explanation for selecting these test configurations:
        # 1. Small edge cases and quickly running through tests: 1, 2, 3, 4
        # 2. Large test case 256, hopefully amplifying any rarely occurring errors
        1,
        2,
        3,
        4,
        256,
    ],
)
@pytest.mark.streamline
def test_round_and_clip_thresholds_fxp_float(op_type, i_dtype, o_dtype, n_elems):
    i_dtype = DataType[i_dtype]
    t_dtype = DataType["FLOAT32"]
    o_dtype = DataType[o_dtype]
    n_thresholds = o_dtype.get_num_possible_values() - 1

    if op_type == "MultiThreshold":
        node = helper.make_node(
            "MultiThreshold",
            domain="qonnx.custom_op.general",
            inputs=["inp", "thresholds"],
            outputs=["out"],
            out_dtype=str(o_dtype),
            out_bias=float(o_dtype.min()),
        )
    else:  # Thresholding
        node = helper.make_node(
            "Thresholding",
            ["inp", "thresholds"],
            ["out"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=n_elems,
            PE=1,
            inputDataType=i_dtype.name,
            weightDataType="INT32",
            outputDataType=o_dtype.name,
            numInputVectors=[1],
            numSteps=n_thresholds,
            ActVal=int(o_dtype.min()),
        )

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, n_elems])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, n_elems])
    thresholds = helper.make_tensor_value_info(
        "thresholds", TensorProto.FLOAT, [n_elems, n_thresholds]
    )
    graph = helper.make_graph([node], "thresholds", [inp, thresholds], [out])
    model = ModelWrapper(helper.make_model(graph))

    inp_vals = gen_finn_dt_tensor(i_dtype, [1, n_elems])
    # Draw uniformly random prototype thresholds in [0,+1] range
    thresholds = np.random.rand(n_elems, n_thresholds)
    # Type alias to 25-bit signed integer type used to set the range of the
    # thresholds
    FXP25 = DataType["FIXED<15,3>"]
    # Map the prototype thresholds into the test integer range and sort
    thresholds = np.sort((FXP25.max() - FXP25.min()) * thresholds + FXP25.min())
    # Set data type annotations for the input and thresholds tensor
    model.set_tensor_datatype("inp", i_dtype)
    model.set_tensor_datatype("thresholds", t_dtype)
    model.set_tensor_datatype("out", o_dtype)
    model.set_initializer("thresholds", thresholds)

    # Execute the model before running the RoundAndClipThresholds transformation
    out_expected = oxe.execute_onnx(model, {"inp": inp_vals})["out"]
    # Before rounding the threshold data type must be as annotated
    assert model.get_tensor_datatype("thresholds") == t_dtype

    model = model.transform(RoundAndClipThresholds())

    new_tdt = expected_threshold_datatype(model, i_dtype)
    assert model.get_tensor_datatype("thresholds") == new_tdt
    assert model.get_tensor_datatype("out") == o_dtype

    # For Thresholding, verify weightDataType attribute is also set
    if op_type == "Thresholding":
        inst = getCustomOp(model.graph.node[0])
        assert DataType[inst.get_nodeattr("weightDataType")] == new_tdt

    # After this transformation, the container type used to store the thresholds
    # values must be float32. No other type-cast or type promotion may happen.
    assert model.get_initializer("thresholds").dtype == np.float32
    # After rounding, all thresholds must be integers represented as float32
    assert all(
        x.is_integer()
        for x in model.get_initializer("thresholds").flatten() / new_tdt.scale_factor()
    )

    out_produced = oxe.execute_onnx(model, {"inp": inp_vals})["out"]

    assert np.allclose(out_produced, out_expected, atol=1.0e-3)


# Test RoundAndClipThresholds for MVAU and VVAU with thresholds
@pytest.mark.parametrize("op_type", ["MVAU", "VVAU"])
@pytest.mark.parametrize(
    "i_dtype",
    [
        # Test various input bitwidths to verify accumulator-based clipping
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT24",
        "UINT24",
    ],
)
@pytest.mark.parametrize(
    "o_dtype",
    [
        # Test different output datatypes
        "INT4",
        "UINT4",
        "INT8",
        "UINT8",
    ],
)
@pytest.mark.parametrize("wdt", [DataType["INT4"], DataType["INT8"]])
@pytest.mark.parametrize(
    "n_channels",
    [
        # Test different sizes like the MultiThreshold/Thresholding tests
        1,
        4,
        16,
    ],
)
@pytest.mark.streamline
def test_round_and_clip_thresholds_mvau_vvau(op_type, i_dtype, o_dtype, wdt, n_channels):
    """Test RoundAndClipThresholds for MVAU and VVAU nodes with integrated thresholding.
    Verifies that thresholds are rounded/clipped based on accumulator datatype
    (not input datatype) and that threshold datatype is set correctly."""

    i_dtype = DataType[i_dtype]
    o_dtype = DataType[o_dtype]
    n_steps = o_dtype.get_num_possible_values() - 1

    if op_type == "MVAU":
        mw = 8
        mh = n_channels

        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
        weights = gen_finn_dt_tensor(wdt, (mw, mh))
        inp_shape = [1, mw]

        node = helper.make_node(
            "MVAU",
            ["inp", "weights", "thresh"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            MW=mw,
            MH=mh,
            SIMD=1,
            PE=1,
            inputDataType=i_dtype.name,
            outputDataType=o_dtype.name,
            weightDataType=wdt.name,
            accDataType="INT32",  # Will be minimized
            ActVal=int(o_dtype.min()),
            noActivation=0,
        )
    else:  # VVAU
        k_h, k_w = 3, 3

        inp = helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT, [1, 8, 8, n_channels * k_h * k_w]
        )
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 8, 8, n_channels])
        weights = gen_finn_dt_tensor(wdt, (n_channels, 1, k_h, k_w))
        inp_shape = [1, 8, 8, n_channels * k_h * k_w]

        node = helper.make_node(
            "VVAU",
            ["inp", "weights", "thresh"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            PE=1,
            Channels=n_channels,
            Dim=[8, 8],
            Kernel=[k_h, k_w],
            inputDataType=i_dtype.name,
            outputDataType=o_dtype.name,
            weightDataType=wdt.name,
            accDataType="INT32",  # Will be minimized
            ActVal=int(o_dtype.min()),
            noActivation=0,
        )

    # Generate thresholds with some non-integer values
    # Use a wide range that may exceed accumulator range
    thresholds = np.random.rand(n_channels, n_steps) * 1000 - 500
    thresholds = np.sort(thresholds, axis=1).astype(np.float32)

    graph = helper.make_graph(
        nodes=[node],
        name=f"{op_type.lower()}_threshold_test",
        inputs=[inp],
        outputs=[outp],
        initializer=[
            helper.make_tensor("weights", TensorProto.FLOAT, weights.shape, weights.flatten()),
            helper.make_tensor("thresh", TensorProto.FLOAT, thresholds.shape, thresholds.flatten()),
        ],
    )

    model = helper.make_model(graph, producer_name=f"{op_type.lower()}-threshold-test")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", i_dtype)
    model.set_tensor_datatype("outp", o_dtype)
    model.set_tensor_datatype("weights", wdt)
    model.set_tensor_datatype("thresh", DataType["FLOAT32"])

    # First run MinimizeAccumulatorWidth to set the actual accumulator datatype
    model = model.transform(MinimizeAccumulatorWidth())

    inst = getCustomOp(model.graph.node[0])
    acc_dt = DataType[inst.get_nodeattr("accDataType")]

    # Generate test input and execute before RoundAndClipThresholds
    test_input = gen_finn_dt_tensor(i_dtype, inp_shape)
    output_before = oxe.execute_onnx(model, {"inp": test_input})["outp"]

    # Now run RoundAndClipThresholds
    model = model.transform(RoundAndClipThresholds())

    # Verify threshold datatype is set correctly (one bit wider than accumulator)
    if not acc_dt.signed():
        expected_tdt = DataType.get_smallest_possible(acc_dt.max() + 1)
    else:
        expected_tdt = DataType.get_smallest_possible(-(acc_dt.max() + 1) - 1)

    assert model.get_tensor_datatype("thresh") == expected_tdt

    # Verify thresholds are rounded to integers
    rounded_thresholds = model.get_initializer("thresh")
    assert all(x.is_integer() for x in rounded_thresholds.flatten())

    # Verify thresholds are clipped to accumulator range
    assert rounded_thresholds.min() >= acc_dt.min()
    assert rounded_thresholds.max() <= acc_dt.max() + 1

    # Verify functional equivalence
    output_after = oxe.execute_onnx(model, {"inp": test_input})["outp"]
    assert np.all(
        output_before == output_after
    ), "Output changed after RoundAndClipThresholds transformation"


@pytest.mark.parametrize("op_type", ["MultiThreshold", "Thresholding"])
@pytest.mark.parametrize(
    "thresholds,kept,num_always,widened",
    [
        # columns never met in all channels are dropped, always met ones go into the bias
        ([[-200, -3, 5, 300], [-129, 0, 9, 128]], [[-3, 5], [0, 9]], 1, False),
        # a channel keeps a never-met threshold: the type needs one more value
        ([[-3, 5, 128], [0, 9, 10]], [[-3, 5, 128], [0, 9, 10]], 0, True),
        # every column constant: the output does not depend on the input
        ([[-200, 300], [-129, 128]], None, 0, False),
        ([[300, 400], [128, 500]], None, 0, False),
        # mixed across channels, yet every channel constant
        ([[-200, -150], [300, 400]], None, 0, False),
        # unsorted thresholds are sorted per channel
        ([[5, -3, 300], [9, 0, 200]], [[-3, 5], [0, 9]], 0, False),
    ],
)
@pytest.mark.streamline
def test_round_and_clip_thresholds_drop_constant(op_type, thresholds, kept, num_always, widened):
    """Threshold columns constant over all channels are removed: never-met ones
    without trace, always-met ones into the output bias. The threshold type is
    the input type unless a channel keeps a never-met threshold. A node whose
    thresholds are all constant is rejected."""
    i_dtype, o_dtype = DataType["INT8"], DataType["INT4"]
    thresholds = np.asarray(thresholds, dtype=np.float32)
    n_elems, n_thresholds = thresholds.shape
    bias = -8
    if op_type == "MultiThreshold":
        node = helper.make_node(
            "MultiThreshold",
            domain="qonnx.custom_op.general",
            inputs=["inp", "thresholds"],
            outputs=["out"],
            out_dtype=str(o_dtype),
            out_bias=float(bias),
        )
    else:
        node = helper.make_node(
            "Thresholding",
            ["inp", "thresholds"],
            ["out"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=n_elems,
            PE=1,
            inputDataType=i_dtype.name,
            weightDataType="INT32",
            outputDataType=o_dtype.name,
            numInputVectors=[1],
            numSteps=n_thresholds,
            ActVal=bias,
        )
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, n_elems])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, n_elems])
    graph = helper.make_graph([node], "thresholds", [inp], [out])
    model = ModelWrapper(helper.make_model(graph))
    model.set_tensor_datatype("inp", i_dtype)
    model.set_tensor_datatype("out", o_dtype)
    model.set_initializer("thresholds", thresholds)
    model.set_tensor_datatype("thresholds", DataType["INT32"])

    inputs = [np.full((1, n_elems), v, dtype=np.float32) for v in (-128, -4, 0, 5, 9, 10, 127)]
    out_expected = [oxe.execute_onnx(model, {"inp": x})["out"] for x in inputs]

    if kept is None:
        with pytest.raises(ValueError, match="output is constant"):
            model.transform(RoundAndClipThresholds())
        return
    model = model.transform(RoundAndClipThresholds())

    assert np.array_equal(model.get_initializer("thresholds"), np.asarray(kept, np.float32))
    expected_tdt = widened_datatype(i_dtype) if widened else i_dtype
    assert model.get_tensor_datatype("thresholds") == expected_tdt
    inst = getCustomOp(model.graph.node[0])
    if op_type == "MultiThreshold":
        assert inst.get_nodeattr("out_bias") == bias + num_always
    else:
        assert inst.get_nodeattr("ActVal") == bias + num_always
        assert inst.get_nodeattr("numSteps") == len(kept[0])
    for x, y in zip(inputs, out_expected):
        assert np.array_equal(oxe.execute_onnx(model, {"inp": x})["out"], y)
