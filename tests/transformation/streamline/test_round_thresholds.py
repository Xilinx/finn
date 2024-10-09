# Copyright (c) 2020-2022, Xilinx, Inc.
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
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

# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Testing framework
import pytest

# Use numpy for python execution / computing the ground truth expected values
import numpy as np

# Utility types and function for creating onnx nodes and graphs
from onnx import TensorProto, helper

# QONNX data types like INT25
from qonnx.core.datatype import DataType

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Generate random tensors of QONNX/FINN data types for testing
from qonnx.util.basic import gen_finn_dt_tensor

# Execution of onnx graphs within FINN
import finn.core.onnx_exec as oxe

# The transformation to be tested
from finn.transformation.streamline import RoundAndClipThresholds


# Tests the RoundAndClipThresholds transformation under various input, output
# data type combinations with purely integer inputs. Without proper rounding,
# this tests only the clipping, range and type-casting behavior of the
# transformation.
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
def test_round_and_clip_thresholds_ints(i_dtype, o_dtype, n_elems):
    i_dtype = DataType[i_dtype]
    t_dtype = DataType["INT25"]  # Note: Matches configuration above
    o_dtype = DataType[o_dtype]  # noqa: Duplicate model setup code
    node = helper.make_node(
        "MultiThreshold",
        domain="qonnx.custom_op.general",
        inputs=["inp", "thresholds"],
        outputs=["out"],
        out_dtype=str(o_dtype),
        out_bias=float(o_dtype.min()),
    )
    n_thresholds = o_dtype.get_num_possible_values() - 1
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, n_elems])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, n_elems])
    thresholds = helper.make_tensor_value_info(
        "thresholds", TensorProto.FLOAT, [n_elems, n_thresholds]
    )
    graph = helper.make_graph([node], "thresholds", [inp, thresholds], [out])
    model = ModelWrapper(helper.make_model(graph))

    inp = gen_finn_dt_tensor(i_dtype, [1, n_elems])
    inp[0][0] = i_dtype.max()
    thresholds = np.sort(gen_finn_dt_tensor(t_dtype, [n_elems, n_thresholds]))
    model.set_tensor_datatype("inp", i_dtype)  # noqa: Duplicate model execution
    model.set_tensor_datatype("thresholds", t_dtype)
    model.set_tensor_datatype("out", o_dtype)
    model.set_initializer("thresholds", thresholds)

    # Execute the model before running the RoundAndClipThresholds transformation
    out_expected = oxe.execute_onnx(model, {"inp": inp})["out"]
    assert model.get_tensor_datatype("thresholds") == t_dtype

    model = model.transform(RoundAndClipThresholds())

    # After this transformation, the thresholds and output data type should be
    # inferred correctly
    if not i_dtype.signed():
        new_tdt = DataType.get_smallest_possible(i_dtype.max() + 1)
    else:
        new_tdt = DataType.get_smallest_possible(-(i_dtype.max() + 1) - 1)
    assert model.get_tensor_datatype("thresholds") == new_tdt
    assert model.get_tensor_datatype("out") == o_dtype

    # After this transformation, the container type used to store the thresholds
    # values must be float32. No other type-cast or type promotion may happen.
    assert model.get_initializer("thresholds").dtype == np.float32

    # After rounding, all thresholds must be integers represented as float32
    assert all(x.is_integer() for x in model.get_initializer("thresholds").flatten())

    # Execute the model after running the RoundAndClipThresholds transformation
    out_produced = oxe.execute_onnx(model, {"inp": inp})["out"]

    assert np.all(out_produced == out_expected)


# Tests the RoundAndClipThresholds transformation under various input, output
# data type combinations with purely integer inputs. This test case tests actual
# rounding of floating-point thresholds.
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
def test_round_and_clip_thresholds_floats(i_dtype, o_dtype, n_elems):
    i_dtype = DataType[i_dtype]
    t_dtype = DataType["FLOAT32"]
    o_dtype = DataType[o_dtype]  # noqa: Duplicate model setup code
    node = helper.make_node(
        "MultiThreshold",
        domain="qonnx.custom_op.general",
        inputs=["inp", "thresholds"],
        outputs=["out"],
        out_dtype=str(o_dtype),
    )
    n_thresholds = o_dtype.get_num_possible_values() - 1
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, n_elems])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, n_elems])
    thresholds = helper.make_tensor_value_info(
        "thresholds", TensorProto.FLOAT, [n_elems, n_thresholds]
    )
    graph = helper.make_graph([node], "thresholds", [inp, thresholds], [out])
    model = ModelWrapper(helper.make_model(graph))

    inp = gen_finn_dt_tensor(i_dtype, [1, n_elems])
    # Draw uniformly random prototype thresholds in [0,+1] range
    thresholds = np.random.rand(n_elems, n_thresholds)
    # Type alias to 25-bit signed integer type used to set the range of the
    # thresholds
    INT25 = DataType["INT25"]  # noqa: Variable name not lowercase
    # Map the prototype thresholds into the test integer range and sort
    thresholds = np.sort((INT25.max() - INT25.min()) * thresholds + INT25.min())
    # Set data type annotations for the input and thresholds tensor
    model.set_tensor_datatype("inp", i_dtype)  # noqa: Duplicate model execution
    model.set_tensor_datatype("thresholds", t_dtype)
    model.set_tensor_datatype("out", o_dtype)
    # Set the thresholds as initializer input to the model
    model.set_initializer("thresholds", thresholds)
    # Execute the model before running the RoundAndClipThresholds transformation
    out_expected = oxe.execute_onnx(model, {"inp": inp})["out"]
    # Before rounding the threshold data type must be as annotated
    assert model.get_tensor_datatype("thresholds") == t_dtype

    model = model.transform(RoundAndClipThresholds())

    if not i_dtype.signed():
        new_tdt = DataType.get_smallest_possible(i_dtype.max() + 1)
    else:
        new_tdt = DataType.get_smallest_possible(-(i_dtype.max() + 1) - 1)
    assert model.get_tensor_datatype("thresholds") == new_tdt
    assert model.get_tensor_datatype("out") == o_dtype

    # After this transformation, the container type used to store the thresholds
    # values must be float32. No other type-cast or type promotion may happen.
    assert model.get_initializer("thresholds").dtype == np.float32
    # After rounding, all thresholds must be integers represented as float32
    assert all(
        x.is_integer() for x in model.get_initializer("thresholds").flatten()
    )
    # Execute the model after running the RoundAndClipThresholds transformation
    out_produced = oxe.execute_onnx(model, {"inp": inp})["out"]
    # Compare the results before and after: This is the floating-point test with
    # actual rounding, this the transformed result may only be equal within some
    # tolerance.
    # Hm, never observed this to be relevant. For all test configurations, exact
    # equality seems to hold, probably due to only integer inputs being tested.
    assert np.allclose(out_produced, out_expected, atol=1.0e-3)
