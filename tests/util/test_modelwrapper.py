# Copyright (C) 2025, Advanced Micro Devices, Inc.
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

import onnx
import onnx.helper as oh
from qonnx.core.modelwrapper import ModelWrapper


@pytest.mark.util
@pytest.mark.parametrize(
    "input_name,output_name",
    [
        ("test_input", "test_output"),
        ("global_in", "global_out"),
        ("inp", "outp"),
    ],
)
def test_get_first_global_in_out(input_name, output_name):
    """Test get_first_global_in() and get_first_global_out() helper methods.

    Verifies that the new helper methods correctly return the names
    of global input/output tensors and match the deprecated pattern
    they replace (.graph.input[0].name and .graph.output[0].name).

    Tests with various input/output tensor names to ensure the methods
    work correctly regardless of naming convention.
    """
    # Create a simple ONNX model for testing
    inp = oh.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [1, 4])
    outp = oh.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 4])

    identity_node = oh.make_node("Identity", [input_name], [output_name])

    graph = oh.make_graph([identity_node], "test_graph", [inp], [outp])
    onnx_model = oh.make_model(graph, producer_name="finn-test")
    model = ModelWrapper(onnx_model)

    # Verify that get_first_global_in returns correct name and type
    result_in = model.get_first_global_in()
    assert result_in == input_name, f"Expected '{input_name}', got '{result_in}'"
    assert isinstance(result_in, str), "get_first_global_in() should return a string"

    # Same for get_first_global_out
    result_out = model.get_first_global_out()
    assert result_out == output_name, f"Expected '{output_name}', got '{result_out}'"
    assert isinstance(result_out, str), "get_first_global_out() should return a string"

    # Verify backward compatibility with deprecated pattern
    assert (
        model.get_first_global_in() == model.graph.input[0].name
    ), "get_first_global_in() does not match deprecated .graph.input[0].name"
    assert (
        model.get_first_global_out() == model.graph.output[0].name
    ), "get_first_global_out() does not match deprecated .graph.output[0].name"
