# Copyright (c) 2020, Xilinx
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

from pkgutil import get_data

import numpy as np
import onnx
import onnx.numpy_helper as np_helper

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.core.datatype import DataType
from finn.util.basic import gen_finn_dt_tensor


def test_mnist_onnx_download_extract_run():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    raw_o = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/output_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    output_tensor = onnx.load_tensor_from_string(raw_o)
    # run using FINN-based execution
    input_dict = {"Input3": np_helper.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    assert np.isclose(
        np_helper.to_array(output_tensor), output_dict["Plus214_Output_0"], atol=1e-3
    ).all()


def test_onnx_exec_internal_rounding():
    inp0 = onnx.helper.make_tensor_value_info("inp0", onnx.TensorProto.FLOAT, [2, 2])
    inp1 = onnx.helper.make_tensor_value_info("inp1", onnx.TensorProto.FLOAT, [1])
    outp = onnx.helper.make_tensor_value_info("outp", onnx.TensorProto.FLOAT, [2, 2])
    mul_node = onnx.helper.make_node("Mul", inputs=["inp0", "inp1"], outputs=["outp"],)
    graph = onnx.helper.make_graph(
        nodes=[mul_node], name="mul_graph", inputs=[inp0, inp1], outputs=[outp]
    )

    model = onnx.helper.make_model(graph, producer_name="mul-model")
    model = ModelWrapper(model)
    idt = DataType.INT2
    model.set_tensor_datatype("inp0", idt)
    model.set_tensor_datatype("inp1", idt)
    model.transform(InferShapes())

    mul_value = np.asarray([2], dtype=np.float32)
    inp_int = gen_finn_dt_tensor(idt, [2, 2])
    scale = np.random.uniform(low=0, high=1, size=(2, 2)).astype(np.float32)
    inp_rounded = (inp_int * scale) / (scale + 1e-7)
    input_dict = {"inp0": inp_rounded, "inp1": mul_value}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict["outp"]
    expected = np.multiply(inp_int, mul_value)
    assert (produced == expected).all()
