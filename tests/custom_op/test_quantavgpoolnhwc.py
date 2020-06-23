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

from onnx import helper, TensorProto

from finn.custom_op.maxpoolnhwc import compute_pool_output_dim
from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe


def test_quantavgpoolnhwc():
    s = 1
    k = 3
    ibits = 8
    obits = 4
    signed = False
    n = 1
    c = 2
    idim = 7
    odim = compute_pool_output_dim(idim, k, s)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [n, c, idim, idim])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [n, c, odim, odim])

    node = helper.make_node(
        "QuantAvgPool2d",
        ["inp"],
        ["outp"],
        domain="finn",
        stride=s,
        kernel=k,
        ibits=ibits,
        obits=obits,
        signed=signed,
    )
    graph = helper.make_graph(
        nodes=[node], name="single-quantavgpool", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save("test.onnx")
    model_transformed = model.transform(ChangeDataLayoutQuantAvgPool2d())
    model_transformed.save("test2.onnx")
    model_transformed = model_transformed.transform(InferShapes())
    model_transformed = model_transformed.transform(InferDataTypes())
    model_transformed = model_transformed.transform(GiveUniqueNodeNames())
    model_transformed = model_transformed.transform(GiveReadableTensorNames())
    inp_values = gen_finn_dt_tensor(DataType.INT2, [n, c, idim, idim])
    idict = {"inp": inp_values}
    assert oxe.compare_execution(model, model_transformed, idict)
