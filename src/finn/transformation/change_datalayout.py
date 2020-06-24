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

from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name


class ChangeDataLayoutQuantAvgPool2d(Transformation):
    """Replace QuantAvgPool2d with datalayout (N,C,H,W) with Transpose nodes
    and QuantAvgPool2dNHWC with datalayout (N,H,W,C)"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "QuantAvgPool2d" and (
                get_by_name(n.attribute, "data_layout") is None
                or get_by_name(n.attribute, "data_layout").s.decode("UTF-8") == "NCHW"
            ):
                graph_modified = True
                node_input = n.input[0]
                node_output = n.output[0]
                s = get_by_name(n.attribute, "stride").i
                k = get_by_name(n.attribute, "kernel").i
                ibits = get_by_name(n.attribute, "ibits").i
                obits = get_by_name(n.attribute, "obits").i
                signed = get_by_name(n.attribute, "signed").i
                batchsize = model.get_tensor_shape(n.input[0])[0]  # assume NCHW
                channels = model.get_tensor_shape(n.input[0])[1]  # assume NCHW
                idim = model.get_tensor_shape(n.input[0])[-1]  # assume NCHW
                odim = model.get_tensor_shape(n.output[0])[-1]  # assume NCHW

                # create new nodes
                # NCHW -> NHWC
                # create new intermediate values
                inp_trans_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (batchsize, idim, idim, channels),  # NHWC
                )
                graph.value_info.append(inp_trans_out)
                inp_trans_out = inp_trans_out.name
                quantavg_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (batchsize, odim, odim, channels),
                )
                graph.value_info.append(quantavg_out)
                quantavg_out = quantavg_out.name
                inp_trans_node = helper.make_node(
                    "Transpose", [node_input], [inp_trans_out], perm=[0, 2, 3, 1]
                )
                quantavg_node = helper.make_node(
                    "QuantAvgPool2d",
                    [inp_trans_out],
                    [quantavg_out],
                    domain="finn",
                    stride=s,
                    kernel=k,
                    ibits=ibits,
                    obits=obits,
                    signed=signed,
                    data_layout="NHWC",
                )
                # NHWC -> NCHW
                out_trans_node = helper.make_node(
                    "Transpose", [quantavg_out], [node_output], perm=[0, 3, 1, 2]
                )
                # insert nodes
                graph.node.insert(node_ind, inp_trans_node)
                graph.node.insert(node_ind + 1, quantavg_node)
                graph.node.insert(node_ind + 2, out_trans_node)
                # remove old nodes
                graph.node.remove(n)

                # set shapes
                model.set_tensor_shape(inp_trans_out, (batchsize, idim, idim, channels))
                model.set_tensor_shape(quantavg_out, (batchsize, odim, odim, channels))
        model = model.transform(InferShapes())
        return (model, graph_modified)
