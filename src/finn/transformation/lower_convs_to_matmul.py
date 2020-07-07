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

import numpy as np
from onnx import TensorProto
from onnx import helper

from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name


class LowerConvsToMatMul(Transformation):
    """Replace Conv layers with pairs of Im2Col-MatMul layers, plus Transpose
    layers to keep the original data layout."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv":
                graph_modified = True
                cnv_input = n.input[0]
                cnv_output = n.output[0]
                idt = model.get_tensor_datatype(cnv_input)
                odt = model.get_tensor_datatype(cnv_output)
                # extract conv parameters
                k = get_by_name(n.attribute, "kernel_shape").ints[-1]
                pad = get_by_name(n.attribute, "pads").ints[-1]
                stride = get_by_name(n.attribute, "strides").ints[-1]
                group = get_by_name(n.attribute, "group").i
                weight_name = n.input[1]
                W_conv = model.get_initializer(weight_name)
                ifm_ch = model.get_tensor_shape(n.input[0])[1]  # assume NCHW
                ofm_ch = model.get_tensor_shape(n.output[0])[1]  # assume NCHW
                ifm_dim = model.get_tensor_shape(n.input[0])[-1]  # assume NCHW
                ofm_dim = model.get_tensor_shape(n.output[0])[-1]  # assume NCHW

                # if depthwise conv create sparse matrix and variable "dw"
                # to store as attribute in Im2Col that indicates that the created
                # Im2Col node belongs to a depthwise convolution
                dw = False
                if group == ifm_ch and ofm_ch == ifm_ch:
                    W_sparse = np.zeros((ofm_ch, ifm_ch, k, k))
                    for ch in range(ifm_ch):
                        W_sparse[ch][ch] = W_conv[ch][0]
                    W_conv = W_sparse.astype(np.float32)
                    # we need to store information of the
                    # sparsity of the weight matrix. For this
                    # we use the sparsity annotation of the
                    # weight tensor
                    sparsity = {"dw": {"kernel_shape": k}}
                    model.set_tensor_sparsity(weight_name, sparsity)
                    # additionally create variable "dw" to store
                    # as attribute in Im2Col that indicates that the created
                    # Im2Col node belongs to a depthwise convolution
                    dw = True

                # reuse conv weights for new matmul weights
                # conv weights are [OFM][IFM][k][k]
                # first convert to [OFM][k][k][IFM] (to remain compatible with
                # finn-hlslib and how it does im2col/sliding window)
                W_matmul = W_conv.transpose(0, 2, 3, 1)
                # reshape into [OFM][k*k*IFM] matrix
                W_matmul = W_matmul.reshape(ofm_ch, ifm_ch * k * k)
                # transpose to get ONNX-compatible [k*k*IFM][OFM] matrix
                W_matmul = W_matmul.T
                model.set_initializer(weight_name, W_matmul)

                # create new intermediate values
                inp_trans_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ifm_dim, ifm_dim, ifm_ch),  # NHWC
                )
                graph.value_info.append(inp_trans_out)
                inp_trans_out = inp_trans_out.name
                model.set_tensor_datatype(inp_trans_out, idt)

                need_im2col = True
                if k == 1 and pad == 0 and stride == 1:
                    need_im2col = False

                if need_im2col:
                    im2col_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ofm_dim, ofm_dim, ifm_ch * k * k),
                    )
                    graph.value_info.append(im2col_out)
                    im2col_out = im2col_out.name
                    model.set_tensor_datatype(im2col_out, idt)

                matmul_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_dim, ofm_dim, ofm_ch),
                )
                graph.value_info.append(matmul_out)
                matmul_out = matmul_out.name
                model.set_tensor_datatype(matmul_out, odt)

                # create new nodes
                # NCHW -> NHWC
                inp_trans_node = helper.make_node(
                    "Transpose", [cnv_input], [inp_trans_out], perm=[0, 2, 3, 1]
                )
                # lower input tensor
                matmul_input = inp_trans_out
                if need_im2col:
                    matmul_input = im2col_out
                    im2col_node = helper.make_node(
                        "Im2Col",
                        [inp_trans_out],
                        [im2col_out],
                        domain="finn",
                        stride=stride,
                        kernel_size=k,
                        pad_amount=pad,
                        input_shape="(1,{},{},{})".format(ifm_dim, ifm_dim, ifm_ch),
                        depthwise=dw,
                    )

                # do matmul
                matmul_node = helper.make_node(
                    "MatMul", [matmul_input, weight_name], [matmul_out]
                )
                # NHWC -> NCHW
                out_trans_node = helper.make_node(
                    "Transpose", [matmul_out], [cnv_output], perm=[0, 3, 1, 2]
                )
                # insert nodes where the conv is to preserve topological ordering
                graph.node.insert(node_ind, inp_trans_node)
                if need_im2col:
                    graph.node.insert(node_ind + 1, im2col_node)
                    graph.node.insert(node_ind + 2, matmul_node)
                    graph.node.insert(node_ind + 3, out_trans_node)
                else:
                    graph.node.insert(node_ind + 1, matmul_node)
                    graph.node.insert(node_ind + 2, out_trans_node)
                # remove old nodes
                graph.node.remove(n)
        model = model.transform(InferShapes())
        return (model, graph_modified)
