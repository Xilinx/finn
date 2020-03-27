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

from finn.custom_op import CustomOp
import numpy as np
from onnx import helper, TensorProto
from finn.core.modelwrapper import ModelWrapper


def compute_pool_output_dim(ifm_dim, k, stride, pad=0):
    "Return spatial output dimension size for pooling with given params."
    return int(((ifm_dim + 2 * pad - k) / stride) + 1)


class MaxPoolNHWC(CustomOp):
    # a MaxPool node, but using the NHWC data layout

    def get_nodeattr_types(self):
        # no specific attributes for MaxPoolNHWC
        return {
            "kernel_shape": ("ints", True, []),
            "pads": ("ints", True, []),
            "strides": ("ints", True, []),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        iname = node.input[0]
        ishape = model.get_tensor_shape(iname)
        kernel_shape = self.get_nodeattr("kernel_shape")
        pads = self.get_nodeattr("pads")
        strides = self.get_nodeattr("strides")
        assert len(kernel_shape) == 2, "Non-2D MaxPoolNHWC not supported"
        assert pads[0] == pads[2], "Uneven padding not supported"
        assert pads[1] == pads[3], "Uneven padding not supported"
        (n, hi, wi, c) = ishape
        ho = compute_pool_output_dim(hi, kernel_shape[0], strides[0], pads[0])
        wo = compute_pool_output_dim(wi, kernel_shape[1], strides[1], pads[2])
        oshape = (n, ho, wo, c)
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_name = node.input[0]
        out_name = node.output[0]
        inp = context[inp_name]
        dummy_out = context[out_name]
        # convert i/o NHWC -> NCHW
        inp = np.transpose(inp, (0, 3, 1, 2))
        dummy_out = np.transpose(dummy_out, (0, 3, 1, 2))
        # execute as regular MaxPool
        node.domain = ""
        node.op_type = "MaxPool"
        inp_vi = helper.make_tensor_value_info(inp_name, TensorProto.FLOAT, inp.shape)
        out_vi = helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, dummy_out.shape
        )
        tmp_graph = helper.make_graph(
            nodes=[node], name="tmp_graph", inputs=[inp_vi], outputs=[out_vi]
        )
        tmp_model = helper.make_model(tmp_graph, producer_name="finn")
        tmp_model = ModelWrapper(tmp_model)
        new_ctx = {inp_name: inp}
        from finn.core.onnx_exec import execute_onnx

        ret = execute_onnx(tmp_model, new_ctx)
        # restore original node props
        node.domain = "finn"
        node.op_type = "MaxPoolNHWC"
        outp = ret[out_name]
        # convert output NCHW -> NHWC
        outp = np.transpose(outp, (0, 2, 3, 1))
        context[out_name] = outp

    def verify_node(self):
        info_messages = []
        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')
        return info_messages
