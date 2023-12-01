# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
import onnxruntime as rt
import warnings
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# TODO: consider splitting this into separate implementations for 1D and 2D
# similar to what we do for ConvolutionInputGenerator


class StreamingMaxPool(HWCustomOp):
    """Abstraction layer for HW implementation of StreamingMaxPool"""

    def get_nodeattr_types(self):
        my_attrs = {
            "ImgDim": ("ints", True, []),  # [H, W] = [Y, X]
            "PoolDim": ("ints", True, []),  # [H, W] = [Y, X]
            "NumChannels": ("i", True, 0),
            # parallelism control - only supported for 1D maxpool
            "PE": ("i", False, 0),
            # round up (instead of down) output size - only supported for 1D maxpool
            "CeilMode": ("i", False, 0),
            # FINN DataTypes for inputs/outputs
            "dataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("dataType")]

    def get_1d_attrs_normalized(self):
        # support both (1, D) and (D, 1) cases transparently:
        # assume the dummy ('1') dimension is the Y-dimension, i.e.
        # images and kernels (and their attributes) of dimension
        # [H, W] = [Y, X] = [D, 1] or [1, D] are always mapped to [1, D]
        ifm_dim = self.get_nodeattr("ImgDim")
        k = self.get_nodeattr("PoolDim")
        ifm_ch = self.get_nodeattr("NumChannels")
        if ifm_dim[1] == 1:
            ifm_dim = ifm_dim[::-1]
            k = k[::-1]
        return (ifm_dim, k, ifm_ch)

    def is_1d(self):
        ifm_dim, k, ifm_ch = self.get_1d_attrs_normalized()
        return (ifm_dim[0] == 1) and (k[0] == 1)

    def get_normal_input_shape(self, ind=0):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("ImgDim")
        ifm_ch = self.get_nodeattr("NumChannels")
        ishape = (1, ifm_dim_h, ifm_dim_w, ifm_ch)
        return ishape

    def get_folded_input_shape(self, ind=0):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("ImgDim")
        ifm_ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        nf = int(ifm_ch / pe)
        if self.is_1d():
            folded_ishape = (1, ifm_dim_h, ifm_dim_w, nf, pe)
        else:
            folded_ishape = (1, ifm_dim_h, ifm_dim_w, 1, ifm_ch)
        return folded_ishape

    def get_normal_output_shape(self, ind=0):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("ImgDim")
        k_h, k_w = tuple(self.get_nodeattr("PoolDim"))
        ifm_ch = self.get_nodeattr("NumChannels")
        ceil_mode = self.get_nodeattr("CeilMode")
        if not self.is_1d():
            assert ifm_dim_h % k_h == 0, "StreamingMaxPool needs ImgDim_h % PoolDim_h == 0"
            assert ifm_dim_w % k_w == 0, "StreamingMaxPool needs ImgDim_w % PoolDim_w == 0"
        ofm_dim_h = compute_pool_output_dim(ifm_dim_h, k_h, k_h, 0, ceil_mode)
        ofm_dim_w = compute_pool_output_dim(ifm_dim_w, k_w, k_w, 0, ceil_mode)
        oshape = (1, ofm_dim_h, ofm_dim_w, ifm_ch)
        return oshape

    def get_folded_output_shape(self, ind=0):
        # even though there is no folding in the current hlslib op,
        # insert a time multiplexing axis to remain compatible with the
        # shapes produced by the rest of the dataflow pipeline
        ifm_ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        nf = int(ifm_ch / pe)
        ret = list(self.get_normal_output_shape())
        if self.is_1d():
            ret[-1] = nf
            ret.append(pe)
        else:
            ret.insert(-1, 1)
        return tuple(ret)

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def get_exp_cycles(self):
        # derived from StreamingMaxPool_Batch loop nest
        ifm_dim, k, ifm_ch = self.get_1d_attrs_normalized()

        warnings.warn(
            """Estimated latency for layer {} can be lower than
             actual latency!""".format(
                self.onnx_node.name
            )
        )
        if self.is_1d():
            _, _, _, nf, _ = self.get_folded_output_shape()
            ceil_mode = self.get_nodeattr("CeilMode")
            ofm_dim = compute_pool_output_dim(ifm_dim[1], k[1], k[1], 0, ceil_mode)
            exp_cycles = ofm_dim * nf * (k[1] + 1)
            return int(exp_cycles)
        else:
            # TODO: adjust inaccurate formula
            return int(ifm_dim[1] * ifm_dim[1] * (1 + 1 / (k[1] * k[1])))

    def get_instream_width(self, ind=0):
        dt_bits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        ifm_ch = self.get_nodeattr("NumChannels")
        if self.is_1d():
            in_width = int(dt_bits * pe)
        else:
            in_width = int(dt_bits * ifm_ch)
        return in_width

    def get_outstream_width(self, ind=0):
        """For streaming maxpool out stream width is the same as in stream width"""
        return self.get_instream_width()

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for StreamingMaxPool."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        pass

    def execute_node(self, context, graph):
        # create a standard add node to help calculate the result
        node = self.onnx_node
        kernel_shape = self.get_nodeattr("PoolDim")
        ceil_mode = self.get_nodeattr("CeilMode")
        inp_values = context[node.input[0]]
        dummy_out = context[node.output[0]]
        # convert i/o NHWC -> NCHW
        inp_values = np.transpose(inp_values, (0, 3, 1, 2))
        dummy_out = np.transpose(dummy_out, (0, 3, 1, 2))
        # handle 1d case
        ishape = inp_values.shape
        if ishape[2] == 1 or ishape[3] == 1:
            inp_values = inp_values.reshape(ishape[0], ishape[1], ishape[2] * ishape[3])
            kernel_shape = [kernel_shape[0] * kernel_shape[1]]
        # execute as regular MaxPool
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, inp_values.shape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, dummy_out.shape)
        node_mp = helper.make_node(
            "MaxPool",
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            kernel_shape=kernel_shape,
            strides=kernel_shape,
            ceil_mode=ceil_mode,
        )
        graph_mp = helper.make_graph(
            nodes=[node_mp],
            name="single-mp-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_mp = qonnx_make_model(graph_mp, **onnx_kwargs)
        idict = {node.input[0]: inp_values}
        sess = rt.InferenceSession(model_mp.SerializeToString())
        result = sess.run(None, idict)
        result = np.asarray(result, dtype=np.float32).reshape(dummy_out.shape)
        # convert output NCHW -> NHWC
        result = np.transpose(result, (0, 2, 3, 1))
        context[node.output[0]] = result
