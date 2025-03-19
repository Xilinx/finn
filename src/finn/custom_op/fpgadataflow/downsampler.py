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
import warnings
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class DownSampler(HWCustomOp):
    """Abstraction layer for HW implementation of DownSampling
    Basically performs a down sampling of the image removing rows and columns."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # spatial size of input images
            "ImgDim": ("i", True, 0),
            # number of channels in input image
            "NumChannels": ("i", True, 0),
            # Number of input columns computed in parallel
            "SIMD": ("i", False, 1),
            "Stride": ("i", True, 2),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # Batch size
            "numInputVectors": ("i", False, 1),
            # 1D (True) or 2D (False) spatial data
            "is1D": ("i", False, 0),
            # for 1D only: (D, 1) (True) or (1, D) dims
            "is1D_unitx": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_downsampled_odim(self):
        "Return the down sampled spatial size of the output."
        idim = self.get_nodeattr("ImgDim")
        stride = self.get_nodeattr("Stride")
        return int(np.floor((idim - 1) / stride) + 1)

    def get_exp_cycles(self):
        is_1D = self.get_nodeattr("is1D")
        idim = self.get_nodeattr("ImgDim")
        idim_total = idim if is_1D else idim * idim
        channels = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        batch_size = self.get_nodeattr("numInputVectors")
        exp_cycles = channels / simd * batch_size * idim_total
        return int(exp_cycles)

    def get_normal_input_shape(self, ind=0):
        is_1D = self.get_nodeattr("is1D")
        is_1D_unitx = self.get_nodeattr("is1D_unitx")
        idim = self.get_nodeattr("ImgDim")
        num_ch = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("numInputVectors")
        if is_1D:
            if is_1D_unitx:
                ishape = (batch, idim, 1, num_ch)
            else:
                ishape = (batch, 1, idim, num_ch)
        else:
            ishape = (batch, idim, idim, num_ch)
        return ishape

    def get_normal_output_shape(self, ind=0):
        is_1D = self.get_nodeattr("is1D")
        is_1D_unitx = self.get_nodeattr("is1D_unitx")
        odim = self.get_downsampled_odim()
        num_ch = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("numInputVectors")
        if is_1D:
            if is_1D_unitx:
                oshape = (batch, odim, 1, num_ch)
            else:
                oshape = (batch, 1, odim, num_ch)
        else:
            oshape = (batch, odim, odim, num_ch)
        return oshape

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        ifm_ch = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide input channels"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        ifm_ch = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide input channels"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        return ret

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def execute_node(self, context, graph):
        # using Im2Col node to calculate output
        node = self.onnx_node
        ifm_dim = self.get_nodeattr("ImgDim")
        stride = self.get_nodeattr("Stride")
        ifm_ch = self.get_nodeattr("NumChannels")
        # check if 1D or 2D case
        if self.get_nodeattr("is1D"):
            if self.get_nodeattr("is1D_unitx"):
                ifm_dim_w = 1
                sw = 1
                ifm_dim_h = ifm_dim
                sh = stride
            else:
                ifm_dim_h = 1
                sh = 1
                ifm_dim_w = ifm_dim
                sw = stride
        else:
            ifm_dim_h = ifm_dim_w = ifm_dim
            sh = sw = stride
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        im2col_node = helper.make_node(
            "Im2Col",
            [node.input[0]],
            [node.output[0]],
            domain="qonnx.custom_op.general",
            stride=[sh, sw],
            kernel_size=[1, 1],
            input_shape="(1,{},{},{})".format(ifm_dim_h, ifm_dim_w, ifm_ch),
        )
        graph_im2col = helper.make_graph(
            nodes=[im2col_node],
            name="single-im2col-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_im2col = ModelWrapper(qonnx_make_model(graph_im2col, **onnx_kwargs))
        model_im2col.set_tensor_datatype(node.input[0], self.get_input_datatype())
        # use execution function from Im2Col node
        # this automatically updates the execution context
        inst = getCustomOp(im2col_node)
        inst.execute_node(context, model_im2col.graph)
