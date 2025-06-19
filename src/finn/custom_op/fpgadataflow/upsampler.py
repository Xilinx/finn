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
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class UpsampleNearestNeighbour(HWCustomOp):
    """Abstraction layer for HW implementation of UpsampleNearestNeighbour."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # Size of the output feature map
            "OFMDim": ("i", True, 0),
            # Size of the input feature map
            "IFMDim": ("i", True, 0),
            # Amount of channels of the input feature map
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # Batch size
            "numInputVectors": ("i", False, 1),
            # Dimensionality mode: 0 = 2D square, 1 = 1D in H dim
            "DimMode": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_exp_cycles(self):
        OFMDim = self.get_nodeattr("OFMDim")
        batch_size = self.get_nodeattr("numInputVectors")
        is_2d = self.get_nodeattr("DimMode") == 0
        reps = 1
        if is_2d:
            OFMDim = OFMDim * OFMDim
            reps = batch_size
        exp_cycles = OFMDim * reps
        return int(exp_cycles)

    def get_normal_input_shape(self, ind=0):
        IFMDim = self.get_nodeattr("IFMDim")
        num_ch = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("numInputVectors")
        is_2d = self.get_nodeattr("DimMode") == 0
        if is_2d:
            ishape = (batch, IFMDim, IFMDim, num_ch)
        else:
            ishape = (batch, IFMDim, 1, num_ch)
        return ishape

    def get_normal_output_shape(self, ind=0):
        OFMDim = self.get_nodeattr("OFMDim")
        num_ch = self.get_nodeattr("NumChannels")
        batch = self.get_nodeattr("numInputVectors")
        is_2d = self.get_nodeattr("DimMode") == 0
        if is_2d:
            oshape = (batch, OFMDim, OFMDim, num_ch)
        else:
            oshape = (batch, OFMDim, 1, num_ch)
        return oshape

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        return tuple(normal_ishape)

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        return tuple(normal_oshape)

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
        ifm_ch = self.get_nodeattr("NumChannels")
        return ibits * ifm_ch

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        ifm_ch = self.get_nodeattr("NumChannels")
        return obits * ifm_ch

    def execute_node(self, context, graph):
        # create a standard resize node to help calculate the result
        node = self.onnx_node
        inp_values = context[node.input[0]]
        ishape = inp_values.shape
        odim = self.get_nodeattr("OFMDim")
        idim = self.get_nodeattr("IFMDim")
        if ishape[1] == ishape[2]:
            scales_val = [1, int(round(odim / idim)), int(round(odim / idim)), 1]
        elif ishape[1] > 1 and ishape[2] == 1:
            scales_val = [1, int(round(odim / idim)), 1, 1]
        else:
            warnings.warn(
                """HW abstraction layer for Upsample cannot be executed.
            Upsampling only supported for 1D H, or 2D square scaling"""
            )
        oshape = context[node.output[0]].shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        node_resize = helper.make_node(
            "Resize",
            inputs=[node.input[0], "", "scales"],
            outputs=[node.output[0]],
            mode="nearest",
        )
        graph_resize = helper.make_graph(
            nodes=[node_resize],
            name="single-resize-exec",
            inputs=[inp, scales],
            outputs=[outp],
        )

        opset_version = 13
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_resize = qonnx_make_model(graph_resize, **onnx_kwargs)
        idict = {node.input[0]: inp_values, "scales": scales_val}
        sess = rt.InferenceSession(model_resize.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)
