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
            "SIMD": ("i", True, 0),
            # Height, width of the output feature map
            "HO": ("i", True, 0),
            "WO": ("i", True, 0),
            # Height, width of the input feature map
            "HI": ("i", True, 0),
            "WI": ("i", True, 0),
            # Amount of channels of the input feature map
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # Batch size
            "batchSize": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_exp_cycles(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_normal_input_shape(self, ind=0):
        batch = self.get_nodeattr("batchSize")
        HI = self.get_nodeattr("HI")
        WI = self.get_nodeattr("WI")
        num_ch = self.get_nodeattr("NumChannels")
        ishape = (batch, HI, WI, num_ch)
        return ishape

    def get_normal_output_shape(self, ind=0):
        batch = self.get_nodeattr("batchSize")
        HO = self.get_nodeattr("HO")
        WO = self.get_nodeattr("WO")
        num_ch = self.get_nodeattr("NumChannels")
        oshape = (batch, HO, WO, num_ch)
        return oshape

    def get_folded_input_shape(self, ind=0):
        spatial_shape = list(self.get_normal_input_shape())[:-1]
        simd = self.get_nodeattr("SIMD")
        folds = self.get_nodeattr("NumChannels") // simd
        return tuple(spatial_shape + [folds, simd])

    def get_folded_output_shape(self, ind=0):
        spatial_shape = list(self.get_normal_output_shape())[:-1]
        simd = self.get_nodeattr("SIMD")
        folds = self.get_nodeattr("NumChannels") // simd
        return tuple(spatial_shape + [folds, simd])

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

    def execute_node(self, context, graph):
        # create a standard resize node to help calculate the result
        node = self.onnx_node
        inp_values = context[node.input[0]]
        ishape = inp_values.shape
        HO = self.get_nodeattr("HO")
        WO = self.get_nodeattr("WO")
        HI = self.get_nodeattr("HI")
        WI = self.get_nodeattr("WI")
        scales_val = [1, int(round(HO / HI)), int(round(WO / WI)), 1]
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

        opset_imports = [helper.make_opsetid("", 13)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_resize = qonnx_make_model(graph_resize, **onnx_kwargs)
        idict = {node.input[0]: inp_values, "scales": scales_val}
        sess = rt.InferenceSession(model_resize.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)
