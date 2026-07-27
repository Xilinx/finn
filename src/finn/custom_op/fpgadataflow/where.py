# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
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
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Where(HWCustomOp):
    """Elementwise ONNX Where with multidirectional broadcasting."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                "Shape": ("ints", True, []),
                "CondShape": ("ints", False, []),
                "XShape": ("ints", False, []),
                "YShape": ("ints", False, []),
                "CondRank": ("i", False, -1),
                "XRank": ("i", False, -1),
                "YRank": ("i", False, -1),
                "PE": ("i", False, 1),
                "conditionDataType": ("s", False, "BINARY"),
                "inputDataType": ("s", True, ""),
                "outputDataType": ("s", False, ""),
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "block", "distributed", "ultra"},
                ),
                "inFIFODepths": ("ints", False, [2, 2, 2]),
                "outFIFODepths": ("ints", False, [2]),
            }
        )
        return my_attrs

    def _shape(self):
        return tuple(self.get_nodeattr("Shape"))

    def _input_shape(self, ind):
        if ind == 0:
            attr_name, rank_name = "CondShape", "CondRank"
        elif ind == 1:
            attr_name, rank_name = "XShape", "XRank"
        elif ind == 2:
            attr_name, rank_name = "YShape", "YRank"
        else:
            raise Exception("Where has exactly three inputs")

        rank = self.get_nodeattr(rank_name)
        shape = tuple(self.get_nodeattr(attr_name))
        if rank >= 0:
            assert len(shape) == rank, "%s length must match %s" % (
                attr_name,
                rank_name,
            )
            return shape
        if len(shape) != 0:
            return shape
        return self._shape()

    def _rtl_shape(self, shape):
        if len(shape) == 0:
            return (1,)
        return tuple(shape)

    def _input_stream_pe(self, ind):
        shape = self._rtl_shape(self.get_normal_input_shape(ind))
        if shape[-1] == 1:
            return 1
        return self._output_stream_pe()

    def _output_stream_pe(self):
        shape = self._rtl_shape(self.get_normal_output_shape())
        if shape[-1] == 1:
            return 1
        return self.get_nodeattr("PE")

    def _folded_shape(self, shape, stream_pe):
        rtl_shape = self._rtl_shape(shape)
        *outer, channels = rtl_shape
        assert channels % stream_pe == 0, "Stream PE must divide the innermost dimension"
        return tuple(outer + [channels // stream_pe, stream_pe])

    def get_normal_input_shape(self, ind=0):
        if ind not in [0, 1, 2]:
            raise Exception("Where has exactly three inputs")
        return self._input_shape(ind)

    def get_folded_input_shape(self, ind=0):
        return self._folded_shape(self.get_normal_input_shape(ind), self._input_stream_pe(ind))

    def get_normal_output_shape(self, ind=0):
        if ind != 0:
            raise Exception("Where has exactly one output")
        return self._shape()

    def get_folded_output_shape(self, ind=0):
        return self._folded_shape(self.get_normal_output_shape(ind), self._output_stream_pe())

    def make_shape_compatible_op(self, model):
        for i, inp in enumerate(self.onnx_node.input):
            ishape = tuple(model.get_tensor_shape(inp))
            assert ishape == self.get_normal_input_shape(i), (
                "Unexpected input shape for Where input %d." % i
            )
        return super().make_const_shape_op(self.get_normal_output_shape())

    def infer_node_datatype(self, model):
        node = self.onnx_node

        cond_dt = model.get_tensor_datatype(node.input[0])
        if cond_dt is None:
            cond_dt = self.get_condition_datatype()
            model.set_tensor_datatype(node.input[0], cond_dt)
        if cond_dt != DataType["BINARY"]:
            raise Exception("Where condition datatype must be BINARY")
        self.set_nodeattr("conditionDataType", cond_dt.name)

        attr_idt = None
        if self.get_nodeattr("inputDataType") != "":
            attr_idt = self.get_input_datatype(1)

        x_dt = model.get_tensor_datatype(node.input[1])
        y_dt = model.get_tensor_datatype(node.input[2])
        idt = x_dt if x_dt is not None else attr_idt
        if idt is None:
            raise Exception("Where input datatype is not set")
        if y_dt is None:
            model.set_tensor_datatype(node.input[2], idt)
        elif y_dt != idt:
            raise Exception("Where X and Y datatypes must match")
        if x_dt is None:
            model.set_tensor_datatype(node.input[1], idt)

        if attr_idt is not None and attr_idt != idt:
            warnings.warn(
                "inputDataType changing for %s: %s -> %s" % (node.name, str(attr_idt), str(idt))
            )
        self.set_nodeattr("inputDataType", idt.name)

        attr_odt = self.get_nodeattr("outputDataType")
        if attr_odt != "" and DataType[attr_odt] != idt:
            warnings.warn(
                "outputDataType changing for %s: %s -> %s"
                % (node.name, str(DataType[attr_odt]), str(idt))
            )
        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        pass

    def get_condition_datatype(self):
        return DataType[self.get_nodeattr("conditionDataType")]

    def get_input_datatype(self, ind=0):
        if ind == 0:
            return self.get_condition_datatype()
        if ind in [1, 2]:
            return DataType[self.get_nodeattr("inputDataType")]
        raise Exception("Where has exactly three inputs")

    def get_output_datatype(self, ind=0):
        odt = self.get_nodeattr("outputDataType")
        if odt == "":
            return self.get_input_datatype(1)
        return DataType[odt]

    def get_instream_width(self, ind=0):
        if ind == 0:
            return self._input_stream_pe(ind)
        if ind in [1, 2]:
            return self.get_input_datatype(ind).bitwidth() * self._input_stream_pe(ind)
        return 0

    def get_outstream_width(self, ind=0):
        return self.get_output_datatype(ind).bitwidth() * self._output_stream_pe()

    def get_number_output_values(self):
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_exp_cycles(self):
        input_cycles = max(int(np.prod(self.get_folded_input_shape(ind)[:-1])) for ind in range(3))
        output_cycles = self.get_number_output_values()
        return input_cycles + output_cycles + 4

    def execute_node(self, context, graph):
        node = self.onnx_node
        cond = context[node.input[0]]
        xval = context[node.input[1]]
        yval = context[node.input[2]]

        result = np.where(cond.astype(bool), xval, yval)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(
            self.get_normal_output_shape()
        )

    def bram_estimation(self):
        return 0

    def lut_estimation(self):
        return int(64 + self.get_nodeattr("PE") * self.get_output_datatype().bitwidth())

    def get_op_and_param_counts(self):
        return {"op_where": int(np.prod(self.get_normal_output_shape()))}
