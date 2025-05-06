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

# ONNX i/o tensor shape assumptions for channelwise ops:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the channelwise parameter tensor, shape (NumChannels, params_per_channel)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


def get_smallest_possible(vals):
    """Returns smallest (fewest bits) possible DataType that can represent
    value. Prefers unsigned integers where possible."""
    vals = np.array(vals, dtype=np.float64)
    for v in vals:
        assert int(v) == v, "Error float value"

    for k in DataType.get_accumulator_dt_cands():
        dt = DataType[k]

        if dt in [DataType["BIPOLAR"], DataType["TERNARY"], DataType["FLOAT32"]]:
            # not currently supported
            continue

        if (dt.min() <= vals).all() and (vals <= dt.max()).all():
            return dt

    warnings.warn(
        """InferChannelwiseLinearLayer: Output values may not be
    representable with supported data types.
    Setting maximum width data type available.
    This will lead to errors if there are no constrains on the input
    """
    )

    if (0 <= vals).all():
        return DataType["UINT64"]
    else:
        return DataType["INT64"]


class ChannelwiseOp(HWCustomOp):
    """Abstraction layer for HW implementation of ChannelwiseOp."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # channelwise "map" function to apply:
            # one of cmp_le, cmp_ge, add, mul
            "Func": ("s", False, "cmp_le", {"cmp_le", "cmp_ge", "add", "mul"}),
            "PE": ("i", True, 0),
            "NumChannels": ("i", True, 0),
            # string defining memory resource type for parameters
            "ram_style": ("s", False, "distributed", {"distributed", "block"}),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "paramDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def calc_tmem(self):
        """Calculates and returns TMEM, the depth of the memory used
        to store the channelwise op parameters."""
        chn = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return chn // pe

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # check input datatype against property
        idt = model.get_tensor_datatype(node.input[0])

        exp_idt_name = self.get_nodeattr("inputDataType")
        if exp_idt_name != idt.name:
            func = self.get_nodeattr("Func")
            assert func in ["add", "mul"], "Bad input DataType for ChannelwiseOp layer"

            self.set_nodeattr("inputDataType", idt.name)
            # update the func in ['add','mul'] cases

            # get parameter ranges
            param = model.get_initializer(node.input[1])
            param_min = min(param.flatten())
            param_max = max(param.flatten())

            # set function and determine output data type
            if func == "add":
                out_min = idt.min() + param_min
                out_max = idt.max() + param_max
                odt = get_smallest_possible([out_min, out_max])
            elif func == "mul":
                possible_limits = []
                possible_limits += [idt.min() * param_min]
                possible_limits += [idt.min() * param_max]
                possible_limits += [idt.max() * param_min]
                possible_limits += [idt.max() * param_max]
                odt = get_smallest_possible(possible_limits)

            self.set_nodeattr("outputDataType", odt.name)

        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            return DataType[self.get_nodeattr("inputDataType")]
        elif ind == 1:
            return DataType[self.get_nodeattr("paramDataType")]
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = self.get_input_datatype().bitwidth()
            return i_bits * self.get_nodeattr("PE")
        elif ind == 1:
            # param input is not exposed so width can default to 0
            return 0
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def get_folded_input_shape(self, ind=0):
        if ind == 0:
            ich = self.get_nodeattr("NumChannels")
            pe = self.get_nodeattr("PE")
            fold = ich // pe
            vecs = list(self.get_nodeattr("numInputVectors"))
            folded_input_shape = tuple(vecs + [fold, pe])
            return folded_input_shape
        elif ind == 1:
            return self.get_normal_input_shape(ind)
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_folded_output_shape(self, ind=0):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        if ind == 0:
            vecs = list(self.get_nodeattr("numInputVectors"))
            normal_input_shape = tuple(vecs + [ich])
        elif ind == 1:
            normal_input_shape = tuple(
                [ich],
            )
        else:
            raise Exception("Undefined input ind for this layer type")
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        # same shape as input
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def execute_node(self, context, graph):
        # create a standard onnx node to help calculate the result
        # depending on Func node attribute either a Mul or an Add node
        node = self.onnx_node
        func = self.get_nodeattr("Func")
        inp_values = context[node.input[0]]
        param_values = context[node.input[1]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        pshape = param_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        param = helper.make_tensor_value_info(node.input[1], TensorProto.FLOAT, pshape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        node_func = helper.make_node(
            func.capitalize(),
            inputs=node.input,
            outputs=[node.output[0]],
        )
        graph_func = helper.make_graph(
            nodes=[node_func],
            name="single-add-exec",
            inputs=[inp, param],
            outputs=[outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_func = qonnx_make_model(graph_func, **onnx_kwargs)
        idict = {node.input[0]: inp_values, node.input[1]: param_values}
        sess = rt.InferenceSession(model_func.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)
