# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import Characteristic_Node


class Pad1D(HWCustomOp):
    """One-dimensional padding for token streams.

    The first input is the streamed token sequence with shape
    ``(1, NumTokens, NumChannels)``. Optional second and third inputs provide
    left and right padding data. Each pad input can be a single token
    ``(1, 1, NumChannels)`` to be repeated, or a full pad sequence
    ``(1, PadLeft/PadRight, NumChannels)``.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                "NumTokens": ("i", True, 0),
                "NumChannels": ("i", True, 0),
                "PadLeft": ("i", False, 0),
                "PadRight": ("i", False, 0),
                "SIMD": ("i", False, 1),
                "inputDataType": ("s", True, ""),
                "outputDataType": ("s", False, ""),
            }
        )
        return my_attrs

    def _get_pad_count(self, ind):
        if ind == 1:
            return self.get_nodeattr("PadLeft")
        elif ind == 2:
            return self.get_nodeattr("PadRight")
        else:
            raise Exception("Pad1D pad inputs are indices 1 and 2")

    def _get_pad_side_name(self, ind):
        if ind == 1:
            return "left"
        elif ind == 2:
            return "right"
        else:
            raise Exception("Pad1D pad inputs are indices 1 and 2")

    def _get_pad_input_shape(self, ind):
        num_channels = self.get_nodeattr("NumChannels")
        pad_count = self._get_pad_count(ind)
        return (1, max(1, pad_count), num_channels)

    def _validate_pad_shape(self, shape, ind):
        pad_count = self._get_pad_count(ind)
        num_channels = self.get_nodeattr("NumChannels")
        valid_shapes = {(1, max(1, pad_count), num_channels)}
        if pad_count > 1:
            valid_shapes.add((1, 1, num_channels))
        assert tuple(shape) in valid_shapes, "Pad1D %s pad shape must be one of %s, got %s" % (
            self._get_pad_side_name(ind),
            sorted(valid_shapes),
            tuple(shape),
        )

    def get_normal_input_shape(self, ind=0):
        num_channels = self.get_nodeattr("NumChannels")
        if ind == 0:
            return (1, self.get_nodeattr("NumTokens"), num_channels)
        elif ind in [1, 2]:
            return self._get_pad_input_shape(ind)
        else:
            raise Exception("Pad1D has at most three inputs")

    def get_folded_input_shape(self, ind=0):
        normal_shape = self.get_normal_input_shape(ind)
        if ind != 0:
            return normal_shape

        simd = self.get_nodeattr("SIMD")
        num_channels = normal_shape[-1]
        assert num_channels % simd == 0, "SIMD must divide NumChannels"
        return normal_shape[:-1] + (num_channels // simd, simd)

    def get_normal_output_shape(self, ind=0):
        num_tokens = self.get_nodeattr("NumTokens")
        num_channels = self.get_nodeattr("NumChannels")
        pad_left = self.get_nodeattr("PadLeft")
        pad_right = self.get_nodeattr("PadRight")
        return (1, num_tokens + pad_left + pad_right, num_channels)

    def get_folded_output_shape(self, ind=0):
        normal_shape = self.get_normal_output_shape(ind)
        simd = self.get_nodeattr("SIMD")
        num_channels = normal_shape[-1]
        assert num_channels % simd == 0, "SIMD must divide NumChannels"
        return normal_shape[:-1] + (num_channels // simd, simd)

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape(0)
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape for Pad1D tokens."

        for ind in [1, 2]:
            if len(self.onnx_node.input) <= ind:
                assert self._get_pad_count(ind) == 0, "Pad1D %s padding requires input index %d" % (
                    self._get_pad_side_name(ind),
                    ind,
                )
                continue

            pad_name = self.onnx_node.input[ind]
            pad_shape = model.get_tensor_shape(pad_name)
            if pad_shape is None:
                pad_init = model.get_initializer(pad_name)
                if pad_init is not None:
                    pad_shape = pad_init.shape
            if pad_shape is not None:
                self._validate_pad_shape(pad_shape, ind)

        return super().make_const_shape_op(self.get_normal_output_shape())

    def infer_node_datatype(self, model):
        node = self.onnx_node
        attr_idt = None
        if self.get_nodeattr("inputDataType") != "":
            attr_idt = self.get_input_datatype()

        idt = model.get_tensor_datatype(node.input[0])
        if idt is None:
            idt = attr_idt
        if idt is None:
            raise Exception("Pad1D input datatype is not set")

        if attr_idt is not None and attr_idt != idt:
            warnings.warn(
                "inputDataType changing for %s: %s -> %s" % (node.name, str(attr_idt), str(idt))
            )
        self.set_nodeattr("inputDataType", idt.name)

        for pad_input in node.input[1:]:
            pad_dt = model.get_tensor_datatype(pad_input)
            if pad_dt is None:
                model.set_tensor_datatype(pad_input, idt)
            else:
                assert pad_dt == idt, "Pad1D pad datatype must match input datatype."

        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        assert self.get_nodeattr("NumTokens") > 0, "NumTokens must be positive"
        assert self.get_nodeattr("NumChannels") > 0, "NumChannels must be positive"
        assert self.get_nodeattr("PadLeft") >= 0, "PadLeft cannot be negative"
        assert self.get_nodeattr("PadRight") >= 0, "PadRight cannot be negative"
        assert self.get_nodeattr("SIMD") > 0, "SIMD must be positive"

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        odt = self.get_nodeattr("outputDataType")
        if odt == "":
            return self.get_input_datatype(ind)
        return DataType[odt]

    def get_instream_width(self, ind=0):
        if ind != 0:
            return 0
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("SIMD")

    def get_outstream_width(self, ind=0):
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("SIMD")

    def get_exp_cycles(self):
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_tree_model(self):
        """Emits the left pad, then the input sequence, then the right pad.

        The layer writes one folded word every cycle for the whole output
        sequence and never stalls: the pad words come from constants held in the
        design, so the input stream is read only while the sequence itself is
        being passed through. That splits one period into three back-to-back
        phases -- ``PadLeft`` tokens written with nothing read, ``NumTokens``
        tokens read and written, ``PadRight`` tokens written with nothing read --
        each token being ``NumChannels / SIMD`` folded words. The period is the
        output frame, which is what ``get_exp_cycles`` returns.

        The single-cycle output register between reading a word and writing it is
        not modelled; the wind-up is shorter than one cycle of the period rtlsim
        measures and so contributes nothing to it.

        Valid for the RTL implementation at any SIMD dividing NumChannels and any
        pad widths. Assumes the module accepts a word every cycle, as
        ``get_exp_cycles`` does.
        """
        if not self.onnx_node.op_type.endswith("_rtl"):
            return None
        simd = self.get_nodeattr("SIMD")
        num_channels = self.get_nodeattr("NumChannels")
        num_tokens = self.get_nodeattr("NumTokens")
        pad_left = self.get_nodeattr("PadLeft")
        pad_right = self.get_nodeattr("PadRight")
        if simd < 1 or num_channels % simd != 0:
            return None
        folds = num_channels // simd
        if folds < 1 or num_tokens < 1 or min(pad_left, pad_right) < 0:
            return None
        phases = []
        if pad_left > 0:
            left = Characteristic_Node("emit the left pad", [(pad_left * folds, [0, 1])], True)
            phases.append((1, left))
        thru = Characteristic_Node("pass the sequence on", [(num_tokens * folds, [1, 1])], True)
        phases.append((1, thru))
        if pad_right > 0:
            right = Characteristic_Node("emit the right pad", [(pad_right * folds, [0, 1])], True)
            phases.append((1, right))
        return Characteristic_Node("pad a sequence", phases, False)

    def _get_expanded_pad_values(self, context, ind):
        pad_count = self._get_pad_count(ind)
        num_channels = self.get_nodeattr("NumChannels")
        if pad_count == 0:
            return np.zeros((1, 0, num_channels), dtype=np.float32)

        if len(self.onnx_node.input) <= ind:
            raise Exception(
                "Pad1D %s padding requires input index %d" % (self._get_pad_side_name(ind), ind)
            )

        pad_values = np.asarray(context[self.onnx_node.input[ind]], dtype=np.float32)
        self._validate_pad_shape(pad_values.shape, ind)
        if pad_values.shape[1] == 1 and pad_count > 1:
            return np.repeat(pad_values, pad_count, axis=1)
        return pad_values

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp = context[node.input[0]]

        values = [self._get_expanded_pad_values(context, 1), inp]
        values.append(self._get_expanded_pad_values(context, 2))

        result = np.concatenate(values, axis=1)
        oshape = self.get_normal_output_shape()
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

    def bram_estimation(self):
        return 0

    def lut_estimation(self):
        pad_tokens = self.get_nodeattr("PadLeft") + self.get_nodeattr("PadRight")
        return int(128 + self.get_nodeattr("NumChannels") * max(1, pad_tokens))

    def get_op_and_param_counts(self):
        num_channels = self.get_nodeattr("NumChannels")
        pad_tokens = self.get_nodeattr("PadLeft") + self.get_nodeattr("PadRight")
        return {"param_pad_tokens": int(num_channels * pad_tokens)}
