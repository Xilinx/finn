# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class HWWhere(HWCustomOp):
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
                # Where has three streaming inputs, unlike the base class default.
                "inFIFODepths": ("ints", False, [2, 2, 2]),
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

    def _input_gen_buffer_specs(self):
        """Return the (width, depth) of each RTL input_gen buffer.

        The depth here must match the buffer size that ``input_gen.sv`` derives
        at elaboration from its FM_SIZE/DIMS/COEFS parameters (see where.sv's
        INIT_FM_SIZE/INIT_COEFS, which feed those parameters). Resource
        estimation has no access to the elaborated RTL, so this recomputes the
        same occupancy bound in Python; keep it in sync with input_gen.sv.
        """

        out_shape = self._rtl_shape(self.get_normal_output_shape())
        out_pe = self._output_stream_pe()
        out_dims = list(out_shape)
        out_dims[-1] //= out_pe
        specs = []

        for ind in range(3):
            in_shape = self._rtl_shape(self.get_normal_input_shape(ind))
            assert len(in_shape) <= len(out_shape), "Input rank must not exceed output rank"
            aligned_shape = (1,) * (len(out_shape) - len(in_shape)) + in_shape

            def word_dim(axis):
                dim = aligned_shape[axis]
                if axis == len(out_shape) - 1 and dim != 1:
                    dim //= out_pe
                return dim

            fm_size = int(np.prod([word_dim(i) for i in range(len(out_shape))]))
            coefs = []
            for axis, dim in enumerate(aligned_shape):
                # Replay (coef 0) only when the operand dim is 1 and the output
                # dim is >1, matching where.sv's INIT_COEFS. When the output dim
                # is also 1 the RTL keeps a real stride, so mirror that here.
                if dim == 1 and out_dims[axis] > 1:
                    coefs.append(0)
                else:
                    coefs.append(
                        int(np.prod([word_dim(i) for i in range(axis + 1, len(out_shape))]))
                    )

            # Mirror the occupancy recurrence input_gen.sv evaluates at
            # elaboration to size its circular replay buffer.
            weights = [fm_size] + coefs
            free_flags = [True]
            for axis, coef in enumerate(coefs):
                free_flags.append(
                    free_flags[-1] and coef > 0 and coef * out_dims[axis] <= weights[axis]
                )

            max_occupancy = 0
            read_rewind = 0
            free_rewind = 0
            for axis in range(len(out_shape) - 1, -1, -1):
                inner_read_rewind = read_rewind
                inner_free_rewind = free_rewind
                read_rewind = (out_dims[axis] - 1) * coefs[axis] + read_rewind
                free_rewind = (
                    (out_dims[axis] - 1) * coefs[axis] + free_rewind if free_flags[axis + 1] else 0
                )
                max_occupancy = max(max_occupancy, read_rewind - free_rewind)
                if free_flags[axis]:
                    burst = max(weights[axis] - free_rewind, 0)
                    required = inner_read_rewind - inner_free_rewind + burst
                    max_occupancy = max(max_occupancy, required)

            # input_gen adds one write-pointer delay and two safety entries,
            # then rounds the buffer depth up to a power of two.
            required_depth = max_occupancy + 3
            buffer_depth = 1 << (required_depth - 1).bit_length()
            specs.append((self.get_instream_width(ind), buffer_depth))

        return specs

    @staticmethod
    def _bram18_estimation(width, depth):
        if width == 1:
            return math.ceil(depth / 16384)
        if width == 2:
            return math.ceil(depth / 8192)
        if width <= 4:
            return math.ceil(depth / 4096) * math.ceil(width / 4)
        if width <= 9:
            return math.ceil(depth / 2048) * math.ceil(width / 9)
        if width <= 18 or depth > 512:
            return math.ceil(depth / 1024) * math.ceil(width / 18)
        return math.ceil(depth / 512) * math.ceil(width / 36)

    def bram_estimation(self):
        ram_style = self.get_nodeattr("ram_style")
        if ram_style == "block":
            buffer_specs = self._input_gen_buffer_specs()
        elif ram_style == "auto":
            # Vivado maps small buffers to distributed RAM; only buffers of at
            # least ~1 kbit are expected to end up in BRAM.
            buffer_specs = [
                (width, depth)
                for width, depth in self._input_gen_buffer_specs()
                if width * depth >= 1024
            ]
        else:
            return 0
        return int(sum(self._bram18_estimation(width, depth) for width, depth in buffer_specs))

    def uram_estimation(self):
        if self.get_nodeattr("ram_style") != "ultra":
            return 0
        return int(
            sum(
                math.ceil(width / 72) * math.ceil(depth / 4096)
                for width, depth in self._input_gen_buffer_specs()
            )
        )

    def bram_efficiency_estimation(self):
        bram_estimate = self.bram_estimation()
        if bram_estimate == 0:
            return 1
        buffer_specs = self._input_gen_buffer_specs()
        if self.get_nodeattr("ram_style") == "auto":
            # Match bram_estimation's ~1 kbit distributed-vs-BRAM split.
            buffer_specs = [
                (width, depth) for width, depth in buffer_specs if width * depth >= 1024
            ]
        used_bits = sum(width * depth for width, depth in buffer_specs)
        return used_bits / (bram_estimate * 36 * 512)

    def uram_efficiency_estimation(self):
        uram_estimate = self.uram_estimation()
        if uram_estimate == 0:
            return 1
        used_bits = sum(width * depth for width, depth in self._input_gen_buffer_specs())
        return used_bits / (uram_estimate * 72 * 4096)

    def lut_estimation(self):
        selection_luts = 64 + self.get_nodeattr("PE") * self.get_output_datatype().bitwidth()
        ram_style = self.get_nodeattr("ram_style")
        if ram_style == "distributed":
            buffer_specs = self._input_gen_buffer_specs()
        elif ram_style == "auto":
            # Complement of bram_estimation: buffers below ~1 kbit stay in LUTRAM.
            buffer_specs = [
                (width, depth)
                for width, depth in self._input_gen_buffer_specs()
                if width * depth < 1024
            ]
        else:
            buffer_specs = []
        if buffer_specs:
            buffer_luts = sum(width * math.ceil(depth / 64) for width, depth in buffer_specs)
        else:
            buffer_luts = 0
        return int(selection_luts + buffer_luts)

    def get_op_and_param_counts(self):
        return {"op_where": int(np.prod(self.get_normal_output_shape()))}
