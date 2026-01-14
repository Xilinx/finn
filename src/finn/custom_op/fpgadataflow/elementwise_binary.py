# Copyright (C) 2025, Advanced Micro Devices, Inc.
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
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


# Generic implementation for elementwise binary operations
class ElementwiseBinaryOperation(HWCustomOp):
    # Specifies the elementwise operation to be implemented
    #   Format: (Identifier, Python, C++, RTL)
    _operation: tuple[str, np.ufunc, str, str] | None = None

    # Numpy operation available as property
    @property
    def npy_op(self) -> np.ufunc:
        return self._operation[1]

    # C++ operation template available as property
    @property
    def cpp_op(self) -> str:
        return self._operation[2]

    # RTL operation template available as property
    @property
    def rtl_op(self) -> str:
        return self._operation[3]

    # Initializes the operator given an onnx graph node
    def __init__(self, onnx_node, **kwargs):
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

    # Defines attributes which must be present on this node
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update(
            {
                # Data type of the left-hand-side input elements
                "lhs_dtype": ("s", True, ""),
                # Data type of the right-hand-side input elements
                "rhs_dtype": ("s", True, ""),
                # Data type of the output elements
                "out_dtype": ("s", True, ""),
                # Shape of the left-hand-side input
                "lhs_shape": ("ints", True, [1]),
                # Shape of the right-hand-side input
                "rhs_shape": ("ints", True, [1]),
                # Shape of the output, mus correspond to multi-directional
                # broadcasting of the left- and right-hand-side
                "out_shape": ("ints", True, [1]),
                # Style specifies how the left-hand-side input is provided
                #   Note: Might be inferred from the context
                "lhs_style": ("s", False, "input", {"input", "const"}),
                # Style specifies how the right-hand-side input is provided
                #   Note: Might be inferred from the context
                "rhs_style": ("s", False, "input", {"input", "const"}),
                # Number of elements in the last dimensions processed in parallel
                "PE": ("i", False, 1),
                # FPGA resource type for memories/internal buffers of the operator
                "ram_style": ("s", False, "auto", {"auto", "block", "distributed", "ultra"}),
                # memory mode for the const value
                # internal_embedded -- embedded parameters
                # internal_decoupled -- streaming parameters with streamer packaged inside IP
                "mem_mode": (
                    "s",
                    False,
                    "internal_embedded",
                    {"internal_embedded", "internal_decoupled"},
                ),
                # Input and output FIFO depths for multi-I/O nodes
                #   Note: Need to override here as there might be two inputs
                "inFIFODepths": ("ints", False, [2, 2]),
                "outFIFODepths": ("ints", False, [2]),
            }
        )
        # Return updated attribute dictionary
        return attrs

    # Datatype attribute as property for convenience
    @property
    def lhs_dtype(self):
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("lhs_dtype")]

    # Datatype attribute as property for convenience
    @property
    def rhs_dtype(self):
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("rhs_dtype")]

    # Datatype attribute as property for convenience
    @property
    def out_dtype(self):
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("out_dtype")]

    # Shape attribute as property for convenience
    @property
    def lhs_shape(self):
        return self.get_nodeattr("lhs_shape")

    # Shape attribute as property for convenience
    @property
    def rhs_shape(self):
        return self.get_nodeattr("rhs_shape")

    # Shape attribute as property for convenience
    @property
    def out_shape(self):
        return self.get_nodeattr("out_shape")

    # Style attribute as property for convenience
    @property
    def lhs_style(self):
        return self.get_nodeattr("lhs_style")

    # Style attribute as property for convenience
    @property
    def rhs_style(self):
        return self.get_nodeattr("rhs_style")

    # Number of parallel processed elements as property for convenience
    @property
    def pe(self):
        return self.get_nodeattr("PE")

    # Checks whether the last axis is broadcast
    @property
    def broadcast_last_axis(self):
        return (self.lhs_shape[-1] == 1) != (self.rhs_shape[-1] == 1)

    # Makes an operation compatible with the output shape for shape inference
    #   Note: Propagates shape forward, i.e., never asks for the shape of the
    #   output, even if it seems easier.
    def make_shape_compatible_op(self, model: ModelWrapper):
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # There must be exactly two inputs to the binary operation
        assert len(node.input) == 2, f"Binary operation {node.name} requires exactly two inputs"
        # Validate input shapes match what is stored as attributes
        assert (
            model.get_tensor_shape(node.input[0]) == self.lhs_shape
        ), f"Input shape mismatch: {node.name} {node.input[0]}"
        assert (
            model.get_tensor_shape(node.input[1]) == self.rhs_shape
        ), f"Input shape mismatch: {node.name} {node.input[1]}"
        # Validate broadcasting of inputs to the output shape
        assert (
            list(np.broadcast_shapes(self.lhs_shape, self.rhs_shape)) == self.out_shape
        ), f"Shape broadcast mismatch: {node.name}"
        # Simulate behavior via the standard ONNX add operation
        return oh.make_node("Add", node.input, node.output)

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Test for changing left-hand-side input datatype
        if model.get_tensor_datatype(node.input[0]) != self.lhs_dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            warnings.warn(f"{node.name}: lhs_dtype changing from {self.lhs_dtype} to {new_dtype}")
            # Set the new datatype attribute
            self.set_nodeattr("lhs_dtype", new_dtype.name)
        # Test for changing right-hand-side input datatype
        if model.get_tensor_datatype(node.input[1]) != self.rhs_dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[1])
            # Issue a warning message
            warnings.warn(f"{node.name}: rhs_dtype changing from {self.rhs_dtype} to {new_dtype}")
            # Set the new datatype attribute
            self.set_nodeattr("rhs_dtype", new_dtype.name)
        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.out_dtype)

    def execute_node(self, context, graph):
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the inputs out of the execution context
        lhs = context[node.input[0]]
        rhs = context[node.input[1]]
        # Note: Need to make sure these have the right type for the Numpy API
        # Note: Always simulate integer inputs in int64, numpy casting is
        # weird....
        lhs = lhs.astype(np.int64) if self.lhs_dtype.is_integer() else lhs
        rhs = rhs.astype(np.int64) if self.rhs_dtype.is_integer() else rhs
        # Apply elementwise operation with broadcasting in numpy and insert
        # result into the execution context
        out = self.npy_op(lhs, rhs)
        # Make sure the output has the right type, e.g. turn all booleans into
        # integers (actually floats as the container type)
        # Note: This is relevant for logical ops, ==, <=, >=, etc.
        # Note: Somehow QONNX does not like boolean tensors
        # context[node.output[0]] = out.astype(self.out_dtype.to_numpy_dt())
        # TODO: Apparently it is not? Verify this behavior...
        context[node.output[0]] = out.astype(np.float32)

    # Note: End of QONNX CustomOp region, below is FINN HWCustomOp stuff

    # Gets the datatype of input at index ind
    def get_input_datatype(self, ind=0):
        # Get input data type by index, order inputs from left to right
        return [self.lhs_dtype, self.rhs_dtype][ind]

    # Gets the datatype of the output at index ind
    def get_output_datatype(self, ind=0):
        # There is only one output, the type is set as an attribute
        return self.out_dtype

    # Gets the shape of the input at index ind without folding
    def get_normal_input_shape(self, ind=0):
        # Input shapes are stored as a node attributes
        return [self.lhs_shape, self.rhs_shape][ind]

    # Gets the shape of the output at index ind without folding
    def get_normal_output_shape(self, ind=0):
        # The output shape is stored as a node attribute
        return self.out_shape

    # Gets the shape of the input at index ind with folding
    def get_folded_input_shape(self, ind=0):
        # Get the normal shape before applying folding
        *num_inputs, num_elems = self.get_normal_input_shape(ind=ind)
        # Folding only applies if the folded axis is not broadcast
        if not self.broadcast_last_axis or num_elems != 1:
            # Valid folding requires the PE to divide the number of elements
            assert num_elems % self.pe == 0, "PE must divide last axis"
            # Folding along the last dimension
            return *num_inputs, num_elems // self.pe, self.pe
        # For broadcast axes return the non-folded shape with dummy axis
        # inserted
        return *num_inputs, 1, num_elems

    # Gets the shape of the output at index ind with folding
    def get_folded_output_shape(self, ind=0):
        # Get the normal shape before applying folding
        *num_inputs, num_elems = self.get_normal_output_shape(ind=ind)
        # Valid folding requires the PE to divide the number of elements
        assert num_elems % self.pe == 0, "PE must divide last axis"
        # Folding along the last dimension
        return *num_inputs, num_elems // self.pe, self.pe

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        folded_shape = self.get_folded_input_shape(ind=1)
        return np.prod(folded_shape[:-1])

    # Widths of the input data stream of the input at index ind
    def get_instream_width(self, ind=0):
        # Get the number of bits used to represent the input
        i_bits = self.get_input_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded input
        *_, elems = self.get_folded_input_shape(ind)
        # apply parallelism if broadcast
        if self.broadcast_last_axis:
            elems = elems * self.pe
        # Width of a stream receiving input elements in parallel
        return elems * i_bits

    # Widths of the output data stream of the output at index ind
    def get_outstream_width(self, ind=0):
        # Get the number of bits used to represent the output
        o_bits = self.get_output_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded output
        *_, elems = self.get_folded_output_shape(ind)
        # Width of a stream producing output elements in parallel
        return elems * o_bits

    # Minimizes the width of the accumulator data type, 'accumulator width' here
    # due to convention, it is actually the output data type
    def minimize_accumulator_width(self, model: ModelWrapper):
        # If any of the inputs is not an integer, the bit-width cannot be
        # minimized
        if not all([self.lhs_dtype.is_integer(), self.rhs_dtype.is_integer()]):
            # Check the annotated tensor data type corresponds to the stored
            # attribute
            assert (
                model.get_tensor_datatype(self.onnx_node.output[0]) == self.out_dtype
            ), f"Output type mismatch for {self.onnx_node.name}"
            # Exit here, returning the not-minimized data type
            return self.out_dtype
        # Call the output type derivation specialized by the concrete operator
        # implementation
        out_dtype = self._derive_out_dtype(model)
        # Set the new output data type as attribute
        self.set_nodeattr("out_dtype", out_dtype.name)
        # Annotate the output tensor with the new data type
        model.set_tensor_datatype(self.onnx_node.output[0], out_dtype)
        # Return the minimized output data type
        # Note: Probably not required by MinimizeAccumulatorWidth transformation
        return out_dtype

    # Derives the optimal width of the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Depends on the actual operation performed and must be specialized by
        # the concrete implementations
        raise NotImplementedError(
            f"_derive_out_dtype of {self.__class__.__name__} is not implemented!"
        )

    # Minimizes the width of the weight data type, 'weight' here due to
    # convention, it actually applies to any constant initializer input
    def minimize_weight_bit_width(self, model: ModelWrapper):
        # Check for an initializer providing the left hand side input
        lhs = model.get_initializer(self.onnx_node.input[0])
        # If the left hand side input is provided as initializer, minimize the
        # bits used for storing this
        if lhs is not None:
            # Remember the "style" of receiving the input for further code
            # generation
            self.set_nodeattr("lhs_style", "const")
            lhs_dtype = self.get_input_datatype(0)
            # ignore minimization for floats
            if not lhs_dtype.get_canonical_name().startswith("FLOAT"):
                if lhs_dtype.is_integer():
                    # Minimum and maximum "weight" on the left hand side, determining
                    # the range of values which needs to be represented
                    _min = lhs.min()
                    _max = lhs.max()
                    # Determine whether signed or unsigned type is required for
                    # representing the weights and select the largest "signed magnitude"
                    _mag = _max if _min > 0 else _min if (abs(_min) > _max) else (-_max - 1)
                    # Smallest data type large enough to represent this range of values
                    lhs_dtype = DataType.get_smallest_possible(_mag)
                elif lhs_dtype.is_fixed_point():
                    # Convert the fixed-point array to corresponding integers and get
                    # smallest integer representation
                    lhs = lhs / lhs_dtype.scale_factor()
                    _min = lhs.min()
                    _max = lhs.max()
                    _mag = _max if _min > 0 else _min if (abs(_min) > _max) else (-_max - 1)
                    dtype = DataType.get_smallest_possible(_mag)
                    _total_bits = dtype.bitwidth() if dtype.signed() else dtype.bitwidth() + 1
                    _integer_bits = _total_bits - lhs_dtype.frac_bits()
                    lhs_dtype = DataType[f"FIXED<{_total_bits},{_integer_bits}>"]

            # Update the corresponding data type attribute of the node
            self.set_nodeattr("lhs_dtype", lhs_dtype.name)
            # Annotate the tensor with the new data type
            model.set_tensor_datatype(self.onnx_node.input[0], lhs_dtype)

        # Check for an initializer providing the right hand side input
        rhs = model.get_initializer(self.onnx_node.input[1])
        # If the right hand side input is provided as initializer, minimize the
        # bits used for storing this
        if rhs is not None:
            # Remember the "style" of receiving the input for further code
            # generation
            self.set_nodeattr("rhs_style", "const")
            rhs_dtype = self.get_input_datatype(1)
            # ignore minimization for floats
            if not rhs_dtype.get_canonical_name().startswith("FLOAT"):
                if rhs_dtype.is_integer():
                    # Minimum and maximum "weight" on the left hand side, determining
                    # the range of values which needs to be represented
                    _min = rhs.min()
                    _max = rhs.max()
                    # Determine whether signed or unsigned type is required for
                    # representing the weights and select the largest "signed magnitude"
                    _mag = _max if _min > 0 else _min if (abs(_min) > _max) else (-_max - 1)
                    # Smallest data type large enough to represent this range of values
                    rhs_dtype = DataType.get_smallest_possible(_mag)
                elif rhs_dtype.is_fixed_point():
                    # Convert the fixed-point array to corresponding integers and get
                    # smallest integer representation
                    rhs = rhs / rhs_dtype.scale_factor()
                    _min = rhs.min()
                    _max = rhs.max()
                    _mag = _max if _min > 0 else _min if (abs(_min) > _max) else (-_max - 1)
                    dtype = DataType.get_smallest_possible(_mag)
                    _total_bits = dtype.bitwidth() if dtype.signed() else dtype.bitwidth() + 1
                    _integer_bits = _total_bits - rhs_dtype.frac_bits()
                    rhs_dtype = DataType[f"FIXED<{_total_bits},{_integer_bits}>"]

            # Update the corresponding data type attribute of the node
            self.set_nodeattr("rhs_dtype", rhs_dtype.name)
            # Annotate the tensor with the new data type
            model.set_tensor_datatype(self.onnx_node.input[1], rhs_dtype)

        # TODO: MVAU returns the data type here, which does not make sense for
        #  potentially two data types changing and apparently, the
        #  MinimizeWeightBitWidth transformations does not even use the returned
        #  value.

    # Derives the expected cycles for the elementwise binary operation given the
    # folding configuration
    def get_exp_cycles(self):
        # Number of iterations required to process the whole folded input stream
        #   Note: This is all but the PE (last, parallelized) dimension
        return np.prod(self.get_folded_output_shape()[:-1])


# Derive a specialization to implement elementwise addition of two inputs
@register_custom_op
class ElementwiseAdd(ElementwiseBinaryOperation):
    # Specialize to implement the addition operation of left hand side and right
    # hand side input
    _operation = "Add", np.add, "({0} + {1})", None

    # Derives the output data type according to UG1399
    def _derive_out_dtype(self, model: ModelWrapper):
        # Get the width of the data types of the inputs and the larger of the
        # two widths
        lhs_width = self.lhs_dtype.bitwidth()
        rhs_width = self.rhs_dtype.bitwidth()
        max_width = max(lhs_width, rhs_width)
        # Check whether the addition operation is a signed addition
        signed = any([self.lhs_dtype.signed(), self.rhs_dtype.signed()])
        # By default, the output is one bit more than the widest of the inputs
        out_width = max_width + 1
        # If the addition is signed, the output might be wider depending on
        # which of the inputs is signed
        if signed:
            # Find the wider and narrower of the two inputs by assuming left to
            # right order first
            wider, narrower = self.lhs_dtype, self.rhs_dtype
            # Swap if the order is not correct
            if narrower.bitwidth() > wider.bitwidth():
                wider, narrower = narrower, wider
            # If and only if the wider is unsigned and the narrower is signed,
            # add two bits to the output width
            if not wider.signed() and narrower.signed():
                # Out has two bits more than the widest input
                out_width = max_width + 2
            # The new output type is a signed integer of the calculated
            # bit-width
            return DataType[f"INT{out_width}"]
        # By default, if both inputs are unsigned, the output is unsigned as
        # well
        return DataType[f"UINT{out_width}"]


# Derive a specialization to implement elementwise subtraction of two inputs
@register_custom_op
class ElementwiseSub(ElementwiseBinaryOperation):
    # Specialize to implement the subtraction operation of left hand side and
    # right hand side input
    _operation = "Sub", np.subtract, "({0} - {1})", None

    # Derives the output data type according to UG1399
    def _derive_out_dtype(self, model: ModelWrapper):
        # Get the width of the data types of the inputs and the larger of the
        # two widths
        lhs_width = self.lhs_dtype.bitwidth()
        rhs_width = self.rhs_dtype.bitwidth()
        max_width = max(lhs_width, rhs_width)
        # Check whether the addition operation is a signed addition
        signed = any([self.lhs_dtype.signed(), self.rhs_dtype.signed()])
        # By default, the output is one bit more than the widest of the inputs
        out_width = max_width + 1
        # If the operation is signed, the output might be wider depending on
        # which of the inputs is signed
        if signed:
            # Find the wider and narrower of the two inputs by assuming left to
            # right order first
            wider, narrower = self.lhs_dtype, self.rhs_dtype
            # Swap if the order is not correct
            if narrower.bitwidth() > wider.bitwidth():
                wider, narrower = narrower, wider
            # If and only if the wider is unsigned and the narrower is signed,
            # add two bits to the output width
            if not wider.signed() and narrower.signed():
                # Out has two bits more than the widest input
                out_width = max_width + 2
        # For subtraction, the output data type is always signed
        return DataType[f"INT{out_width}"]


# Derive a specialization to implement elementwise multiplication of two inputs
@register_custom_op
class ElementwiseMul(ElementwiseBinaryOperation):
    # Specialize to implement the multiplication operation of left hand side and
    # right hand side input
    _operation = "Mul", np.multiply, "({0} * {1})", None

    # Derives the output data type according to UG1399
    def _derive_out_dtype(self, model: ModelWrapper):
        # Get the width of the data types of the inputs
        lhs_width = self.lhs_dtype.bitwidth()
        rhs_width = self.rhs_dtype.bitwidth()
        # Check whether the addition operation is a signed addition
        signed = any([self.lhs_dtype.signed(), self.rhs_dtype.signed()])
        # The width of the product is the sum of the widths of the operands.
        out_width = lhs_width + rhs_width
        # The product is treated as a signed type if either of the operands is
        # of a signed type.
        return DataType[f"INT{out_width}" if signed else f"UINT{out_width}"]


# Derive a specialization to implement elementwise division of two inputs
@register_custom_op
class ElementwiseDiv(ElementwiseBinaryOperation):
    # TODO: Not tested due to divide by zero from randomly generated inputs...
    # Specialize to implement the division operation of left hand side and
    # right hand side input
    _operation = "Div", np.divide, "({0} / {1})", None

    # Derives the output data type according to UG1399
    def _derive_out_dtype(self, model: ModelWrapper):
        # Get the width of the data types of the inputs
        lhs_width = self.lhs_dtype.bitwidth()
        # Check whether the addition operation is a signed addition
        signed = any([self.lhs_dtype.signed(), self.rhs_dtype.signed()])
        # The width of the quotient is the width of the dividend if the divisor
        # is an unsigned type. Otherwise, it is the width of the dividend plus
        # one.
        out_width = lhs_width if not self.rhs_dtype.signed() else lhs_width + 1
        # The quotient is treated as a signed type if either of the operands is
        # of a signed type.
        return DataType[f"INT{out_width}" if signed else f"UINT{out_width}"]


# TODO: ElementwiseMod - Requires extra attribute selecting the function


# Derive a specialization to implement elementwise logical and of two inputs
@register_custom_op
class ElementwiseAnd(ElementwiseBinaryOperation):
    # Specialize to implement the logical and operation of left hand side and
    # right hand side input
    _operation = "And", np.logical_and, "({0} && {1})", None

    # Derives the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Treat the boolean output of a logical operation as unsigned integer of
        # width 1, i.e., a single bit True/False
        return DataType["BINARY"]


# Derive a specialization to implement elementwise logical or of two inputs
@register_custom_op
class ElementwiseOr(ElementwiseBinaryOperation):
    # Specialize to implement the logical or operation of left hand side and
    # right hand side input
    _operation = "Or", np.logical_or, "({0} || {1})", None

    # Derives the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Treat the boolean output of a logical operation as unsigned integer of
        # width 1, i.e., a single bit True/False
        return DataType["BINARY"]


# Derive a specialization to implement elementwise logical xor of two inputs
@register_custom_op
class ElementwiseXor(ElementwiseBinaryOperation):
    # Specialize to implement the logical xor operation of left hand side and
    # right hand side input
    _operation = "Xor", np.logical_xor, "(bool({0}) != bool({1}))", None

    # Derives the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Treat the boolean output of a logical operation as unsigned integer of
        # width 1, i.e., a single bit True/False
        return DataType["BINARY"]


# Derive a specialization to implement elementwise equality of two inputs
@register_custom_op
class ElementwiseEqual(ElementwiseBinaryOperation):
    # Specialize to implement the logical equal operation of left hand side and
    # right hand side input
    _operation = "Equal", np.equal, "({0} == {1})", None

    # Derives the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Treat the boolean output of a logical operation as unsigned integer of
        # width 1, i.e., a single bit True/False
        return DataType["BINARY"]


# Derive a specialization to implement elementwise less of two inputs
@register_custom_op
class ElementwiseLess(ElementwiseBinaryOperation):
    # Specialize to implement the logical less operation of left hand side and
    # right hand side input
    _operation = "Less", np.less, "({0} < {1})", None

    # Derives the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Treat the boolean output of a logical operation as unsigned integer of
        # width 1, i.e., a single bit True/False
        return DataType["BINARY"]


# Derive a specialization to implement elementwise less or equal of two inputs
@register_custom_op
class ElementwiseLessOrEqual(ElementwiseBinaryOperation):
    # Specialize to implement the logical less or equal operation of left hand
    # side and right hand side input
    _operation = "LessOrEqual", np.less_equal, "({0} <= {1})", None

    # Derives the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Treat the boolean output of a logical operation as unsigned integer of
        # width 1, i.e., a single bit True/False
        return DataType["BINARY"]


# Derive a specialization to implement elementwise greater of two inputs
@register_custom_op
class ElementwiseGreater(ElementwiseBinaryOperation):
    # Specialize to implement the logical greater operation of left hand side
    # and right hand side input
    _operation = "Greater", np.greater, "({0} > {1})", None

    # Derives the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Treat the boolean output of a logical operation as unsigned integer of
        # width 1, i.e., a single bit True/False
        return DataType["BINARY"]


# Derive a specialization to implement elementwise greater or equal of two
# inputs
@register_custom_op
class ElementwiseGreaterOrEqual(ElementwiseBinaryOperation):
    # Specialize to implement the logical greater or equal operation of left
    # hand side and right hand side input
    _operation = "GreaterOrEqual", np.greater_equal, "({0} >= {1})", None

    # Derives the output data type
    def _derive_out_dtype(self, model: ModelWrapper):
        # Treat the boolean output of a logical operation as unsigned integer of
        # width 1, i.e., a single bit True/False
        return DataType["BINARY"]


# Derive a specialization to implement elementwise bitwise and of two inputs
@register_custom_op
class ElementwiseBitwiseAnd(ElementwiseBinaryOperation):
    # Specialize to implement the bitwise and operation of left hand side and
    # right hand side input
    _operation = "BitwiseAnd", np.bitwise_and, "({0} & {1})", None

    # Derives the output data type according to UG1399
    def _derive_out_dtype(self, model: ModelWrapper):
        # Get the width of the data types of the inputs
        lhs_width = self.lhs_dtype.bitwidth()
        rhs_width = self.rhs_dtype.bitwidth()
        # Check whether the addition operation is a signed addition
        signed = any([self.lhs_dtype.signed(), self.rhs_dtype.signed()])
        # The bitwise logical operators all return a value with a width that is
        # the maximum of the widths of the two operands.
        out_width = max(lhs_width, rhs_width)
        # The product is treated as a signed type if either of the operands is
        # of a signed type.
        return DataType[f"INT{out_width}" if signed else f"UINT{out_width}"]


# Derive a specialization to implement elementwise bitwise or of two inputs
@register_custom_op
class ElementwiseBitwiseOr(ElementwiseBinaryOperation):
    # Specialize to implement the bitwise or operation of left hand side and
    # right hand side input
    _operation = "BitwiseOr", np.bitwise_or, "({0} | {1})", None

    # Derives the output data type according to UG1399
    def _derive_out_dtype(self, model: ModelWrapper):
        # Get the width of the data types of the inputs
        lhs_width = self.lhs_dtype.bitwidth()
        rhs_width = self.rhs_dtype.bitwidth()
        # Check whether the addition operation is a signed addition
        signed = any([self.lhs_dtype.signed(), self.rhs_dtype.signed()])
        # The bitwise logical operators all return a value with a width that is
        # the maximum of the widths of the two operands.
        out_width = max(lhs_width, rhs_width)
        # The product is treated as a signed type if either of the operands is
        # of a signed type.
        return DataType[f"INT{out_width}" if signed else f"UINT{out_width}"]


# Derive a specialization to implement elementwise bitwise xor of two inputs
@register_custom_op
class ElementwiseBitwiseXor(ElementwiseBinaryOperation):
    # Specialize to implement the bitwise xor operation of left hand side and
    # right hand side input
    _operation = "BitwiseXor", np.bitwise_xor, "({0} ^ {1})", None

    # Derives the output data type according to UG1399
    def _derive_out_dtype(self, model: ModelWrapper):
        # Get the width of the data types of the inputs
        lhs_width = self.lhs_dtype.bitwidth()
        rhs_width = self.rhs_dtype.bitwidth()
        # Check whether the addition operation is a signed addition
        signed = any([self.lhs_dtype.signed(), self.rhs_dtype.signed()])
        # The bitwise logical operators all return a value with a width that is
        # the maximum of the widths of the two operands.
        out_width = max(lhs_width, rhs_width)
        # The product is treated as a signed type if either of the operands is
        # of a signed type.
        return DataType[f"INT{out_width}" if signed else f"UINT{out_width}"]


# ElementwiseBitShift - Requires extra attribute selecting the direction
@register_custom_op
class ElementwiseBitShift(ElementwiseBinaryOperation):
    # Defines attributes which must be present on this node
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = ElementwiseBinaryOperation.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update(
            {
                # Direction of the bit-shift
                "direction": ("s", True, "", {"LEFT", "RIGHT"}),
            }
        )
        # Return updated attribute dictionary
        return attrs

    @property
    def npy_op(self):
        return {"LEFT": np.left_shift, "RIGHT": np.right_shift}[self.get_nodeattr("direction")]

    # C++ operation template available as property
    @property
    def cpp_op(self) -> str:
        return {"LEFT": "({0} << {1})", "RIGHT": "({0} >> {1})"}[self.get_nodeattr("direction")]

    # RTL operation template available as property
    @property
    def rtl_op(self) -> str:
        return None

    # Derives the output data type just as annotated...
    def _derive_out_dtype(self, model: ModelWrapper):
        # The attributes decide the output datatype
        return DataType[self.get_nodeattr("out_dtype")]


# # Derive a specialization to implement elementwise power of two inputs
# TODO: std::pow does not work for HLS types and hls::pow fails to link for some
#  reason
# @register_custom_op
# class ElementwisePow(ElementwiseBinaryOperation):
#     # Specialize to implement the power operation of left hand side and
#     # right hand side input
#     _operation = "Pow", np.power, "(std::pow({0}, {1}))", None
