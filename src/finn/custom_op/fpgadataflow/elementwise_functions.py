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


# Generic implementation for elementwise function operations
class ElementwiseFunctionOperation(HWCustomOp):
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
                # Data type of the input elements
                "inp_dtype": ("s", True, ""),
                # Data type of the output elements
                "out_dtype": ("s", True, ""),
                # Shape of the input
                "inp_shape": ("ints", True, [1]),
                # Shape of the output, must be equal to the input shape
                "out_shape": ("ints", True, [1]),
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
                "inFIFODepths": ("ints", False, [2]),
                "outFIFODepths": ("ints", False, [2]),
            }
        )
        # Return updated attribute dictionary
        return attrs

    # Datatype attribute as property for convenience
    @property
    def inp_dtype(self):
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("inp_dtype")]

    # Datatype attribute as property for convenience
    @property
    def out_dtype(self):
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("out_dtype")]

    # Shape attribute as property for convenience
    @property
    def inp_shape(self):
        return self.get_nodeattr("inp_shape")

    # Shape attribute as property for convenience
    @property
    def out_shape(self):
        return self.get_nodeattr("out_shape")

    # Number of parallel processed elements as property for convenience
    @property
    def pe(self):
        return self.get_nodeattr("PE")

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Test for changing left-hand-side input datatype
        if model.get_tensor_datatype(node.input[0]) != self.inp_dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            warnings.warn(f"{node.name}: inp_dtype changing from {self.inp_dtype} to {new_dtype}")
            # Set the new datatype attribute
            self.set_nodeattr("inp_dtype", new_dtype.name)
        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.out_dtype)

    def execute_node(self, context, graph):
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the inputs out of the execution context
        inp = context[node.input[0]]
        # Note: Need to make sure these have the right type for the Numpy API
        # Note: Always simulate integer inputs in int64, numpy casting is
        # weird....
        inp = inp.astype(np.int64) if self.inp_dtype.is_integer() else inp
        # Apply elementwise operation in numpy and insert
        # result into the execution context
        out = self.npy_op(inp)
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
        # There is only one input
        return self.inp_dtype

    # Gets the datatype of the output at index ind
    def get_output_datatype(self, ind=0):
        # There is only one output, the type is set as an attribute
        return self.out_dtype

    # Gets the shape of the input at index ind without folding
    def get_normal_input_shape(self, ind=0):
        # Input shape is stored as a node attribute
        return self.inp_shape

    # Gets the shape of the output at index ind without folding
    def get_normal_output_shape(self, ind=0):
        # The output shape is stored as a node attribute
        return self.out_shape

    # Gets the shape of the input at index ind with folding
    def get_folded_input_shape(self, ind=0):
        # Get the normal shape before applying folding
        *num_inputs, num_elems = self.get_normal_input_shape(ind=ind)
        # Valid folding requires the PE to divide the number of elements
        assert num_elems % self.pe == 0, "PE must divide last axis"
        # Folding along the last dimension
        return *num_inputs, num_elems // self.pe, self.pe

    # Gets the shape of the output at index ind with folding
    def get_folded_output_shape(self, ind=0):
        # Get the normal shape before applying folding
        *num_inputs, num_elems = self.get_normal_output_shape(ind=ind)
        # Valid folding requires the PE to divide the number of elements
        assert num_elems % self.pe == 0, "PE must divide last axis"
        # Folding along the last dimension
        return *num_inputs, num_elems // self.pe, self.pe

    # Widths of the input data stream of the input at index ind
    def get_instream_width(self, ind=0):
        # Get the number of bits used to represent the input
        i_bits = self.get_input_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded input
        *_, elems = self.get_folded_input_shape(ind)
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
        # If the input is not an integer, the bit-width cannot be
        # minimized
        if not self.inp_dtype.is_integer():
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

    # Derives the expected cycles for the elementwise operation given the
    # folding configuration
    def get_exp_cycles(self):
        # Number of iterations required to process the whole folded input stream
        #   Note: This is all but the PE (last, parallelized) dimension
        return np.prod(self.get_folded_output_shape()[:-1])


# Derive a specialization to implement the Relu activation function
@register_custom_op
class ElementwiseRelu(ElementwiseFunctionOperation):
    @property
    def npy_op(self):
        def relu(x):
            return np.maximum(x, 0)
        return relu

    @property
    def cpp_op(self):
        odt_hls_name = self.out_dtype.get_hls_datatype_str()
        return "({0} > 0 ? (%s){0} : (%s)0)" % (odt_hls_name, odt_hls_name)

    @property
    def rtl_op(self):
        return None

    def _derive_out_dtype(self, model: ModelWrapper):
        if self.inp_dtype.is_integer():
            inp_bw = self.inp_dtype.bitwidth()
            # The output would be unsigned with same bit-width as input
            # if input was unsigned, else one bit less
            out_bw = inp_bw - 1 if self.inp_dtype.signed() else inp_bw
            return DataType[f"UINT{out_bw}"]

        # output datatype is input datatype for all other data-formats
        return self.inp_dtype


# Derive a specialization to implement elementwise exponent of the input
@register_custom_op
class ElementwiseExp(ElementwiseFunctionOperation):
    @property
    def npy_op(self):
        return np.exp

    @property
    def cpp_op(self):
        # TODO: extend to fixed-point datatypes
        assert self.out_dtype.get_canonical_name().startswith("FLOAT")
        odt_hls_name = self.out_dtype.get_hls_datatype_str()
        # Explicitly use the overloads, using hls::exp results in minor errors
        if self.out_dtype.get_canonical_name() == "FLOAT32":
            return "(hls::expf((%s){0}))" % (odt_hls_name)
        elif self.out_dtype.get_canonical_name() == "FLOAT16":
            return "(hls::half_exp((%s){0}))" % (odt_hls_name)

    @property
    def rtl_op(self):
        return None

    def _derive_out_dtype(self, model: ModelWrapper):
        if self.inp_dtype.get_canonical_name() == "FLOAT16":
            return DataType["FLOAT16"]
        return DataType["FLOAT32"]


# Derive a specialization to implement elementwise erf of the input
@register_custom_op
class ElementwiseErf(ElementwiseFunctionOperation):
    @property
    def npy_op(self):
        import scipy.special
        return scipy.special.erf

    @property
    def cpp_op(self):
        # TODO: extend to fixed-point datatypes
        assert self.out_dtype.get_canonical_name().startswith("FLOAT")
        odt_hls_name = self.out_dtype.get_hls_datatype_str()
        # Explicitly use the overloads, using hls::erf results in minor errors
        if self.out_dtype.get_canonical_name() == "FLOAT32":
            return "(hls::erff((%s){0}))" % (odt_hls_name)
        elif self.out_dtype.get_canonical_name() == "FLOAT16":
            return "(hls::half_erf((%s){0}))" % (odt_hls_name)

    @property
    def rtl_op(self):
        return None

    def _derive_out_dtype(self, model: ModelWrapper):
        if self.inp_dtype.get_canonical_name() == "FLOAT16":
            return DataType["FLOAT16"]
        return DataType["FLOAT32"]
