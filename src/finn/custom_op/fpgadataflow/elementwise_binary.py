# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Numpy math and arrays
import numpy as np

# Operating system stuff, e.g. paths
import os

# Python warning subsystem
import warnings

# Helper for creating ONNX nodes
from onnx import helper as oh

# QONNX/FINN datatypes
from qonnx.core.datatype import DataType

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Utility for registering HWCustomOp implementations into the module scope
from finn.custom_op.fpgadataflow import register_custom_op

# Derive custom operators form the FINN base custom op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Converts inputs/outputs to/from RTL simulation format
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


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
        attrs.update({
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
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": (
                "s", False, "python", {"", "rtlsim", "cppsim", "python"}
            ),
            # FPGA resource type for memories/internal buffers of the operator
            "ram_style": (
                "s", False, "auto", {"auto", "block", "distributed", "ultra"}
            ),
            # Input and output FIFO depths for multi-I/O nodes
            #   Note: Need to override here as there might be two inputs
            "inFIFODepths": ("ints", False, [2, 2]),
            "outFIFODepths": ("ints", False, [2]),
        })
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
    def make_shape_compatible_op(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # There must be exactly two inputs to the binary operation
        assert len(node.input) == 2, \
            f"Binary operation {node.name} requires exactly two inputs"
        # Validate input shapes match what is stored as attributes
        assert model.get_tensor_shape(node.input[0]) == self.lhs_shape, \
            f"Input shape mismatch: {node.name} {node.input[0]}"
        assert model.get_tensor_shape(node.input[1]) == self.rhs_shape, \
            f"Input shape mismatch: {node.name} {node.input[1]}"
        # Validate broadcasting of inputs to the output shape
        assert (list(np.broadcast_shapes(self.lhs_shape, self.rhs_shape))
                == self.out_shape), f"Shape broadcast mismatch: {node.name}"
        # Simulate behavior via the standard ONNX add operation
        return oh.make_node("Add", node.input, node.output)

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op  # noqa Duplicate
        node = self.onnx_node
        # Test for changing left-hand-side input datatype
        if model.get_tensor_datatype(node.input[0]) != self.lhs_dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            warnings.warn(
                f"{node.name}: lhs_dtype changing from"
                f" {self.lhs_dtype} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("lhs_dtype", new_dtype.name)
        # Test for changing right-hand-side input datatype
        if model.get_tensor_datatype(node.input[1]) != self.rhs_dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[1])
            # Issue a warning message
            warnings.warn(
                f"{node.name}: rhs_dtype changing from"
                f" {self.rhs_dtype} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("rhs_dtype", new_dtype.name)
        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.out_dtype)

    # Executes elementwise operation in python
    def _execute_node_python(self, context, graph):  # noqa: graph unused
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
        context[node.output[0]] = out.astype(np.float32)

    # Executes elementwise operation in C++ simulation
    def _execute_node_cppsim(self, context, graph):  # noqa: graph unused
        # C++ Simulation needs to be implemented in HLS backend specialization
        raise NotImplementedError(
            f"exec_mode cppsim of {self.__class__.__name__} is not implemented!"
        )

    # Executes elementwise operation in RTL simulation
    def _execute_node_rtlsim(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op  # noqa Duplicate
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Get the inputs out of the execution context
        lhs = context[node.input[0]]  # noqa: Duplicate code prepare simulation
        rhs = context[node.input[1]]  # noqa: Duplicate code prepare simulation
        # Validate the shape of the inputs
        assert list(lhs.shape) == self.get_normal_input_shape(ind=0), \
            f"Input shape mismatch for {node.input[0]}"
        assert list(rhs.shape) == self.get_normal_input_shape(ind=1), \
            f"Input shape mismatch for {node.input[1]} {rhs.shape=}"
        # Reshape the inputs into folded form
        lhs = lhs.reshape(self.get_folded_input_shape(ind=0))
        rhs = rhs.reshape(self.get_folded_input_shape(ind=1))
        # Path to store the intermediate inputs in numpy format
        lhs_filename = os.path.join(code_gen_dir, "lhs.npy")
        rhs_filename = os.path.join(code_gen_dir, "rhs.npy")
        # Save the folded inputs to file to be used by simulation
        np.save(lhs_filename, lhs)
        np.save(rhs_filename, rhs)
        # Start collecting inputs/outputs to the RTL simulation in a dictionary
        #   Note: Prepare one output empty output list
        io_dict = {
            "inputs": {},
            "outputs": {"out": []}
        }
        # Type and width of the input tensors
        lhs_dtype = self.get_input_datatype(ind=0)
        lhs_width = self.get_instream_width(ind=0)
        rhs_dtype = self.get_input_datatype(ind=1)
        rhs_width = self.get_instream_width(ind=1)

        # If the left-hand-side is provided as runtime input it needs to be
        # inserted into the RTL simulation inputs
        if self.lhs_style == "input":
            # Convert inputs to RTL simulation format
            io_dict["inputs"]["lhs"] = npy_to_rtlsim_input(
                lhs_filename, lhs_dtype, lhs_width
            )

        # If the right-hand-side is provided as runtime input it needs to be
        # inserted into the RTL simulation inputs
        if self.rhs_style == "input":
            # Convert inputs to RTL simulation format
            io_dict["inputs"]["rhs"] = npy_to_rtlsim_input(
                rhs_filename, rhs_dtype, rhs_width
            )

        # Setup PyVerilator simulation of the node
        sim = self.get_rtlsim()  # noqa: Duplicate code prepare simulation
        # Reset the RTL simulation
        super().reset_rtlsim(sim)
        super().toggle_clk(sim)
        # Run the RTL Simulation
        self.rtlsim_multi_io(sim, io_dict)
        # free up resources
        self.close_rtlsim(sim)

        # Collect the output from RTL simulation
        out = io_dict["outputs"]["out"]
        # Type and sizes of the output tensor
        dtype = self.get_output_datatype(ind=0)  # noqa: Duplicate readout code
        width = self.get_outstream_width(ind=0)
        shape = self.get_folded_output_shape(ind=0)
        # Path to store the intermediate numpy file
        filename = os.path.join(code_gen_dir, "out.npy")
        # Convert from RTL simulation format to numpy format
        rtlsim_output_to_npy(
            out, filename, dtype, shape, width, dtype.bitwidth()
        )
        # Load the generated output numpy file
        out = np.load(filename)
        # Reshape the folded output and insert into the execution context
        context[node.output[0]] = out.reshape(
            self.get_normal_output_shape(ind=0)
        )

    # Executes elementwise op in simulation (either python c++ or rtl sim)
    def execute_node(self, context, graph):
        # Get the configured execution mode
        mode = self.get_nodeattr("exec_mode")
        # Lookup table mapping execution modes to implementing methods
        exec_fns = {
            "python": self._execute_node_python,
            "cppsim": self._execute_node_cppsim,
            "rtlsim": self._execute_node_rtlsim,
        }
        # Select and execute the function by mode string
        exec_fns[mode](context, graph)

    # Verifies the node attributes, inputs and outputs
    def verify_node(self):
        # TODO: Implement
        return []

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

    # Gets the number of expected output values, i.e. how many times read()
    # could/should be called on any output stream of this operator
    def get_number_output_values(self):
        # Elements over all but the last dimension of the output folded along
        # the embedding dimension.
        return np.prod(self.get_folded_output_shape()[:-1])

    # Minimizes the width of the accumulator data type, 'accumulator width' here
    # due to convention, it is actually the output data type
    def minimize_accumulator_width(self, model: ModelWrapper):
        # If any of the inputs is not an integer, the bit-width cannot be
        # minimized
        if not all([self.lhs_dtype.is_integer(), self.rhs_dtype.is_integer()]):
            # Check the annotated tensor data type corresponds to the stored
            # attribute
            assert (model.get_tensor_datatype(self.onnx_node.output[0])
                    == self.out_dtype), \
                f"Output type mismatch for {self.onnx_node.name}"
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
            f"_derive_out_dtype of {self.__class__.__name__}"
            f" is not implemented!"
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
            # Minimum and maximum "weight" on the left hand side, determining
            # the range of values which needs to be represented
            _min = lhs.min()
            _max = lhs.max()
            # Determine whether signed or unsigned type is required for
            # representing the weights and select the largest "signed magnitude"
            _mag = _max if _min > 0 else \
                _min if (abs(_min) > _max) else (-_max - 1)
            # Smallest data type large enough to represent this range of values
            dtype = DataType.get_smallest_possible(_mag)
            # Update the corresponding data type attribute of the node
            self.set_nodeattr("lhs_dtype", dtype.name)
            # Annotate the tensor with the new data type
            model.set_tensor_datatype(self.onnx_node.input[0], dtype)

        # Check for an initializer providing the right hand side input
        rhs = model.get_initializer(self.onnx_node.input[1])
        # If the right hand side input is provided as initializer, minimize the
        # bits used for storing this
        if rhs is not None:
            # Remember the "style" of receiving the input for further code
            # generation
            self.set_nodeattr("rhs_style", "const")
            # Minimum and maximum "weight" on the right hand side, determining
            # the range of values which needs to be represented
            _min = rhs.min()
            _max = rhs.max()
            assert _min != 0
            assert _max != 0
            # Determine whether signed or unsigned type is required for
            # representing the weights and select the largest "signed magnitude"
            _mag = _max if _min > 0 else \
                _min if (abs(_min) > _max) else (-_max - 1)
            # Smallest data type large enough to represent this range of values
            dtype = DataType.get_smallest_possible(_mag)
            # Update the corresponding data type attribute of the node
            self.set_nodeattr("rhs_dtype", dtype.name)
            # Annotate the tensor with the new data type
            model.set_tensor_datatype(self.onnx_node.input[1], dtype)

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
        # Get the width of the data types of the inputs  # noqa: Duplicate
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
        # Get the width of the data types of the inputs  # noqa: Duplicate
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
        # Get the width of the data types of the inputs  # noqa: Duplicate
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

# TODO: ElementwiseBitShift - Requires extra attribute selecting the direction


# # Derive a specialization to implement elementwise power of two inputs
# TODO: std::pow does not work for HLS types and hls::pow fails to link for some
#  reason
# @register_custom_op
# class ElementwisePow(ElementwiseBinaryOperation):
#     # Specialize to implement the power operation of left hand side and
#     # right hand side input
#     _operation = "Pow", np.power, "(std::pow({0}, {1}))", None
