# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Copies of python objects, copy.deepcopy
import copy

# Numpy math and arrays
import numpy as np

# Operating system stuff, e.g. paths
import os

# Python warning subsystem
import warnings

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


# Squeeze operation: Removes single-dimension entries from the shape of a tensor
@register_custom_op
class Squeeze(HWCustomOp):
    # Initializes the operator given an onnx graph node
    def __init__(self, onnx_node, **kwargs):
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

    # Defines attributes which must be present on this node
    def get_nodeattr_types(self):
        # Start from parent operator class attributes  # noqa: Duplicate
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Axes to be squeezed can be given as an attribute for opset < 13
            "axes": ("ints", False, None),
            # Data type of the input elements
            "inp_dtype": ("s", True, ""),
            # Data type of the output elements
            "out_dtype": ("s", True, ""),
            # Shape of the input
            "inp_shape": ("ints", True, [1]),
            # Shape of the output
            "out_shape": ("ints", True, [1]),
            # Number of elements in the last dimensions processed in parallel
            "PE": ("i", False, 1),
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": (
                "s", False, "python", {"", "rtlsim", "cppsim", "python"}
            ),
        })
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

    # Makes an operation compatible with the output shape for shape inference
    # Note: Propagates shape forward, i.e., never asks for the shape of the
    # output, even if it seems easier.
    def make_shape_compatible_op(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = copy.deepcopy(self.onnx_node)
        # Though providing squeezed axes via a second input is supported by the
        # implementation, the inferred shapes might eb incorrect if this is
        # truly a dynamic list of axes changing at runtime.
        if len(node.input) > 1:
            # Issue a warning to make the user aware of this potential issue
            warnings.warn(
                f"{node.name}: Providing dimensions to squeeze as an input"
                f" might invalidate shape inference if these are not constant."
            )
        # Transplant this operator back into the standard ONNX domain
        node.domain = ""
        # Shape inference should now work on this standard ONNX node
        return node

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op  # noqa Duplicate
        node = self.onnx_node  # noqa: Duplicate
        # Test for changing input datatype
        if model.get_tensor_datatype(node.input[0]) != self.inp_dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            warnings.warn(
                f"{node.name}: inp_dtype changing from"
                f" {self.inp_dtype} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("inp_dtype", new_dtype.name)
        # Though providing squeezed axes via a second input is supported by the
        # implementation, the datatype of this input is ignored here
        if len(node.input) > 1:
            # Issue a warning to make the user aware of this potential issue
            warnings.warn(
                f"{node.name}: Providing dimensions to squeeze as an input"
                f" will be ignored by datatype inference."
            )
        # Make sure the output always has the same type as the input
        if self.out_dtype != self.inp_dtype:
            # Issue a warning message
            warnings.warn(
                f"{node.name}: out_dtype changing from"
                f" {self.out_dtype} to {self.inp_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("out_dtype", self.inp_dtype.name)
        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.out_dtype)

    # Executes squeeze operation in python
    def _execute_node_python(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node  # noqa: Duplicate
        # Get the input from the execution context
        inp = context[node.input[0]]
        # Try with axes specified as attribute first
        axes = self.get_nodeattr("axes")
        # If there are exes specified via attribute but there is a second input
        # to the operator, this input specifies the axes to be squeezed
        if axes is None and len(node.input) > 1:
            # Get the axes list from the execution context
            axes = context[node.input[1]]
        # If axes are specified convert them to tuple as required by numpy
        axes = tuple(axes) if axes is not None else None
        # Squeeze the input along the optionally specified axes
        out = np.squeeze(inp, axis=axes)
        # Make sure the output has the right type (always use float32 as the
        # container type) and insert into the execution context
        context[node.output[0]] = out.astype(np.float32)

    # Executes squeeze operation in C++ simulation
    def _execute_node_cppsim(self, context, graph):  # noqa: graph unused
        # C++ Simulation needs to be implemented in HLS backend specialization
        raise NotImplementedError(
            f"exec_mode cppsim of {self.__class__.__name__} is not implemented!"
        )

    # Executes squeeze operation in RTL simulation
    def _execute_node_rtlsim(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op  # noqa Duplicate
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Get the inputs out of the execution context
        inp = context[node.input[0]]  # noqa: Duplicate code prepare simulation
        # Validate the shape of the inputs
        assert list(inp.shape) == self.get_normal_input_shape(ind=0), \
            f"Input shape mismatch for {node.input[0]}"
        # Reshape the input into folded form
        inp = inp.reshape(self.get_folded_input_shape(ind=0))
        # Path to store the intermediate input in numpy format
        inp_filename = os.path.join(code_gen_dir, "inp.npy")
        # Save the folded input to file to be used by simulation
        np.save(inp_filename, inp)
        # Start collecting inputs/outputs to the RTL simulation in a dictionary
        #   Note: Prepare one output empty output list
        io_dict = {
            "inputs": {},
            "outputs": {"out": []}
        }
        # Type and width of the input tensors
        inp_dtype = self.get_input_datatype(ind=0)
        inp_width = self.get_instream_width(ind=0)

        # Convert input to RTL simulation format
        io_dict["inputs"]["inp"] = npy_to_rtlsim_input(
            inp_filename, inp_dtype, inp_width
        )

        # Setup PyVerilator simulation of the node
        sim = self.get_rtlsim()  # noqa: Duplicate code prepare simulation
        # Reset the RTL simulation
        super().reset_rtlsim(sim)
        super().toggle_clk(sim)
        # Run the RTL Simulation
        self.rtlsim_multi_io(sim, io_dict)

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

    # Executes squeeze operation in simulation (either python c++ or rtl sim)
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
        # There is only one proper input (we ignore the optional axes input
        # here)
        return self.inp_dtype

    # Gets the datatype of the output at index ind
    def get_output_datatype(self, ind=0):
        # There is only one output, the type is set as an attribute
        return self.out_dtype

    # Gets the shape of the input at index ind without folding
    def get_normal_input_shape(self, ind=0):
        # There is only one proper input (we ignore the optional axes input
        # here)
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
        *num_outputs, num_elems = self.get_normal_output_shape(ind=ind)
        # Valid folding requires the PE to divide the number of elements
        assert num_elems % self.pe == 0, "PE must divide last axis"
        # Folding along the last dimension
        return *num_outputs, num_elems // self.pe, self.pe

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

    # Derives the expected cycles for the squeeze operation given the folding
    # configuration
    def get_exp_cycles(self):
        # Number of iterations required to process the whole folded stream
        #   Note: This is all but the PE (last, parallelized) dimension
        return np.prod(self.get_folded_output_shape()[:-1])
