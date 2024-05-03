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
# Derive custom operators form the FINN base custom op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
# Converts inputs/outputs to/from RTL simulation format
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


# Replicates an input stream to arbitrary many output streams
#   See DuplicateStreams_Batch for feeding exactly two streams
class ReplicateStream(HWCustomOp):
    # Initializes the operator given an onnx graph node
    def __init__(self, onnx_node, **kwargs):
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

        # Need to override the default depths of outputs FIFOs here as these
        # depend on the number of replicas, which are not known during calls to
        # get_nodeattr_types.
        if not self.get_nodeattr("outFIFODepths"):
            self.set_nodeattr("outFIFODepths", [2 for _ in range(self.num)])

    # Defines attributes which must be present on this node
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Number of replicas to produce
            "num": ("i", True, 1),
            # Data type of input and output elements
            "dtype": ("s", True, ""),
            # Number of input elements in the last dimension
            "num_elems": ("i", True, 1),
            # Number of elements in the last dimensions processed in parallel
            "PE": ("i", True, 1),
            # Number of inputs to be processed sequentially
            "num_inputs": ("ints", True, [1]),
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": (
                "s", False, "python", {"", "rtlsim", "cppsim", "python"}
            ),
            # Input and output FIFO depths for multi-I/O nodes
            #   Note: Need to override here as there multiple outputs
            "inFIFODepths": ("ints", False, [2]),
            "outFIFODepths": ("ints", False, []),  # Default will be override
        })
        # Return updated attribute dictionary
        return attrs

    # Number of replicas attribute as property for convenience
    @property
    def num(self):
        return self.get_nodeattr("num")

    # Datatype attribute as property for convenience
    @property
    def dtype(self):
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("dtype")]

    # Number of elements attribute as property for convenience
    @property
    def num_elems(self):
        return self.get_nodeattr("num_elems")

    # Number of parallel processed elements as property for convenience
    @property
    def pe(self):
        return self.get_nodeattr("PE")

    # Number of inputs attribute as property for convenience
    @property
    def num_inputs(self):
        return self.get_nodeattr("num_inputs")

    # Makes an operation compatible with the output shape for shape inference
    #   Note: Propagates shape forward, i.e., never asks for the shape of the
    #   output, even if it seems easier.
    def make_shape_compatible_op(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Prepare a dummy input to simulate a large input that can be split into
        # the desired number and shapes of outputs
        mock_input = model.make_new_valueinfo_name()
        # Simulate an input of number of replicas many elements
        model.set_tensor_shape(
            mock_input, [*self.num_inputs, self.num * self.num_elems]
        )
        # Simulate behavior via the standard ONNX split operation
        return oh.make_node(
            "Split", [mock_input], node.output, num_outputs=self.num, axis=-1
        )

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op  # noqa Duplicate
        node = self.onnx_node
        # Test for changing input datatype
        if model.get_tensor_datatype(node.input[0]) != self.dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            warnings.warn(
                f"{node.name}: dtype changing from {self.dtype} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("dtype", new_dtype.name)
        # Propagate the type from the input to each output tensor
        for o in node.output:
            # Replicating simply propagates the dtype to the output
            model.set_tensor_datatype(o, self.dtype)

    # Executes replicating inputs in python
    def _execute_node_python(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the input out of the execution context
        inp = context[node.input[0]]
        # Copy the input into each of the outputs
        for o in node.output:
            # Insert copy of input into the execution context at output
            context[o] = inp

    # Executes replicating inputs in C++ simulation
    def _execute_node_cppsim(self, context, graph):  # noqa: graph unused
        # C++ Simulation needs to be implemented in HLS backend specialization
        raise NotImplementedError(
            f"exec_mode cppsim of {self.__class__.__name__} is not implemented!"
        )

    # Executes replicating inputs in RTL simulation
    def _execute_node_rtlsim(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op  # noqa Duplicate
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Get the input out of the execution context
        inp = context[node.input[0]]
        # Validate the shape of the input
        assert inp.shape == self.get_normal_input_shape(ind=0), \
            f"Input shape mismatch for {node.input[0]}"
        # Reshape the input into folded form
        inp = inp.reshape(self.get_folded_input_shape(ind=0))
        # Path to store the intermediate input in numpy format
        filename = os.path.join(code_gen_dir, "in.npy")
        # Save the folded inputs to file to be used by simulation
        np.save(filename, inp)
        # Start collecting inputs/outputs to the RTL simulation in a dictionary
        #   Note: Prepare one output list per replica
        io_dict = {
            "inputs": {}, "outputs": {f"out{i}": [] for i in range(self.num)}
        }
        # Type and width of the input tensor
        dtype = self.get_input_datatype(ind=0)
        width = self.get_instream_width(ind=0)
        # Convert inputs to RTL simulation format
        io_dict["inputs"]["in"] = npy_to_rtlsim_input(filename, dtype, width)

        # Setup PyVerilator simulation of the node
        sim = self.get_rtlsim()
        # Reset the RTL simulation
        super().reset_rtlsim(sim)
        super().toggle_clk(sim)
        # Run the RTL Simulation
        self.rtlsim_multi_io(sim, io_dict)

        # Enumerate the node outputs
        for i, name in enumerate(node.output):
            # Collect the output from RTL simulation
            out = io_dict["outputs"][f"out{i}"]
            # Type and sizes of the output tensor
            dtype = self.get_output_datatype(ind=i)
            width = self.get_outstream_width(ind=i)
            shape = self.get_folded_output_shape(ind=i)
            # Path to store the intermediate numpy file
            filename = os.path.join(code_gen_dir, f"out{i}.npy")
            # Convert from RTL simulation format to numpy format
            rtlsim_output_to_npy(
                out, filename, dtype, shape, width, dtype.bitwidth()
            )
            # Load the generated output numpy file
            out = np.load(filename)
            # Reshape the folded output and insert into the execution context
            context[name] = out.reshape(self.get_normal_output_shape(ind=i))

    # Executes replicating inputs in simulation (either python c++ or rtl sim)
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
        # All inputs (there should only be one) have the same type
        return self.dtype

    # Gets the datatype of the output at index ind
    def get_output_datatype(self, ind=0):
        # All outputs will hae the same type, which is the same as the input
        return self.dtype

    # Gets the shape of the input at index ind without folding
    def get_normal_input_shape(self, ind=0):
        # There is only one input with shape configured as attributes
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems

    # Gets the shape of the output at index ind without folding
    def get_normal_output_shape(self, ind=0):
        # All outputs have the same shape, which is the same as the input
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems

    # Gets the shape of the input at index ind with folding
    def get_folded_input_shape(self, ind=0):
        # Valid folding requires the PE to divides the number of elements
        assert self.num_elems % self.pe == 0, "PE must divide num_elems"
        # Folding along the last dimension
        return *self.num_inputs, self.num_elems // self.pe, self.pe

    # Gets the shape of the output at index ind with folding
    def get_folded_output_shape(self, ind=0):
        # Valid folding requires the PE to divides the number of elements
        assert self.num_elems % self.pe == 0, "PE must divide num_elems"
        # Folding along the last dimension
        return *self.num_inputs, self.num_elems // self.pe, self.pe

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
        # the embedding dimension. Need to count across the number of replicas,
        # as RTL simulation actually counts individual outputs, not cycles with
        # outputs, i.e., producing N replica outputs per cycle in parallel,
        # count N outputs per cycle...
        return np.prod(self.get_folded_output_shape()[:-1]) * self.num

    # Derives the expected cycles for the stream replication operation given the
    # folding configuration
    def get_exp_cycles(self):
        # Number of iterations required to process the whole folded input stream
        #   Note: This is all but the PE (last, parallelized) dimension
        return np.prod(self.get_folded_output_shape()[:-1])
