# Operating system stuff, e.g. paths
import os
# Numpy math and arrays
import numpy as np

# Protobuf onnx graph node type
from onnx import NodeProto  # noqa
# Helper for creating ONNX nodes
from onnx import helper as oh  # noqa

# QONNX/FINN datatypes
from qonnx.core.datatype import DataType  # noqa qonnx dependency is specified
# in setup.cfg as well as in fetch-repos.sh
# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper  # noqa

# Converts inputs/outputs to/from RTL simulation format
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
# Derive custom operators form the FINN base custom op
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp


# Splitting of attention heads (after input projections) custom operator
class SplitMultiHeads(HLSCustomOp):
    # Initializes the operator given an onnx graph node
    def __init__(self, onnx_node, **kwargs):
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

    # Defines attributes which must be present on this node
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = super().get_nodeattr_types()
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Number of attention heads
            "heads": ("i", True, 1),
            # Specifies whether the output is packed as a single output tensor
            # or split as multiple output tensors
            "packed": ("i", True, 1),
            # Data type of input and output elements
            "dtype": ("s", True, ""),
            # Number of input elements to be split
            "num_elems": ("i", True, 1),
            # Number of inputs to be processed sequentially
            "num_inputs": ("ints", True, [1]),
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": ("s", False, "", {"", "rtlsim", "cppsim", "python"})
        })
        # Return updated attribute dictionary
        return attrs

    # Number of attention heads attribute as property for convenience
    @property
    def heads(self):
        return self.get_nodeattr("heads")

    # Packed attribute as property for convenience
    @property
    def packed(self):
        # Note: Converts from int to bool
        return bool(self.get_nodeattr("packed"))

    # Datatype attribute as property for convenience
    @property
    def dtype(self):
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("dtype")]

    # Number of elements attribute as property for convenience
    @property
    def num_elems(self):
        return self.get_nodeattr("num_elems")

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
        # Get the shape of the input tensor for inferring the number of
        # heads and correctly propagating shapes
        shape = model.get_tensor_shape(node.input[0])
        # Determine the rank of the input tensor to support batched and
        # non-batched inputs
        rank = len(shape)
        # The input shape determines the sequence length
        seq, _, dim = shape if (rank == 3) else (shape[0], 1, shape[1])
        # Packed outputs a represented by a reshape operation producing one
        # tensor
        if self.packed:
            # Create a new name for the temporary shape tensor
            shape = model.make_new_valueinfo_name()
            # Set the target shape of slices heads
            model.set_initializer(
                shape, np.asarray([self.heads, seq, dim // self.heads])
            )
            # Return a node simulating the shape effect of slicing into
            # multi-heads
            return oh.make_node(
                "Reshape", [node.input[0], shape], [node.output[0]]
            )
        # Prepare a dummy input to simulate reordering of batch/head dimension
        # to the front
        mock_input = model.make_new_valueinfo_name()
        # Set the target shape of slices heads
        model.set_tensor_shape(
            mock_input, [1, seq, dim] if rank == 3 else [seq, dim]
        )
        # If the outputs are not packed, the operation is represented as a split
        # operation producing number of heads outputs along the last axis
        return oh.make_node(
            "Split", [mock_input], node.output, num_outputs=self.heads, axis=-1
        )

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Propagate the type from the input to each output tensor
        for o in node.output:
            # Slicing simply propagates the type of the input to the output
            model.set_tensor_datatype(
                o, model.get_tensor_datatype(node.input[0])
            )

    # Executes multi-head slicing in python
    def _execute_node_python(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the input out of the execution context
        #   Note: Shape must be either seq x 1 x dim or seq x dim
        inp = context[node.input[0]]
        # Packed execution boils down to a reshape of the single input to a
        # single output
        if self.packed:
            # Reshape to separate the heads out of the embedding dimensions,
            # finally transpose to heads first layout
            out = inp.reshape(inp.shape[0], self.heads, -1).transpose(1, 0, 2)
            # Write the output into the execution context
            context[node.output[0]] = out
        # Split is realized as the split operation of numpy
        else:
            # Produces multiple outputs as a list
            splits = np.split(inp, indices_or_sections=self.heads, axis=-1)
            # Correspondence between outputs and splits in order
            for o, out in zip(node.output, splits):
                # Write the output into the execution context
                context[o] = out

    # Executes multi-head slicing in C++ simulation
    def _execute_node_cppsim(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # Get the input out of the execution context
        #   Note: Shape must be either seq x 1 x dim or seq x dim
        inp = context[node.input[0]]
        # Validate the shape of the input
        assert inp.shape == self.get_normal_input_shape(ind=0), \
            f"Input shape mismatch for {node.input[0]}"
        # Reshape the input into folded form
        inp = inp.reshape(self.get_folded_input_shape(ind=0))
        # Save the folded inputs to file to be used by simulation
        np.save(os.path.join(code_gen_dir, f"in.npy"), inp)

        # Execute the precompiled model
        super().exec_precompiled_singlenode_model()

        # Enumerate the node outputs
        for i, name in enumerate(node.output):
            # Load the output numpy file generated by the C++ simulation
            out = np.load(os.path.join(code_gen_dir, f"out{i}.npy"))
            # Reshape the folded output and insert into the execution context
            context[name] = out.reshape(self.get_normal_output_shape(ind=i))

    # Executes multi-head slicing in RTL simulation
    def _execute_node_rtlsim(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Get the input out of the execution context
        #   Note: Shape must be either seq x 1 x dim or seq x dim
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
        #   Note: Prepare one output list per head
        io_dict = {
            "inputs": {}, "outputs": {f"out{i}": [] for i in range(self.heads)}
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

    # Executes multi-head slicing in simulation (either python c++ or rtl sim)
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

    # Note: End of QONNX CustomOp region, below is FINN HLSCustomOp stuff

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
        # Packed layout is currently not implemented
        assert not self.packed, "Packed multi-heads are not implemented yet"
        # All output have the same shape, which correspond to distributing the
        # number of input elements to the heads specified as attributes
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems // self.heads

    # Gets the shape of the input at index ind with folding
    def get_folded_input_shape(self, ind=0):
        # No folding for now, normal and folded shape are the same
        return self.get_normal_input_shape(ind=ind)

    # Gets the shape of the output at index ind with folding
    def get_folded_output_shape(self, ind=0):
        # No folding for now, normal and folded shape are the same
        return self.get_normal_output_shape(ind=ind)

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

    # Maximum width of any ap_int used in this operator
    def get_ap_int_max_w(self):
        # Find the widths of the widest input
        # Note: There is just one input.
        i_bits_max = self.get_instream_width(ind=0)
        # Find the widths of the widest output
        # Note: there is one output per head
        o_bits_max = max(
            (self.get_outstream_width(ind) for ind in range(self.heads))
        )
        # Find the biggest of the inputs/outputs
        return max([i_bits_max, o_bits_max])

    # Gets the number of expected output values, i.e. how many times read()
    # could/should be called on any output stream of this operator
    def get_number_output_values(self):
        # Elements over all but the last dimension of the output folded along
        # the embedding dimension. Need to count across the number of heads, as
        # RTL simulation actually counts individual inputs, not cycles with
        # inputs, i.e., producing N heads outputs per cycle in parallel, count
        # N outputs per cycle...
        return np.prod(self.get_folded_output_shape()[:-1]) * self.heads

    # Note: End of shape and datatype utilities

    # Generates list of C++ includes to be placed at the top of the generated
    # code
    def global_includes(self):
        # Currently nothing to include
        self.code_gen_dict["$GLOBALS$"] = []

    # Generates C++ code of type alias, global constant and macro definitions
    def defines(self, var):
        # Insert constants and type aliases into the dictionary
        self.code_gen_dict["$DEFINES$"] = [
            # Input and output element datatypes
            f"using IType = {self.dtype.get_hls_datatype_str()};",
            f"using OType = {self.dtype.get_hls_datatype_str()};",
            # Datatype of elements packed into the input stream
            f"using IPacked = ap_uint<{self.get_instream_width()}>;",
            # Datatype of elements packed into the output stream
            f"using OPacked = ap_uint<{self.get_outstream_width()}>;",
            # Input and output HLS stream datatypes
            f"using IStream = hls::stream<"
            f"  ap_uint<{self.get_instream_width()}>"
            f">;",
            f"using OStream = hls::stream<"
            f"  ap_uint<{self.get_outstream_width()}>"
            f">;",
        ]

    # Generates C++ code for reading data from .npy (numpy format) for testing
    # in C++ simulation
    def read_npy_data(self):
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # Generate function calls for reading the input files into the input
        # streams
        self.code_gen_dict["$READNPYDATA$"] = [
            # Generate function call reading from file into the input stream
            #   Note: Inputs are always represented as numpy floats
            f'npy2apintstream<IPacked, IType, IType::width, float>(',
            f'"{code_gen_dir}/in.npy", in_{self.hls_sname()}, false',
            f');'
        ]

    # Generates C++ code for declaring all streams involved in C++ simulation
    # for testing
    def strm_decl(self):
        # Declare input and output streams
        # Note: Assumes stream type aliases to be set in defines
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            # There is one input datastream
            f"IStream in_{self.hls_sname()};",
            # There is one output datastream per head
            *(f"OStream out{i}_{self.hls_sname()};" for i in range(self.heads))
        ]

    # Generates C++ code for calling the computation part of the operator
    def docompute(self):
        # Generates the bit-slicing indices string for the ith split of the
        # input
        def split(i):
            # Assemble a C++ indexing/bit-slicing string
            return f"({i + 1} * OPacked::width - 1, {i} * OPacked::width)"

        # Generates the name of the ith output stream
        def out(i):
            return f"out{i}_{self.hls_sname()}"

        # Write the body of the head-splitting top-level function
        self.code_gen_dict["$DOCOMPUTE$"] = [
            # Repeat for the number of inputs
            # Note: Repeat for all num_inputs dimensions
            f"for(std::size_t i = 0; i < {np.prod(self.num_inputs)}; ++i) {{",
            # Pipeline the steps of this loop
            f"#pragma HLS pipeline II=1 style=flp",
            # Read the next input element from the stream
            f"const auto x = in_{self.hls_sname()}.read();",
            # Split the next element from the input stream into the number of
            # output elements per head and write into the corresponding stream
            *(f"{out(i)}.write(x{split(i)});" for i in range(self.heads)),
            # End of for-loop over repetitions body
            f"}}"
        ]

    # Generates C++ code for reading the output stream and converting back to
    # numpy format for testing in C** simulation
    def dataoutstrm(self):
        # Output data will be stored in numpy files in the code generation
        # dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # Get the expected shape of the folded output array formatted as a C++
        # vector initializer
        # Note: Valid formatting relies on correct placement of curly braces
        # and line breaks: Open/close all three braces on the same line of code
        # to avoid '\n' to be inserted into the string
        shape = f"""{{{
        ','.join((str(i) for i in self.get_folded_output_shape()))
        }}}"""
        # Start collecting function calls to write the output data stream
        self.code_gen_dict["$DATAOUTSTREAM$"] = []

        # Generates the name of the ith output stream
        def out(i):
            return f"out{i}_{self.hls_sname()}"

        # Generate code for each output stream
        for i in range(self.heads):
            # Append each reading/writing function call
            self.code_gen_dict["$DATAOUTSTREAM$"] += [
                # Generate function call reading from stream into the output
                # file
                #   Note: Outputs are always represented as numpy floats
                f'apintstream2npy<OPacked, OType, OType::width, float>(',
                f'{out(i)}, {shape}, "{code_gen_dir}/out{i}.npy", false',
                f');'
            ]

    # Generates C++ code for saving the output of C++ simulation to a file in
    # numpy format
    def save_as_npy(self):
        # Note: This seems to be empty in ALL HLSCustomOps. Probably it was used
        # for something before, which is now integrated into dataoutstrm()?
        self.code_gen_dict["$SAVEASCNPY$"] = []

    # Generates essentially the head of the C++ function from which the IP block
    # will be generated during ipgen, i.e. actual synthesis
    def blackboxfunction(self):
        # Insert function head describing the top level interface of the head
        # splitting operator
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # @formatter:off Prevent Python formatter from messing with C++
            # formatting
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            # Input HLS stream
            f"  IStream &in_{self.hls_sname()}, ", ",".join([
            # One output HLS stream per head  # noqa: Formatting
            f"  OStream &out{i}_{self.hls_sname()}" for i in range(self.heads)
            ]),
            f")",
            # @formatter:off
        ]

    # Generates C++ pragmas to be inserted into the main function of the C++
    # simulation and the ipgen-blackboxfunction as well
    def pragmas(self):
        # Add HLS interface directives specifying how to create RTL ports for
        # the top-level function arguments
        self.code_gen_dict["$PRAGMAS$"] = [
            # Connect the input stream with an axi stream interface
            f"#pragma HLS INTERFACE axis port=in_{self.hls_sname()}"
        ]
        # Connect each output stream with an axi stream interface
        for i in range(self.heads):
            # Add new interface directive for the output stream
            self.code_gen_dict["$PRAGMAS$"] += [
                f"#pragma HLS INTERFACE axis port=out{i}_{self.hls_sname()}"
            ]
        # No block-level I/O protocol for the function return value
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )


# Merging of attention heads (before output projections) custom operator
class MergeMultiHeads(HLSCustomOp):
    # Initializes the operator given an onnx graph node
    def __init__(self, onnx_node, **kwargs):
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

    # Defines attributes which must be present on this node
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = super().get_nodeattr_types()
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Number of attention heads
            "heads": ("i", True, 1),
            # Specifies whether the output is packed as a single output tensor
            # or split as multiple output tensors
            "packed": ("i", True, 1),
            # Data type of input and output elements
            "dtype": ("s", True, ""),
            # Number of input elements to be split
            "num_elems": ("i", True, 1),
            # Number of inputs to be processed sequentially
            "num_inputs": ("ints", True, [1]),
            # Output needs to be squeezed
            "squeezed": ("i", True, 0),
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": ("s", False, "", {"", "rtlsim", "cppsim", "python"})
        })
        # Return updated attribute dictionary
        return attrs

    # Number of attention heads attribute as property for convenience
    @property
    def heads(self):
        return self.get_nodeattr("heads")

    # Packed attribute as property for convenience
    @property
    def packed(self):
        # Note: Converts from int to bool
        return bool(self.get_nodeattr("packed"))

    # Datatype attribute as property for convenience
    @property
    def dtype(self):
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("dtype")]

    # Number of elements attribute as property for convenience
    @property
    def num_elems(self):
        return self.get_nodeattr("num_elems")

    # Number of inputs attribute as property for convenience
    @property
    def num_inputs(self):
        return self.get_nodeattr("num_inputs")

    # Squeezed output attribute as property for convenience
    @property
    def squeezed(self):
        # Note: Converts from int to bool
        return bool(self.get_nodeattr("squeezed"))

    # Makes an operation compatible with the output shape for shape inference
    #   Note: Propagates shape forward, i.e., never asks for the shape of the
    #   output, even if it seems easier.
    def make_shape_compatible_op(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Squeeze single-element batch dimension from the output?
        squeezed = self.squeezed
        # Packed inputs a represented by a reshape operation consuming one
        # tensor
        if self.packed:
            # Get the shape of the input tensor for inferring the number of
            # heads and correctly propagating shapes
            h, seq, dim = model.get_tensor_shape(node.input[0])
            # Attribute heads must match wht is annotated at the input
            assert h == self.heads, \
                f"Shape annotation and number of heads differ: {node.name}"
            # Distribute the heads into the embedding dimension
            dim = self.heads * dim
            # Create a new name for the temporary shape tensor
            shape = model.make_new_valueinfo_name()
            # Set the target shape of slices heads
            model.set_initializer(
                shape, np.asarray([seq, dim] if squeezed else [seq, 1, dim])
            )
            # Return a node simulating the shape effect of merging multi-heads
            return oh.make_node(
                "Reshape", [node.input[0], shape], [node.output[0]]
            )
        # If the inputs are not packed, the operation is represented as a concat
        # operation consuming number of heads inputs concatenating along the
        # last axis
        return oh.make_node("Concat", node.input, node.output, axis=-1)

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Merging simply propagates the type of the input to the output
        model.set_tensor_datatype(
            node.output[0], model.get_tensor_datatype(node.input[0])
        )

    # Executes multi-head merging in python
    def _execute_node_python(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the input out of the execution context
        #   Note: Shape must be heads x seq x dim
        inp = context[node.input[0]]
        # Packed execution boils down to a reshape of the single input to a
        # single output
        if self.packed:
            # Transpose back into sequence first layout then reintegrate the
            # heads via reshape
            out = inp.transpose(1, 0, 2).reshape(
                inp.shape[1], 1, self.heads * inp.shape[-1]
            )
        # Split is realized as the concat operation of numpy
        else:
            # Collect the list of inputs from the execution context and
            # concatenate along the last axis
            out = np.concatenate([context[i] for i in node.input], axis=-1)
            # Reshape to simulate the batch dimensions if it is not present
            out = out.reshape(out.shape[0], 1, out.shape[-1])
        # Optionally squeeze the output (remove batch dimension of size 1)
        if self.squeezed:
            # Squeeze batch dimension via reshape
            out = out.reshape(out.shape[0], out.shape[-1])
        # Write the output into the execution context. Force output shape
        # which might be squeezed
        context[node.output[0]] = out

    # Executes multi-head slicing in C++ simulation
    def _execute_node_cppsim(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")

        # Enumerate the node outputs
        for i, name in enumerate(node.input):
            # Get the input out of the execution context
            #   Note: Shape must be either 1 x seq x dim or seq x dim
            inp = context[name]
            # Validate the shape of the input
            assert inp.shape == self.get_normal_input_shape(ind=i), \
                f"Input shape mismatch for {name}"
            # Reshape the input into folded form
            inp = inp.reshape(self.get_folded_input_shape(ind=i))
            # Save the folded inputs to file to be used by simulation
            np.save(os.path.join(code_gen_dir, f"in{i}.npy"), inp)

        # Execute the precompiled model
        super().exec_precompiled_singlenode_model()

        # Load the output numpy file generated by the C++ simulation
        out = np.load(os.path.join(code_gen_dir, f"out.npy"))
        # Reshape the folded output and insert into the execution context
        context[node.output[0]] = out.reshape(
            self.get_normal_output_shape(ind=0)
        )

    # Executes multi-head slicing in RTL simulation
    def _execute_node_rtlsim(self, context, graph):  # noqa: graph unused
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        # Start collecting inputs/outputs to the RTL simulation in a dictionary
        #   Note: Prepare one output list per head
        io_dict = {
            "inputs": {}, "outputs": {"out": []}
        }

        # Enumerate the node outputs
        for i, name in enumerate(node.input):
            # Get the input out of the execution context
            #   Note: Shape must be either 1 x seq x dim or seq x dim
            inp = context[name]
            # Validate the shape of the input
            assert inp.shape == self.get_normal_input_shape(ind=i), \
                f"Input shape mismatch for {name}"
            # Reshape the input into folded form
            inp = inp.reshape(self.get_folded_input_shape(ind=i))
            # Path to store the intermediate input in numpy format
            filename = os.path.join(code_gen_dir, f"in{i}.npy")
            # Save the folded inputs to file to be used by simulation
            np.save(filename, inp)
            # Type and width of the input tensor
            dtype = self.get_input_datatype(ind=i)
            width = self.get_instream_width(ind=i)
            # Convert inputs to RTL simulation format
            io_dict["inputs"][f"in{i}"] = npy_to_rtlsim_input(
                filename, dtype, width
            )

        # Setup PyVerilator simulation of the node
        sim = self.get_rtlsim()
        # Reset the RTL simulation
        super().reset_rtlsim(sim)
        super().toggle_clk(sim)
        # Run the RTL Simulation
        self.rtlsim_multi_io(sim, io_dict)

        # Collect the output from RTL simulation
        out = io_dict["outputs"]["out"]
        # Type and sizes of the output tensor
        dtype = self.get_output_datatype(ind=0)
        width = self.get_outstream_width(ind=0)
        shape = self.get_folded_output_shape(ind=0)
        # Path to store the intermediate numpy file
        filename = os.path.join(code_gen_dir, "out.npy")
        # Convert from RTL simulation format to numpy format
        rtlsim_output_to_npy(
            out, filename, dtype, shape, width, dtype.bitwidth()
        )
        # Load the output numpy file generated by the RTL simulation
        out = np.load(filename)
        # Reshape the folded output and insert into the execution context
        context[node.output[0]] = out.reshape(
            self.get_normal_output_shape(ind=0)
        )

    # Executes multi-head slicing in simulation (either python c++ or rtl sim)
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

    # Note: End of QONNX CustomOp region, below is FINN HLSCustomOp stuff

    # Gets the datatype of input at index ind
    def get_input_datatype(self, ind=0):
        # All inputs (there should only be one) have the same type
        return self.dtype

    # Gets the datatype of the output at index ind
    def get_output_datatype(self, ind=0):
        # All outputs will have the same type, which is the same as the input
        return self.dtype

    # Gets the shape of the input at index ind without folding
    def get_normal_input_shape(self, ind=0):
        # Packed layout is currently not implemented
        assert not self.packed, "Packed multi-heads are not implemented yet"
        # There is only one input with shape configured as attributes
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems

    # Gets the shape of the output at index ind without folding
    def get_normal_output_shape(self, ind=0):
        # All output have the same shape, which correspond to collecting the
        # number of input elements from the heads specified as attributes
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems * self.heads

    # Gets the shape of the input at index ind with folding
    def get_folded_input_shape(self, ind=0):
        # No folding for now, normal and folded shape are the same
        return self.get_normal_input_shape(ind=ind)

    # Gets the shape of the output at index ind with folding
    def get_folded_output_shape(self, ind=0):
        # No folding for now, normal and folded shape are the same
        return self.get_normal_output_shape(ind=ind)

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

    # Maximum width of any ap_int used in this operator
    def get_ap_int_max_w(self):
        # Find the widths of the widest input
        # Note: There is just one input.
        i_bits_max = self.get_instream_width(ind=0)
        # Find the widths of the widest output
        # Note: there is one output per head
        o_bits_max = max(
            (self.get_outstream_width(ind) for ind in range(self.heads))
        )
        # Find the biggest of the inputs/outputs
        return max([i_bits_max, o_bits_max])

    # Gets the number of expected output values, i.e. how many times read()
    # could/should be called on any output stream of this operator
    def get_number_output_values(self):
        # Elements over all but the last dimension of the output folded along
        # the embedding dimension
        return np.prod(self.get_folded_output_shape()[:-1])

    # Note: End of shape and datatype utilities

    # Generates list of C++ includes to be placed at the top of the generated
    # code
    def global_includes(self):
        # Currently nothing to include
        self.code_gen_dict["$GLOBALS$"] = []

    # Generates C++ code of type alias, global constant and macro definitions
    def defines(self, var):
        # Insert constants and type aliases into the dictionary
        self.code_gen_dict["$DEFINES$"] = [
            # Input and output element datatypes
            f"using IType = {self.dtype.get_hls_datatype_str()};",
            f"using OType = {self.dtype.get_hls_datatype_str()};",
            # Datatype of elements packed into the input stream
            f"using IPacked = ap_uint<{self.get_instream_width()}>;",
            # Datatype of elements packed into the output stream
            f"using OPacked = ap_uint<{self.get_outstream_width()}>;",
            # Input and output HLS stream datatypes
            f"using IStream = hls::stream<"
            f"  ap_uint<{self.get_instream_width()}>"
            f">;",
            f"using OStream = hls::stream<"
            f"  ap_uint<{self.get_outstream_width()}>"
            f">;",
        ]

    # Generates C++ code for reading data from .npy (numpy format) for testing
    # in C++ simulation
    def read_npy_data(self):
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # Generate function calls for reading the input files into the input
        # streams
        self.code_gen_dict["$READNPYDATA$"] = []
        # Generate code for each input stream
        for i in range(self.heads):
            # Append each reading/writing function call
            self.code_gen_dict["$READNPYDATA$"] += [
                # Generate function call reading from file into the input stream
                #   Note: Inputs are always represented as numpy floats
                f'npy2apintstream<IPacked, IType, IType::width, float>(',
                f'"{code_gen_dir}/in{i}.npy", in{i}_{self.hls_sname()}, false',
                f');'
            ]

    # Generates C++ code for declaring all streams involved in C++ simulation
    # for testing
    def strm_decl(self):
        # Declare input and output streams
        # Note: Assumes stream type aliases to be set in defines
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            # There is one output stream
            f"OStream out_{self.hls_sname()};",
            # There is one input stream per head
            *(f"IStream in{i}_{self.hls_sname()};" for i in range(self.heads))
        ]

    # Generates C++ code for calling the computation part of the operator
    def docompute(self):
        reversed_reads = ", ".join([
            f"in{i}_{self.hls_sname()}.read()"
            for i in reversed(range(self.heads))
        ])

        # Write the body of the head-splitting top-level function
        self.code_gen_dict["$DOCOMPUTE$"] = [
            # Repeat for the number of inputs
            # Note: Repeat for all num_inputs dimensions
            f"for(std::size_t i = 0; i < {np.prod(self.num_inputs)}; ++i) {{",
            # Pipeline the steps of this loop
            f"#pragma HLS pipeline II=1 style=flp",
            # Read the next input element from each input stream and concatenate
            # using the comma operator overload of ap_uint, writing into the
            # output stream
            f"out_{self.hls_sname()}.write(({reversed_reads}));"
            # End of for-loop over repetitions body
            f"}}"
        ]

    # Generates C++ code for reading the output stream and converting back to
    # numpy format for testing in C** simulation
    def dataoutstrm(self):
        # Output data will be stored in numpy files in the code generation
        # dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # Get the expected shape of the folded output array formatted as a C++
        # vector initializer
        # Note: Valid formatting relies on correct placement of curly braces
        # and line breaks: Open/close all three braces on the same line of code
        # to avoid '\n' to be inserted into the string
        shape = f"""{{{
        ','.join((str(i) for i in self.get_folded_output_shape()))
        }}}"""
        # Generate function call for reading from the output stream into the
        # output file
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            # Generate function call reading from stream into the output file
            #   Note: Outputs are always represented as numpy floats
            f'apintstream2npy<OPacked, OType, OType::width, float>(',
            f'out_{self.hls_sname()}, {shape}, "{code_gen_dir}/out.npy", false',
            f');',
        ]

    # Generates C++ code for saving the output of C++ simulation to a file in
    # numpy format
    def save_as_npy(self):
        # Note: This seems to be empty in ALL HLSCustomOps. Probably it was used
        # for something before, which is now integrated into dataoutstrm()?
        self.code_gen_dict["$SAVEASCNPY$"] = []

    # Generates essentially the head of the C++ function from which the IP block
    # will be generated during ipgen, i.e. actual synthesis
    def blackboxfunction(self):
        # Insert function head describing the top level interface of the head
        # splitting operator
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # @formatter:off Prevent Python formatter from messing with C++
            # formatting
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            # Output HLS stream
            f"  OStream &out_{self.hls_sname()}, ", ",".join([
            # One input HLS stream per head  # noqa: Formatting
            f"  IStream &in{i}_{self.hls_sname()}" for i in range(self.heads)
            ]),
            f")",
            # @formatter:off
        ]

    # Generates C++ pragmas to be inserted into the main function of the C++
    # simulation and the ipgen-blackboxfunction as well
    def pragmas(self):
        # Add HLS interface directives specifying how to create RTL ports for
        # the top-level function arguments
        self.code_gen_dict["$PRAGMAS$"] = [
            # Connect the output stream with an axi stream interface
            f"#pragma HLS INTERFACE axis port=out_{self.hls_sname()}"
        ]
        # Connect each input stream with an axi stream interface
        for i in range(self.heads):
            # Add new interface directive for the input stream
            self.code_gen_dict["$PRAGMAS$"] += [
                f"#pragma HLS INTERFACE axis port=in{i}_{self.hls_sname()}"
            ]
        # No block-level I/O protocol for the function return value
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )
