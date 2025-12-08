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
import os
import textwrap
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import roundup_to_integer_multiple

import finn.custom_op.fpgadataflow.elementwise_functions as elementwise_functions
from finn.custom_op.fpgadataflow.elementwise_functions import ElementwiseFunctionOperation
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)

# Mapping of memory resource attributes to the corresponding C++ HLS
# pragma directives
RAM_STYLES = {"auto": "AUTO", "block": "BRAM", "distributed": "LUTRAM", "ultra": "URAM"}


# HLS Backend specialization of the elementwise function operation operator
class ElementwiseFunctionOperation_hls(
    # CapWords convention
    ElementwiseFunctionOperation,
    HLSBackend,
):
    # Node attributes matching the HLS operator
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = ElementwiseFunctionOperation.get_nodeattr_types(self)
        # Add the HLSBackend default attributes on top
        attrs.update(HLSBackend.get_nodeattr_types(self))
        # Add/Specialize implementation specific attributes here...
        # Return the updated attributes dictionary
        return attrs

    # Maximum width of any ap_int used in this operator
    def get_ap_int_max_w(self):
        # Find the width of the input
        i_bits_max = self.get_instream_width(ind=0)
        # Width of the output, there is just one output
        # Note: there is one output per replica
        o_bits_max = self.get_outstream_width(ind=0)
        # Find the biggest of the inputs/outputs
        return max([i_bits_max, o_bits_max])

    # Note: End of shape and datatype utilities

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates c++ code and tcl script for ip generation."""
        super().code_generation_ipgen(model, fpgapart, clk)
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            self.generate_hdl_memstream(fpgapart)

    # Generates list of C++ includes to be placed at the top of the generated
    # code
    def global_includes(self):
        # Currently nothing to include
        self.code_gen_dict["$GLOBALS$"] = ['#include "flatten.hpp"']

    # Generates C++ code of type alias, global constant and macro definitions
    def defines(self, var):
        # Insert constants and type aliases into the dictionary
        self.code_gen_dict["$DEFINES$"] = [
            # Input and output element datatypes
            f"using InpType = {self.inp_dtype.get_hls_datatype_str()};",
            f"using OutType = {self.out_dtype.get_hls_datatype_str()};",
            # Width of single elements to avoid using ::width attribute which is
            # not present for datatype float
            f"static constexpr auto InpWidth = {self.inp_dtype.bitwidth()};",
            f"static constexpr auto OutWidth = {self.out_dtype.bitwidth()};",
            # Datatype of elements packed into the input stream
            f"using InpPacked = ap_uint<{self.get_instream_width(ind=0)}>;",
            # Datatype of elements packed into the output stream
            f"using OutPacked = ap_uint<{self.get_outstream_width(ind=0)}>;",
            # Input and output HLS stream datatypes
            "using InpStream = hls::stream<InpPacked>;",
            "using OutStream = hls::stream<OutPacked>;",
        ]

    # Generates C++ code for reading data from .npy (numpy format) for testing
    # in C++ simulation
    def read_npy_data(self):
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # Prepare empty stream reading to append optionals
        self.code_gen_dict["$READNPYDATA$"] = []

        # Generate function calls for reading the input files into the input
        # streams
        npy_type = "half" if self.inp_dtype.get_hls_datatype_str() == "half" else "float"
        self.code_gen_dict["$READNPYDATA$"] += [
            # Generate function call reading from file into the input stream
            #   Note: Inputs can be represented as numpy floats or halfs
            f"npy2apintstream<InpPacked, InpType, InpWidth, {npy_type}>(",
            f'"{code_gen_dir}/input_0.npy", in0_V, false',
            ");",
        ]

    # Generates C++ code for declaring all streams involved in C++ simulation
    # for testing
    def strm_decl(self):
        # Allways add the output stream to the declarations
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            # Note: Assumes stream type aliases to be set in defines
            "OutStream out0_V;"
        ]

        # Generate a stream declaration
        self.code_gen_dict["$STREAMDECLARATIONS$"] += [
            # Note: Assumes stream type aliases to be set in defines
            "InpStream in0_V;"
        ]

    # Generates C++ code for calling the computation part of the operator
    def docompute(self):
        # Get the folded shapes of all tensors involved without PE axis
        inp_shape = self.get_folded_input_shape(ind=0)[:-1]
        out_shape = self.get_folded_output_shape(ind=0)[:-1]

        # Code generation of array index strings
        def make_index_string(shape):
            # Generate index operation [i]
            return "".join([f"[i{d}]" for d in range(len(shape))])

        inp_index = make_index_string(inp_shape)

        # Generate C++ code for declaring an array of the buffer shapes
        inp_shape = "".join([f"[{size}]" for size in inp_shape])

        # Number of dimensions of the output. All shapes will be
        # aligned to this number of dimensions.
        # Note: +1 for the PE dimension
        ndim = len(out_shape) + 1

        # For-Loop template for nested loops over arbitrary many levels
        def for_loop(level, size):
            return f"for(std::size_t i{level} = 0; i{level}<{size}; ++i{level})"

        # Type of memory to use for storing constant parameters
        ram_style = RAM_STYLES[self.get_nodeattr("ram_style")]

        # Write the body of the top-level function
        self.code_gen_dict["$DOCOMPUTE$"] = [
            # @formatter:off  Disable formatter for mixed Python and C++
            # For streamed inputs, generate local buffer of non-broadcast size
            # but broadcasts dimensions un-squeezed to size 1. For constant
            # inputs, use the generated parameters of the same name.
            # For streamed inputs, implement a simple dual-port RAM partitioned
            # on the last, i.e., the PE, axis for parallel access.
            f"""
            InpType inp{inp_shape}[{self.pe}];
            #pragma HLS ARRAY_PARTITION variable=inp complete dim={ndim}
            #pragma HLS BIND_STORAGE variable=inp type=RAM_S2P impl={ram_style}
            """,
            # Buffer to hold the parallel output elements: Implement a simple
            # dual-port RAM for the output buffer, partitioned on the last,
            # i.e., the PE, axis for parallel access.
            # Note: The PE output should be rather small, force this into
            # distributed memory here.
            # TODO: Maybe reconsider this later?
            f"""
            OutType out[{self.pe}];
            #pragma HLS ARRAY_PARTITION variable=out complete dim=1
            #pragma HLS BIND_STORAGE variable=out type=RAM_S2P impl=LUTRAM
            """,
            # Perfect loop nest over all folded output dimensions
            *[for_loop(dim, size) + " {" for dim, size in enumerate(out_shape)],
            # Pipeline the loops. This should be possible as there is no code
            # between the loop levels, i.e., this is a perfect loop nest.
            """
            #pragma HLS pipeline II=1 style=flp
            """,
            # Read from the input stream
            f"""
            const auto buffer = Slice<InpType>{{}}(
                in0_V.read()
            );
            for(std::size_t pe = 0; pe < {self.pe}; ++pe) {{
            #pragma HLS unroll
                inp{inp_index}[pe] = buffer(pe, 0);
            }}
            """,
            # Apply PE parallel elementwise operations by filling the operation
            # template
            f"""
            for(std::size_t pe = 0; pe < {self.pe}; ++pe) {{
            #pragma HLS unroll
                out[pe] = {self.cpp_op.format(
                    f"inp{inp_index}[pe]"
                )};
            }}
            """,
            # Write the PE group into the output stream
            """
            out0_V.write(flatten(out));
            """,
            # Close all for-loop bodies of the generated nest
            *["}" for _ in enumerate(out_shape)]
            # @formatter:on  End of code generation
        ]

        # Post-process the generated code to remove unnecessary white space
        self.code_gen_dict["$DOCOMPUTE$"] = [
            textwrap.dedent(code) for code in self.code_gen_dict["$DOCOMPUTE$"]
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
        ','.join((str(i) for i in self.get_folded_output_shape(ind=0)))
        }}}"""
        # Generate function call for reading from the output stream into the
        # output file
        npy_type = "half" if self.out_dtype.get_hls_datatype_str() == "half" else "float"
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            # Generate function call reading from stream into the output file
            #   Note: Outputs can be numpy floats or halfs
            f"apintstream2npy<OutPacked, OutType, OutWidth, {npy_type}>(",
            f'out0_V, {shape}, "{code_gen_dir}/output_0.npy", false',
            ");",
        ]

    # Generates C++ code for saving the output of C++ simulation to a file in
    # numpy format
    def save_as_npy(self):
        # Note: This seems to be empty in ALL HLSBackends. Probably it was used
        # for something before, which is now integrated into dataoutstrm()?
        self.code_gen_dict["$SAVEASCNPY$"] = []

    # Generates essentially the head of the C++ function from which the IP block
    # will be generated during ipgen, i.e. actual synthesis
    def blackboxfunction(self):
        # Insert function head describing the top level interface of the
        # attention operator
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            "  InpStream &in0_V,",
            "  OutStream &out0_V",
            ")",
        ]

    # Generates C++ pragmas to be inserted into the main function of the C++
    # simulation and the ipgen-blackboxfunction as well
    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")
        # Check whether there are already pragmas in the code generation
        # dictionary
        if "$PRAGMAS$" not in self.code_gen_dict:
            # If not, insert an empty list to collect more pragmas
            self.code_gen_dict["$PRAGMAS$"] = []

        # Add HLS interface directives specifying how to create RTL ports for
        # the top-level function arguments
        self.code_gen_dict["$PRAGMAS$"] += [
            # Connect the output stream with an axi stream interface
            "#pragma HLS INTERFACE axis port=out0_V",
        ]
        # Connect the lhs input stream with an axi stream interface
        self.code_gen_dict["$PRAGMAS$"] += [
            "#pragma HLS INTERFACE axis port=in0_V",
        ]

        # No block-level I/O protocol for the function return value
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    # Returns the names of input and output interfaces grouped by protocol
    def get_verilog_top_module_intf_names(self):
        # Start collecting interface names in a dictionary starting with clock
        # and reset
        intf_names = {"clk": ["ap_clk"], "rst": ["ap_rst_n"]}
        # AXI stream input interfaces
        intf_names["s_axis"] = [("in0_V", self.get_instream_width_padded(ind=0))]
        # AXI stream output interfaces
        intf_names["m_axis"] = [("out0_V", self.get_outstream_width_padded(ind=0))]
        # No AXI-MM, AXI-Lite or protocol-less interfaces
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        # Return the interface name dictionary
        return intf_names

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            HLSBackend.execute_node(self, context, graph)
        elif mode == "rtlsim":
            # rtlsim execution needs to be overwritten here because the HLS code
            # is dynamically generated which results in different interfaces
            # Get the node wrapped by this custom op
            node = self.onnx_node
            # Input data is stored in numpy files in the code generation dictionary
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            # Get the inputs out of the execution context
            inp = context[node.input[0]]
            # Validate the shape of the inputs
            assert list(inp.shape) == self.get_normal_input_shape(
                ind=0
            ), f"Input shape mismatch for {node.input[0]}"
            # Reshape the inputs into folded form
            inp = inp.reshape(self.get_folded_input_shape(ind=0))
            # Path to store the intermediate inputs in numpy format
            inp_filename = os.path.join(code_gen_dir, "input_0.npy")
            # Save the folded inputs to file to be used by simulation
            np.save(inp_filename, inp)
            # Start collecting inputs/outputs to the RTL simulation in a dictionary
            # Note: Prepare one output empty output list
            io_dict = {"inputs": {}, "outputs": {"out0": []}}
            # Type and width of the input tensors
            inp_dtype = self.get_input_datatype(ind=0)
            inp_width = self.get_instream_width(ind=0)

            # Convert inputs to RTL simulation format
            io_dict["inputs"]["in0"] = npy_to_rtlsim_input(inp_filename, inp_dtype, inp_width)

            # Setup PyVerilator simulation of the node
            sim = self.get_rtlsim()
            # Reset the RTL simulation; finnxsi toggles the clock
            super().reset_rtlsim(sim)
            # Run the RTL Simulation
            self.rtlsim_multi_io(sim, io_dict)

            # Collect the output from RTL simulation
            out = io_dict["outputs"]["out0"]
            # Type and sizes of the output tensor
            dtype = self.get_output_datatype(ind=0)
            width = self.get_outstream_width(ind=0)
            shape = self.get_folded_output_shape(ind=0)
            # Path to store the intermediate numpy file
            filename = os.path.join(code_gen_dir, "output_0.npy")
            # Convert from RTL simulation format to numpy format
            rtlsim_output_to_npy(out, filename, dtype, shape, width, dtype.bitwidth())
            # Load the generated output numpy file
            out = np.load(filename)
            # Reshape the folded output and insert into the execution context
            context[node.output[0]] = out.reshape(self.get_normal_output_shape(ind=0))
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )


# Derive a specialization to implement elementwise relu of the input
@register_custom_op
class ElementwiseRelu_hls(ElementwiseFunctionOperation_hls, elementwise_functions.ElementwiseRelu):
    pass


# Derive a specialization to implement elementwise exponent of the input
@register_custom_op
class ElementwiseExp_hls(ElementwiseFunctionOperation_hls, elementwise_functions.ElementwiseExp):
    # Generates list of C++ includes to be placed at the top of the generated
    # code
    def global_includes(self):
        super().global_includes()
        # additional hls_math include
        self.code_gen_dict["$GLOBALS$"] += ['#include "hls_math.h"']


# Derive a specialization to implement elementwise erf of the input
@register_custom_op
class ElementwiseErf_hls(ElementwiseFunctionOperation_hls, elementwise_functions.ElementwiseErf):
    # Generates list of C++ includes to be placed at the top of the generated
    # code
    def global_includes(self):
        super().global_includes()
        # additional hls_math include
        self.code_gen_dict["$GLOBALS$"] += ['#include "hls_math.h"']
