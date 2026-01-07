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

import finn.custom_op.fpgadataflow.elementwise_binary as elementwise_binary
from finn.custom_op.fpgadataflow.elementwise_binary import ElementwiseBinaryOperation
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


# HLS Backend specialization of the binary elementwise operation operator
class ElementwiseBinaryOperation_hls(
    # CapWords convention
    ElementwiseBinaryOperation,
    HLSBackend,
):
    # Node attributes matching the HLS operator
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = ElementwiseBinaryOperation.get_nodeattr_types(self)
        # Add the HLSBackend default attributes on top
        attrs.update(HLSBackend.get_nodeattr_types(self))
        # Add/Specialize implementation specific attributes here...
        # Return the updated attributes dictionary
        return attrs

    # Maximum width of any ap_int used in this operator
    def get_ap_int_max_w(self):
        # Find the widths of the widest of the two inputs
        i_bits_max = max(self.get_instream_width(ind=0), self.get_instream_width(ind=1))
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

    # Generates C++ parameters file, i.e., constant initializer inputs
    def generate_params(self, model: ModelWrapper, path: str):
        # The code generation directory is specified as an argument, so this
        # will work for both RTL and C++ simulation
        code_gen_dir = path
        # By default, assume runtime inputs not requiring code to be generated
        lhs_code = rhs_code = ""
        # Check for an initializer providing the left hand side input
        lhs = model.get_initializer(self.onnx_node.input[0])
        # Folded output shape for broadcasting/aligning the input shapes
        out_shape = self.get_folded_output_shape(ind=0)
        # Type of memory to use for storing constant parameters
        ram_style = RAM_STYLES[self.get_nodeattr("ram_style")]

        # Check whether there are already pragmas in the code generation
        # dictionary
        if "$PRAGMAS$" not in self.code_gen_dict:
            # If not, insert an empty list to collect more pragmas
            # Note: Do this here as it is easier to add the array partition and
            # bind storage pragmas for generated parameter here, where the shape
            # is computed.
            self.code_gen_dict["$PRAGMAS$"] = []

        # If the left hand side input is provided as initializer, generate
        # initializer parameters code
        if lhs is not None:
            # Remember the "style" of receiving the input for further code
            # generation
            self.set_nodeattr("lhs_style", "const")
            # Reshape the parameter tensor into folded shape
            lhs = lhs.reshape(*self.get_folded_input_shape(ind=0))
            # Need to make sure there are PE many elements which can be accessed
            # in parallel
            if lhs.shape[-1] != self.pe:
                # Broadcast the parameter tensor "offline" to have PE elements
                # TODO: This replicates all parameters and might be inefficient
                #  in terms of memory utilization. It might be ore efficient to
                #  replicate the PEs when needed in docompute, probably at the
                #  cost of some latency for extra reads and registers.
                lhs = np.broadcast_to(lhs, lhs.shape[:-1] + (self.pe,))
            # Current, maybe non-aligned input shape
            lhs_shape = lhs.shape
            # Fill up shape from the left to match the broadcast output shape
            lhs_shape = (len(out_shape) - len(lhs_shape)) * (1,) + lhs_shape
            # Reshape the input to align with the output shape
            lhs = lhs.reshape(*lhs_shape)
            if self.get_nodeattr("mem_mode") == "internal_embedded":
                # Generate C++ array initialization code
                # Note: no packing, but with variable name/type declaration
                lhs_code = numpy_to_hls_code(lhs, self.lhs_dtype, "lhs", False, False)
                # Add pragma configuring the storage type to use for the parameter
                # tensors: This is a constant parameter implemented as dual-port ROM
                self.code_gen_dict["$PRAGMAS$"].append(
                    f"#pragma HLS BIND_STORAGE variable=lhs type=ROM_2P impl={ram_style}"
                )
                # Add pragma to partition the parameter tensor along the last
                # dimensions, i.e., the PE dimension for parallel access
                self.code_gen_dict["$PRAGMAS$"].append(
                    f"#pragma HLS ARRAY_PARTITION variable=lhs complete dim={len(lhs_shape)}"
                )

        # Check for an initializer providing the right hand side input
        rhs = model.get_initializer(self.onnx_node.input[1])
        # If the right hand side input is provided as initializer, generate
        # initializer parameters code
        if rhs is not None:
            # Remember the "style" of receiving the input for further code
            # generation
            self.set_nodeattr("rhs_style", "const")
            # Reshape the parameter tensor into folded shape
            rhs = rhs.reshape(*self.get_folded_input_shape(ind=1))
            # Need to make sure there are PE many elements which can be accessed
            # in parallel
            if rhs.shape[-1] != self.pe:
                # Broadcast the parameter tensor "offline" to have PE elements
                # TODO: This replicates all parameters and might be inefficient
                #  in terms of memory utilization. It might be ore efficient to
                #  replicate the PEs when needed in docompute, probably at the
                #  cost of some latency for extra reads and registers.
                rhs = np.broadcast_to(rhs, rhs.shape[:-1] + (self.pe,))
            # Current, maybe non-aligned input shape
            rhs_shape = rhs.shape
            # Fill up shape from the left to match the broadcast output shape
            rhs_shape = (len(out_shape) - len(rhs_shape)) * (1,) + rhs_shape
            # Reshape the input to align with the output shape
            rhs = rhs.reshape(*rhs_shape)
            if self.get_nodeattr("mem_mode") == "internal_embedded":
                # Generate C++ array initialization code
                # Note: no packing, but with variable name/type declaration
                rhs_code = numpy_to_hls_code(rhs, self.rhs_dtype, "rhs", False, False)
                # Add pragma configuring the storage type to use for the parameter
                # tensors: This is a constant parameter implemented as dual-port ROM
                self.code_gen_dict["$PRAGMAS$"].append(
                    f"#pragma HLS BIND_STORAGE variable=rhs type=ROM_2P impl={ram_style}"
                )
                # Add pragma to partition the parameter tensor along the last
                # dimensions, i.e., the PE dimension for parallel access
                self.code_gen_dict["$PRAGMAS$"].append(
                    f"#pragma HLS ARRAY_PARTITION variable=rhs complete dim={len(rhs_shape)}"
                )
            else:
                # merge first dimensions together
                rhs = rhs.reshape(-1, self.pe)
                # flip PE dimension
                rhs = np.flip(rhs, axis=-1)
                rhs_width = self.get_instream_width(1)
                # pad to nearest 4 bits to get hex strings
                rhs_width_padded = roundup_to_integer_multiple(rhs_width, 4)
                rhs_tensor = pack_innermost_dim_as_hex_string(
                    rhs, self.rhs_dtype, rhs_width_padded, prefix=""
                )
                rhs_stream = rhs_tensor.flatten()
                rhs_stream = rhs_stream.copy()
                with open(f"{code_gen_dir}/memblock.dat", "w") as f:
                    for val in rhs_stream:
                        f.write(val + "\n")

        # Open a file to store the thresholds parameters as C++ code
        with open(f"{code_gen_dir}/params.hpp", "w") as file:
            file.write(
                "\n".join(
                    [
                        # Insert left-hand-side and right-hand-side parameter code and
                        # append a newline at the end of the file (to avoid problems
                        # when including, required by C standard?)
                        lhs_code,
                        rhs_code,
                        "\n",
                    ]
                )
            )

    # Generates C++ code of type alias, global constant and macro definitions
    def defines(self, var):
        # Insert constants and type aliases into the dictionary
        self.code_gen_dict["$DEFINES$"] = [
            # Input and output element datatypes
            f"using LhsType = {self.lhs_dtype.get_hls_datatype_str()};",
            f"using RhsType = {self.rhs_dtype.get_hls_datatype_str()};",
            f"using OutType = {self.out_dtype.get_hls_datatype_str()};",
            # Width of single elements to avoid using ::width attribute which is
            # not present for datatype float
            f"static constexpr auto LhsWidth = {self.lhs_dtype.bitwidth()};",
            f"static constexpr auto RhsWidth = {self.rhs_dtype.bitwidth()};",
            f"static constexpr auto OutWidth = {self.out_dtype.bitwidth()};",
            # Datatype of elements packed into the input stream
            f"using LhsPacked = ap_uint<{self.get_instream_width(ind=0)}>;",
            f"using RhsPacked = ap_uint<{self.get_instream_width(ind=1)}>;",
            # Datatype of elements packed into the output stream
            f"using OutPacked = ap_uint<{self.get_outstream_width(ind=0)}>;",
            # Include the activation function type definitions and parameters
            #   Note: The typedefs in this header require the typedefs above,
            #   thus adding this to the global includes is not possible.
            '#include "params.hpp"',
            # Input and output HLS stream datatypes
            "using LhsStream = hls::stream<LhsPacked>;",
            "using RhsStream = hls::stream<RhsPacked>;",
            "using OutStream = hls::stream<OutPacked>;",
        ]

    # Generates C++ code for reading data from .npy (numpy format) for testing
    # in C++ simulation
    def read_npy_data(self):
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # Prepare empty stream reading to append optionals
        self.code_gen_dict["$READNPYDATA$"] = []
        # If the left-hand-side is provided as runtime input or internal decoupled const,
        # read code needs to be generated
        mem_mode = self.get_nodeattr("mem_mode")
        lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
        if self.lhs_style == "input" or lhs_decoupled:
            # Generate function calls for reading the input files into the input
            # streams
            npy_type = "half" if self.lhs_dtype.get_hls_datatype_str() == "half" else "float"
            self.code_gen_dict["$READNPYDATA$"] += [
                # Generate function call reading from file into the input stream
                #   Note: Inputs can be represented as numpy floats or halfs
                f"npy2apintstream<LhsPacked, LhsType, LhsWidth, {npy_type}>(",
                f'"{code_gen_dir}/input_0.npy", in0_V, false',
                ");",
            ]
        # If the right-hand-side is provided as runtime input or internal decoupled const,
        # read code needs to be generated
        rhs_decoupled = self.rhs_style == "const" and mem_mode == "internal_decoupled"
        if self.rhs_style == "input" or rhs_decoupled:
            # Generate function calls for reading the input files into the input
            # streams
            npy_type = "half" if self.rhs_dtype.get_hls_datatype_str() == "half" else "float"
            self.code_gen_dict["$READNPYDATA$"] += [
                # Generate function call reading from file into the input stream
                #   Note: Inputs can be represented as numpy floats or halfs
                f"npy2apintstream<RhsPacked, RhsType, RhsWidth, {npy_type}>(",
                f'"{code_gen_dir}/input_1.npy", in1_V, false',
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
        # If the left-hand-side is provided as runtime input or internal decoupled const,
        # read code needs to be generated
        mem_mode = self.get_nodeattr("mem_mode")
        lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
        if self.lhs_style == "input" or lhs_decoupled:
            # Generate a stream declaration
            self.code_gen_dict["$STREAMDECLARATIONS$"] += [
                # Note: Assumes stream type aliases to be set in defines
                "LhsStream in0_V;"
            ]
        # If the right-hand-side is provided as runtime input or internal decoupled const,
        # read code needs to be generated
        rhs_decoupled = self.rhs_style == "const" and mem_mode == "internal_decoupled"
        if self.rhs_style == "input" or rhs_decoupled:
            # Generate a stream declaration
            self.code_gen_dict["$STREAMDECLARATIONS$"] += [
                # Note: Assumes stream type aliases to be set in defines
                "RhsStream in1_V;"
            ]

    # Generates C++ code for calling the computation part of the operator
    def docompute(self):
        # Add padding ones to a shape to match the broadcast output shape
        def pad_shape(shape):
            return (len(out_shape) - len(shape)) * (1,) + shape

        # Get the folded shapes of all tensors involved without PE axis
        lhs_shape = self.get_folded_input_shape(ind=0)[:-1]
        rhs_shape = self.get_folded_input_shape(ind=1)[:-1]
        out_shape = self.get_folded_output_shape(ind=0)[:-1]
        # Expanded shape of the inputs, filling with dimensions of size 1 from
        # the left to align the shape with the broadcast shape
        lhs_shape = pad_shape(lhs_shape)
        rhs_shape = pad_shape(rhs_shape)

        # Removes contiguous matching dimensions from a shape
        def drop_matching_dims(shape, like):
            # Core functionality for this is implemented in itertools
            from itertools import dropwhile

            # Compare shapes from left to right removing dimensions as long as
            # they match
            return (*[size for size, _ in dropwhile(lambda x: x[0] == x[1], zip(shape, like))],)

        # Take away all contiguous dimensions where these align with the output
        # shape, as these can be consumed directly without buffering to be
        # repeated
        lhs_buffer_shape = drop_matching_dims(lhs_shape, out_shape)
        rhs_buffer_shape = drop_matching_dims(rhs_shape, out_shape)
        # Expand once again, filling with dimensions of size 1 from the left to
        # align the shape with the broadcast shape
        lhs_buffer_shape = pad_shape(lhs_buffer_shape)
        rhs_buffer_shape = pad_shape(rhs_buffer_shape)

        # Code generation of array index strings with broadcasting
        def make_index_string(shape):
            # Generate index operation [i] for "normal" dimensions but reduce to
            # hardcoded [0] for broadcast dimensions to repeat from a single
            # buffer slot
            return "".join([f"[i{d}]" if s != 1 else "[0]" for d, s in enumerate(shape)])

        # Generate the C++ code for indexing the buffers depending on if the input/parameter
        # gets streamed in or if it is provided as a constant
        mem_mode = self.get_nodeattr("mem_mode")
        lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
        lhs_key = "stream_in" if self.lhs_style == "input" or lhs_decoupled else "const"
        lhs_index = {
            "stream_in": make_index_string(lhs_buffer_shape),
            "const": make_index_string(lhs_shape),
        }[lhs_key]
        rhs_decoupled = self.rhs_style == "const" and mem_mode == "internal_decoupled"
        rhs_key = "stream_in" if self.rhs_style == "input" or rhs_decoupled else "const"
        rhs_index = {
            "stream_in": make_index_string(rhs_buffer_shape),
            "const": make_index_string(rhs_shape),
        }[rhs_key]

        # Generate C++ code for declaring an array of the buffer shapes
        lhs_buffer_shape = "".join([f"[{size}]" for size in lhs_buffer_shape])
        rhs_buffer_shape = "".join([f"[{size}]" for size in rhs_buffer_shape])

        # Number of dimensions of the (broadcast) output. All shapes will be
        # aligned to this number of dimensions.
        # Note: +1 for the PE dimension
        ndim = len(out_shape) + 1

        # For-Loop template for nested loops over arbitrary many levels
        def for_loop(level, size):
            return f"for(std::size_t i{level} = 0; i{level}<{size}; ++i{level})"

        # Generate code testing for the condition when the next element needs to
        # be read from the input stream according to broadcasting semantics
        def read_stream_condition(shape):
            # Start with the assumption that none of the dimensions is
            # broadcast, meaning each individual element needs to be read from
            # the stream
            condition = "true"
            # Search for the dimensions which are broadcast
            for dim, size in enumerate(shape):
                # If this dimension has a size of 1 in the input but not in the
                # output, it is broadcast and contributes to the conjunctive
                # reading condition if this index wraps around
                if size == 1 and out_shape[dim] != 1:
                    # Add testing for index wrap-around to the condition
                    condition += f" && (i{dim} == 0)"
            # Return the composed reading condition
            return condition

        # Generate code for unpacking elements read from the stream into the PE-
        # parallel buffer according to broadcasting semantics
        def unpack_buffer(shape):
            # Unpacking behavior depends on whether the last, i.e., folded PE
            # dimension is broadcast
            if shape[-1] == 1 and self.pe != self.out_shape[-1]:
                # PE axis is broadcast, i.e., slice yields just one element
                # which needs to be replicated
                return "buffer(0, 0)"
            # PE axis is not broadcast, i.e., slice actually yields parallel
            # elements to be unpacked
            return "buffer(pe, 0)"

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
            LhsType lhs{lhs_buffer_shape}[{self.pe}];
            #pragma HLS ARRAY_PARTITION variable=lhs complete dim={ndim}
            #pragma HLS BIND_STORAGE variable=lhs type=RAM_S2P impl={ram_style}
            """
            if self.lhs_style == "input" or lhs_decoupled
            else """""",
            f"""
            RhsType rhs{rhs_buffer_shape}[{self.pe}];
            #pragma HLS ARRAY_PARTITION variable=rhs complete dim={ndim}
            #pragma HLS BIND_STORAGE variable=rhs type=RAM_S2P impl={ram_style}
            """
            if self.rhs_style == "input" or rhs_decoupled
            else """""",
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
            # Read from the left-hand-side input stream if new elements are
            # needed according to broadcasting semantics
            f"""
            if({read_stream_condition(lhs_shape)}) {{
                const auto buffer = Slice<LhsType>{{}}(
                    in0_V.read()
                );
                for(std::size_t pe = 0; pe < {self.pe}; ++pe) {{
                #pragma HLS unroll
                    lhs{lhs_index}[pe] = {unpack_buffer(lhs_shape)};
                }}
            }}
            """
            if self.lhs_style == "input" or lhs_decoupled
            else """""",
            # Read from the right-hand-side input stream if new elements are
            # needed according to broadcasting semantics
            f"""
            if({read_stream_condition(rhs_shape)}) {{
                const auto buffer = Slice<RhsType>{{}}(
                    in1_V.read()
                );
                for(std::size_t pe = 0; pe < {self.pe}; ++pe) {{
                #pragma HLS unroll
                    rhs{rhs_index}[pe] = {unpack_buffer(rhs_shape)};
                }}
            }}
            """
            if self.rhs_style == "input" or rhs_decoupled
            else """""",
            # Apply PE parallel elementwise operations by filling the operation
            # template
            f"""
            for(std::size_t pe = 0; pe < {self.pe}; ++pe) {{
            #pragma HLS unroll
                out[pe] = {self.cpp_op.format(
                    f"lhs{lhs_index}[pe]", f"rhs{rhs_index}[pe]"
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
        # Check whether the inputs are provided at runtime to generate stream
        # inputs to the toplevel interface
        mem_mode = self.get_nodeattr("mem_mode")
        lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
        rhs_decoupled = self.rhs_style == "const" and mem_mode == "internal_decoupled"
        runtime_lhs = self.lhs_style == "input" or lhs_decoupled
        runtime_rhs = self.rhs_style == "input" or rhs_decoupled
        # Insert function head describing the top level interface of the
        # attention operator
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            "  LhsStream &in0_V," if runtime_lhs else "",
            "  RhsStream &in1_V," if runtime_rhs else "",
            "  OutStream &out0_V",
            ")",
        ]

    # Generates C++ pragmas to be inserted into the main function of the C++
    # simulation and the ipgen-blackboxfunction as well
    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")
        lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
        rhs_decoupled = self.rhs_style == "const" and mem_mode == "internal_decoupled"
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

        # If the left-hand-side is provided as runtime input interface pragmas
        # need to be inserted
        if self.lhs_style == "input" or lhs_decoupled:
            # Connect the lhs input stream with an axi stream interface
            self.code_gen_dict["$PRAGMAS$"] += [
                "#pragma HLS INTERFACE axis port=in0_V",
            ]

        # If the right-hand-side is provided as runtime input interface pragmas
        # need to be inserted
        if self.rhs_style == "input" or rhs_decoupled:
            # Connect the rhs input stream with an axi stream interface
            self.code_gen_dict["$PRAGMAS$"] += [
                "#pragma HLS INTERFACE axis port=in1_V",
            ]

        # No block-level I/O protocol for the function return value
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    # Returns the names of input and output interfaces grouped by protocol
    def get_verilog_top_module_intf_names(self):
        # Start collecting interface names in a dictionary starting with clock
        # and reset
        intf_names = {"clk": ["ap_clk"], "rst": ["ap_rst_n"]}
        # AXI stream input interfaces
        intf_names["s_axis"] = []
        # If the left-hand-side is provided as runtime input interface names
        # need to be inserted
        if self.lhs_style == "input":
            intf_names["s_axis"] += [("in0_V", self.get_instream_width_padded(ind=0))]
        # If the right-hand-side is provided as runtime input interface names
        # need to be inserted
        if self.rhs_style == "input":
            intf_names["s_axis"] += [("in1_V", self.get_instream_width_padded(ind=1))]
        # AXI stream output interfaces
        intf_names["m_axis"] = [("out0_V", self.get_outstream_width_padded(ind=0))]
        # No AXI-MM, AXI-Lite or protocol-less interfaces
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        # Return the interface name dictionary
        return intf_names

    def code_generation_ipi(self):
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]
        # add streamer if needed
        mem_mode = self.get_nodeattr("mem_mode")
        lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
        rhs_decoupled = self.rhs_style == "const" and mem_mode == "internal_decoupled"

        # lhs_decoupled XOR rhs_decoupled
        if lhs_decoupled != rhs_decoupled:
            node_name = self.onnx_node.name
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )
            # instantiate the hls ip
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s"
                % (self.get_nodeattr("ip_vlnv"), node_name, node_name)
            )
            # instantiate a streamer and connect it to the IP
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
            ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
            file_suffix = "_memstream_wrapper.v"
            # automatically find memstream verilog component in code generation directory
            for fname in os.listdir(code_gen_dir):
                if fname.endswith(file_suffix):
                    strm_tmpl = fname
            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                os.path.join(code_gen_dir, strm_tmpl),
                axi_dir + "axilite.sv",
                ms_rtllib_dir + "memstream_axi.sv",
                ms_rtllib_dir + "memstream.sv",
            ]
            for f in sourcefiles:
                cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (strm_tmpl_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/in1_V]" % (node_name, strm_inst, node_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            # 2x clock is not used for decoupled elementwise ops
            # simply connect input to the 1x clock for now
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, rst_name, node_name, node_name, rst_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, clk_name, node_name, node_name, clk_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, din_name, node_name, node_name, din_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, dout_name, node_name, node_name, dout_name)
            )
            cmd.append("save_bd_design")
        else:
            # base class impl sufficient
            return super().code_generation_ipi()
        return cmd

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
            lhs = context[node.input[0]]
            rhs = context[node.input[1]]
            # Validate the shape of the inputs
            assert list(lhs.shape) == self.get_normal_input_shape(
                ind=0
            ), f"Input shape mismatch for {node.input[0]}"
            assert list(rhs.shape) == self.get_normal_input_shape(
                ind=1
            ), f"Input shape mismatch for {node.input[1]} {rhs.shape=}"
            # Reshape the inputs into folded form
            lhs = lhs.reshape(self.get_folded_input_shape(ind=0))
            rhs = rhs.reshape(self.get_folded_input_shape(ind=1))
            # Path to store the intermediate inputs in numpy format
            lhs_filename = os.path.join(code_gen_dir, "input_0.npy")
            rhs_filename = os.path.join(code_gen_dir, "input_1.npy")
            # Save the folded inputs to file to be used by simulation
            np.save(lhs_filename, lhs)
            np.save(rhs_filename, rhs)
            # Start collecting inputs/outputs to the RTL simulation in a dictionary
            # Note: Prepare one output empty output list
            io_dict = {"inputs": {}, "outputs": {"out0": []}}
            # Type and width of the input tensors
            lhs_dtype = self.get_input_datatype(ind=0)
            lhs_width = self.get_instream_width(ind=0)
            rhs_dtype = self.get_input_datatype(ind=1)
            rhs_width = self.get_instream_width(ind=1)

            mem_mode = self.get_nodeattr("mem_mode")
            lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
            rhs_decoupled = self.rhs_style == "const" and mem_mode == "internal_decoupled"
            # If the left-hand-side is provided as runtime input it needs to be
            # inserted into the RTL simulation inputs
            if self.lhs_style == "input" or lhs_decoupled:
                # Convert inputs to RTL simulation format
                io_dict["inputs"]["in0"] = npy_to_rtlsim_input(lhs_filename, lhs_dtype, lhs_width)

            # If the right-hand-side is provided as runtime input it needs to be
            # inserted into the RTL simulation inputs
            if self.rhs_style == "input" or rhs_decoupled:
                # Convert inputs to RTL simulation format
                io_dict["inputs"]["in1"] = npy_to_rtlsim_input(rhs_filename, rhs_dtype, rhs_width)

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


# Derive a specialization to implement elementwise addition of two inputs
@register_custom_op
class ElementwiseAdd_hls(ElementwiseBinaryOperation_hls, elementwise_binary.ElementwiseAdd):
    pass


# Derive a specialization to implement elementwise subtraction of two inputs
@register_custom_op
class ElementwiseSub_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseSub,
):
    pass


# Derive a specialization to implement elementwise multiplication of two inputs
@register_custom_op
class ElementwiseMul_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseMul,
):
    pass


# Derive a specialization to implement elementwise division of two inputs
@register_custom_op
class ElementwiseDiv_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseDiv,
):
    pass


# TODO: ElementwiseMod_hls - Requires extra attribute selecting the function


# Derive a specialization to implement elementwise logical and of two inputs
@register_custom_op
class ElementwiseAnd_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseAnd,
):
    pass


# Derive a specialization to implement elementwise logical or of two inputs
@register_custom_op
class ElementwiseOr_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseOr,
):
    pass


# Derive a specialization to implement elementwise logical xor of two inputs
@register_custom_op
class ElementwiseXor_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseXor,
):
    pass


# Derive a specialization to implement elementwise equal of two inputs
@register_custom_op
class ElementwiseEqual_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseEqual,
):
    pass


# Derive a specialization to implement elementwise less of two inputs
@register_custom_op
class ElementwiseLess_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseLess,
):
    pass


# Derive a specialization to implement elementwise less or equal of two inputs
@register_custom_op
class ElementwiseLessOrEqual_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseLessOrEqual,
):
    pass


# Derive a specialization to implement elementwise greater of two inputs
@register_custom_op
class ElementwiseGreater_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseGreater,
):
    pass


# Derive a specialization to implement elementwise greater or equal of two
# inputs
@register_custom_op
class ElementwiseGreaterOrEqual_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseGreaterOrEqual,
):
    pass


# Derive a specialization to implement elementwise bitwise and of two inputs
@register_custom_op
class ElementwiseBitwiseAnd_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseBitwiseAnd,
):
    pass


# Derive a specialization to implement elementwise bitwise or of two inputs
@register_custom_op
class ElementwiseBitwiseOr_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseBitwiseOr,
):
    pass


# Derive a specialization to implement elementwise bitwise xor of two inputs
@register_custom_op
class ElementwiseBitwiseXor_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseBitwiseXor,
):
    pass


# ElementwiseBitShift_hls - Requires extra attribute selecting the direction
@register_custom_op
class ElementwiseBitShift_hls(
    # CapWords convention
    ElementwiseBinaryOperation_hls,
    elementwise_binary.ElementwiseBitShift,
):
    # We need to resolve the attribute types due to multiple inheritance
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = elementwise_binary.ElementwiseBitShift.get_nodeattr_types(self)
        # Add the HLSBackend default attributes on top
        attrs.update(HLSBackend.get_nodeattr_types(self))
        # Add/Specialize implementation specific attributes here...
        # Return the updated attributes dictionary
        return attrs


# # Derive a specialization to implement elementwise power of two inputs
# TODO: std::pow does not work for HLS types and hls::pow fails to link for some
#  reason
# @register_custom_op
# class ElementwisePow_hls(
#     # CapWords convention
#     ElementwiseBinaryOperation_hls, elementwise_binary.ElementwisePow
# ):
#     pass
