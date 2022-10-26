# Copyright (c) 2020, Xilinx
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

import math
import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp

# the IODMA inerfaces a memory-mapped AXI interface and an AXI stream
# direction "in": pulls data from AXI-MM to AXI stream
# direction "out": pushes data from AXI stream to AXI-MM

# DMA Addressing
# - burst mode can be "wrap" or "increment"
# - "increment" bursts will increment the address when moving to the next image
# - "wrap" bursts will reinitialize the address to the start address,
#   and are useful for e.g. streaming weights, where the same buffer is
#   repeatedly read into the FPGA
# - no additional alignment restrictions beyond anything specified in the AXI spec

# Interfaces
# - AXI-MM name specified by intfName unless this is set to "" (empty, the default)
#   in which case output AXI-MM are named "out" and input AXI-MM are named "in0"
# - AXI-MM interface width (in bits) is specified by intfWidth
# - AXI-Stream interface width (in bits) is specified by streamWidth
# - If inftWidth and streamWidth are not equal, the DMA core performs
#   width conversion by going up to the least common multiple of bitwidths
#   e.g. intfWidth=32b -> 96b -> sreamWidth=24b
# - transfers occur in multiples of the AXI-MM interface width, therefore
#   the total number of bits in the tensor must be a multiple of intfWidth
# - transfers occur in multiples of the AXI-Stream interface width, therefore
#   the total number of bits in the tensor must be a multiple of streamWidth
# - both interface widths must be a multiple of 8b (AXI protocol requirement)
# - in most systems, intfWidth is also restricted to a power of 2 (e.g. Vitis)
#   but this is not universal so we don't check here explicitly

# Input/output tensor sizes shapes
# - The data being moved is a tensor of shape numInputVectors+[NumChannels]
# - The data type of the tensor elements is specified by dataType
# - on the stream side
#       -the normal shape is the same as the ONNX tensor attached to it
#       -the folded shape is computed from the stream width and normal shape
# - on the AXI-MM side
#       -the normal shape is the same as the one on the stream side
#       -the folded shape is not defined


class IODMA(HLSCustomOp):
    """Class that corresponds to finn-hlslib DMA function(s)."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "dataType": ("s", True, ""),
            # Width of input or output stream
            "streamWidth": ("i", False, 32),
            # DMA-specific parameters
            # width of axi-mm interface
            "intfWidth": ("i", False, 32),
            # burst mode for axi-mm interface (wrap used for DRAM weights)
            "burstMode": ("s", False, "increment", {"wrap", "increment"}),
            # IODMA direction: in = read from DRAM, out = write to DRAM
            "direction": ("s", False, "in", {"in", "out"}),
            # shape describing input vecs per execution
            "numInputVectors": ("ints", False, [1]),
            # name of axi-mm interface
            "intfName": ("s", False, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        vecs = list(self.get_nodeattr("numInputVectors"))
        num_ch = self.get_nodeattr("NumChannels")
        ishape = tuple(vecs + [num_ch])
        return ishape

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind=0):
        if self.get_nodeattr("direction") == "in":
            raise ValueError("Folded input shape not defined for input IODMA")
        else:
            shape = list(self.get_normal_input_shape())
            itype_bits = self.get_input_datatype().bitwidth()
            intfw = self.get_nodeattr("streamWidth")
            assert (
                intfw % itype_bits == 0
            ), "Input stream width must be a multiple of datatype bits"
            elems_per_word = intfw // itype_bits
            assert shape[-1] % elems_per_word == 0, "Fold depth must be integer"
            fold_depth = shape[-1] // elems_per_word
            shape[-1] = fold_depth
            shape.append(elems_per_word)
            return tuple(shape)

    def get_folded_output_shape(self, ind=0):
        if self.get_nodeattr("direction") == "out":
            raise ValueError("Folded output shape not defined for output IODMA")
        else:
            shape = list(self.get_normal_output_shape())
            itype_bits = self.get_output_datatype().bitwidth()
            intfw = self.get_nodeattr("streamWidth")
            assert (
                intfw % itype_bits == 0
            ), "Input stream width must be a multiple of datatype bits"
            elems_per_word = intfw // itype_bits
            assert shape[-1] % elems_per_word == 0, "Fold depth must be integer"
            fold_depth = shape[-1] // elems_per_word
            shape[-1] = fold_depth
            shape.append(elems_per_word)
            return tuple(shape)

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        if self.get_nodeattr("direction") == "in":
            return self.get_nodeattr("intfWidth")
        elif self.get_nodeattr("direction") == "out":
            return self.get_nodeattr("streamWidth")
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

    def get_outstream_width(self, ind=0):
        if self.get_nodeattr("direction") == "out":
            return self.get_nodeattr("intfWidth")
        elif self.get_nodeattr("direction") == "in":
            return self.get_nodeattr("streamWidth")
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

    def get_number_output_values(self):
        oshape = self.get_normal_output_shape()
        itype_bits = self.get_input_datatype().bitwidth()
        stream_width = self.get_nodeattr("streamWidth")
        nelems = np.prod(oshape)
        nbits = nelems * itype_bits
        assert (
            nbits % stream_width == 0
        ), "DMA: total transfer size must be word multiple"
        ovalues = nbits // stream_width
        return ovalues

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "dma.h"']
        self.code_gen_dict["$GLOBALS$"].append('#include "streamtools.h"')

    def defines(self, var):
        itype_bits = self.get_input_datatype().bitwidth()
        total_bits = itype_bits * np.prod(self.get_normal_input_shape())
        assert total_bits % 8 == 0, "DMA input not a multiple of 1 Byte"
        total_bytes = total_bits // 8
        self.code_gen_dict["$DEFINES$"] = [
            """#define NumBytes1 {}\n#define DataWidth1 {}\n""".format(
                total_bytes, self.get_nodeattr("intfWidth")
            )
        ]

    def get_ap_int_max_w(self):
        "Return the maximum width of any ap_int used in this module."
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        width_lcm = (instream * outstream) // math.gcd(instream, outstream)
        return width_lcm

    def docompute(self):
        direction = self.get_nodeattr("direction")
        mode = self.get_nodeattr("burstMode")
        dwc_func = "StreamingDataWidthConverter_Batch"
        if direction == "in":
            if mode == "wrap":
                func = "Mem2Stream_Batch_external_wmem"
            else:
                func = "Mem2Stream_Batch"
        elif direction == "out":
            func = "Stream2Mem_Batch"
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")
        # define templates for instantiation
        dma_inst_template = func + "<DataWidth1, NumBytes1>(%s, %s, numReps);"
        dwc_inst_template = dwc_func + "<%d, %d, %d>(%s, %s, numReps);"
        # do stream infrastructure and instantiations
        intfw = self.get_nodeattr("intfWidth")
        strmw = self.get_nodeattr("streamWidth")
        width_lcm = (strmw * intfw) // math.gcd(strmw, intfw)
        # we always need two streams: one of width_lcm, and one of intfw width
        # because we use WidthAdjustedInputStream,
        dtype_bits = self.get_input_datatype().bitwidth()
        total_bits = dtype_bits * np.prod(self.get_normal_input_shape())

        if direction == "in":
            # AXI MM -> IODMA -> (DWCs) -> out
            # DWCs depend on AXI MM and out interface width
            if strmw == intfw:
                # case 0: AXI MM width = out width, no DWCs needed
                self.code_gen_dict["$DOCOMPUTE$"] = [dma_inst_template % ("in0", "out")]
            elif (strmw % intfw == 0) or (intfw % strmw == 0):
                # case 1: AXI MM width divisible by out width or vice versa
                # single DWC + single extra stream needed
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dma2dwc;" % intfw,
                    dma_inst_template % ("in0", "dma2dwc"),
                    dwc_inst_template
                    % (intfw, strmw, total_bits // intfw, "dma2dwc", "out"),
                ]
            else:
                # case 2: AXI MM width not divisible by out width or vice versa
                # need 2 DWCs (going through the least common multiple width)
                # and 2 streams
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dma2lcm;" % intfw,
                    "hls::stream<ap_uint<%d> > lcm2out;" % width_lcm,
                    dma_inst_template % ("in0", "dma2lcm"),
                    dwc_inst_template
                    % (intfw, width_lcm, total_bits // intfw, "dma2lcm", "lcm2out"),
                    dwc_inst_template
                    % (width_lcm, strmw, total_bits // width_lcm, "lcm2out", "out"),
                ]
        elif direction == "out":
            # in0 -> (DWCs) -> IODMA -> AXI MM
            # DWCs depend on AXI MM and out interface width
            if strmw == intfw:
                # case 0: in width = AXI MM width, no DWCs needed
                self.code_gen_dict["$DOCOMPUTE$"] = [dma_inst_template % ("in0", "out")]
            elif (strmw % intfw == 0) or (intfw % strmw == 0):
                # case 1: AXI MM width divisible by in width or vice versa
                # single DWC + single extra stream needed
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dwc2dma;" % intfw,
                    dwc_inst_template
                    % (strmw, intfw, total_bits // strmw, "in0", "dwc2dma"),
                    dma_inst_template % ("dwc2dma", "out"),
                ]
            else:
                # case 2: AXI MM width not divisible by out width or vice versa
                # need 2 DWCs (going through the least common multiple width)
                # and 2 streams
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > in2lcm;" % width_lcm,
                    "hls::stream<ap_uint<%d> > lcm2dma;" % intfw,
                    dwc_inst_template
                    % (strmw, width_lcm, total_bits // strmw, "in0", "in2lcm"),
                    dwc_inst_template
                    % (width_lcm, intfw, total_bits // width_lcm, "in2lcm", "lcm2dma"),
                    dma_inst_template % ("lcm2dma", "out"),
                ]
        else:
            raise Exception("Unknown IODMA direction: %s" % direction)

    def blackboxfunction(self):
        packed_ibits = self.get_instream_width()
        packed_hls_type_in = "ap_uint<%d>" % packed_ibits
        packed_obits = self.get_outstream_width()
        packed_hls_type_out = "ap_uint<%d>" % packed_obits
        direction = self.get_nodeattr("direction")
        if direction == "in":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(%s *in0, hls::stream<%s > &out, unsigned int numReps)"
                % (self.onnx_node.name, packed_hls_type_in, packed_hls_type_out)
            ]
        elif direction == "out":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(hls::stream<%s > &in0, %s *out, unsigned int numReps)"
                % (self.onnx_node.name, packed_hls_type_in, packed_hls_type_out)
            ]
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE s_axilite port=numReps bundle=control"
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE s_axilite port=return bundle=control"
        )
        direction = self.get_nodeattr("direction")
        intfname = self.get_nodeattr("intfName")
        if direction == "in":
            if intfname == "":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=in0"
                )
            else:
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=%s" % (intfname)
                )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=in0 bundle=control"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE axis port=out name=out_" + self.hls_sname()
            )
        elif direction == "out":
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE axis port=in0 name=in0_" + self.hls_sname()
            )
            if intfname == "":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=out"
                )
            else:
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=%s" % (intfname)
                )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=out bundle=control"
            )
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS DATAFLOW")

    def execute_node(self, context, graph):
        pass

    def dataoutstrm(self):
        pass

    def read_npy_data(self):
        pass

    def save_as_npy(self):
        pass

    def strm_decl(self):
        pass

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("direction") == "out":
            intf_names["m_axis"] = []
        else:
            intf_names["s_axis"] = []
        intf_names["axilite"] = ["s_axi_control"]
        intf_names["aximm"] = [("m_axi_gmem", self.get_nodeattr("intfWidth"))]
        return intf_names
