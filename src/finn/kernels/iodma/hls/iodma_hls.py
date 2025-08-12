from finn.kernels import Kernel, KernelProjection
from finn.util import templates
from dataclasses import dataclass, field
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from qonnx.core.datatype import DataType
import numpy as np
import math


@dataclass(frozen=True, init=False)
class IODMAHLS(Kernel):
    """ IODMA HLS kernel class """

    ######################### Kernel attributes #########################
    NumChannels:int
    # FINN input datatype
    dataType:str
    # Width of input or output stream
    streamWidth:int
    # DMA-specific parameters
    # width of axi-mm interface
    intfWidth:int = 32
    # burst mode for axi-mm interface (wrap used for DRAM weights)
    burstMode:str = "increment"
    # IODMA direction: in = read from DRAM, out = write to DRAM
    direction:str = "in"
    # shape describing input vecs per execution
    numInputVectors:list[int] = field(default_factory=lambda: [1])
    # name of axi-mm interface
    intfName:str = ""

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style:str = "hls"

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = ()

    ######################### Code Generation #########################
    sharedFiles: FrozenSet[Tuple[str,Path]] = frozenset({
        ("finn-hlslib", Path("."))
    })

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return { 
            (self.code_generation_ipgen, Path(f"{self.name}.cpp"))
        }

    def code_generation_ipgen(self, ctx):
        """Generates c++ code and tcl script for ip generation."""

        node_dir = ctx.directory

        # generate top cpp file for ip generation
        code_gen_dict = {}
        code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        code_gen_dict |= self.global_includes()
        code_gen_dict |= self.defines("ipgen")
        code_gen_dict |= self.blackboxfunction()
        code_gen_dict |= self.pragmas()
        code_gen_dict |= self.docompute()

        template = templates.ipgen_template

        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        with open(node_dir / f"{self.name}.cpp", "w") as f:
            f.write(template)

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        if self.direction == "out":
            intf_names["m_axis"] = []
        else:
            intf_names["s_axis"] = []
        intf_names["axilite"] = ["s_axi_control"]
        intf_names["aximm"] = [("m_axi_gmem", self.intfWidth)]
        return intf_names

    def global_includes(self):
        code_gen_dict = {}
        code_gen_dict["$GLOBALS$"] = ['#include "dma.h"']
        code_gen_dict["$GLOBALS$"].append('#include "streamtools.h"')
        return code_gen_dict

    def defines(self, var):
        itype_bits = self.get_input_datatype().bitwidth()
        total_bits = itype_bits * np.prod(self.get_normal_input_shape())
        assert total_bits % 8 == 0, "DMA input not a multiple of 1 Byte"
        total_bytes = total_bits // 8
        code_gen_dict = {}
        code_gen_dict["$DEFINES$"] = [
            """#define NumBytes1 {}\n#define DataWidth1 {}\n""".format(
                total_bytes, self.intfWidth
            )
        ]
        return code_gen_dict

    def get_ap_int_max_w(self):
        "Return the maximum width of any ap_int used in this module."
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        width_lcm = (instream * outstream) // math.gcd(instream, outstream)
        return width_lcm

    def docompute(self):
        direction = self.direction
        mode = self.burstMode
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
        intfw = self.intfWidth
        strmw = self.streamWidth
        width_lcm = (strmw * intfw) // math.gcd(strmw, intfw)
        # we always need two streams: one of width_lcm, and one of intfw width
        # because we use WidthAdjustedInputStream,
        dtype_bits = self.get_input_datatype().bitwidth()
        total_bits = dtype_bits * np.prod(self.get_normal_input_shape())

        code_gen_dict = {}
        if direction == "in":
            # AXI MM -> IODMA -> (DWCs) -> out
            # DWCs depend on AXI MM and out interface width
            if strmw == intfw:
                # case 0: AXI MM width = out width, no DWCs needed
                code_gen_dict["$DOCOMPUTE$"] = [dma_inst_template % ("in0_V", "out0_V")]
            elif (strmw % intfw == 0) or (intfw % strmw == 0):
                # case 1: AXI MM width divisible by out width or vice versa
                # single DWC + single extra stream needed
                code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dma2dwc;" % intfw,
                    dma_inst_template % ("in0_V", "dma2dwc"),
                    dwc_inst_template
                    % (
                        intfw,
                        strmw,
                        total_bits // intfw,
                        "dma2dwc",
                        "out0_V",
                    ),
                ]
            else:
                # case 2: AXI MM width not divisible by out width or vice versa
                # need 2 DWCs (going through the least common multiple width)
                # and 2 streams
                code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dma2lcm;" % intfw,
                    "hls::stream<ap_uint<%d> > lcm2out;" % width_lcm,
                    dma_inst_template % ("in0_V", "dma2lcm"),
                    dwc_inst_template
                    % (intfw, width_lcm, total_bits // intfw, "dma2lcm", "lcm2out"),
                    dwc_inst_template
                    % (
                        width_lcm,
                        strmw,
                        total_bits // width_lcm,
                        "lcm2out",
                        "out0_V",
                    ),
                ]
        elif direction == "out":
            # in0 -> (DWCs) -> IODMA -> AXI MM
            # DWCs depend on AXI MM and out interface width
            if strmw == intfw:
                # case 0: in width = AXI MM width, no DWCs needed
                code_gen_dict["$DOCOMPUTE$"] = [dma_inst_template % ("in0_V", "out0_V")]
            elif (strmw % intfw == 0) or (intfw % strmw == 0):
                # case 1: AXI MM width divisible by in width or vice versa
                # single DWC + single extra stream needed
                code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > dwc2dma;" % intfw,
                    dwc_inst_template
                    % (
                        strmw,
                        intfw,
                        total_bits // strmw,
                        "in0_V",
                        "dwc2dma",
                    ),
                    dma_inst_template % ("dwc2dma", "out0_V"),
                ]
            else:
                # case 2: AXI MM width not divisible by out width or vice versa
                # need 2 DWCs (going through the least common multiple width)
                # and 2 streams
                code_gen_dict["$DOCOMPUTE$"] = [
                    "hls::stream<ap_uint<%d> > in2lcm;" % width_lcm,
                    "hls::stream<ap_uint<%d> > lcm2dma;" % intfw,
                    dwc_inst_template
                    % (
                        strmw,
                        width_lcm,
                        total_bits // strmw,
                        "in0_V",
                        "in2lcm",
                    ),
                    dwc_inst_template
                    % (width_lcm, intfw, total_bits // width_lcm, "in2lcm", "lcm2dma"),
                    dma_inst_template % ("lcm2dma", "out0_V"),
                ]
        else:
            raise Exception("Unknown IODMA direction: %s" % direction)
        
        return code_gen_dict

    def blackboxfunction(self):
        packed_ibits = self.get_instream_width()
        packed_hls_type_in = "ap_uint<%d>" % packed_ibits
        packed_obits = self.get_outstream_width()
        packed_hls_type_out = "ap_uint<%d>" % packed_obits
        direction = self.direction
        code_gen_dict = {}
        if direction == "in":
            code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(%s *in0_V, hls::stream<%s > &out0_V, unsigned int numReps)"
                % (
                    self.name,
                    packed_hls_type_in,
                    packed_hls_type_out,
                )
            ]
        elif direction == "out":
            code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(hls::stream<%s > &in0_V, %s *out0_V, unsigned int numReps)"
                % (
                    self.name,
                    packed_hls_type_in,
                    packed_hls_type_out,
                )
            ]
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

        return code_gen_dict

    def pragmas(self):
        code_gen_dict = {}
        code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE s_axilite port=numReps bundle=control"
        ]
        code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE s_axilite port=return bundle=control"
        )
        direction = self.direction
        intfname = self.intfName
        if direction == "in":
            if intfname == "":
                code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=in0_V"
                )
            else:
                code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=%s" % (intfname)
                )
            code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=in0_V bundle=control"
            )
            code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        elif direction == "out":
            code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in0_V")
            if intfname == "":
                code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=out0_V"
                )
            else:
                code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=%s" % (intfname)
                )
            code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=out0_V bundle=control"
            )
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")
        code_gen_dict["$PRAGMAS$"].append("#pragma HLS DATAFLOW")

        return code_gen_dict

    ######################### Projections #########################
    def projection(self, fpgapart: str) -> KernelProjection:
        return KernelProjection(
            cycles = self.get_exp_cycles(),
            LUT = None,
            DSP = None,
            BRAM_18K= None,
            URAM = None,
            BRAM_efficiency = None,
            URAM_efficiency = None

        )

    def get_exp_cycles(self) -> int:
        return 0

    ######################### Other methods #########################
    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.dataType]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_normal_input_shape(self, ind=0):
        vecs = list(self.numInputVectors)
        num_ch = self.NumChannels
        ishape = tuple(vecs + [num_ch])
        return ishape

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind=0):
        if self.direction == "in":
            raise ValueError("Folded input shape not defined for input IODMA")
        else:
            shape = list(self.get_normal_input_shape())
            itype_bits = self.get_input_datatype().bitwidth()
            intfw = self.streamWidth
            assert intfw % itype_bits == 0, "Input stream width must be a multiple of datatype bits"
            elems_per_word = intfw // itype_bits
            assert shape[-1] % elems_per_word == 0, "Fold depth must be integer"
            fold_depth = shape[-1] // elems_per_word
            shape[-1] = fold_depth
            shape.append(elems_per_word)
            return tuple(shape)

    def get_folded_output_shape(self, ind=0):
        if self.direction == "out":
            raise ValueError("Folded output shape not defined for output IODMA")
        else:
            shape = list(self.get_normal_output_shape())
            itype_bits = self.get_output_datatype().bitwidth()
            intfw = self.streamWidth
            assert intfw % itype_bits == 0, "Input stream width must be a multiple of datatype bits"
            elems_per_word = intfw // itype_bits
            assert shape[-1] % elems_per_word == 0, "Fold depth must be integer"
            fold_depth = shape[-1] // elems_per_word
            shape[-1] = fold_depth
            shape.append(elems_per_word)
            return tuple(shape)

    def get_instream_width(self, ind=0):
        if self.direction == "in":
            return self.intfWidth
        elif self.direction == "out":
            return self.streamWidth
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

    def get_outstream_width(self, ind=0):
        if self.direction == "out":
            return self.intfWidth
        elif self.direction == "in":
            return self.streamWidth
        else:
            raise ValueError("Invalid IODMA direction, please set to in or out")

    def get_number_output_values(self):
        oshape = self.get_normal_output_shape()
        itype_bits = self.get_input_datatype().bitwidth()
        stream_width = self.streamWidth
        nelems = np.prod(oshape)
        nbits = nelems * itype_bits
        assert nbits % stream_width == 0, "DMA: total transfer size must be word multiple"
        ovalues = nbits // stream_width
        return ovalues

    ######################### Simulation #########################
    def execute_rtlsim(self, context, graph, code_gen_dir, node, rtlsim_trace):
        pass
