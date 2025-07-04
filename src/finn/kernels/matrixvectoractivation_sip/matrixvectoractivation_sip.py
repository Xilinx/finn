import math
import numpy as np
from dataclasses import dataclass, field

from finn.kernels import Kernel, KernelProjection
from finn.kernels.matrixvectoractivation_hls import MVAUHLS
from finn.kernels.memstream_rtl import MemstreamRTL
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from pkgutil import get_data
from qonnx.core.datatype import DataType


@dataclass(frozen=True, init=False)
class MVAUSIP(Kernel):
    """Corresponds to finn-hlslib MatrixVectorActivation_Batch function."""

    ######################### Kernel attributes #########################
    name: str
    PE: int
    SIMD: int
    MW: int
    MH: int
    resType: str = "auto"
    ActVal: int = 0
    # FINN DataTypes for inputs, weights, outputs
    inputDataType: str
    weightDataType: str
    outputDataType: str
    # FINN DataType for accumulator -- auto-computed and updated
    accDataType: str = "INT32"
    # use xnor-popcount for binary weights/inputs, thus treating them
    # as bipolar
    binaryXnorMode: int = 0
    # no-activation mode (produce accumulators)
    noActivation: int = 0
    # number of input vectors, examples:
    # [1] is a single vector (like a FC layer with batch=1)
    # [4] is four vectors (like a FC layer with batch=4)
    # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
    numInputVectors: list[int] = field(default_factory=lambda: [1])
    # memory mode for the FC weights
    # internal_embedded -- embedded weights, long compile/synth times
    # internal_decoupled -- default, streaming weights with streamer packaged inside IP
    # external -- streaming weights with external streamer
    mem_mode: str = "internal_decoupled"
    # FPGA resource type for memories in internal_decoupled mode
    # auto -- let Vivado decide
    # block -- use BRAM
    # distributed -- use LUTRAM
    # ultra -- use UltraRAM (URAM), must have runtime_writeable_weights=1
    # see also https://www.xilinx.com/support/answers/38070.html
    ram_style: str = "auto"
    # FPGA resource type for threshold memories (if noActivation is False)
    # auto -- let Vivado decide
    # block -- use BRAM
    # distributed -- use LUTRAM
    ram_style_thresholds: str = "auto"
    # (mem_mode = internal_decoupled only) whether weights will be
    # writeable through an AXI-lite interface during runtime
    # 1 for enabled, 0 for disabled.
    # see finn-rtllib/memstream/doc/README for more about the memory
    # address map used for writable weights
    # IMPORTANT: After using AXI lite to either read or write the weights,
    # always "flush" the accelerator by first passing a dummy input
    # vector through the accelerator. This will get rid of any old
    # weight data from the weight FIFOs.
    runtime_writeable_weights: int = 0
    pumpedMemory: int = 0
    pumpedCompute: int
    ip_vlnv: str
    weights: np.ndarray
    thresholds: np.ndarray = None

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = () 

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style:str = "sip"

    ######################### Code Generation #########################
    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return { 
            (self.toplevel, Path(f"{self.name}.v"))
        }

    ######################### Stitched kernel things #########################
    def _init_subkernels(self, **kwargs) -> Tuple["Kernel"]:
        subkernels_uninited = [MVAUHLS]
        mem_mode = kwargs.get("mem_mode")
        if mem_mode == "internal_decoupled" or mem_mode == None:
            subkernels_uninited.append(MemstreamRTL)

        subkernels_inited = []

        for sk in subkernels_uninited:
            kwargs["name"] = self.name + "_" + sk.__name__
            subkernels_inited.append(sk(**kwargs))

        return tuple(subkernels_inited)

    ######################### Projections #########################
    def projection(self)->KernelProjection:
        return KernelProjection(
            cycles = None,
            LUTs = self.lut_estimation(),
            DSPs = self.dsp_estimation(),
            BRAMs= None
        )

    def lut_estimation(self):
        """Calculates resource estimations for LUTs based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018
        """
        # TODO add in/out FIFO contributions
        P = self.PE
        Q = self.SIMD
        MW = self.MW
        wdt = self.get_input_datatype(ind=1)
        W = wdt.bitwidth()
        # determine tdt with input and weight data types
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        # parameters from experiments in paper mentioned above
        c0 = 300
        c1 = 1.1
        c2 = 0
        mmode = self.mem_mode
        mstyle = self.ram_style
        if (mmode == "internal_decoupled" and mstyle == "distributed") or (
            mmode == "internal_embedded" and self.calc_wmem() <= 128
        ):
            c2 = (P * Q * W) * math.ceil(self.calc_wmem() / 64)

        # multiplication
        res_type = self.resType
        if res_type == "dsp":
            mult_luts = 0
        else:
            mult_luts = Q * (2 * math.ceil((W + A) / 6) - 1) * (W + A)
        # adder tree
        addertree_luts = (W + A) * (2 * Q - 1)
        # accumulator
        acc_datatype = self.get_accumulator_datatype()
        # if accDataType is not set, then it will default to INT32, which would
        # be a large overestimate in most (if not all) cases. In this scenario,
        # we would use the minimum accumulator as determined by the data types
        # bound, derived in https://arxiv.org/abs/2301.13376
        alpha = math.log(MW, 2) + W + A - 1 - int(idt.signed())
        acc_bits = min(
            acc_datatype.bitwidth(),
            np.ceil(alpha + math.log(1 + pow(2, -alpha), 2) + 1),
        )
        acc_luts = acc_bits
        # thresholds and threshold comparators
        thr_luts = 0
        comp_luts = 0
        noact = self.noActivation
        tmem_style = self.ram_style_thresholds
        if (noact == 0) and (tmem_style == "distributed"):
            odt = self.get_output_datatype()
            B = odt.bitwidth()
            thr_luts = (2**B - 1) * acc_bits * math.ceil(self.calc_tmem() / 64)
            comp_luts = (2**B - 1) * acc_bits

        return int(
            c0 + c1 * (P * (mult_luts + addertree_luts + acc_luts + thr_luts + comp_luts)) + c2
        )

    def dsp_estimation(self, fpgapart):
        # multiplication
        P = self.PE
        res_type = self.resType
        Q = self.SIMD
        wdt = self.get_input_datatype(ind=1)
        W = wdt.bitwidth()
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        if res_type == "dsp":
            mult_dsp = P * Q * np.ceil((W + A) / 48)  # TODO: more accurate modelling
        else:
            mult_dsp = 0
        return int(mult_dsp)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        # when performing FIFO insertion on an FC layer with ext weights, the ind
        # parameter can be > 0 (referring to the weights) so handle that here
        if ind == 0:
            return DataType[self.inputDataType]
        elif ind == 1:
            return DataType[self.weightDataType]
        else:
            raise Exception("Undefined input ind for this layer type")

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        mw = self.MW
        mh = self.MH
        pe = self.PE
        simd = self.SIMD
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
        wmem = mw * mh // (pe * simd)
        return wmem

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        if self.noActivation == 1:
            return 0
        else:
            mh = self.MH
            pe = self.PE
            return mh // pe

    def get_accumulator_datatype(self):
        """Returns FINN DataType of accumulator"""
        return DataType[self.accDataType]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.outputDataType]

    ######################### Other methods #########################

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = DataType[self.inputDataType].bitwidth()
            width = i_bits * self.SIMD
        elif ind == 1:
            if (
                self.mem_mode == "internal_decoupled"
                or self.mem_mode == "external"
            ):
                pe = self.PE
                simd = self.SIMD
                wp = DataType[self.weightDataType].bitwidth()
                width = pe * simd * wp
            else:
                width = 0
        elif ind == 2:
            # check if integrated thresholding and return 0
            # because threshold values are always embedded
            # or raise expection if there shouldn't be
            # a third input to the node
            act = not self.noActivation
            if act:
                width = 0
            else:
                raise Exception("Index out of range")
        else:
            raise Exception("Index out of range")
        return width

    def get_outstream_width(self, ind=0):
        o_bits = DataType[self.outputDataType].bitwidth()
        out_width = o_bits * self.PE
        return out_width

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        try:
            pumped_compute = self.pumpedCompute
        except AttributeError:
            pumped_compute = 0

        if pumped_compute or self.pumpedMemory:
            intf_names["clk2x"] = ["ap_clk2x"]

        mem_mode = self.mem_mode
        if mem_mode == "external":
            intf_names["s_axis"].append(("in1_V", self.get_instream_width_padded(1)))
        else:
            intf_names["s_axis"].remove(("in1_V", self.get_instream_width_padded(1)))
        if mem_mode == "internal_decoupled":
            # only expose axilite interface if attribute is set
            runtime_writeable = self.runtime_writeable_weights
            if runtime_writeable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def toplevel(self, ctx):

        top_inft_names = self.get_verilog_top_module_intf_names()

        code_gen_dict = {}
        code_gen_dict["$TOP_MODULE_NAME$"] = self.name
        code_gen_dict["$WIDTH$"] = self.get_instream_width_padded(1)
        code_gen_dict["$DEPTH$"] = self.calc_wmem()

        code_gen_dict["$CLK_NAME$"] = top_inft_names["clk"][0]
        code_gen_dict["$RST_NAME$"] = top_inft_names["rst"][0]

        code_gen_dict["$CLK2X$"] = ""
        if top_inft_names.get("clk2x") != None:
            code_gen_dict["$CLK2X$"] = f"""
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 {top_inft_names.get("clk2x")} CLK" *)
input	{top_inft_names.get("clk2x")},
"""

        code_gen_dict["$M_AXIS$"] = ""
        for m_axi in top_inft_names["m_axis"]:
            code_gen_dict["$M_AXIS$"] += f"""
input	{m_axi[0]}_TREADY,
output	{m_axi[0]}_TVALID,
output	[{m_axi[1]}-1:0]  {m_axi[0]}_TDATA
"""

        code_gen_dict["$S_AXIS$"] = ""
        for s_axi in top_inft_names["s_axis"]:
            code_gen_dict["$S_AXIS$"] += f"""
output	{s_axi[0]}_TREADY,
input	{s_axi[0]}_TVALID,
input	[{s_axi[1]}-1:0]  {s_axi[0]}_TDATA,
"""

        code_gen_dict["$AXILITE$"] = ""
        for axilite in top_inft_names["axilite"]:
            code_gen_dict["$AXILITE$"] += f"""
input	       {axilite[0]}_AWVALID,
output	       {axilite[0]}_AWREADY,
input	[AXILITE_ADDR_WIDTH-1:0]  {axilite[0]}_AWADDR,

input	        {axilite[0]}_WVALID,
output	        {axilite[0]}_WREADY,
input	[31:0]  {axilite[0]}_WDATA,
input	[ 3:0]  {axilite[0]}_WSTRB,

output	       {axilite[0]}_BVALID,
input	       {axilite[0]}_BREADY,
output	[1:0]  {axilite[0]}_BRESP,

input	       {axilite[0]}_ARVALID,
output	       {axilite[0]}_ARREADY,
input	[AXILITE_ADDR_WIDTH-1:0]  {axilite[0]}_ARADDR,

output	        {axilite[0]}_RVALID,
input	        {axilite[0]}_RREADY,
output	[31:0]  {axilite[0]}_RDATA,
output	[ 1:0]  {axilite[0]}_RRESP,
"""
        top_ports = top_inft_names['axilite'] + top_inft_names['s_axis'] + top_inft_names['m_axis']
        top_port_names = [port[0] for port in top_ports]
        code_gen_dict["$TOP_PORT_NAMES$"] = ':'.join(top_port_names)

        din_name = top_inft_names["s_axis"][0][0]
        dout_name = top_inft_names["m_axis"][0][0]
        wstrm_name = 'in1_V'

        code_gen_dict["$MVAU$"] = f"""
wire WSTRM_TREADY;
wire WSTRM_TVALID;
wire [WIDTH-1:0] WSTRM_TDATA;

{self.name + "_MVAUHLS"} impl
(
 .{top_inft_names["clk"][0]}({top_inft_names["clk"][0]}),
 .{top_inft_names["rst"][0]}({top_inft_names["rst"][0]}),
 .{din_name}_TREADY({din_name}_TREADY),
 .{din_name}_TVALID({din_name}_TVALID),
 .{din_name}_TDATA({din_name}_TDATA),
 .{wstrm_name}_TREADY(WSTRM_TREADY),
 .{wstrm_name}_TVALID(WSTRM_TVALID),
 .{wstrm_name}_TDATA(WSTRM_TDATA),
 .{dout_name}_TREADY({dout_name}_TREADY),
 .{dout_name}_TVALID({dout_name}_TVALID),
 .{dout_name}_TDATA({dout_name}_TDATA)
);
"""

        if self.pumpedMemory:
            wstrm_clk2x_name = top_inft_names["clk2x"][0]
        else:
            wstrm_clk2x_name = top_inft_names["clk"][0]
        code_gen_dict["$MEMSTREAM$"] = f"""
{self.name + "_MemstreamRTL"} impl_wstrm
(
 .ap_clk({top_inft_names["clk"][0]}),
 .ap_clk2x({wstrm_clk2x_name}),
 .ap_rst_n({top_inft_names["rst"][0]}),
 .m_axis_0_tready(WSTRM_TREADY),
 .m_axis_0_tvalid(WSTRM_TVALID),
 .m_axis_0_tdata(WSTRM_TDATA),

 .s_axilite_ARADDR({{AXILITE_ADDR_WIDTH{{1'b0}}}}),
 .s_axilite_ARPROT(3'b0),
 .s_axilite_ARVALID(1'b0),
 .s_axilite_AWADDR({{AXILITE_ADDR_WIDTH{{1'b0}}}}),
 .s_axilite_AWPROT(3'b0),
 .s_axilite_AWVALID(1'b0),
 .s_axilite_BREADY(1'b0),
 .s_axilite_RREADY(1'b0),
 .s_axilite_WDATA(32'b0),
 .s_axilite_WSTRB(4'b1),
 .s_axilite_WVALID(1'b0)
);
""" # CONNECT AXILITE FOR RUNTIME WRITABLE CASE, CURRENTLY SET TO 0s ##############################################################################

        # Find and replace parameters in template, then return
        node_dir = ctx.directory
        template_path = "matrixvectoractivation_sip/hdl/mvau_sip_template.v"
        template = get_data('finn.kernels', template_path).decode('utf-8')
        for key in code_gen_dict:
            template = template.replace(key, str(code_gen_dict[key]))
        with open(node_dir / Path(f'{self.name}.v'), 'w') as f:
            f.write(template)

    def code_generation_ipi(self, node_ctx) -> list[str]:
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""

        sourcefiles = [
            f"{self.name}.v",
        ]

        cmd = []
        for f in sourcefiles:
            cmd += [f"add_files -norecurse {'../'+str((node_ctx.directory / Path(f)).relative_to(node_ctx.top_ctx.directory))}"]
        cmd += [f"create_bd_cell -type module -reference {self.name} {self.name}"]

        for subkernel in self.subkernels:
            cmd += subkernel.code_generation_ipi(node_ctx.get_subcontext(Path(subkernel.name)))

        return cmd
