from finn.kernels import Kernel, KernelProjection
from finn.util import templates
from dataclasses import dataclass, field
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from pkgutil import get_data
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions
import numpy as np
from typing import List
import math
import textwrap
from math import ceil, log2
import os

from finn.kernels.thresholding.hls import ThresholdingHLS
from finn.kernels.memstream.rtl import MemstreamRTL
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)

# ONNX i/o tensor shape assumptions for Thresholding:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the threshold tensor, shape (NumChannels, n_thres)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


@dataclass(frozen=True, init=False)
class ThresholdingSIP(Kernel):
    """Class that corresponds to finn-hls Thresholding_Batch function."""

    ######################### Kernel attributes #########################
    # whether weights (thresholds) will be
    # writable through an AXI-lite interface during runtime
    # 1 for enabled, 0 for disabled.
    runtime_writeable_weights:bool = False
    # parallelization; channels thresholded per cycle
    PE:int
    # number of channels (each may have different thresholds)
    NumChannels:int
    # number of steps in thresholding function. Used only in decoupled mode
    numSteps:int
    # FINN DataTypes for inputs, outputs
    inputDataType:str
    weightDataType:str
    outputDataType:str
    # number of input vectors, examples:
    # [1] is a single vector (like a FC layer with batch=1)
    # [4] is four vectors (like a FC layer with batch=4)
    # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
    numInputVectors:list[int] = field(default_factory=lambda: [1])
    # initialization value for the thresholding accumulator
    ActVal:int = 0
    # memory mode for the thresholds
    # internal_embedded -- embedded thresholds
    # internal_decoupled -- default, streaming thresholds with  streamer packaged inside IP
    mem_mode:str = "internal_decoupled"
    # string defining memory type
    ram_style:str = "distributed"
    # (mem_mode = internal_decoupled only) whether weights (thresholds) will be
    # writable through an AXI-lite interface during runtime
    # 1 for enabled, 0 for disabled.
    # see finn-rtllib/memstream/doc/README for more about the memory
    # address map used for writable weights
    # IMPORTANT: After using AXI lite to either read or write the weights,
    # always "flush" the accelerator by first passing a dummy input
    # vector through the accelerator. This will get rid of any old
    # weight data from the weight FIFOs.
    runtime_writeable_weights:bool = False
    thresholds: np.ndarray = None  # From: thresholds = model.get_initializer(self.onnx_node.input[1])

    def input_init_map(self, input_initializers: list[np.ndarray]) -> dict[str, np.ndarray]:
        init_map = {}
        init_map["thresholds"] = input_initializers[1]
        return init_map

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style:str = "sip"

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = ()

    ######################### Stitched kernel things #########################
    def _init_subkernels(self, **kwargs) -> Tuple["Kernel"]:
        subkernels_uninited = [ThresholdingHLS]
        mem_mode = kwargs.get("mem_mode")
        if mem_mode == "internal_decoupled" or mem_mode == None:
            subkernels_uninited.append(MemstreamRTL)

        subkernels_inited = []

        for sk in subkernels_uninited:
            kwargs["name"] = self.name + "_" + sk.__name__
            kwargs["sip_depth"] = self.calc_tmem()
            kwargs["sip_padded_width"] = self.get_instream_width_padded(1)
            subkernels_inited.append(sk(**kwargs))

        return tuple(subkernels_inited)

    ######################### Code Generation #########################
    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:

        out = set({
            (self.toplevel, Path(f"{self.name}.v"))
        })

        if self.mem_mode == "internal_decoupled":
            out.add((self.generate_params, Path(f"{self.name}_MemstreamRTL/memblock.dat")))

        return frozenset(out)

    def toplevel(self, ctx):

        top_inft_names = self.get_verilog_top_module_intf_names()

        code_gen_dict = {}
        code_gen_dict["$TOP_MODULE_NAME$"] = self.name
        code_gen_dict["$WIDTH$"] = self.get_instream_width_padded(1)
        code_gen_dict["$DEPTH$"] = self.calc_tmem()

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

{self.name + "_ThresholdingHLS"} impl
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
        template_path = "thresholding/sip/hdl/thresholding_sip_template.v"
        template = get_data('finn.kernels', template_path).decode('utf-8')
        for key in code_gen_dict:
            template = template.replace(key, str(code_gen_dict[key]))
        with open(node_dir / Path(f'{self.name}.v'), 'w') as f:
            f.write(template)

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for unsigned inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """

        mh = self.NumChannels
        pe = self.PE
        tmem = mh // pe
        assert mh % pe == 0, "Requirement NumChannels divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.numSteps, "Mismatch in threshold steps"
        if not self.get_input_datatype(0).signed():
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
        # ensure all thresholds are integer
        assert np.equal(np.mod(orig_thres_matrix, 1), 0).all(), "Need int threshold tensor"
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        assert ret.shape[0] == mh, "Channels of threshold matrix are not as expected (mh)"
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        assert (
            ret.shape[0] == pe
        ), """First dimension after distribution of the
        rows between PEs is not as expected (pe)"""
        assert (
            ret.shape[1] == tmem
        ), """Second dimension after distribution of the
        rows between PEs is not as expected (tmem)"""
        assert (
            ret.shape[2] == n_thres_steps
        ), """Third dimension after distribution of the
        rows between PEs is not as expected (n_thres_steps)"""
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights (thresholds) in appropriate
        format for this layer. This file can be used for either synthesis or
        run-time reconfig of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated

        """
        threshold_tensor = self.get_hw_compatible_threshold_tensor(weights)
        tdt = self.get_input_datatype(1)
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        if weight_file_mode == "hls_header":
            # save thresholds in thresh.h
            thresholds_hls_code = numpy_to_hls_code(
                threshold_tensor, tdt, "thresholds", False, True
            )
            # write thresholds into thresh.h
            f_thresh = open(weight_file_name, "w")
            tdt_hls = tdt.get_hls_datatype_str()
            # use binary to export bipolar activations
            export_odt = self.get_output_datatype()
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                export_odt = DataType["BINARY"]
            odt_hls = export_odt.get_hls_datatype_str()
            f_thresh.write(
                "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs \
                = ".format(
                    self.calc_tmem(),
                    self.PE,
                    threshold_tensor.shape[-1],
                    tdt_hls,
                    odt_hls,
                    self.ActVal,
                    "comp::less_equal<%s, %s>" % (tdt_hls, tdt_hls),
                )
            )
            f_thresh.write(thresholds_hls_code)
            f_thresh.close()
        elif "decoupled" in weight_file_mode:
            # streaming thresholds need to be organized differently
            # (1, pe, tmem, n_thres_steps) -> (1, tmem, pe, n_thres_steps)
            decoupled_thres = np.transpose(threshold_tensor, (0, 2, 1, 3))
            # TODO add flips/reversals as needed here
            # (1, tmem, pe, n_thres_steps) -(1, tmem, pe * n_thres_steps)
            pe = self.PE
            n_thres_steps = self.numSteps
            decoupled_thres_pe_flipped = np.flip(decoupled_thres, axis=-2)
            decoupled_thres = decoupled_thres.reshape(1, -1, pe * n_thres_steps)
            decoupled_thres = decoupled_thres.copy()
            decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.reshape(
                1, -1, pe * n_thres_steps
            )
            decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.copy()

            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, decoupled_thres)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_instream_width(1)
                words_per_memwidth = 2 ** ceil(log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        # split into groups of 8 hex digits (= 32 bits)
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise Exception("Decoupled weight export not yet implemented")
        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, node_ctx):
        code_gen_dir = node_ctx.directory
        thresholds = self.thresholds
        mem_mode = self.mem_mode
        if mem_mode == "internal_embedded":
            # save thresholds in thresh.h
            weight_filename = "{}/thresh.h".format(code_gen_dir)
            self.make_weight_file(thresholds, "hls_header", weight_filename)
        elif mem_mode == "internal_decoupled":
            # save internal_decoupled weights for cppsim
            weight_filename_sim = "{}/thresholds.npy".format(code_gen_dir)
            self.make_weight_file(thresholds, "decoupled_npy", weight_filename_sim)
            memstream_ipgen_path = Path(f"{code_gen_dir}/{self.name}_MemstreamRTL")
            memstream_ipgen_path.mkdir(exist_ok=True)
            # also save weights as Verilog .dat file
            weight_filename_rtl = "{}/{}_MemstreamRTL/memblock.dat".format(code_gen_dir, self.name)
            self.make_weight_file(thresholds, "decoupled_verilog_dat", weight_filename_rtl)
        else:
            raise Exception("Unrecognized mem_mode")

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.mem_mode
        if mem_mode == "internal_decoupled":
            # only expose axilite interface if attribute is set
            runtime_writable = self.runtime_writeable_weights == 1
            if runtime_writable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

    ######################### Projections #########################
    def projection(self, fpgapart: str) -> KernelProjection:
        return KernelProjection(
            cycles = self.get_exp_cycles(),
            LUT = self.lut_estimation(),
            DSP = None,
            BRAM_18K= self.bram_estimation(),
            URAM = None,
            BRAM_efficiency = None,
            URAM_efficiency = None

        )

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def lut_estimation(self):
        """Calculates LUT cost, taking memory resource type into account"""
        # TODO add in/out FIFO contributions
        style = self.ram_style
        P = self.PE
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()
        tmem = self.calc_tmem()
        # cost of comparators
        comparator_cost = A * P
        # cost of LUTRAM
        if style == "distributed" and tmem > 1:
            lutram_cost = P * A * int(ceil(tmem / 64))
        else:
            lutram_cost = 0
        # total cost
        return int(comparator_cost + lutram_cost)

    def bram_estimation(self):
        """Calculates BRAM cost if resource set to BRAM"""
        style = self.ram_style
        P = self.PE
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()
        tmem = self.calc_tmem()

        if style == "block" and tmem > 1:
            return int(ceil(A * P / 16)) * int(ceil(tmem / 1024))
        else:
            return 0

    ######################### Other methods #########################
    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            dt = DataType[self.inputDataType]
        elif ind == 1:
            dt = DataType[self.weightDataType]
        else:
            raise Exception("Index out of range")
        return dt

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.outputDataType]

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.PE
        elif ind == 1:
            # try to access mem_mode attribute, doesn't exist for RTL Thresholding
            try:
                mem_mode = self.mem_mode
            except AttributeError:
                mem_mode = 0
            if mem_mode == "internal_decoupled":
                pe = self.PE
                wp = self.get_input_datatype(1).bitwidth()
                n_thres_steps = self.numSteps
                width = pe * wp * n_thres_steps
            else:
                width = 0
        else:
            raise Exception("Index out of range")
        return width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.PE

    def get_folded_input_shape(self, ind=0):
        pe = self.PE
        fold = self.calc_tmem()
        vecs = list(self.numInputVectors)
        folded_input_shape = tuple(vecs + [fold, pe])
        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        ich = self.NumChannels
        vecs = list(self.numInputVectors)
        normal_input_shape = tuple(vecs + [ich])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        # same shape as input
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        num_channels = self.NumChannels
        pe = self.PE
        return num_channels // pe

    def get_op_and_param_counts(self):
        ret_dict = {}
        weight_bits = self.get_input_datatype(1).bitwidth()
        out_features = self.NumChannels
        num_steps = self.numSteps
        # thresholds are called weights in this layer
        thres_param_type = "param_threshold_%db" % (weight_bits)
        thres_count = out_features * num_steps
        ret_dict[thres_param_type] = thres_count
        return ret_dict

    ######################### Simulation #########################
    def execute_rtlsim(self, context, graph, code_gen_dir, node, rtlsim_trace):
        # create a npy file fore each input of the node (in_ind is input index)
        in_ind = 0
        for inputs in node.input:
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
            if in_ind == 0:
                assert (
                    str(context[inputs].dtype) == "float32"
                ), """Input datatype is
                not float32 as expected."""
                expected_inp_shape = self.get_folded_input_shape()
                reshaped_input = context[inputs].reshape(expected_inp_shape)
                if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype(0)
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for Thresholding_Batch")
            in_ind += 1

        sim = self.get_rtlsim(code_gen_dir, rtlsim_trace)
        nbits = self.get_instream_width(0)
        inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
        super().reset_rtlsim(sim)
        if self.mem_mode == "internal_decoupled":
            wnbits = self.get_instream_width(1)
            export_wdt = self.get_input_datatype(1)
            wei = npy_to_rtlsim_input(
                "{}/thresholds.npy".format(code_gen_dir), export_wdt, wnbits
            )
            num_w_reps = np.prod(self.numInputVectors)
            io_dict = {
                "inputs": {"in0": inp, "in1": wei * num_w_reps},
                "outputs": {"out0": []},
            }
        elif self.mem_mode == "internal_embedded":
            io_dict = {
                "inputs": {"in0": inp},
                "outputs": {"out0": []},
            }
        else:
            raise Exception("Unrecognized mem_mode")
        self.rtlsim_multi_io(sim, io_dict, node)
        super().close_rtlsim(sim)
        output = io_dict["outputs"]["out0"]
        odt = self.get_output_datatype()
        target_bits = odt.bitwidth()
        packed_bits = self.get_outstream_width()
        out_npy_path = "{}/output_0.npy".format(code_gen_dir)
        out_shape = self.get_folded_output_shape()
        rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)

        # load and reshape output
        output = np.load(out_npy_path)
        oshape = self.get_normal_output_shape()
        output = np.asarray([output], dtype=np.float32).reshape(*oshape)
        context[node.output[0]] = output

    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out0": []},
        }
        mem_mode = self.mem_mode
        if mem_mode in ["internal_decoupled", "external"]:
            n_weight_inps = self.calc_tmem()
            num_w_reps = np.prod(self.numInputVectors)
            io_dict["inputs"]["in1"] = [0 for i in range(num_w_reps * n_weight_inps)]
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)
