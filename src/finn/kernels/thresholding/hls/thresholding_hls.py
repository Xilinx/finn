from finn.kernels import Kernel, KernelProjection
from finn.util import templates
from dataclasses import dataclass, field
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions
import numpy as np
from typing import List
import math
import textwrap
from math import ceil, log2
import os

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
class ThresholdingHLS(Kernel):
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
    impl_style:str = "hls"

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = ()

    ######################### Code Generation #########################
    sharedFiles: FrozenSet[Tuple[str,Path]] = frozenset({
        ("finn-hlslib", Path("."))
    })

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:

        out = set({
            (self.code_generation_ipgen, Path(f"{self.name}.cpp"))
        })

        out.add((self.generate_params, Path("thresh.h")))

        return frozenset(out)

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
            # also save weights as Verilog .dat file
            weight_filename_rtl = "{}/memblock.dat".format(code_gen_dir)
            self.make_weight_file(thresholds, "decoupled_verilog_dat", weight_filename_rtl)
        else:
            raise Exception("Unrecognized mem_mode")

    def code_generation_ipgen(self, node_ctx):
        """Generates c++ code and tcl script for ip generation."""

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
        code_gen_dir = node_ctx.directory
        f = open(os.path.join(code_gen_dir, "{}.cpp".format(self.name)), "w")
        f.write(template)
        f.close()

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        # fill in TSrcI
        ret["TSrcI"] = "Slice<%s>" % inp_hls_str
        # fill in TDstI
        ret["TDstI"] = "Slice<%s>" % out_hls_str

        return ret

    def global_includes(self):
        code_gen_dict = {}
        code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        if self.mem_mode == "internal_embedded":
            code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']
        return code_gen_dict

    # TODO check and add whatever missing
    def defines(self, var):
        numReps = 1
        numInputVectors = list(self.numInputVectors)
        total_spatial_size = int(np.prod(numInputVectors))

        code_gen_dict = {}
        code_gen_dict["$DEFINES$"] = [
            """#define NumChannels1 {}\n #define PE1 {}\n #define numReps {}\n
               #define ImgDim1 {}""".format(
                self.NumChannels,
                self.PE,
                numReps,
                total_spatial_size,
            )
        ]
        if self.mem_mode == "internal_decoupled":
            code_gen_dict["$DEFINES$"].append(
                "#define ActVal1 %d" % self.ActVal
            )
            code_gen_dict["$DEFINES$"].append(
                "#define ThresType1 %s" % self.get_input_datatype(1).get_hls_datatype_str()
            )
            code_gen_dict["$DEFINES$"].append(
                "#define NumSteps1 %d" % self.numSteps
            )
        return code_gen_dict

    def strm_decl(self):
        code_gen_dict = {}
        code_gen_dict["$STREAMDECLARATIONS$"] = []
        code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width(0))
        )
        code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width())
        )
        mem_mode = self.mem_mode
        if mem_mode == "internal_decoupled":
            code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> in1_V ("in1_V");'.format(self.get_instream_width(1))
            )
        return code_gen_dict

    def docompute(self):
        tmpl_args = self.get_template_param_values()
        mem_mode = self.mem_mode
        code_gen_dict = {}
        if mem_mode == "internal_embedded":
            code_gen_dict["$DOCOMPUTE$"] = [
                """Thresholding_Batch<ImgDim1, NumChannels1, PE1, {}, {}>
                (in0_V, out0_V, threshs, numReps);""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                )
            ]
        elif mem_mode == "internal_decoupled":
            # note that numReps is set to 1 in the invocation below, since
            # - for cppsim the repetition comes from the threshold stream reader+input
            # - for synth the unit runs continuously anyway (ap_ctrl_none)
            code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ImgDim1, NumChannels1, PE1, {}, {}, ActVal1, ThresType1, NumSteps1>
                (in0_V, out0_V, in1_V, numReps);""".format(
                    "Thresholding_Stream_Batch",
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                )
            ]
        else:
            raise Exception("Unrecognized mem_mode")
        return code_gen_dict

    def get_ap_int_max_w(self):
        ap_int_max_w = Kernel.get_ap_int_max_w(self)
        if self.mem_mode == "internal_decoupled":
            weightstream = self.get_instream_width(1)
            ap_int_max_w = max([weightstream, ap_int_max_w])
        return ap_int_max_w

    def blackboxfunction(self):
        code_gen_dict = {}
        if self.mem_mode == "internal_embedded":
            code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0_V,
                    hls::stream<ap_uint<{}>> &out0_V
                    )""".format(
                    self.name,
                    self.get_instream_width(0),
                    self.get_outstream_width(),
                )
            ]
        elif self.mem_mode == "internal_decoupled":
            code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0_V,
                    hls::stream<ap_uint<{}>> &in1_V,
                    hls::stream<ap_uint<{}>> &out0_V
                    )""".format(
                    self.name,
                    self.get_instream_width(0),
                    self.get_instream_width(1),
                    self.get_outstream_width(),
                )
            ]
        else:
            raise Exception("Unrecognized mem_mode")
        return code_gen_dict

    def pragmas(self):
        code_gen_dict = {}
        code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if self.mem_mode == "internal_embedded":
            # the threshold tensor is acc_type [PE][TMEM][N_THRES]
            # partition for parallel access along PE and N_THRES
            # dimensions (dims 1 and 3)
            code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=1")
            )
            code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=3")
            )
            # set resource type
            ram_style = self.ram_style
            pe = self.PE
            ich = self.NumChannels
            # if PE less than NumChannels, assign cores according to ram_style;
            # otherwise if PE == NumChannels, Vivado HLS will unroll to FFs
            if pe < ich:
                if ram_style == "distributed":
                    code_gen_dict["$PRAGMAS$"].append(
                        ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_LUTRAM")
                    )
                elif ram_style == "block":
                    code_gen_dict["$PRAGMAS$"].append(
                        ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_BRAM")
                    )
                else:
                    raise Exception(
                        """Invalid value for attribute ram_style! Is currently set to: {}
                    has to be set to one of ("block", "distributed")""".format(
                            ram_style
                        )
                    )
        elif self.mem_mode == "internal_decoupled":
            code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")
        return code_gen_dict

    def ipgen_extra_directives(self):
        "Return a list of extra tcl directives for HLS synthesis."

        return ["config_compile -pipeline_style frp"]

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
            URAM = self.uram_estimation(),
            BRAM_efficiency = None,
            URAM_efficiency = None

        )

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

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
        return comparator_cost + lutram_cost

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
