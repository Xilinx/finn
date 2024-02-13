# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.matrixvectoractivation import MatrixVectorActivation
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from pyverilator.util.axi_utils import toggle_clk, reset_rtlsim

# ONNX i/o tensor shape assumptions for MatrixVectorActivation:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)


class MatrixVectorActivation_hls(MatrixVectorActivation, HLSBackend):
    """Corresponds to finn-hlslib MatrixVectorActivation_Batch function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(MatrixVectorActivation.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def lut_estimation(self):
        """Calculates resource estimations for LUTs based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018
        """
        # TODO add in/out FIFO contributions
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        MW = self.get_nodeattr("MW")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        # determine tdt with input and weight data types
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        # parameters from experiments in paper mentioned above
        c0 = 300
        c1 = 1.1
        c2 = 0
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (mmode == "decoupled" and mstyle == "distributed") or (
            mmode == "const" and self.calc_wmem() <= 128
        ):
            c2 = (P * Q * W) * math.ceil(self.calc_wmem() / 64)

        # multiplication
        res_type = self.get_nodeattr("resType")
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
        noact = self.get_nodeattr("noActivation")
        tmem_style = self.get_nodeattr("ram_style_thresholds")
        if (noact == 0) and (tmem_style == "distributed"):
            odt = self.get_output_datatype()
            B = odt.bitwidth()
            thr_luts = (2**B - 1) * acc_bits * math.ceil(self.calc_tmem() / 64)
            comp_luts = (2**B - 1) * acc_bits

        return int(
            c0 + c1 * (P * (mult_luts + addertree_luts + acc_luts + thr_luts + comp_luts)) + c2
        )

    def dsp_estimation(self):
        # multiplication
        P = self.get_nodeattr("PE")
        res_type = self.get_nodeattr("resType")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        if res_type == "dsp":
            mult_dsp = P * Q * np.ceil((W + A) / 48)  # TODO: more accurate modelling
        else:
            mult_dsp = 0
        return int(mult_dsp)

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype().get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype() == DataType["BINARY"]
        # out_is_binary = self.get_output_datatype() == DataType["BINARY"]
        wt_is_binary = self.get_weight_datatype() == DataType["BINARY"]
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise Exception("True binary (non-bipolar) inputs not yet supported")
        inp_is_bipolar = self.get_input_datatype() == DataType["BIPOLAR"]
        # out_is_bipolar = self.get_output_datatype() == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_weight_datatype() == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # fill in TSrcI and TWeightI
        # TODO check these with Giulio
        # TODO handle non-bipolar binary inputs
        if inp_is_bipolar and wt_is_bipolar:
            ret["TSrcI"] = "Recast<XnorMul>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and wt_is_bipolar:
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Recast<Binary>"
        elif inp_is_bipolar and (not wt_is_bipolar):
            ret["TSrcI"] = "Recast<Binary>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and (not wt_is_bipolar):
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Identity"

        # fill in TDstI
        ret["TDstI"] = "Slice<%s>" % out_hls_str

        return ret

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        sname = self.hls_sname()
        if mem_mode == "external":
            intf_names["s_axis"].append(("weights_" + sname, self.get_weightstream_width_padded()))
        if mem_mode == "decoupled":
            # only expose axilite interface if attribute is set
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            if runtime_writable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode not in ["const", "decoupled", "external"]:
            raise Exception(
                """Please set mem_mode to "const", "decoupled", or "external",
                currently no other parameter value is supported!"""
            )
        self.code_gen_dict["$GLOBALS$"] += ['#include "mvau.hpp"']
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var):
        # Only ipgen mode: Make sure that SIMD parameter satisfies minimum requirements.
        if var == "ipgen":
            SIMD = self.get_nodeattr("SIMD")
            MW = self.get_nodeattr("MW")
            condition = SIMD >= (MW / 1024)
            msg = (
                f"HLS synthesis of MatrixVectorActivation requires: "
                f"SIMD >= MW / 1024. This is not fulfilled with: SIMD={SIMD} "
                f"and MW={MW} for node: {self.onnx_node.name}."
            )
            assert condition, msg
        mem_mode = self.get_nodeattr("mem_mode")
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        numReps = np.prod(numInputVectors)
        self.code_gen_dict["$DEFINES$"] = [
            """#define MW1 {}\n #define MH1 {}\n
            #define SIMD1 {}\n #define PE1 {}\n #define WMEM1 {}\n
            #define TMEM1 {}\n #define numReps {}""".format(
                self.get_nodeattr("MW"),
                self.get_nodeattr("MH"),
                self.get_nodeattr("SIMD"),
                self.get_nodeattr("PE"),
                self.calc_wmem(),
                self.calc_tmem(),
                numReps,
            )
        ]
        if mem_mode == "decoupled" or mem_mode == "external":
            wdt = self.get_weight_datatype()
            self.code_gen_dict["$DEFINES$"].append("#define WP1 {}\n".format(wdt.bitwidth()))

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0_%s, false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                npy_in,
                self.hls_sname(),
            )
        )

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled" or mem_mode == "external":
            wdt = self.get_weight_datatype()
            elem_bits = wdt.bitwidth()
            packed_bits = self.get_weightstream_width()
            packed_hls_type = "ap_uint<%d>" % packed_bits
            elem_hls_type = wdt.get_hls_datatype_str()
            npy_type = "float"
            npy_in = "%s/weights.npy" % code_gen_dir

            self.code_gen_dict["$READNPYDATA$"].append(
                'npy2apintstream<%s, %s, %d, %s>("%s", weights_%s, false, numReps);'
                % (
                    packed_hls_type,
                    elem_hls_type,
                    elem_bits,
                    npy_type,
                    npy_in,
                    self.hls_sname(),
                )
            )

    def strm_decl(self):
        mem_mode = self.get_nodeattr("mem_mode")
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_{} ("in0_{}");'.format(
                self.get_instream_width(), self.hls_sname(), self.hls_sname()
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out_{} ("out_{}");'.format(
                self.get_outstream_width(), self.hls_sname(), self.hls_sname()
            )
        )

        if mem_mode == "decoupled" or mem_mode == "external":
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> weights_{} ("weights_{}");'.format(
                    self.get_weightstream_width(), self.hls_sname(), self.hls_sname()
                )
            )

    def docompute(self):
        mem_mode = self.get_nodeattr("mem_mode")
        map_to_hls_mult_style = {
            "auto": "ap_resource_dflt()",
            "lut": "ap_resource_lut()",
            "dsp": "ap_resource_dsp()",
        }
        tmpl_args = self.get_template_param_values()
        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            threshs = "PassThroughActivation<%s>()" % odtype_hls_str
        else:
            threshs = "threshs"
        if mem_mode == "const":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, 1, {}, {}, {}>
                (in0_{}, out_{}, weights, {}, numReps, {});""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    self.hls_sname(),
                    self.hls_sname(),
                    threshs,
                    map_to_hls_mult_style[self.get_nodeattr("resType")],
                )
            ]
        elif mem_mode == "decoupled" or mem_mode == "external":
            wdt = self.get_weight_datatype()
            if wdt == DataType["BIPOLAR"]:
                export_wdt = DataType["BINARY"]
            else:
                export_wdt = wdt
            wdtype_hls_str = export_wdt.get_hls_datatype_str()
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, {}, {}, {}, {} >
                (in0_{}, out_{}, weights_{}, {}, numReps, {});""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    wdtype_hls_str,
                    self.hls_sname(),
                    self.hls_sname(),
                    self.hls_sname(),
                    threshs,
                    map_to_hls_mult_style[self.get_nodeattr("resType")],
                )
            ]

        else:
            raise Exception(
                """Please set mem_mode to "const", "decoupled", or "external",
                currently no other parameter value is supported!"""
            )

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out_%s, %s, "%s", false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                self.hls_sname(),
                shape_cpp_str,
                npy_out,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0_{},
                    hls::stream<ap_uint<{}>> &out_{}
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(),
                    self.hls_sname(),
                    self.get_outstream_width(),
                    self.hls_sname(),
                )
            ]
        elif mem_mode == "decoupled" or mem_mode == "external":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(
                    hls::stream<ap_uint<{}>> &in0_{},
                    hls::stream<ap_uint<{}>> &weights_{},
                    hls::stream<ap_uint<{}>> &out_{}
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(),
                    self.hls_sname(),
                    self.get_weightstream_width(),
                    self.hls_sname(),
                    self.get_outstream_width(),
                    self.hls_sname(),
                )
            ]

        else:
            raise Exception(
                """Please set mem_mode to "const" or "decoupled", currently no other
                    parameter value is supported!"""
            )

    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")
        ram_style_thresholds = self.get_nodeattr("ram_style_thresholds")
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_" + self.hls_sname()
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if mem_mode == "const":
            self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
            # the weight tensor is ap_uint<simd*prec> [PE][WMEM]
            # partition for parallel access along the PE dimension (dim 1)
            self.code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=weights.m_weights " "complete dim=1")
            )
        elif mem_mode == "decoupled" or mem_mode == "external":
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE axis port=weights_" + self.hls_sname()
            )

        else:
            raise Exception(
                """Please set mem_mode to "const", "decoupled", or external,
                currently no other parameter value is supported!"""
            )

        # the threshold tensor is acc_type [PE][TMEM][N_THRES]
        # partition for parallel access along PE and N_THRES
        # dimensions (dims 1 and 3)
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=1")
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=3")
            )
            # add resource pragma for thresholds if set
            if ram_style_thresholds == "distributed":
                self.code_gen_dict["$PRAGMAS$"].append(
                    ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_LUTRAM")
                )
            elif ram_style_thresholds == "block":
                self.code_gen_dict["$PRAGMAS$"].append(
                    ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_BRAM")
                )
            elif ram_style_thresholds == "auto":
                # no pragma needed
                pass
            else:
                raise Exception("Unrecognized ram_style_thresholds value:" + ram_style_thresholds)

    def get_ap_int_max_w(self):
        # base class impl (max of inp/out stream widths)
        max_of_io = super().get_ap_int_max_w()
        # decoupled mode weight stream
        weightstream = self.get_weightstream_width()
        # single PE weight entry
        weight_bits = self.get_weight_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        single_pe_w = simd * weight_bits
        return max([weightstream, max_of_io, single_pe_w])

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.get_nodeattr("mem_mode")
        node = self.onnx_node

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

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
                if self.get_input_datatype() == DataType["BIPOLAR"]:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype()
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for MatrixVectorActivation")
            in_ind += 1

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            assert (
                context[node.output[0]].shape == self.get_normal_output_shape()
            ), "cppsim did not produce expected output shape"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
            reset_rtlsim(sim)
            toggle_clk(sim)
            if mem_mode == "external" or mem_mode == "decoupled":
                wnbits = self.get_weightstream_width()
                export_wdt = self.get_weight_datatype()
                # we have converted bipolar weights to binary for export,
                # so use it as such for weight generation
                if self.get_weight_datatype() == DataType["BIPOLAR"]:
                    export_wdt = DataType["BINARY"]
                wei = npy_to_rtlsim_input("{}/weights.npy".format(code_gen_dir), export_wdt, wnbits)
                num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
                io_dict = {
                    "inputs": {"in0": inp, "weights": wei * num_w_reps},
                    "outputs": {"out": []},
                }
                self.rtlsim_multi_io(sim, io_dict)
                output = io_dict["outputs"]["out"]
            else:
                output = self.rtlsim(sim, inp)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def instantiate_ip(self, cmd):
        # instantiate the HLS IP
        vlnv = self.get_nodeattr("ip_vlnv")
        node_name = self.onnx_node.name
        if self.get_nodeattr("mem_mode") == "decoupled":
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s"
                % (vlnv, node_name, node_name)
            )
        else:
            cmd.append("create_bd_cell -type ip -vlnv %s %s" % (vlnv, self.onnx_node.name))