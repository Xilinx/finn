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
import os

import numpy as np
from pyverilator import PyVerilator

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp
from finn.util.basic import interleave_matrix_outer_dim_from_partitions
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    rtlsim_output_to_npy,
)

# ONNX i/o tensor shape assumptions for StreamingFCLayer:
# input 0 is the input vector, shape (1, i_size) = (1, MW)
# input 1 is the weight vector, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the threshold vector, shape (o_size, n_thres)
# output 0 is the output vector, shape (1, o_size) = (1, MH)


class StreamingFCLayer_Batch(HLSCustomOp):
    """Class that corresponds to finn-hls StreamingFCLayer_Batch function."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "PE": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "MW": ("i", True, 0),
            "MH": ("i", True, 0),
            "resType": ("s", True, ""),
            "ActVal": ("i", False, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # use xnor-popcount for binary weights/inputs, thus treating them
            # as bipolar
            "binaryXnorMode": ("i", False, 0),
            # no-activation mode (produce accumulators)
            "noActivation": ("i", False, 0),
            # input and output FIFO depths
            "inFIFODepth": ("i", False, 0),
            "outFIFODepth": ("i", False, 0),
            "mem_mode": ("s", True, "const"),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
        wmem = mw * mh // (pe * simd)
        return wmem

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        if self.get_nodeattr("noActivation") == 1:
            return 0
        else:
            mh = self.get_nodeattr("MH")
            pe = self.get_nodeattr("PE")
            return mh // pe

    def make_shape_compatible_op(self):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        info_messages = []

        # verify number of attributes
        num_of_attr = 14
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    self.onnx_node.op_type, num_of_attr
                )
            )

        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')

        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        # TODO collect automatically from get_nodeattr_types
        try:
            self.get_nodeattr("code_gen_dir_npysim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("resType")
            self.get_nodeattr("MW")
            self.get_nodeattr("MH")
            self.get_nodeattr("SIMD")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("weightDataType")
            self.get_nodeattr("outputDataType")
            self.get_nodeattr("ActVal")
            self.get_nodeattr("binaryXnorMode")
            self.get_nodeattr("noActivation")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                StreamingFCLayer_Batch needs the following attributes:
                code_gen_dir_npysim, executable_path, resType, MW, MH, SIMD, PE,
                inputDataType, weightDataType, outputDataType, ActVal,
                binaryXnorMode, noActivation"""
            )

        # verify the number of inputs depending on noActivation value
        # check noActivation value to determine the number of inputs
        no_act = self.get_nodeattr("noActivation")

        if no_act == 1:
            if len(self.onnx_node.input) == 2:
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append(
                    """StreamingFCLayer_Batch needs in no
                            activation mode 2 inputs (data input and weights)"""
                )
        elif no_act == 0:
            if len(self.onnx_node.input) == 3:
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append(
                    """StreamingFCLayer_Batch needs 3 inputs
                            (data input and weights and threshold values)"""
                )
        else:
            info_messages.append(
                """noActivation attribute contains {} should
                be 0 or 1""".format(
                    no_act
                )
            )

        return info_messages

    def bram_estimation(self):
        """Calculates resource estimation for BRAM based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018
        """
        # TODO add in/out FIFO contributions
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_instream_width()
        D_out = self.get_outstream_width()
        omega = (D_in * D_out) / (Q * P)
        return P * (math.ceil(omega / 512)) * (math.ceil((Q * W) / 36))

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
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        # determine tdt with input and weight data types
        idt = self.get_input_datatype()
        A = idt.bitwidth()
        # parameters from experiments in paper mentioned above
        c0 = 300
        c1 = 1.1

        return c0 + c1 * (P * Q) * (W * A)

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_weight_datatype(self):
        """Returns FINN DataType of weights."""
        return DataType[self.get_nodeattr("weightDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self):
        i_bits = self.get_input_datatype().bitwidth()
        return i_bits * self.get_nodeattr("SIMD")

    def get_outstream_width(self):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def get_weightstream_width(self):
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wp = self.get_weight_datatype().bitwidth()
        return pe * simd * wp

    def get_folded_input_shape(self):
        mw = self.get_nodeattr("MW")
        simd = self.get_nodeattr("SIMD")
        sf = mw // simd
        return (1, sf, simd)

    def get_folded_output_shape(self):
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        nf = mh // pe
        return (1, nf, pe)

    def get_number_output_values(self):
        nf = self.get_folded_output_shape()[1]
        return nf

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype().get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        wt_hls_str = self.get_weight_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype() == DataType.BINARY
        out_is_binary = self.get_output_datatype() == DataType.BINARY
        wt_is_binary = self.get_weight_datatype() == DataType.BINARY
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise Exception("True binary (non-bipolar) inputs not yet supported")
        inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
        out_is_bipolar = self.get_output_datatype() == DataType.BIPOLAR
        wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # fill in TSrcI and TWeightI
        # TODO check these with Giulio
        # TODO handle non-bipolar binary inputs
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
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

        elif mem_mode == "decoupled":
            if inp_is_bipolar and wt_is_bipolar:
                ret["TSrcI"] = "Recast<XnorMul>"
                ret["TWeightI"] = "Identity"
            elif (not inp_is_bipolar) and wt_is_bipolar:
                ret["TSrcI"] = "Slice<%s>" % inp_hls_str
                ret["TWeightI"] = "Recast<Binary>"
            elif inp_is_bipolar and (not wt_is_bipolar):
                ret["TSrcI"] = "Recast<Binary>"
                ret["TWeightI"] = "Slice<%s>" % wt_hls_str
            elif (not inp_is_bipolar) and (not wt_is_bipolar):
                ret["TSrcI"] = "Slice<%s>" % inp_hls_str
                ret["TWeightI"] = "Slice<%s>" % wt_hls_str

        # fill in TDstI
        if out_is_bipolar or out_is_binary:
            ret["TDstI"] = "Identity"
        else:
            ret["TDstI"] = "Slice<%s>" % out_hls_str
        return ret

    def get_hls_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0 and MW % SIMD == 0
        * for bipolar {-1,+1} weights, convert to binary {0, 1}
        * interleave rows between PEs
        * reshape into (1, PE, WMEM, SIMD) and return
        """
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wmem = self.calc_wmem()
        assert orig_weight_matrix.shape == (
            mw,
            mh,
        ), """Weights matrix doesn't
        have expected shape (mw, mh)"""
        assert mw % simd == 0, "Requirement MH divisable by SIMD is violated."
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        # start by transposing the original weight matrix, since ONNX and
        # finn-hlslib use different assumptions
        # ONNX uses (in_features, out_features) and matmul(x, W)
        # finn-hlslib uses (out_features, in_features) and matmul(W, x)
        ret = orig_weight_matrix.T
        if self.get_weight_datatype() == DataType.BIPOLAR:
            # convert bipolar to binary
            ret = (ret + 1) / 2
        # interleave rows between PEs and reshape
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        # create SIMD as innermost dimension and add a dummy outer dim
        ret = ret.reshape(1, pe, wmem, simd)
        # reverse the SIMD dimension
        ret = np.flip(ret, axis=-1)
        return ret

    def get_hls_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for bipolar weights&inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
        wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = self.get_input_datatype() == DataType.BINARY
        wt_is_binary = self.get_weight_datatype() == DataType.BINARY
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        if inp_is_bipolar and wt_is_bipolar:
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
            # ensure all thresholds are integer
            assert (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all()
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        assert (
            ret.shape[0] == mh
        ), "Channels of threshold matrix are not as expected (mh)"
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

    def generate_params(self, model, path):
        mem_mode = self.get_nodeattr("mem_mode")
        # weights
        weights = model.get_initializer(self.onnx_node.input[1])
        # convert weights into hlslib-compatible format
        weight_tensor = self.get_hls_compatible_weight_tensor(weights)
        export_wdt = self.get_weight_datatype()
        code_gen_dir = path

        if mem_mode == "const":
            """Saves weights into params.h"""
            # we have converted bipolar weights to binary for export,
            # so use it as such for weight generation
            if self.get_weight_datatype() == DataType.BIPOLAR:
                export_wdt = DataType.BINARY
            weight_hls_code = numpy_to_hls_code(
                weight_tensor, export_wdt, "weights", True, True
            )
            # write weights into params.h
            f_weights = open("{}/params.h".format(code_gen_dir), "w")

            if export_wdt.bitwidth() != 1:
                f_weights.write(
                    "static FixedPointWeights<{},{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        export_wdt.get_hls_datatype_str(),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            else:
                f_weights.write(
                    "static BinaryWeights<{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"), self.get_nodeattr("PE"), self.calc_wmem()
                    )
                )
            f_weights.write(weight_hls_code)
            f_weights.close()

        elif mem_mode == "decoupled":
            """Saves weights into .npy file"""
            # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
            weight_tensor = np.transpose(weight_tensor, (0, 2, 1, 3))
            # flip PE dimension
            weight_tensor = np.flip(weight_tensor, axis=-2)
            weight_tensor = np.flip(weight_tensor, axis=-1)
            # reshape weight tensor to desired shape
            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")
            weight_tensor = weight_tensor.reshape(1, -1, pe*simd)
            np.save(
                    os.path.join(code_gen_dir, "weights.npy"),
                    weight_tensor,
                )
        else:
            raise Exception("""Please set mem_mode to "const"i or "decoupled", currently no other
                    parameter value is supported!""")


        # save thresholds in thresh.h
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if thresholds is not None:
                threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
                tdt = DataType.INT32
                # use UINT32 threshold export for bipolar times bipolar
                inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
                wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
                # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
                inp_is_binary = self.get_input_datatype() == DataType.BINARY
                wt_is_binary = self.get_weight_datatype() == DataType.BINARY
                bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
                inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
                wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
                if inp_is_bipolar and wt_is_bipolar:
                    tdt = DataType.UINT32
                thresholds_hls_code = numpy_to_hls_code(
                    threshold_tensor, tdt, "thresholds", False, True
                )
                # write thresholds into thresh.h
                f_thresh = open("{}/thresh.h".format(code_gen_dir), "w")
                tdt_hls = tdt.get_hls_datatype_str()
                # use binary to export bipolar activations
                export_odt = self.get_output_datatype()
                if self.get_output_datatype() == DataType.BIPOLAR:
                    export_odt = DataType.BINARY
                odt_hls = export_odt.get_hls_datatype_str()
                f_thresh.write(
                    "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs \
                    = ".format(
                        self.calc_tmem(),
                        self.get_nodeattr("PE"),
                        threshold_tensor.shape[-1],
                        tdt_hls,
                        odt_hls,
                        self.get_nodeattr("ActVal"),
                        "std::less_equal<%s>" % tdt_hls,
                    )
                )
                f_thresh.write(thresholds_hls_code)
                f_thresh.close()

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        sf = mw // simd
        nf = mh // pe

        # TODO ensure codegen dir exists
        if mode == "npysim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("npysim", "rtlsim")""".format(
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
                expected_inp_shape = (1, sf, simd)
                reshaped_input = context[inputs].reshape(expected_inp_shape)
                if self.get_input_datatype() == DataType.BIPOLAR:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType.BINARY
                else:
                    export_idt = self.get_input_datatype()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for StreamingFCLayer")
            in_ind += 1

        if mode == "npysim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType.BIPOLAR:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            assert context[node.output[0]].shape == (
                1,
                nf,
                pe,
            ), """Output shape is not
            as expected (1, nf, pe)"""
            # reshape output to have expected shape
            context[node.output[0]] = context[node.output[0]].reshape(1, mh)
        elif mode == "rtlsim":
            prefixed_top_name = "%s_%s" % (node.name, node.name)
            # check if needed file exists
            verilog_file = "{}/project_{}/sol1/impl/verilog/{}.v".format(
                code_gen_dir, node.name, prefixed_top_name
            )
            if os.path.isfile(verilog_file):
                nbits = self.get_instream_width()
                inp = npy_to_rtlsim_input(
                    "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
                )
                sim = PyVerilator.build(
                    verilog_file,
                    verilog_path=[
                        "{}/project_{}/sol1/impl/verilog/".format(
                            code_gen_dir, node.name
                        )
                    ],
                )
                super().reset_rtlsim(sim)
                super().toggle_clk(sim)
                output = self.rtlsim(sim, inp)
                odt = self.get_output_datatype()
                target_bits = odt.bitwidth()
                packed_bits = self.get_outstream_width()
                out_npy_path = "{}/output.npy".format(code_gen_dir)
                rtlsim_output_to_npy(
                    output, out_npy_path, odt, (1, nf, pe), packed_bits, target_bits
                )

                # load and reshape output
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(1, mh)
                context[node.output[0]] = output

            else:
                raise Exception(
                    """Found no verilog files for this node,
                    did you run the codegen_ipgen transformation?"""
                )

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("npysim", "rtlsim")""".format(
                    mode
                )
            )

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']
        
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            self.code_gen_dict["$GLOBALS$"] += ['#include "params.h"']
        elif mem_mode == "decoupled":
            self.code_gen_dict["$GLOBALS$"] += ['#include "stream_custom.h"']
        else:
            raise Exception("""Please set mem_mode to "const" or "decoupled", currently no other
                    parameter value is supported!""")
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']


    def defines(self, var):
        mem_mode = self.get_nodeattr("mem_mode")
        numReps = 1
        self.code_gen_dict["$DEFINES$"] = [
            """#define MW1 {}\n #define MH1 {}\n #define SIMD1 {}\n
            #define PE1 {}\n #define WMEM1 {}\n #define TMEM1 {}\n
            #define numReps {}""".format(
                self.get_nodeattr("MW"),
                self.get_nodeattr("MH"),
                self.get_nodeattr("SIMD"),
                self.get_nodeattr("PE"),
                self.calc_wmem(),
                self.calc_tmem(),
                numReps,
            )
        ]
        if var == "ipgen":
            self.code_gen_dict["$DEFINES$"].append("#define PRAGMA_SUB(x) _Pragma (#x)")
            self.code_gen_dict["$DEFINES$"].append("#define DO_PRAGMA(x) PRAGMA_SUB(x)")

        if mem_mode == "decoupled":
            wdt = self.get_weight_datatype()
            self.code_gen_dict["$DEFINES$"].append(
                "#define WP1 {}\n".format(
                    wdt.bitwidth()
                )
            )

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        dtype = self.get_input_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0, false);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "decoupled":
            wdt = self.get_weight_datatype()
            elem_bits = wdt.bitwidth()
            packed_bits = self.get_weightstream_width()
            packed_hls_type = "ap_uint<%d>" % packed_bits
            elem_hls_type = wdt.get_hls_datatype_str()
            npy_type = "float"
            npy_in = "%s/weights.npy" % code_gen_dir

            self.code_gen_dict["$READNPYDATA$"].append(
                'npy2apintstream<%s, %s, %d, %s>("%s", weights, false);'
                % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
            )

    def strm_decl(self):
        mem_mode = self.get_nodeattr("mem_mode")
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

        if mem_mode == "decoupled":
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> weights ("weights");'.format(
                    self.get_weightstream_width()    
                )
            )
        

    def docompute(self):
        mem_mode = self.get_nodeattr("mem_mode")
        tmpl_args = self.get_template_param_values()
        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            threshs = "PassThroughActivation<%s>()" % odtype_hls_str
        else:
            threshs = "threshs"
        if mem_mode == "const":
            node = self.onnx_node
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<MW1, MH1, SIMD1, PE1, {}, {}, {}>
                (in0, out, weights, {}, numReps, {});""".format(
                    node.op_type,
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    threshs,
                    self.get_nodeattr("resType"),
                )
            ]
        elif mem_mode == "decoupled":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, WP1, {}, {}, {}>
                (in0, out, weights, {}, numReps, {});""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    threshs,
                    self.get_nodeattr("resType"),
                )
            ]            

        else:
            raise Exception("""Please set mem_mode to "const" or "decoupled", currently no other
                    parameter value is supported!""")



    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        dtype = self.get_output_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        nf = int(self.get_nodeattr("MH") / self.get_nodeattr("PE"))
        shape = (1, nf, self.get_nodeattr("PE"))
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s", false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
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
                """void {}(hls::stream<ap_uint<{}>> &in0,
                    hls::stream<ap_uint<{}>> &out
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(),
                    self.get_outstream_width(),
                )
            ]
        else:
            raise Exception("""Please set mem_mode to "const", currently no other
                    parameter value is supported!""")

    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
            in_fifo_depth = self.get_nodeattr("inFIFODepth")
            out_fifo_depth = self.get_nodeattr("outFIFODepth")
            # insert depth pragmas only if specified
            if in_fifo_depth != 0:
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS stream depth=%d variable=in0" % in_fifo_depth
                )
            if out_fifo_depth != 0:
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS stream depth=%d variable=out" % out_fifo_depth
                )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE ap_ctrl_none port=return"
            )
            # the weight tensor is ap_uint<simd*prec> [PE][WMEM]
            # partition for parallel access along the PE dimension (dim 1)
            self.code_gen_dict["$PRAGMAS$"].append(
                (
                    "DO_PRAGMA(HLS ARRAY_PARTITION "
                    "variable=weights.m_weights complete dim=1)"
                )
            )
            # the threshold tensor is acc_type [PE][TMEM][N_THRES]
            # partition for parallel access along PE and N_THRES dimensions (dims 1 and 3)
            if self.calc_tmem() != 0:
                # TODO find a better way of checking for no pregenerated thresholds
                self.code_gen_dict["$PRAGMAS$"].append(
                    (
                        "DO_PRAGMA(HLS ARRAY_PARTITION variable=threshs.m_thresholds "
                        "complete dim=1)"
                    )
                )
                self.code_gen_dict["$PRAGMAS$"].append(
                    (
                        "DO_PRAGMA(HLS ARRAY_PARTITION variable=threshs.m_thresholds "
                        "complete dim=3)"
                    )
                )
        else:
            raise Exception("""Please set mem_mode to "const", currently no other
                    parameter value is supported!""")
