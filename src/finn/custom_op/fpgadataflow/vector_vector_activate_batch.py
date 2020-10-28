import os
import numpy as np

from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import interleave_matrix_outer_dim_from_partitions
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    rtlsim_output_to_npy,
)


class Vector_Vector_Activate_Batch(HLSCustomOp):
    """Class that corresponds to finn-hlslib Vector_Vector_Activate_Batch function"""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "PE": ("i", True, 0),
            "Dim": ("i", True, 0),
            "Channels": ("i", True, 0),
            "Kernel": ("i", True, 0),
            "resType": ("s", True, ""),
            "ActVal": ("i", False, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # no-activation mode (produce accumulators)
            "noActivation": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        ch = self.get_nodeattr("Channels")
        k = self.get_nodeattr("Kernel")
        pe = self.get_nodeattr("PE")
        wmem = k * k * ch // pe
        return wmem

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        if self.get_nodeattr("noActivation") == 1:
            return 0
        else:
            ch = self.get_nodeattr("Channels")
            pe = self.get_nodeattr("PE")
            return ch // pe

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # check input datatype against property
        idt_name = self.get_input_datatype().name
        exp_idt_name = self.get_nodeattr("inputDataType")
        assert exp_idt_name == idt_name, "Bad input DataType for VVAU  node"
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        pass

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
        in_width = i_bits * self.get_nodeattr("PE")
        return in_width

    def get_outstream_width(self):
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("PE")
        return out_width

    def get_folded_input_shape(self):
        k = self.get_nodeattr("Kernel")
        sf = k * k
        dim = self.get_nodeattr("Dim")
        ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        nf = ch // pe
        folded_input_shape = tuple([1, dim, dim, sf * nf, pe])
        return folded_input_shape

    def get_folded_output_shape(self):
        ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        nf = ch // pe
        dim = self.get_nodeattr("Dim")
        folded_output_shape = tuple([1, dim, dim, nf, pe])
        return folded_output_shape

    def get_normal_input_shape(self):
        dim = self.get_nodeattr("Dim")
        ch = self.get_nodeattr("Channels")
        k = self.get_nodeattr("Kernel")
        normal_input_shape = tuple([1, dim, dim, k * k * ch])
        return normal_input_shape

    def get_normal_output_shape(self):
        ch = self.get_nodeattr("Channels")
        dim = self.get_nodeattr("Dim")
        normal_output_shape = tuple([1, dim, dim, ch])
        return normal_output_shape

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def get_exp_cycles(self):
        pe = self.get_nodeattr("PE")
        ch = self.get_nodeattr("Channels")
        dim = self.get_nodeattr("Dim")
        k = self.get_nodeattr("Kernel")
        # currently FINN supports for vvau a batch size of 1
        batch_size = 1
        # since mmv != 1 is not supported yet, we set mmv for now to 1
        mmv = 1
        exp_cycles = ((ch * k * k) / pe) * batch_size * (dim * dim) / mmv
        return int(exp_cycles)

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype().get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_bipolar = self.get_input_datatype() == DataType.BIPOLAR
        wt_is_bipolar = self.get_weight_datatype() == DataType.BIPOLAR
        # fill in TSrcI and TWeightI
        # TODO handle bipolar inputs
        if inp_is_bipolar or wt_is_bipolar:
            raise Exception("VVAU node doesn't support bipolar values yet.")
        else:
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Identity"

        # fill in TDstI
        ret["TDstI"] = "Slice<%s>" % out_hls_str

        return ret

    def get_hls_compatible_weight_tensor(self, orig_weight_matrix):
        pe = self.get_nodeattr("PE")
        ch = self.get_nodeattr("Channels")
        k = self.get_nodeattr("Kernel")
        wmem = self.calc_wmem()
        assert orig_weight_matrix.shape == (
            ch,
            1,
            k,
            k,
        ), """Weights matrix doesn't
        have expected shape (channels, 1, kernel_size, kernel_size)"""
        ret = orig_weight_matrix
        ret = ret.reshape(ch, k * k)
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        ret = ret.reshape(1, pe, wmem, 1)
        return ret

    def get_hls_compatible_threshold_tensor(self, orig_thres_matrix):
        ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        tmem = self.calc_tmem()
        assert ch % pe == 0, "Requirement Channels divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        ret = orig_thres_matrix
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
        # weights
        weights = model.get_initializer(self.onnx_node.input[1])
        # convert weights into hlslib-compatible format
        weight_tensor = self.get_hls_compatible_weight_tensor(weights)
        wdt = self.get_weight_datatype()
        code_gen_dir = path

        """Saves weights into params.h"""
        weight_hls_code = numpy_to_hls_code(weight_tensor, wdt, "weights", True, True)
        # write weights into params.h
        f_weights = open("{}/params.h".format(code_gen_dir), "w")

        if wdt.bitwidth() != 1:
            f_weights.write(
                "const FixedPointWeights<1,{},{},{}> weights = ".format(
                    wdt.get_hls_datatype_str(),
                    self.get_nodeattr("PE"),
                    self.calc_wmem(),
                )
            )
        else:
            f_weights.write(
                "const BinaryWeights<1,{},{}> weights = ".format(
                    self.get_nodeattr("PE"), self.calc_wmem()
                )
            )
        f_weights.write(weight_hls_code)
        f_weights.close()

        # save thresholds in thresh.h
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if thresholds is not None:
                threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
                tdt = DataType.INT32
                assert np.vectorize(tdt.allowed)(
                    threshold_tensor
                ).all(), "Thresholds are not int"
                thresholds_hls_code = numpy_to_hls_code(
                    threshold_tensor, tdt, "thresholds", False, True
                )
                # write thresholds into thresh.h
                f_thresh = open("{}/thresh.h".format(code_gen_dir), "w")
                tdt_hls = tdt.get_hls_datatype_str()
                odt = self.get_output_datatype()
                odt_hls = odt.get_hls_datatype_str()
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
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception(
                    "Unexpected input found for Vector_Vector_Activate_Unit"
                )
            in_ind += 1

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == self.get_folded_output_shape()
            ), """Output shape is not as expected"""
            # reshape output to have expected shape
            oshape = self.get_normal_output_shape()
            context[node.output[0]] = context[node.output[0]].reshape(*oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            idt = self.get_input_datatype()
            inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), idt, nbits)
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            output = self.rtlsim(sim, inp)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

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

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']
        if self.calc_tmem() != 0:
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var):
        dim = self.get_nodeattr("Dim")
        numReps = 1 * dim * dim
        self.code_gen_dict["$DEFINES$"] = [
            """#define Channels1 {}\n #define Kernel1 {}\n
            #define SIMD1 1\n #define PE1 {}\n #define numReps {}""".format(
                self.get_nodeattr("Channels"),
                self.get_nodeattr("Kernel"),
                self.get_nodeattr("PE"),
                numReps,
            )
        ]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
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

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def docompute(self):
        tmpl_args = self.get_template_param_values()
        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            threshs = "PassThroughActivation<%s>()" % odtype_hls_str
        else:
            threshs = "threshs"
        node = self.onnx_node
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<Channels1, Kernel1, SIMD1, PE1, 1, {}, {}, {}>
            (in0, out, weights, {}, numReps, {});""".format(
                node.op_type,
                tmpl_args["TSrcI"],
                tmpl_args["TDstI"],
                tmpl_args["TWeightI"],
                threshs,
                self.get_nodeattr("resType"),
            )
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
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
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}>> &in0,
            hls::stream<ap_uint<{}>> &out
            )""".format(
                self.onnx_node.name,
                self.get_instream_width(),
                self.get_outstream_width(),
            )
        ]

    def pragmas(self):
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

        self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
        # the weight tensor is ap_uint<ch*prec> [PE][WMEM]
        # partition for parallel access along the PE dimension (dim 1)
        self.code_gen_dict["$PRAGMAS$"].append(
            ("#pragma HLS ARRAY_PARTITION variable=weights.m_weights " "complete dim=1")
        )
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$PRAGMAS$"].append(
                (
                    "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds "
                    "complete dim=1"
                )
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                (
                    "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds "
                    "complete dim=3"
                )
            )
