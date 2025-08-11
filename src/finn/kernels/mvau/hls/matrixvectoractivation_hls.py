import numpy as np
from dataclasses import dataclass, field
import os

from finn.kernels import Kernel
from finn.util import templates
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from qonnx.core.datatype import DataType

from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy, numpy_to_hls_code


@dataclass(frozen=True, init=False)
class MVAUHLS(Kernel):
    """MVAUHLS sub-kernel, used by MVAUSIP."""

    ######################### Kernel attributes #########################
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
    binaryXnorMode: bool = False
    # no-activation mode (produce accumulators)
    noActivation: bool = False
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
    runtime_writeable_weights: bool = False
    pumpedMemory: bool = False
    pumpedCompute: bool = False
    weights: np.ndarray     # From: weights = model.get_initializer(self.onnx_node.input[1])
    thresholds: np.ndarray = None  # From: if len(self.onnx_node.input) > 2: thresholds = model.get_initializer(self.onnx_node.input[2])
    # dynamic input
    dynamic_input: bool = False

    def input_init_map(self, input_initializers: list[np.ndarray]) -> dict[str, np.ndarray]:
        init_map = {}
        init_map["weights"] = input_initializers[1]
        if len(input_initializers) > 2:
            init_map["thresholds"] = input_initializers[2]
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

        if (not self.dynamic_input) and (self.mem_mode == "internal_embedded"):
            out.add((self.generate_params, Path("params.h")))

        if self.thresholds is not None:
            out.add((self.generate_thresh, Path("thresh.h")))

        return frozenset(out)

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

    def generate_params(self, ctx):
        code_gen_dir = ctx.directory
        # save hlslib-compatible weights in params.h
        weight_filename = "{}/params.h".format(code_gen_dir)
        self.make_weight_file(self.weights, weight_filename)

    def generate_thresh(self, ctx):
        code_gen_dir = ctx.directory
        # save thresholds in thresh.h
        threshold_tensor = self.get_hw_compatible_threshold_tensor(self.thresholds)
        # use UINT32 threshold export for bipolar times bipolar
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.binaryXnorMode == 1
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # get computed threshold datatype from attribute
        tdt = DataType[self.accDataType]

        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds in %s can't be expressed with type %s" % (
            self.name,
            str(tdt),
        )
        thresholds_hls_code = numpy_to_hls_code(
            threshold_tensor, tdt, "thresholds", False, True
        )
        # write thresholds into thresh.h
        f_thresh = open("{}/thresh.h".format(code_gen_dir), "w")
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

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        # out_is_binary = self.get_output_datatype() == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.binaryXnorMode == 1
        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise Exception("True binary (non-bipolar) inputs not yet supported")
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        # out_is_bipolar = self.get_output_datatype() == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
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

    def global_includes(self):
        code_gen_dict = {}
        code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']

        mem_mode = self.mem_mode
        if mem_mode not in ["internal_embedded", "internal_decoupled", "external"]:
            raise Exception(
                """Please set mem_mode to "internal_embedded", "internal_decoupled", or "external",
                currently no other parameter value is supported!"""
            )
        code_gen_dict["$GLOBALS$"] += ['#include "mvau.hpp"']
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']
        return code_gen_dict

    def defines(self, var):
        # Only ipgen mode: Make sure that SIMD parameter satisfies minimum requirements.
        if var == "ipgen":
            SIMD = self.SIMD
            MW = self.MW
            condition = SIMD >= (MW / 1024)
            msg = (
                f"HLS synthesis of MatrixVectorActivation requires: "
                f"SIMD >= MW / 1024. This is not fulfilled with: SIMD={SIMD} "
                f"and MW={MW} for node: {self.name}."
            )
            assert condition, msg
        mem_mode = self.mem_mode
        numInputVectors = list(self.numInputVectors)
        numReps = np.prod(numInputVectors)
        code_gen_dict = {}
        code_gen_dict["$DEFINES$"] = [
            """#define MW1 {}\n #define MH1 {}\n
            #define SIMD1 {}\n #define PE1 {}\n #define WMEM1 {}\n
            #define TMEM1 {}\n #define numReps {}""".format(
                self.MW,
                self.MH,
                self.SIMD,
                self.PE,
                self.calc_wmem(),
                self.calc_tmem(),
                numReps,
            )
        ]
        if mem_mode == "internal_decoupled" or mem_mode == "external":
            wdt = self.get_input_datatype(1)
            code_gen_dict["$DEFINES$"].append("#define WP1 {}\n".format(wdt.bitwidth()))
        return code_gen_dict

    def get_ap_int_max_w(self):
        # base class impl (max of inp/out stream widths)
        """Return the maximum width of any ap_int used in this module. Used to set the
        AP_INT_MAX_W definition for HLS."""
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        max_of_io = max([instream, outstream])
        assert max_of_io <= 8191, "AP_INT_MAX_W=%d is larger than allowed maximum of 8191" % max_of_io
        # internal_decoupled mode weight stream
        weightstream = self.get_instream_width(1)
        # single PE weight entry
        weight_bits = self.get_input_datatype(1).bitwidth()
        simd = self.SIMD
        single_pe_w = simd * weight_bits
        return max([weightstream, max_of_io, single_pe_w])

    def blackboxfunction(self):
        mem_mode = self.mem_mode
        code_gen_dict = {}
        if mem_mode == "internal_embedded":
            code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0_V,
                    hls::stream<ap_uint<{}>> &out0_V
                    )""".format(
                    self.name,
                    self.get_instream_width(0),
                    self.get_outstream_width(),
                )
            ]
        elif mem_mode == "internal_decoupled" or mem_mode == "external":
            code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(
                    hls::stream<ap_uint<{}>> &in0_V,
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
            raise Exception(
                """Please set mem_mode to "internal_embedded" or "internal_decoupled",
                    currently no other parameter value is supported!"""
            )
        return code_gen_dict

    def pragmas(self):
        mem_mode = self.mem_mode
        ram_style_thresholds = self.ram_style_thresholds
        code_gen_dict = {}
        code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if mem_mode == "internal_embedded":
            code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
            # the weight tensor is ap_uint<simd*prec> [PE][WMEM]
            # partition for parallel access along the PE dimension (dim 1)
            code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=weights.m_weights " "complete dim=1")
            )
        elif mem_mode == "internal_decoupled" or mem_mode == "external":
            code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")

        else:
            raise Exception(
                """Please set mem_mode to "internal_embedded", "internal_decoupled", or external,
                currently no other parameter value is supported!"""
            )

        # the threshold tensor is acc_type [PE][TMEM][N_THRES]
        # partition for parallel access along PE and N_THRES
        # dimensions (dims 1 and 3)
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=1")
            )
            code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=3")
            )
            # add resource pragma for thresholds if set
            if ram_style_thresholds == "distributed":
                code_gen_dict["$PRAGMAS$"].append(
                    ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_LUTRAM")
                )
            elif ram_style_thresholds == "block":
                code_gen_dict["$PRAGMAS$"].append(
                    ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_BRAM")
                )
            elif ram_style_thresholds == "auto":
                # no pragma needed
                pass
            else:
                raise Exception("Unrecognized ram_style_thresholds value:" + ram_style_thresholds)
        return code_gen_dict

    def docompute(self):
        mem_mode = self.mem_mode
        map_to_hls_mult_style = {
            "auto": "ap_resource_dflt()",
            "lut": "ap_resource_lut()",
            "dsp": "ap_resource_dsp()",
        }
        tmpl_args = self.get_template_param_values()
        code_gen_dict = {}
        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            threshs = "PassThroughActivation<%s>()" % odtype_hls_str
        else:
            threshs = "threshs"
        if mem_mode == "internal_embedded":
            code_gen_dict["$DOCOMPUTE$"] = [
                """Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, 1, {}, {}, {}>
                (in0_V, out0_V, weights, {}, numReps, {});""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    threshs,
                    map_to_hls_mult_style[self.resType],
                )
            ]
        elif mem_mode == "internal_decoupled" or mem_mode == "external":
            wdt = self.get_input_datatype(1)
            if wdt == DataType["BIPOLAR"]:
                export_wdt = DataType["BINARY"]
            else:
                export_wdt = wdt
            wdtype_hls_str = export_wdt.get_hls_datatype_str()
            code_gen_dict["$DOCOMPUTE$"] = [
                """Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, {}, {}, {}, {} >
                (in0_V, out0_V, in1_V, {}, numReps, {});""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    wdtype_hls_str,
                    threshs,
                    map_to_hls_mult_style[self.resType],
                )
            ]

        else:
            raise Exception(
                """Please set mem_mode to "internal_embedded", "internal_decoupled", or "external",
                currently no other parameter value is supported!"""
            )
        return code_gen_dict

    def make_weight_file(self, weights, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for synthesis.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_name : filename for the weight file to be generated

        """
        # convert weights into hlslib/rtllib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        export_wdt = self.get_input_datatype(1)
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            export_wdt = DataType["BINARY"]
        weight_hls_code = numpy_to_hls_code(weight_tensor, export_wdt, "weights", True, True)
        # write weights into C++ header file as dictated by finn-hlslib
        f_weights = open(weight_file_name, "w")
        if export_wdt.bitwidth() != 1:
            f_weights.write(
                "const FixedPointWeights<{},{},{},{}> weights = ".format(
                    self.SIMD,
                    export_wdt.get_hls_datatype_str(),
                    self.PE,
                    self.calc_wmem(),
                )
            )
        else:
            f_weights.write(
                "const BinaryWeights<{},{},{}> weights = ".format(
                    self.SIMD,
                    self.PE,
                    self.calc_wmem(),
                )
            )
        f_weights.write(weight_hls_code)
        f_weights.close()

    ######################### Other Methods #########################
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

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.outputDataType]

    def get_normal_input_shape(self, ind=0):
        mw = self.MW
        if ind == 0:
            vecs = list(self.numInputVectors)
            shape = tuple(vecs + [mw])
        elif ind == 1:
            mh = self.MH
            shape = tuple([mw, mh])
        else:
            raise Exception("Undefined input shape for requested input")
        return shape

    def get_normal_output_shape(self, ind=0):
        mh = self.MH
        vecs = list(self.numInputVectors)
        normal_output_shape = tuple(vecs + [mh])
        return normal_output_shape

    def get_folded_input_shape(self, ind=0):
        mw = self.MW
        mh = self.MH
        simd = self.SIMD
        pe = self.PE
        sf = mw // simd
        nf = mh // pe
        vecs = list(self.numInputVectors)

        if ind == 0:
            # calculate shape of input 0
            folded_input_shape = tuple(vecs + [sf, simd])
        elif ind == 1:
            if self.dynamic_input:
                # calculate shape of input 1 (weights dynamic)
                folded_input_shape = tuple(vecs[:2] + [mw] + [nf, pe])
            elif self.mem_mode == "external":
                # calculate shape of input 1 (weights static and external)
                folded_input_shape = tuple(vecs + [sf * nf, simd * pe])
            else:
                raise Exception("Undefined input shape for requested input")
        else:
            raise Exception("Undefined input shape for requested input")

        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        mh = self.MH
        pe = self.PE
        nf = mh // pe
        vecs = list(self.numInputVectors)
        folded_output_shape = tuple(vecs + [nf, pe])
        return folded_output_shape

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.SIMD
        elif ind == 1:
            if (
                self.mem_mode == "internal_decoupled"
                or self.mem_mode == "external"
            ):
                pe = self.PE
                simd = self.SIMD
                wp = self.get_input_datatype(1).bitwidth()
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

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for bipolar weights&inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.MH
        pe = self.PE
        tmem = mh // pe
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.binaryXnorMode == 1
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

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0 and MW % SIMD == 0
        * for bipolar {-1,+1} weights, convert to binary {0, 1}
        * interleave rows between PEs
        * reshape into (1, PE, WMEM, SIMD) and return
        """
        mw = self.MW
        mh = self.MH
        pe = self.PE
        simd = self.SIMD
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
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
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

    ######################### Simulation #########################
    def execute_rtlsim(self, context, graph, code_gen_dir, node, rtlsim_trace):
        dynamic_input = self.dynamic_input
        mem_mode = self.mem_mode

        if (not self.dynamic_input) and (mem_mode in ["internal_decoupled", "external"]):
            weight_filename_sim = "{}/input_1.npy".format(code_gen_dir)
            # save internal_decoupled weights for rtlsim
            self.make_weight_file_rtlsim(self.weights, weight_filename_sim)

        # create a npy file fore each input of the node (in_ind is input index)
        for in_ind, inputs in enumerate(node.input):
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            assert (
                str(context[inputs].dtype) == "float32"
            ), """Input datatype is
            not float32 as expected."""

            if in_ind == 0:
                expected_inp_shape = self.get_folded_input_shape(in_ind)

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
                    os.path.join(code_gen_dir, "input_0.npy"),
                    reshaped_input,
                )

            if in_ind == 1:
                if dynamic_input:
                    reshaped_input = context[inputs].reshape(-1, context[inputs].shape[-1])
                    
                    self.make_weight_file_rtlsim(
                        reshaped_input, "{}/input_1.npy".format(code_gen_dir)
                    )

        sim = self.get_rtlsim(code_gen_dir, rtlsim_trace)
        nbits = self.get_instream_width(0)
        inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
        self.reset_rtlsim(sim)

        if dynamic_input or mem_mode in ["external", "internal_decoupled"]:
            wnbits = self.get_instream_width(1)
            export_wdt = self.get_input_datatype(1)

            # we have converted bipolar weights to binary for export,
            # so use it as such for weight generation
            if self.get_input_datatype(1) == DataType["BIPOLAR"]:
                export_wdt = DataType["BINARY"]

            wei = npy_to_rtlsim_input("{}/input_1.npy".format(code_gen_dir), export_wdt, wnbits)
            num_w_reps = np.prod(self.numInputVectors)

            io_dict = {
                "inputs": {"in0": inp, "in1": wei * num_w_reps},
                "outputs": {"out0": []},
            }
        else:
            io_dict = {
                "inputs": {"in0": inp},
                "outputs": {"out0": []},
            }

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

    def make_weight_file_rtlsim(self, weights, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for rtlsim.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_name : filename for the weight file to be generated

        """
        # convert weights into hlslib/rtllib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        # create a weight stream for various flavors of internal_decoupled mode:
        # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
        weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
        # reverse SIMD flip for saving weights in .npy
        weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
        # simd_flipped
        weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, self.PE * self.SIMD)
        weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
        # save weight stream into npy for cppsim
        np.save(weight_file_name, weight_tensor_simd_flipped)
