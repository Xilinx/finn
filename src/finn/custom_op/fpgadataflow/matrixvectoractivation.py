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

import math
import numpy as np
import onnx.numpy_helper as np_helper
import qonnx.custom_op.general.xnorpopcount as xp
import textwrap
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string

# ONNX i/o tensor shape assumptions for MatrixVectorActivation:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)


class MVAU(HWCustomOp):
    """Abstraction layer for HW implementation of MatrixVectorActivation layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "PE": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "MW": ("i", True, 0),
            "MH": ("i", True, 0),
            "resType": ("s", False, "auto", {"auto", "lut", "dsp"}),
            "ActVal": ("i", False, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # FINN DataType for accumulator -- auto-computed and updated
            "accDataType": ("s", False, "INT32"),
            # use xnor-popcount for binary weights/inputs, thus treating them
            # as bipolar
            "binaryXnorMode": ("i", False, 0, {0, 1}),
            # no-activation mode (produce accumulators)
            "noActivation": ("i", False, 0, {0, 1}),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # memory mode for the FC weights
            # internal_embedded -- embedded weights, long compile/synth times
            # internal_decoupled -- default, streaming weights with streamer packaged inside IP
            # external -- streaming weights with external streamer
            "mem_mode": (
                "s",
                False,
                "internal_decoupled",
                {"internal_embedded", "internal_decoupled", "external"},
            ),
            # FPGA resource type for memories in internal_decoupled mode
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use UltraRAM (URAM), must have runtime_writeable_weights=1
            # see also https://www.xilinx.com/support/answers/38070.html
            "ram_style": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed", "ultra"},
            ),
            # FPGA resource type for threshold memories (if noActivation is False)
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            "ram_style_thresholds": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed"},
            ),
            # (mem_mode = internal_decoupled only) whether weights will be
            # writeable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            # see finn-rtllib/memstream/doc/README for more about the memory
            # address map used for writable weights
            # IMPORTANT: After using AXI lite to either read or write the weights,
            # always "flush" the accelerator by first passing a dummy input
            # vector through the accelerator. This will get rid of any old
            # weight data from the weight FIFOs.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def execute_node(self, context, graph):
        node = self.onnx_node
        in_act = context[node.input[0]]
        # ensure that shape is compatible
        in_act = in_act.reshape(self.get_normal_input_shape())
        mvau_w_init = [x for x in graph.initializer if x.name == node.input[1]][0]
        mvau_w = np_helper.to_array(mvau_w_init)
        # Matrix multiplication
        if self.get_nodeattr("binaryXnorMode"):
            # Note: activation/weights are expected to be binary
            # (by design coming from the transformation inferring this operation mode)
            result = xp.xnorpopcountmatmul(in_act, mvau_w)
        elif (
            self.get_nodeattr("inputDataType") == "BIPOLAR"
            and self.get_nodeattr("weightDataType") == "BIPOLAR"
        ):
            # Convert to binary and use xnorpopcountmatmul function
            result = xp.xnorpopcountmatmul((in_act + 1) / 2, (mvau_w + 1) / 2)
        else:
            # Regular matrix multiplication
            result = np.matmul(in_act, mvau_w)
        if self.get_nodeattr("noActivation") == 0:
            mvau_thr_init = [x for x in graph.initializer if x.name == node.input[2]][0]
            mvau_thr = np_helper.to_array(mvau_thr_init)
            odt_is_bipolar = self.get_nodeattr("outputDataType") == "BIPOLAR"
            out_scale = 2 if odt_is_bipolar else 1
            out_bias = -1 if odt_is_bipolar else self.get_nodeattr("ActVal")
            if result.ndim == 4:
                # NHWC to NCHW for multithreshold node
                result = result.transpose((0, 3, 1, 2))
            result = multithreshold(result, mvau_thr, out_scale, out_bias)
            if result.ndim == 4:
                # NCHW to NHWC
                result = result.transpose((0, 2, 3, 1))
        oshape = context[node.output[0]].shape
        context[node.output[0]] = result.reshape(oshape)

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        # TODO collect automatically from get_nodeattr_types
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("resType")
            self.get_nodeattr("MW")
            self.get_nodeattr("MH")
            self.get_nodeattr("SIMD")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("weightDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required MatrixVectorActivation attributes do not exist.""")

        # verify the number of inputs depending on noActivation value
        # check noActivation value to determine the number of inputs
        no_act = self.get_nodeattr("noActivation")

        if no_act == 1:
            if len(self.onnx_node.input) == 2:
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append(
                    """MatrixVectorActivation needs in no
                            activation mode 2 inputs (data input and weights)"""
                )
        elif no_act == 0:
            if len(self.onnx_node.input) == 3:
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append(
                    """MatrixVectorActivation needs 3 inputs
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

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
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
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        # when performing FIFO insertion on an FC layer with ext weights, the ind
        # parameter can be > 0 (referring to the weights) so handle that here
        if ind == 0:
            return DataType[self.get_nodeattr("inputDataType")]
        elif ind == 1:
            return DataType[self.get_nodeattr("weightDataType")]
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_weight_datatype(self):
        """Returns FINN DataType of weights."""
        return DataType[self.get_nodeattr("weightDataType")]

    def get_accumulator_datatype(self):
        """Returns FINN DataType of accumulator"""
        return DataType[self.get_nodeattr("accDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self, ind=0):
        i_bits = self.get_input_datatype().bitwidth()
        in_width = i_bits * self.get_nodeattr("SIMD")
        return in_width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("PE")
        return out_width

    def get_weightstream_width(self):
        """Returns weight stream width.
        Used only in internal_decoupled and external mode."""
        if (
            self.get_nodeattr("mem_mode") == "internal_decoupled"
            or self.get_nodeattr("mem_mode") == "external"
        ):
            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")
            wp = self.get_weight_datatype().bitwidth()
            w_width = pe * simd * wp
            return w_width
        else:
            return 0

    def get_weightstream_width_padded(self):
        """Returns weight stream width padded to a multiple of 8. This is required
        by the AXI Stream spec. Used in internal_decoupled mode."""
        weight_width = self.get_weightstream_width()
        return roundup_to_integer_multiple(weight_width, 8)

    def get_folded_input_shape(self, ind=0):
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        sf = mw // simd
        nf = mh // pe
        vecs = list(self.get_nodeattr("numInputVectors"))

        if ind == 0:
            # calculate shape of input 0
            folded_input_shape = tuple(vecs + [sf, simd])
        elif ind == 1 and self.get_nodeattr("mem_mode") == "external":
            # calculate shape of input 1 (weights)
            folded_input_shape = tuple(vecs + [sf * nf, simd * pe])
        else:
            raise Exception("Undefined input shape for requested input")

        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        nf = mh // pe
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_output_shape = tuple(vecs + [nf, pe])
        return folded_output_shape

    def get_normal_input_shape(self, ind=0):
        mw = self.get_nodeattr("MW")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_input_shape = tuple(vecs + [mw])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        mh = self.get_nodeattr("MH")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_output_shape = tuple(vecs + [mh])
        return normal_output_shape

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

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

    def uram_estimation(self):
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        omega = (D_in * D_out) / (Q * P)
        mem_width = Q * W * P
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (
            (mmode == "internal_decoupled" and mstyle != "ultra")
            or (mmode == "internal_embedded" and self.calc_wmem() <= 128)
            or (mmode == "external")
        ):
            return 0
        width_multiplier = math.ceil(mem_width / 72)
        depth_multiplier = math.ceil(omega / 4096)
        return width_multiplier * depth_multiplier

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
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        omega = (D_in * D_out) / (Q * P)
        mem_width = Q * W * P
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (
            (mmode == "internal_decoupled" and mstyle in ["distributed", "ultra"])
            or (mmode == "internal_embedded" and self.calc_wmem() <= 128)
            or (mmode == "external")
        ):
            return 0
        # assuming SDP mode RAMB18s (see UG573 Table 1-10)
        # assuming internal_decoupled (RTL) memory,
        # which is more efficient than internal_embedded (HLS)
        if mem_width == 1:
            return math.ceil(omega / 16384)
        elif mem_width == 2:
            return math.ceil(omega / 8192)
        elif mem_width <= 4:
            return (math.ceil(omega / 4096)) * (math.ceil(mem_width / 4))
        elif mem_width <= 9:
            return (math.ceil(omega / 2048)) * (math.ceil(mem_width / 9))
        elif mem_width <= 18 or omega > 512:
            return (math.ceil(omega / 1024)) * (math.ceil(mem_width / 18))
        else:
            return (math.ceil(omega / 512)) * (math.ceil(mem_width / 36))

    def bram_efficiency_estimation(self):
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        wbits = W * D_in * D_out
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def uram_efficiency_estimation(self):
        """Function for URAM efficiency estimation: actual parameter storage
        needed divided by the allocated URAM storage (from estimation)"""
        wdt = self.get_weight_datatype()
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        uram_est = self.uram_estimation()
        if uram_est == 0:
            return 1
        wbits = W * D_in * D_out
        uram_est_capacity = uram_est * 72 * 4096
        return wbits / uram_est_capacity

    def get_exp_cycles(self):
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        num_inp_vec = self.get_nodeattr("numInputVectors")
        mh = self.get_nodeattr("MH")
        mw = self.get_nodeattr("MW")
        # since mmv != 1 is not supported yet, we set mmv for now to 1
        mmv = 1
        exp_cycles = (mh / pe) * (mw / simd) * np.prod(num_inp_vec) / mmv
        return int(exp_cycles)

    def minimize_accumulator_width(self, model):
        """Minimize the accumulator bit width according to the weight values,
        input data types, and size of dot product"""
        weights = model.get_initializer(self.onnx_node.input[1])
        # since in the calculation the values of the weight matrix are used,
        # for the bipolar case they need to be converted to bipolar
        if self.get_nodeattr("binaryXnorMode"):
            weights = 2 * weights - 1

        thresholds = None
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])

        idt = self.get_input_datatype()

        (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)
        # if runtime-writeable weights, then the values of the weights can
        # change and we need to use the worst-case values from the datatypes
        if self.get_nodeattr("runtime_writeable_weights"):
            wdt = self.get_weight_datatype()
            lower_worst = wdt.min() * np.ones_like(weights)
            lower_range = calculate_matvec_accumulator_range(lower_worst, idt)
            upper_worst = wdt.max() * np.ones_like(weights)
            upper_range = calculate_matvec_accumulator_range(upper_worst, idt)
            acc_min = min(min(lower_range), min(upper_range))
            acc_max = max(max(upper_range), max(upper_range))

        # if the thresholds can be used to determine range, then adjust the range
        # according to the known values of the thresholds
        if thresholds is not None:
            threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
            # set threshold datatype (and accumulator datatype implicitly)
            min_threshold = thresholds.min()
            max_threshold = thresholds.max()
            # clip threshold values
            if max_threshold > acc_max or min_threshold < acc_min:
                warnings.warn("Clipping some thresholds in %s" % self.onnx_node.name)
                thresholds = np.clip(thresholds, acc_min, acc_max)
                model.set_initializer(self.onnx_node.input[2], thresholds)
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
                min_threshold = thresholds.min()
                max_threshold = thresholds.max()
            acc_min = min(min_threshold, acc_min)
            acc_max = max(max_threshold, acc_max)

        # if the acc_range is always greater than 0, then acc_max <= 2^P - 1
        if acc_min >= 0:
            acc_bit_width = np.log2(acc_max + 1)
            acc_bit_width = math.ceil(acc_bit_width)
            adt = DataType[f"UINT{acc_bit_width}"]
        # if the acc_range is signed, then acc_min >= -2^{P-1} and acc_max <=
        # 2^{P - 1} - 1, which means 2^{P - 1} >= max(-acc_min, 1 + acc_max)
        else:
            _acc_max = max(-acc_min, 1 + acc_max)
            acc_bit_width = np.log2(_acc_max) + 1
            acc_bit_width = math.ceil(acc_bit_width)
            adt = DataType[f"INT{acc_bit_width}"]

        # if activation, assert that the thresholds can be expressed with adt
        if thresholds is not None:
            assert np.vectorize(adt.allowed)(
                threshold_tensor
            ).all(), "Thresholds in %s can't be expressed with type %s" % (
                self.onnx_node.name,
                str(adt),
            )

        # if no activation, output and accumulator datatypes are the same
        if self.get_nodeattr("noActivation"):
            # if this is the last node in the graph, then ensure the datatype is
            # divisibly by 8 bits
            if model.find_direct_successors(self.onnx_node) is None:
                bw = roundup_to_integer_multiple(adt.bitwidth(), 8)
                new_adt_name = adt.name.replace(str(adt.bitwidth()), str(bw))
                adt = DataType[new_adt_name]
            # for no-activation nodes, output dt = acc dt
            self.set_nodeattr("outputDataType", adt.name)
        self.set_nodeattr("accDataType", adt.name)
        return DataType[self.get_nodeattr("accDataType")]

    def minimize_weight_bit_width(self, model):
        """Minimize the bit width based on the values of the weights"""
        if not self.get_nodeattr("runtime_writeable_weights"):
            weights = model.get_initializer(self.onnx_node.input[1])
            w_min = weights.min()
            w_max = weights.max()
            if w_min < 0:
                if abs(w_min) > w_max:
                    wdt = DataType.get_smallest_possible(w_min)
                else:
                    wdt = DataType.get_smallest_possible(-w_max - 1)
            else:
                wdt = DataType.get_smallest_possible(w_max)
            self.set_nodeattr("weightDataType", wdt.name)
        return DataType[self.get_nodeattr("weightDataType")]

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
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
        inp_is_bipolar = self.get_input_datatype() == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_weight_datatype() == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = self.get_input_datatype() == DataType["BINARY"]
        wt_is_binary = self.get_weight_datatype() == DataType["BINARY"]
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
        if self.get_weight_datatype() == DataType["BIPOLAR"]:
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

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for either synthesis or run-time reconfig
        of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated

        """
        # convert weights into hlslib/rtllib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        export_wdt = self.get_weight_datatype()
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation
        if self.get_weight_datatype() == DataType["BIPOLAR"]:
            export_wdt = DataType["BINARY"]
        if weight_file_mode == "hls_header":
            weight_hls_code = numpy_to_hls_code(weight_tensor, export_wdt, "weights", True, True)
            # write weights into C++ header file as dictated by finn-hlslib
            f_weights = open(weight_file_name, "w")
            if export_wdt.bitwidth() != 1:
                f_weights.write(
                    "const FixedPointWeights<{},{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        export_wdt.get_hls_datatype_str(),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            else:
                f_weights.write(
                    "const BinaryWeights<{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            f_weights.write(weight_hls_code)
            f_weights.close()
        elif "decoupled" in weight_file_mode:
            # create a weight stream for various flavors of internal_decoupled mode:
            # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
            weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
            # reverse SIMD flip for saving weights in .npy
            weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
            # PE flip for saving weights in .dat
            weight_tensor_pe_flipped = np.flip(weight_tensor_unflipped, axis=-2)
            # reshape weight tensor (simd_flipped and pe_flipped) to desired shape
            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")
            # simd_flipped
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
            # flipped
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()
            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, weight_tensor_simd_flipped)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_weightstream_width()
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                # add zeroes to pad out file to 1024 entries
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_weightstream_width()
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
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
                raise Exception("Unknown weight_file_mode")

        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model, path):
        mem_mode = self.get_nodeattr("mem_mode")
        code_gen_dir = path
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])
        if mem_mode == "internal_embedded":
            # save hlslib-compatible weights in params.h
            weight_filename = "{}/params.h".format(code_gen_dir)
            self.make_weight_file(weights, "hls_header", weight_filename)
        elif mem_mode == "internal_decoupled" or mem_mode == "external":
            weight_filename_sim = "{}/weights.npy".format(code_gen_dir)
            # save internal_decoupled weights for cppsim
            self.make_weight_file(weights, "decoupled_npy", weight_filename_sim)
            if mem_mode == "internal_decoupled":
                # also save weights as Verilog .dat file
                # This file will be ignored when synthesizing UltraScale memory.
                weight_filename_rtl = "{}/memblock.dat".format(code_gen_dir)
                self.make_weight_file(weights, "decoupled_verilog_dat", weight_filename_rtl)
        else:
            raise Exception(
                """Please set mem_mode to "internal_embedded", "internal_decoupled", or "external",
                currently no other parameter value is supported!"""
            )

        # save thresholds in thresh.h
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if thresholds is not None:
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
                # use UINT32 threshold export for bipolar times bipolar
                inp_is_bipolar = self.get_input_datatype() == DataType["BIPOLAR"]
                wt_is_bipolar = self.get_weight_datatype() == DataType["BIPOLAR"]
                # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
                inp_is_binary = self.get_input_datatype() == DataType["BINARY"]
                wt_is_binary = self.get_weight_datatype() == DataType["BINARY"]
                bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
                inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
                wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
                # get computed threshold datatype from attribute
                tdt = DataType[self.get_nodeattr("accDataType")]

                assert np.vectorize(tdt.allowed)(
                    threshold_tensor
                ).all(), "Thresholds in %s can't be expressed with type %s" % (
                    self.onnx_node.name,
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
                        self.get_nodeattr("PE"),
                        threshold_tensor.shape[-1],
                        tdt_hls,
                        odt_hls,
                        self.get_nodeattr("ActVal"),
                        "comp::less_equal<%s, %s>" % (tdt_hls, tdt_hls),
                    )
                )
                f_thresh.write(thresholds_hls_code)
                f_thresh.close()

    def get_op_and_param_counts(self):
        in_features = self.get_nodeattr("MW")
        out_features = self.get_nodeattr("MH")
        weight_bits = self.get_weight_datatype().bitwidth()
        inp_bits = self.get_input_datatype().bitwidth()
        num_inp_vec = self.get_nodeattr("numInputVectors")
        num_repetitions = int(np.prod(num_inp_vec))
        mac_count = in_features * out_features * num_repetitions
        # cannonicalize op type: highest bitwidth operand first s.t.
        # e.g. mac_8bx4b and mac_4bx8b don't appear as two different op types
        bw1 = min(inp_bits, weight_bits)
        bw2 = max(inp_bits, weight_bits)
        mac_op_type = "op_mac_%dbx%db" % (bw1, bw2)
        weight_param_type = "param_weight_%db" % (weight_bits)
        weight_count = in_features * out_features
        ret_dict = {mac_op_type: mac_count, weight_param_type: weight_count}
        if self.get_nodeattr("noActivation") == 0:
            tdt = DataType[self.get_nodeattr("accDataType")]
            thres_bits = tdt.bitwidth()
            thres_param_type = "param_threshold_%db" % (thres_bits)
            thres_count = out_features
            ret_dict[thres_param_type] = thres_count
        return ret_dict

    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out": []},
        }
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["internal_decoupled", "external"]:
            n_weight_inps = self.calc_wmem()
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            io_dict["inputs"]["weights"] = [0 for i in range(num_w_reps * n_weight_inps)]
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        sname = self.hls_sname()
        if mem_mode == "external":
            intf_names["s_axis"].append(("weights_" + sname, self.get_weightstream_width_padded()))
        if mem_mode == "internal_decoupled":
            # only expose axilite interface if attribute is set
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            if runtime_writable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def code_generation_ipi(self):
        cmd = []
        # add streamer if needed
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            if self.get_nodeattr("ram_style") == "ultra":
                assert (
                    runtime_writable == 1
                ), "Layer with URAM weights must have runtime_writeable_weights=1"
            node_name = self.onnx_node.name
            sname = self.hls_sname()
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
            # Instantiate either the HLS or RTL IP depending on operator
            self.instantiate_ip(cmd)

            # instantiate a streamer and connect it to the HLS IP
            strm_vlnv = "amd.com:finn:memstream:1.0"
            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s" % (strm_vlnv, node_name, strm_inst)
            )
            cmd.append(
                "set_property -dict [list "
                "CONFIG.DEPTH {%d} "
                "CONFIG.WIDTH {%d} "
                "CONFIG.INIT_FILE {%s} "
                "CONFIG.RAM_STYLE {%s} "
                "] [get_bd_cells /%s/%s]"
                % (
                    self.calc_wmem(),
                    self.get_weightstream_width_padded(),
                    self.get_nodeattr("code_gen_dir_ipgen") + "/memblock.dat",
                    self.get_nodeattr("ram_style"),
                    node_name,
                    strm_inst,
                )
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/weights_%s]"
                % (node_name, strm_inst, node_name, node_name, sname)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
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
            if runtime_writable:
                # expose axi lite interface for writeable weights
                axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s" % (node_name, axilite_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, axilite_name, node_name, strm_inst, axilite_name)
                )
                # TODO calculate and pass in segment size here
                cmd.append("assign_bd_address")
            cmd.append("save_bd_design")
        elif mem_mode == "internal_embedded" or mem_mode == "external":
            # base class impl sufficient for internal_embedded/external modes
            self.instantiate_ip(cmd)
        else:
            raise Exception("Unrecognized mem_mode for MatrixVectorActivation")
        return cmd
