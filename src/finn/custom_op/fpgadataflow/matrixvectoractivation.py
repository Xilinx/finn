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
import os
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
from finn.kernels.kernel_registry import gkr
from finn.util.kernel_util import get_node_attr

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
            "pumpedMemory": ("i", False, 0, {0, 1}),
            # dynamic input
            "dynamic_input": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def execute_node(self, context, graph):
        node = self.onnx_node
        in_act = context[node.input[0]]
        # ensure that shape is compatible
        kernel = gkr.kernel(node.op_type, get_node_attr(node))
        in_act = in_act.reshape(kernel.get_normal_input_shape())

        if self.get_nodeattr("dynamic_input"):
            mvau_w = context[node.input[1]]
        else:
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

    def infer_node_datatype(self, model):
        node = self.onnx_node
        kernel = gkr.kernel(node.op_type, get_node_attr(node))
        idt = model.get_tensor_datatype(node.input[0])
        if idt != kernel.get_input_datatype(0):
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(kernel.get_input_datatype(0)),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = kernel.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def minimize_accumulator_width(self, model):
        """Minimize the accumulator bit width according to the weight values,
        input data types, and size of dot product"""
        node = self.onnx_node
        kernel = gkr.kernel(node.op_type, get_node_attr(node, model))

        weights = model.get_initializer(self.onnx_node.input[1])
        # since in the calculation the values of the weight matrix are used,
        # for the bipolar case they need to be converted to bipolar
        if self.get_nodeattr("binaryXnorMode"):
            weights = 2 * weights - 1

        thresholds = None
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])

        idt = kernel.get_input_datatype(0)

        if not self.get_nodeattr("dynamic_input"):
            (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)

        # if runtime-writeable weights or dynamic input, then the values of the weights can
        # change and we need to use the worst-case values from the datatypes
        if self.get_nodeattr("runtime_writeable_weights") or self.get_nodeattr("dynamic_input"):
            mw = self.get_nodeattr("MW")
            mh = self.get_nodeattr("MH")
            wdt = kernel.get_input_datatype(1)
            lower_worst = wdt.min() * np.ones((mw, mh))
            lower_range = calculate_matvec_accumulator_range(lower_worst, idt)
            upper_worst = wdt.max() * np.ones((mw, mh))
            upper_range = calculate_matvec_accumulator_range(upper_worst, idt)
            acc_min = min(min(lower_range), min(upper_range))
            acc_max = max(max(lower_range), max(upper_range))

        # if the thresholds can be used to determine range, then adjust the range
        # according to the known values of the thresholds
        if thresholds is not None:
            node = self.onnx_node
            kernel = gkr.kernel(node.op_type, get_node_attr(node, model))
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
        if not (
            self.get_nodeattr("runtime_writeable_weights") or self.get_nodeattr("dynamic_input")
        ):
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

    # Redundant methods
    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for bipolar weights&inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        node = self.onnx_node
        kernel = gkr.kernel(node.op_type, get_node_attr(node))

        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        inp_is_bipolar = kernel.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = kernel.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = kernel.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = kernel.get_input_datatype(1) == DataType["BINARY"]
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
