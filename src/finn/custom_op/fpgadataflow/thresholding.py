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
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import (
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import get_memutil_alternatives, mem_primitives_versal
from finn.util.data_packing import pack_innermost_dim_as_hex_string
from finn.util.mlo import mlo_set_select_bits


class Thresholding(HWCustomOp):
    """Abstraction layer for HW implementation of Thresholding."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # whether weights (thresholds) will be
            # writable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            # memory depth triggers for threshold storage: the per-stage memories
            # of this depth or more go into URAM (priority) or BRAM, shallower ones
            # into LUTs once the BRAM trigger is set; 0 disables a trigger, both 0
            # leave the choice to the tools. Vivado maps every per-PE stage memory
            # from a depth of about 64 to its own BRAM18 even if filled to a few
            # percent; 512 keeps BRAMs at least half full (for thresholds up to 18
            # bits). Depth rather than bits decides as BRAM slices to narrow widths
            # while a shallow memory cannot fill it.
            "depth_trigger_uram": ("i", False, 0),
            "depth_trigger_bram": ("i", False, 512),
            # parallelization; channels thresholded per cycle
            "PE": ("i", True, 0),
            # number of channels (each may have different thresholds)
            "NumChannels": ("i", True, 0),
            # number of steps in thresholding function. Used only in decoupled mode
            "numSteps": ("i", True, 1),
            # FINN DataTypes for inputs, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # initialization value for the thresholding accumulator
            "ActVal": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype(0):
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(0).name),
                str(idt.name),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

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
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required Threshold_Batch attributes do not exist.""")

        return info_messages

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            dt = DataType[self.get_nodeattr("inputDataType")]
        elif ind == 1:
            dt = DataType[self.get_nodeattr("weightDataType")]
        else:
            raise Exception("Index out of range")
        return dt

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def minimize_weight_bit_width(self, model, datatype_only=False):
        """Minimize threshold datatype bitwidth based on actual threshold values.
        This function should not round or clip the threshold values,
        that is done in RoundAndClipThresholds.

        Parameters
        ----------
        datatype_only : bool
            If True, skip value-based minimization.
        """
        if datatype_only:
            return DataType[self.get_nodeattr("weightDataType")]

        thresholds = model.get_initializer(self.onnx_node.input[1])
        if self.get_nodeattr("runtime_writeable_weights") or self.get_nodeattr("mlo_max_iter"):
            return DataType[self.get_nodeattr("weightDataType")]
        threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
        # TODO: extend this for fixed point
        if self.get_input_datatype(0).is_integer() and self.get_input_datatype(1).is_integer():
            # minimize threshold width only if input and thresholds are integer
            # Use double precision for intermediate calculations to prevent overflow
            min_threshold = np.float64(thresholds.min())
            max_threshold = np.float64(thresholds.max())
            # Check if input datatype is signed
            input_is_signed = self.get_input_datatype(0).signed()
            # Special case: all thresholds are zero
            # get_smallest_possible(-1) returns BIPOLAR which can't represent 0
            if min_threshold == max_threshold == 0:
                if input_is_signed:
                    tdt = DataType["INT2"]
                else:
                    tdt = DataType["UINT1"]
            elif min_threshold < 0:
                if abs(min_threshold) > max_threshold:
                    tdt = DataType.get_smallest_possible(min_threshold)
                else:
                    tdt = DataType.get_smallest_possible(-max_threshold - 1)
            else:
                # If input is signed, use signed threshold datatype even if thresholds are positive
                if input_is_signed:
                    tdt = DataType.get_smallest_possible(-max_threshold - 1)
                else:
                    tdt = DataType.get_smallest_possible(max_threshold)
        else:
            # special case: if input is float, we keep thresholds as is
            tdt = self.get_input_datatype(1)
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        self.set_nodeattr("weightDataType", tdt.name)
        # Update QONNX DataType of tensor for consistency
        model.set_tensor_datatype(self.onnx_node.input[1], tdt)
        return DataType[self.get_nodeattr("weightDataType")]

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.get_nodeattr("PE")
        elif ind == 1:
            # thresholds are held in the kernel's memories, not streamed
            width = 0
        else:
            raise Exception("Index out of range")
        return width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def get_folded_input_shape(self, ind=0):
        pe = self.get_nodeattr("PE")
        fold = self.calc_tmem()
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_input_shape = tuple(vecs + [fold, pe])
        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_input_shape = tuple(vecs + [ich])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        # same shape as input
        return self.get_normal_input_shape()

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for unsigned inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        assert mh % pe == 0, "Requirement NumChannels divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.get_nodeattr("numSteps"), "Mismatch in threshold steps"
        if not self.get_input_datatype(0).signed():
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
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

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_values = context[node.input[0]]
        th_val = context[node.input[1]]
        out_bias = self.get_nodeattr("ActVal")

        # HW activations are always channels-last, so threshold the last axis
        # regardless of rank.
        y = multithreshold(inp_values, th_val, out_bias=out_bias, channels_last=True)

        act = DataType[self.get_nodeattr("outputDataType")]
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y = 2 * y - 1
        context[node.output[0]] = y.astype(np.float32)

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return num_channels // pe

    def get_verilog_top_module_intf_names(self):
        """Return dict of names of input and output interfaces for MLO support."""
        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = []
        intf_names["s_axis"].append(("in0_V", self.get_instream_width_padded(0)))
        intf_names["m_axis"] = []
        intf_names["m_axis"].append(("out0_V", self.get_outstream_width_padded(0)))
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        mlo_max_iter = self.get_nodeattr("mlo_max_iter")
        if mlo_max_iter:
            intf_names["s_axis"].append(("in1_V", mlo_set_select_bits(mlo_max_iter)))
        if self.get_nodeattr("runtime_writeable_weights") == 1:
            intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def get_search_stages(self):
        """Return the number of binary search stages, $clog2(numSteps+1), each of
        which owns one threshold memory per PE."""
        return math.ceil(math.log2(self.get_nodeattr("numSteps") + 1))

    def get_min_output_bits(self):
        """Return the narrowest output width the RTL accepts for the biased result
        (O_BITS default of thresholding.sv)."""
        n = self.get_nodeattr("numSteps")
        bias = self.get_nodeattr("ActVal")
        if bias >= 0:
            return math.ceil(math.log2(n + bias + 1))
        return 1 + math.ceil(math.log2(-bias if -bias >= n + bias + 1 else n + bias + 1))

    def get_pe_mem_geometries(self):
        """return a list of (bitwidth, depth) for PE memory configurations to be used
        in resource estimation

        for each bitwidth, the depth is calculated as the
        number of thresholds that can be stored in a single
        memory block
        the bitwidth is the bitwidth of the threshold values
        the depth is the number of thresholds that can be stored
        in a single memory block
        the number of memory blocks is calculated as the number
        of thresholds divided by the depth
        the number of memory blocks is then multiplied by the
        number of PEs to get the total number of memory blocks
        required for the entire layer
        """
        wdt_bits = self.get_input_datatype(1).bitwidth()
        cf = self.get_mem_rows()
        return [(wdt_bits, cf * (2**x)) for x in range(self.get_search_stages())]

    def get_mem_rows(self):
        """Return the rows of a stage memory per searched threshold: the channel
        fold, or for MLO the sets stacked over a power-of-two channel fold (as
        DEPTH in thresholding.sv and ROWS in finn-hlslib thresholding.hpp)."""
        cf = self.calc_tmem()
        mlo_sets = self.get_nodeattr("mlo_max_iter")
        if mlo_sets > 1:
            cf = mlo_sets * (1 << (cf - 1).bit_length())
        return cf

    def _get_memory_estimate_details(self):
        """return resource count, used bits and allocated capacity by memory type"""
        res_dict = {}
        used_bits = {}
        capacity_bits = {}
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        pe = self.get_nodeattr("PE")
        ret = self.get_pe_mem_geometries()
        for mem_cfg in ret:
            (width, depth) = mem_cfg
            # as RAM_STYLE in thresholding.sv and select_mem() in thresholding.hpp
            if depth_trigger_uram and depth >= depth_trigger_uram:
                kind = "URAM"
            elif depth_trigger_bram and depth >= depth_trigger_bram:
                kind = "BRAM"
            elif depth_trigger_bram:
                kind = "LUTRAM"
            else:
                kind = ""  # Vivado's choice, approximated by the least waste
            primitives = {k: v for (k, v) in mem_primitives_versal.items() if kind in k}
            alts = get_memutil_alternatives(mem_cfg, primitives)
            primary_alt = alts[0]
            primitive_name = primary_alt[0]
            if primitive_name.startswith("BRAM"):
                res_type = "BRAM"
            else:
                res_type = primitive_name.split("_")[0]
            res_count, eff, waste = primary_alt[1]
            primitive_width, primitive_depth = mem_primitives_versal[primitive_name]
            res_dict[res_type] = res_dict.get(res_type, 0) + pe * res_count
            used_bits[res_type] = used_bits.get(res_type, 0) + pe * width * depth
            capacity_bits[res_type] = capacity_bits.get(res_type, 0) + (
                pe * res_count * primitive_width * primitive_depth
            )
        return res_dict, used_bits, capacity_bits

    def get_memory_estimate(self):
        """return the memory estimate for this node"""
        res_dict, _, _ = self._get_memory_estimate_details()
        return res_dict

    def bram_estimation(self, fpgapart):
        """return the number of BRAMs required for this node"""
        res_dict = self.get_memory_estimate()
        return res_dict.get("BRAM", 0)

    def uram_estimation(self, fpgapart):
        """return the number of URAMs required for this node"""
        res_dict = self.get_memory_estimate()
        return res_dict.get("URAM", 0)

    def lut_estimation(self, fpgapart):
        """return the number of LUTs required for this node"""
        res_dict = self.get_memory_estimate()
        # comparison and pipeline logic, about half a LUT per threshold bit, stage
        # and PE (fit to out-of-context results on xczu7ev for 8- and 16-bit
        # thresholds)
        logic = (
            self.get_nodeattr("PE")
            * self.get_search_stages()
            * self.get_input_datatype(1).bitwidth()
            // 2
        )
        return res_dict.get("LUTRAM", 0) + logic

    def bram_efficiency_estimation(self, fpgapart):
        """return BRAM parameter storage efficiency for this node"""
        _, used_bits, capacity_bits = self._get_memory_estimate_details()
        if capacity_bits.get("BRAM", 0) == 0:
            return 1
        return used_bits["BRAM"] / capacity_bits["BRAM"]

    def uram_efficiency_estimation(self, fpgapart):
        """return URAM parameter storage efficiency for this node."""
        # TODO: Versal URAM supports flexible bit widths (9/18/36/72) unlike
        # UltraScale+ which only supports 72-bit. This could improve efficiency
        # for narrow data types on Versal devices.
        _, used_bits, capacity_bits = self._get_memory_estimate_details()
        if capacity_bits.get("URAM", 0) == 0:
            return 1
        return used_bits["URAM"] / capacity_bits["URAM"]

    def get_stage_thresholds(self, thresholds):
        """Split the thresholds into the per-stage memories of the binary search:
        stage s holds for each PE, at row*2^s + k, the separator (2k+1)*2^(M-1-s)-1
        of the channel in that row; separators beyond numSteps are zero. Returns
        one array of shape (PE, CF*2^s) per stage. Thresholds shared by all
        channels (a single row) are stored once per PE, i.e. as C = PE."""
        pe = self.get_nodeattr("PE")
        stages = self.get_search_stages()
        n = self.get_nodeattr("numSteps")
        thresholds = np.asarray(thresholds)
        if thresholds.shape[0] == 1:
            thresholds = np.broadcast_to(thresholds, (pe, n))
        cf = thresholds.shape[0] // pe
        padded = np.zeros((thresholds.shape[0], 2**stages - 1), dtype=thresholds.dtype)
        padded[:, :n] = thresholds
        ret = []
        for s in range(stages):
            sn = stages - 1 - s
            idx = ((2 * np.arange(2**s) + 1) << sn) - 1
            # (cf, pe, 2^s) -> (pe, cf, 2^s) -> (pe, cf*2^s)
            sel = padded[:, idx].reshape(cf, pe, 2**s).transpose(1, 0, 2)
            ret.append(sel.reshape(pe, cf * 2**s))
        return ret

    def make_runtime_weight_file(self, thresholds, weight_file_name):
        """Write the thresholds in the AXI-lite word layout of runtime-writeable
        thresholds, which the driver writes as is: threshold n of channel
        cf*PE + pe at word ((cf << clog2(PE)) + pe) << clog2(numSteps) | n, one
        32-bit word per line, unused words zero."""
        # Runtime updates bypass inference and codegen, so this file sorts them
        # for the binary search. This never changes the result (see
        # InferThresholdingLayer).
        thresholds = np.sort(thresholds, axis=-1)
        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")
        n_thres_steps = self.get_nodeattr("numSteps")
        wdt = self.get_input_datatype(1)
        # If a single threshold value is found, broadcast the value
        if thresholds.shape[0] == 1:
            thresholds = np.broadcast_to(thresholds, (pe, n_thres_steps))
            num_channels = pe
        # The address space per channel is 2^clog2(N) words; for N=1, clog2(1)=0,
        # so only 1 slot per channel.
        width_padded = 1 << max(0, math.ceil(math.log2(n_thres_steps)))
        thresh_padded = np.zeros((thresholds.shape[0], width_padded))
        thresh_padded[:, :n_thres_steps] = thresholds[:, :n_thres_steps]
        thresh_stream = []
        bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 32)
        padding = np.zeros(width_padded, dtype=np.int32)

        chan_ind = 0
        cf = num_channels // pe
        for fold in range(cf):
            for c in range(2 ** (pe - 1).bit_length()):
                if (c == 0 or c % pe != 0) and c < pe:
                    for t in thresh_padded[chan_ind]:
                        t_packed = pack_innermost_dim_as_hex_string(
                            [t], wdt, bw_hexdigit, prefix=""
                        ).item()
                        thresh_stream.append(t_packed)
                    chan_ind += 1
                else:
                    for z in padding:
                        t_packed = pack_innermost_dim_as_hex_string(
                            [z], wdt, bw_hexdigit, prefix=""
                        ).item()
                        thresh_stream.append(t_packed)
        with open(weight_file_name, "w") as f:
            for val in thresh_stream:
                f.write(val + "\n")
