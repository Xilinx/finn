# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
import warnings
from onnx import helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.custom_op.fpgadataflow.hls import custom_op as hls_variants
from finn.custom_op.fpgadataflow.rtl import custom_op as rtl_variants
from finn.util.basic import get_dsp_block, get_dsp_datapath_limits, is_versal


def _determine_impl_style(node, fpgapart, model):
    optype = node.op_type

    # check if there is an HLS or RTL variant or both
    hls_variant = optype + "_hls" in hls_variants.keys()
    rtl_variant = optype + "_rtl" in rtl_variants.keys()

    # check if user has specified a preferred_impl_style
    node_inst = getCustomOp(node)
    impl_style = node_inst.get_nodeattr("preferred_impl_style")

    # if impl_style not set, for "simple" layers always try
    # to use rtl variant if available
    if impl_style == "":
        if rtl_variant:
            if optype == "DotProd":
                idt = node_inst.get_input_datatype(0)
                wdt = node_inst.get_input_datatype(1)
                inp_width_fit = idt.bitwidth() >= 4
                weight_width_fit = wdt.bitwidth() >= 4
                if inp_width_fit and weight_width_fit and _dotprod_rtl_possible(node, fpgapart, model):
                    return "rtl"
                else:
                    return "hls"
            elif optype in ["ElementwiseAdd", "ElementwiseSub", "ElementwiseMul"]:
                if _elementwise_rtl_possible(node, fpgapart):
                    return "rtl"
                else:
                    return "hls"
            elif optype == "LayerNorm":
                if _layernorm_rtl_possible(node, fpgapart):
                    return "rtl"
                else:
                    return "hls"
            elif optype == "PWPolyF":
                _pwpolyf_rtl_possible(node, fpgapart)
                return "rtl"
            elif optype == "HWSoftmax":
                if _softmax_rtl_possible(node, fpgapart):
                    return "rtl"
                else:
                    return "hls"
            elif optype == "Requant":
                if _requant_rtl_possible(node, fpgapart):
                    return "rtl"
                else:
                    return "hls"
            return "rtl"
        # but if no rtl variant, set impl_style to hls
        elif hls_variant:
            return "hls"
        # if there is neither an rtl nor hls variant
        # throw error
        else:
            raise Exception(
                """Node {} with optype {} has no hw implementation variant)""".format(
                    node.name, optype
                )
            )

    # check if user setting can be fulfilled
    # otherwise change impl_style
    elif impl_style == "hls":
        if hls_variant:
            return "hls"
        elif rtl_variant:
            if optype == "PWPolyF":
                _pwpolyf_rtl_possible(node, fpgapart)
            warn_str = """There is no HLS variant of %s. Node %s will automatically be
                        set to RTL variant.""" % (
                node.op_type,
                node.name,
            )
            warnings.warn(warn_str)
            return "rtl"
        else:
            raise Exception(
                """Node {} with optype {} has no hw implementation variant)""".format(
                    node.name, optype
                )
            )
    elif impl_style == "rtl":
        if optype == "DotProd":
            if _dotprod_rtl_possible(node, fpgapart, model):
                return "rtl"
            else:
                warn_str = """There is no RTL variant for %s. The node will automatically be
                        set to HLS variant. Please check the bit-widths and ensure
                        weights are signed.""" % (
                    node.name,
                )
                warnings.warn(warn_str)
                return "hls"

        elif optype == "PWPolyF":
            _pwpolyf_rtl_possible(node, fpgapart)
            return "rtl"
        elif optype == "LayerNorm":
            if _layernorm_rtl_possible(node, fpgapart):
                return "rtl"
            else:
                warn_str = """There is no RTL variant for %s. The node will automatically be
                        set to HLS variant. The RTL Layernorm layer currently only supports
                        float32 inputs and uses DSP58, so only versal devices supported.""" % (
                    node.name,
                )
                warnings.warn(warn_str)
                return "hls"
        elif optype == "HWSoftmax":
            if _softmax_rtl_possible(node, fpgapart):
                return "rtl"
            else:
                warn_str = """There is no RTL variant for %s. The node will automatically be
                        set to HLS variant. The RTL SoftMax layer uses DSPFP32, so only
                        versal devices are supported.""" % (
                    node.name,
                )
                warnings.warn(warn_str)
                return "hls"
        elif optype in ["ElementwiseAdd", "ElementwiseSub", "ElementwiseMul"]:
            if _elementwise_rtl_possible(node, fpgapart):
                return "rtl"
            else:
                warn_str = """There is no RTL variant for %s. The node will automatically be
                        set to HLS variant. The RTL Elementwise layers use DSP58 and require
                        Versal devices. For int/int, both operand widths and signedness must
                        match, and MUL width is limited by DSP58 capacity.""" % (
                    node.name,
                )
                warnings.warn(warn_str)
                return "hls"
        elif optype == "Requant":
            if _requant_rtl_possible(node, fpgapart):
                return "rtl"
            else:
                warn_str = """There is no RTL variant for %s. The node will automatically be
                        set to HLS variant. The RTL Requant layers require non-narrow
                        quantization and either integer inputs or FLOAT32 inputs on a Versal
                        device (the float requantf core uses DSPFP32).""" % (
                    node.name,
                )
                warnings.warn(warn_str)
                return "hls"

        if rtl_variant:
            return "rtl"
        elif hls_variant:
            warn_str = """There is no RTL variant of %s. Node %s will automatically be
                        set to HLS variant.""" % (
                node.op_type,
                node.name,
            )
            warnings.warn(warn_str)
            return "hls"
        else:
            raise Exception(
                """Node {} with optype {} has no hw implementation variant)""".format(
                    node.name, optype
                )
            )
    else:
        raise Exception(
            """Invalid value for attribute preferred_impl_style! Is currently set to: {}
            has to be set to one of the following value ("hls", "rtl")""".format(
                impl_style
            )
        )


def _elementwise_rtl_possible(n, fpgapart):
    # Checks whether RTL-based ElementwiseOp is possible.
    # Only supports float/float and int/float paths on Versal.
    # Int/int uses HLS to avoid bitwidth mismatch issues after MinimizeBitWidth.
    if not is_versal(fpgapart):
        return False

    node_inst = getCustomOp(n)
    lhs_dtype = node_inst.get_input_datatype(0)
    rhs_dtype = node_inst.get_input_datatype(1)

    lhs_float = lhs_dtype == "FLOAT32"
    rhs_float = rhs_dtype == "FLOAT32"

    # Only use RTL for float/float or int/float scenarios
    # Int/int defaults to HLS to avoid bitwidth mismatch after MinimizeBitWidth
    if not lhs_float and not rhs_float:
        return False

    # Float inputs must be FLOAT32 (not FLOAT16 etc.)
    if lhs_float and lhs_dtype != "FLOAT32":
        return False
    if rhs_float and rhs_dtype != "FLOAT32":
        return False

    lhs_style = node_inst.get_nodeattr("lhs_style")
    rhs_style = node_inst.get_nodeattr("rhs_style")

    # Both const makes no sense for streaming
    if lhs_style == "const" and rhs_style == "const":
        return False

    # Check shape constraints for input/const mode
    lhs_shape = node_inst.get_nodeattr("lhs_shape")
    out_shape = node_inst.get_nodeattr("out_shape")
    if lhs_style == "input" and list(lhs_shape) != list(out_shape):
        return False

    # Broadcasting check for const side
    if rhs_style == "const":
        rhs_shape = node_inst.get_nodeattr("rhs_shape")
        if len(rhs_shape) != len(out_shape) and len(rhs_shape) != len(out_shape) - 1:
            for dim_c, dim_o in zip(rhs_shape, out_shape[-len(rhs_shape) :]):
                if dim_c != 1 and dim_c != dim_o:
                    return False

    return True


def _layernorm_rtl_possible(n, fpgapart):
    # Checks whether RTL-based Layernorm is supported
    # Currently, we only support float32 inputs and versal fabric
    if not is_versal(fpgapart):
        return False
    node_inst = getCustomOp(n)
    idt = node_inst.get_input_datatype(0)
    if idt != "FLOAT32":
        return False
    else:
        return True


def _pwpolyf_rtl_possible(n, fpgapart):
    # PWPolyF uses the Versal DSPFP32 primitive.
    if not is_versal(fpgapart):
        raise Exception(
            "PWPolyF node %s cannot be specialized for FPGA part %s. "
            "PWPolyF_rtl uses the Versal DSPFP32 primitive and is only supported "
            "on Versal devices." % (n.name, fpgapart)
        )
    return True


def _softmax_rtl_possible(n, fpgapart):
    # Checks whether RTL-based SoftMax is supported.
    # The RTL softmax core uses DSPFP32, so only Versal devices are supported.
    return is_versal(fpgapart)


def _dotprod_rtl_possible(n, fpgapart, model):
    """Checks whether RTL-based DotProd is supported.

    DotProd always has standalone thresholds (no noActivation attribute),
    so the only constraints are DSP datapath width limits and signed weights.
    For depthwise mode, VVU RTL is Versal-only (DSP58).
    """
    node_inst = getCustomOp(n)
    depthwise = node_inst.get_nodeattr("depthwise")

    idt = node_inst.get_input_datatype(0)
    wdt = node_inst.get_input_datatype(1)

    # Weights must be signed for the RTL MVU/VVU core
    if not wdt.signed():
        return False

    if depthwise:
        # VVU RTL core requires Versal (DSP58)
        if not is_versal(fpgapart):
            return False
        in_width_in_range = (idt.bitwidth() <= 8) or (idt.bitwidth() == 9 and idt.min() < 0)
        weight_width_in_range = wdt.bitwidth() <= 8
        return in_width_in_range and weight_width_in_range
    else:
        # Standard MVU path — same checks as _mvu_rtl_possible minus noActivation
        dsp_block = get_dsp_block(fpgapart)
        weights = model.get_initializer(n.input[1])
        if weights is None:
            weights_min = wdt.min()
        else:
            weights_min = np.min(weights)
        narrow_weights = False if weights_min == wdt.min() else True
        if not narrow_weights and dsp_block == "DSP48E1":
            return False

        inp_width_in_range = 2 <= idt.bitwidth()
        weight_width_in_range = 2 <= wdt.bitwidth()

        max_act, max_weight, max_acc = get_dsp_datapath_limits(dsp_block)
        acc_width = node_inst.get_output_datatype().bitwidth()
        signed_act = 1 if idt.signed() else 0
        act_fits = (idt.bitwidth() - signed_act) < max_act
        widths_in_range = act_fits and wdt.bitwidth() <= max_weight and acc_width <= max_acc

        return inp_width_in_range and weight_width_in_range and widths_in_range


def _requant_rtl_possible(n, fpgapart):
    # Checks whether RTL-based Requant is supported
    # RTL Requant requires full range (narrow=0) and either:
    # - Integer input   -> integer requant.sv path, or
    # - FLOAT32 input    -> requantf.sv path (DSPFP32, Versal only)
    node_inst = getCustomOp(n)
    idt = node_inst.get_input_datatype(0)
    narrow = node_inst.get_nodeattr("narrow")
    if narrow != 0:
        return False
    if idt.is_integer():
        return True  # existing integer requant.sv path
    if idt == "FLOAT32":
        return is_versal(fpgapart)  # requantf.sv path (DSPFP32 -> Versal only)
    return False


class SpecializeLayers(Transformation):
    """Specialize all layers to either HLS or RTL variants"""

    def __init__(self, fpgapart):
        super().__init__()
        self.fpgapart = fpgapart

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            # Skip nodes that are not hw layers
            if not node.domain == "finn.custom_op.fpgadataflow":
                continue
            # For shuffle nodes the specialisation happens after
            # the ShuffleDecomposition transformation with a
            # dedicated InferInnerOuterShuffle transformation
            if node.op_type == "Shuffle":
                continue
            node_ind += 1
            impl_style = _determine_impl_style(node, self.fpgapart, model)
            optype = node.op_type + "_" + impl_style

            new_node = helper.make_node(
                optype,
                node.input,
                node.output,
                domain="finn.custom_op.fpgadataflow." + impl_style,
            )
            # add all attributes
            for attribute in node.attribute:
                if attribute.name != "preferred_impl_style":
                    new_node.attribute.append(attribute)
            graph.node.insert(node_ind, new_node)
            # remove old nodes
            graph.node.remove(node)
            graph_modified = True
        return (model, graph_modified)
