# Copyright (c) 2021, Xilinx
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
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.quant import max_int, min_int
from qonnx.custom_op.general.trunc import trunc_v2
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

# Largest window-sum width for which a non-FLOOR Trunc is expressed as thresholds:
# the thresholds are stored as float32, which represents every integer exactly
# only up to 2**24.
MAX_TRUNC_THRESHOLD_SUM_BITS = 24


def _get_trunc_v2_thresholds(model, trunc_node, sum_dt):
    """
    Compute MultiThreshold thresholds that reproduce a Trunc (v2) node applied
    to the integer sum of an average pooling window.

    The Trunc output is a monotonically non-decreasing function of the window
    sum for every rounding mode, so it can be expressed exactly by thresholding
    the sum. The threshold of each output level is the smallest window sum in
    sum_dt that reaches it, found with a binary search (run for all levels at
    once) that evaluates the QONNX reference implementation.

    Returns the thresholds (shape (1, n_levels - 1), shared by all channels) and
    the lowest output level, to be used as the MultiThreshold out_bias.
    """
    scale, zeropt, in_bits, out_scale, out_bits = [
        model.get_initializer(inp) for inp in trunc_node.input[1:]
    ]
    rounding_mode = get_by_name(trunc_node.attribute, "rounding_mode")
    rounding_mode = rounding_mode.s.decode() if rounding_mode is not None else "FLOOR"
    narrow = get_by_name(trunc_node.attribute, "narrow")
    narrow = narrow.i if narrow is not None else 0
    signed = get_by_name(trunc_node.attribute, "signed")
    signed = signed.i if signed is not None else 1

    def trunc_level(window_sums):
        # The Trunc input is the window average, i.e. the window sum times the
        # Trunc input scale
        out = trunc_v2(
            window_sums.astype(np.float64) * scale,
            scale,
            zeropt,
            in_bits,
            narrow,
            signed,
            out_scale,
            out_bits,
            rounding_mode,
        )
        return np.round(out / out_scale)

    min_level = int(min_int(signed, narrow, out_bits))
    max_level = int(max_int(signed, narrow, out_bits))
    levels = np.arange(min_level + 1, max_level + 1)
    # Binary search for the smallest window sum reaching each level in [lo, hi),
    # hi = sum_dt.max() + 1 marks levels that are never reached
    lo = np.full(levels.shape, sum_dt.min(), dtype=np.int64)
    hi = np.full(levels.shape, sum_dt.max() + 1, dtype=np.int64)
    while np.any(lo < hi):
        active = lo < hi
        mid = (lo + hi) // 2
        reached = trunc_level(np.minimum(mid, sum_dt.max())) >= levels
        hi = np.where(active & reached, mid, hi)
        lo = np.where(active & ~reached, mid + 1, lo)
    return lo.astype(np.float32).reshape(1, -1), min_level


def _get_signed_from_upstream(model, trunc_node):
    """
    Find out what the sign of the input to the trunc node is,
    by looking at the upstream nodes.
    """
    node = trunc_node
    # Check if the input of this node already has a FINN datatype
    signed = None
    inp_dt = model.get_tensor_datatype(node.input[0])
    if inp_dt is not None and inp_dt != DataType["FLOAT32"]:
        signed = inp_dt.signed()
    # Go further up the graph, since the datatype inference works top down
    # these nodes should either be sign preserving ops or they already have a
    # datatype defined for the output tensor.
    curr_node = node
    if signed is None:
        while curr_node is not None:
            if model.is_join_node(curr_node):
                raise RuntimeError(
                    "Datatype Inference for the Trunc node only supports "
                    "linear nodes in the upstream path."
                )
            next_node = model.find_direct_predecessors(curr_node)
            if next_node is None:
                raise RuntimeError(
                    "Could not infere the Datatype for the Trunc node due to "
                    "missing upstream ndoes."
                )
            next_node = next_node[0]
            out_dt = model.get_tensor_datatype(next_node.output[0])
            if out_dt is not None and out_dt != DataType["FLOAT32"]:
                signed = out_dt.signed()
                break
            # Special cases where the node has an internal or intrinsic datatype.
            if next_node.op_type == "MultiThreshold":
                mt_inst = getCustomOp(next_node, onnx_opset_version=9)
                out_dt = DataType[mt_inst.get_nodeattr("out_dtype")]
                if out_dt is not None and out_dt != DataType["FLOAT32"]:
                    signed = out_dt.signed()
                    break
            if next_node.op_type == "BipolarQuant":
                signed = True
                break
            if next_node.op_type == "Quant":
                q_inst = getCustomOp(next_node, onnx_opset_version=9)
                out_dt = q_inst.get_integer_datatype(model)
                if out_dt is not None and out_dt != DataType["FLOAT32"]:
                    signed = out_dt.signed()
                    break

            # Check if we are allowed to move on to the next op
            sign_preserving_ops = ["Add", "Mul", "AveragePool", "Pad"]
            if next_node.op_type not in sign_preserving_ops:
                raise RuntimeError(
                    f"Could not infere the Datatype for the Trunc node, "
                    f"because the sign of the input datatype could not be infered "
                    f"from upstream nodes. And traversal further up the graph was "
                    f"disallowed, since the next node type {next_node.op_type} "
                    f"is not in the list of "
                    f"sign preserving ops {sign_preserving_ops}."
                )
            curr_node = next_node

    if signed is None:
        raise RuntimeError(
            "Could not infere the Datatype for the Trunc node, "
            "because the sign of the input datatype could not be infered "
            "from upstream nodes."
        )

    return signed


class AvgPoolAndTruncToQuantAvgPool(Transformation):
    """
    Convert a section of nodes of the pattern:
    AveragePool -> Mul (scalar) -> Trunc
    To the FINN op: QuantAvgPool2d
    """

    def apply(self, model):
        opset_imports = model.get_opset_imports()
        if "qonnx.custom_op.general" in opset_imports:
            trunc_opset = opset_imports["qonnx.custom_op.general"]
        elif "onnx.brevitas" in opset_imports:
            trunc_opset = opset_imports["onnx.brevitas"]
        else:
            trunc_opset = 1  # Default to v1 if no opset found
        if trunc_opset == 1:
            model = model.transform(AvgPoolAndTruncv1ToQuantAvgPool())
            return model, False
        elif trunc_opset == 2:
            model = model.transform(AvgPoolAndTruncv2ToQuantAvgPool())
            return model, False
        else:
            raise NotImplementedError(
                f"AvgPoolAndTruncToQuantAvgPool not implemented for "
                f"Trunc opset version {trunc_opset}."
            )


class AvgPoolAndTruncv1ToQuantAvgPool(Transformation):
    """
    Convert a section of nodes of the pattern:
    AveragePool -> Mul (scalar) -> Trunc (v1)
    To the FINN op: Div -> QuantAvgPool2d -> Mul
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "AveragePool":
                mul_node = model.find_direct_successors(n)
                if mul_node is not None and len(mul_node) == 1 and mul_node[0].op_type == "Mul":
                    mul_node = mul_node[0]
                    t_node = model.find_direct_successors(mul_node)
                    if t_node is not None and len(t_node) == 1 and t_node[0].op_type == "Trunc":
                        t_node = t_node[0]
                        running_node_index = node_ind
                        # Check node for compatibility
                        # Avg pooling node
                        k_s = get_by_name(n.attribute, "kernel_shape")
                        if k_s is None or len(k_s.ints) != 2 or len(set(k_s.ints)) != 1:
                            raise ValueError(
                                "FINN only supports average pooling with " "2D square kernels."
                            )
                        k_s = k_s.ints[0]

                        pads = get_by_name(n.attribute, "pads")
                        if pads is None or len(set(pads.ints)) != 1 or pads.ints[0] != 0:
                            raise ValueError("FINN dosn't support padding for average pooling.")

                        stride = get_by_name(n.attribute, "strides")
                        if stride is None or len(stride.ints) != 2 or len(set(stride.ints)) != 1:
                            raise ValueError(
                                "FINN only supports 2D strides with equal values in "
                                "each direction."
                            )
                        stride = stride.ints[0]

                        # Mul node
                        mul_val = model.get_initializer(mul_node.input[1])
                        if mul_val is None or len(mul_val.shape) != 0 or mul_val != k_s * k_s:
                            raise ValueError(
                                f"The Mul node after the AveragePool node must have "
                                f"static initialization at the second input, "
                                f"further the initialization must be of zero dimension "
                                f"and the value of the initialization must be "
                                f"the quadratic value of the kernel size, "
                                f"in this case {k_s * k_s}."
                            )

                        # Trunc node
                        rounding_mode = get_by_name(t_node.attribute, "rounding_mode")
                        normalized_mode_string = rounding_mode.s.upper()
                        if rounding_mode is None or normalized_mode_string != b"FLOOR":
                            raise ValueError(
                                "The Trunc node must have the rounding_mode " "set to 'FLOOR'."
                            )
                        for inp in t_node.input[1:]:
                            if model.get_initializer(inp) is None:
                                raise ValueError(
                                    f"All inputs of the Trunc node, "
                                    f"except the first, must be statically "
                                    f"initialized. However, {inp} is not."
                                )
                        zero_pt = model.get_initializer(t_node.input[2])
                        if len(zero_pt.shape) != 0 or zero_pt != 0:
                            raise ValueError(
                                f"Finn only supports 0 as the zero point for "
                                f"the Trunc node, it currently is {zero_pt}."
                            )
                        trunc_in_bits = model.get_initializer(t_node.input[3]).flatten()
                        trunc_out_bits = model.get_initializer(t_node.input[4]).flatten()
                        if len(trunc_in_bits.shape) != 1 or len(trunc_out_bits.shape) != 1:
                            raise ValueError(
                                f"Finn only supports scalar bit widths "
                                f"for the Trunc node. The input bit width "
                                f"currently is: {trunc_in_bits}, "
                                f"while the output bit width is: {trunc_out_bits}."
                            )
                        trunc_in_bits = int(trunc_in_bits[0])
                        trunc_out_bits = int(trunc_out_bits[0])

                        # Calculate parameters for the QuantAvgPool2d node,
                        # Calculate input bit width. Basically this backwards:
                        # https://github.com/Xilinx/finn-base/blob/
                        # 7c2603a95e90e4de2575020e575c24eab6a15889/src/finn/custom_op/
                        # general/quantavgpool2d.py#L94
                        # TODO: This underestimates ibits by one bit for small input bit
                        #  widths with non-power-of-two windows (e.g. 3 bit and 3x3), so
                        #  the QuantAvgPool2d shift no longer matches the Trunc. Use the
                        #  same derivation as AvgPoolAndTruncv2ToQuantAvgPool.
                        ibits = math.floor(math.log(2**trunc_in_bits / (k_s * k_s), 2))
                        # Get sign
                        signed = _get_signed_from_upstream(model, t_node)
                        # ToDo: Change this to NHWC,
                        #  when the channels last layout comes around.
                        data_layout = "NCHW"

                        # Insert scale nodes, QuantAvgPool2d node and required tensors
                        scale = model.get_initializer(t_node.input[1])
                        scale_div_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(scale_div_tensor)
                        model.set_initializer(scale_div_tensor.name, scale)

                        act_scale_div_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(act_scale_div_tensor)

                        scale_div_node = helper.make_node(
                            "Div",
                            [n.input[0], scale_div_tensor.name],
                            [act_scale_div_tensor.name],
                        )
                        graph.node.insert(running_node_index, scale_div_node)
                        running_node_index += 1

                        act_scale_mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(act_scale_mul_tensor)

                        QuantAvgPool2d_node = helper.make_node(
                            "QuantAvgPool2d",
                            [act_scale_div_tensor.name],
                            [act_scale_mul_tensor.name],
                            domain="qonnx.custom_op.general",
                            stride=stride,
                            kernel=k_s,
                            ibits=ibits,
                            obits=trunc_out_bits,
                            signed=int(signed),
                            data_layout=data_layout,
                        )
                        graph.node.insert(running_node_index, QuantAvgPool2d_node)
                        running_node_index += 1

                        scale_mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(scale_mul_tensor)
                        model.set_initializer(scale_mul_tensor.name, scale)

                        scale_mul_node = helper.make_node(
                            "Mul",
                            [act_scale_mul_tensor.name, scale_mul_tensor.name],
                            [t_node.output[0]],
                        )
                        graph.node.insert(running_node_index, scale_mul_node)
                        running_node_index += 1

                        # Remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(mul_node)
                        graph.node.remove(t_node)

                        # Recompute shapes and datatypes
                        model = model.transform(InferShapes())
                        model = model.transform(InferDataTypes())

                        return model, True

        return model, False


class AvgPoolAndTruncv2ToQuantAvgPool(Transformation):
    """
    Convert a section of nodes of the pattern:
    AveragePool -> Trunc (v2)
    To the FINN op: Div -> QuantAvgPool2d -> Mul

    QuantAvgPool2d truncates with a right shift, i.e. FLOOR rounding. For other
    rounding modes the QuantAvgPool2d outputs the full window sum instead and a
    MultiThreshold with thresholds shared by all channels applies the exact
    Trunc rounding: Div -> QuantAvgPool2d -> MultiThreshold -> Mul
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for node in graph.node:
            node_ind += 1
            if node.op_type == "AveragePool":
                t_node = model.find_direct_successors(node)
                if t_node is not None and len(t_node) == 1 and t_node[0].op_type == "Trunc":
                    t_node = t_node[0]
                    running_node_index = node_ind
                    # Check node for compatibility
                    # Avg pooling node
                    k_s = get_by_name(node.attribute, "kernel_shape")
                    if k_s is None or len(k_s.ints) != 2 or len(set(k_s.ints)) != 1:
                        raise ValueError(
                            "FINN only supports average pooling with " "2D square kernels."
                        )
                    k_s = k_s.ints[0]

                    pads = get_by_name(node.attribute, "pads")
                    if pads is None or len(set(pads.ints)) != 1 or pads.ints[0] != 0:
                        raise ValueError("FINN dosn't support padding for average pooling.")

                    stride = get_by_name(node.attribute, "strides")
                    if stride is None or len(stride.ints) != 2 or len(set(stride.ints)) != 1:
                        raise ValueError(
                            "FINN only supports 2D strides with equal values in " "each direction."
                        )
                    stride = stride.ints[0]

                    # Trunc node
                    rounding_mode = get_by_name(t_node.attribute, "rounding_mode")
                    rounding_mode = (
                        rounding_mode.s.upper() if rounding_mode is not None else b"FLOOR"
                    )
                    for inp in t_node.input[1:]:
                        if model.get_initializer(inp) is None:
                            raise ValueError(
                                f"All inputs of the Trunc node, "
                                f"except the first, must be statically "
                                f"initialized. However, {inp} is not."
                            )
                    zero_pt = model.get_initializer(t_node.input[2])
                    if len(zero_pt.shape) != 0 or zero_pt != 0:
                        raise ValueError(
                            f"Finn only supports 0 as the zero point for "
                            f"the Trunc node, it currently is {zero_pt}."
                        )
                    scale = model.get_initializer(t_node.input[1]).flatten()
                    out_scale = model.get_initializer(t_node.input[4]).flatten()

                    trunc_in_bits = model.get_initializer(t_node.input[3]).flatten()
                    trunc_out_bits = model.get_initializer(t_node.input[5]).flatten()
                    if len(trunc_in_bits.shape) != 1 or len(trunc_out_bits.shape) != 1:
                        raise ValueError(
                            f"Finn only supports scalar bit widths "
                            f"for the Trunc node. The input bit width "
                            f"currently is: {trunc_in_bits}, "
                            f"while the output bit width is: {trunc_out_bits}."
                        )
                    trunc_in_bits = int(trunc_in_bits[0])
                    trunc_out_bits = int(trunc_out_bits[0])
                    if np.round(np.log2(out_scale / scale) != trunc_in_bits - trunc_out_bits):
                        raise ValueError(
                            f"The scale values for the Trunc node are not "
                            f"compatible with the specified bit widths. "
                            f"Input scale: {scale}, output scale: {out_scale}, "
                            f"input bits: {trunc_in_bits}, output bits: {trunc_out_bits}."
                        )

                    # Calculate parameters for the QuantAvgPool2d node,
                    # Calculate input bit width. The Trunc input bit width is sized
                    # for the largest window sum, (2**ibits - 1) * k_s**2, so take
                    # the largest ibits whose window sum still fits. This makes the
                    # QuantAvgPool2d accumulator match the Trunc input bit width,
                    # see get_accum_size() in qonnx's QuantAvgPool2d.
                    ibits = 1
                    while ((2 ** (ibits + 1) - 1) * k_s * k_s).bit_length() <= trunc_in_bits:
                        ibits += 1
                    # Get sign
                    signed = _get_signed_from_upstream(model, t_node)
                    # ToDo: Change this to NHWC,
                    #  when the channels last layout comes around.
                    data_layout = "NCHW"
                    # Range of the window sum. For signed inputs it can exceed the
                    # Trunc input range, since that is sized for the largest unsigned
                    # window sum.
                    k2 = k_s * k_s
                    if signed:
                        sum_min = -(2 ** (ibits - 1)) * k2
                        sum_max = (2 ** (ibits - 1) - 1) * k2
                    else:
                        sum_min, sum_max = 0, (2**ibits - 1) * k2
                    # The right shift of the QuantAvgPool2d reproduces the Trunc only
                    # for FLOOR rounding and if the shifted sum stays within the Trunc
                    # output range, since the Trunc clamps its result. Otherwise the
                    # full window sum is thresholded, which covers any rounding mode
                    # and the clamping.
                    trunc_signed = get_by_name(t_node.attribute, "signed")
                    trunc_signed = trunc_signed.i if trunc_signed is not None else 1
                    trunc_narrow = get_by_name(t_node.attribute, "narrow")
                    trunc_narrow = trunc_narrow.i if trunc_narrow is not None else 0
                    shift = max(trunc_in_bits - trunc_out_bits, 0)
                    shift_clamps = (sum_min >> shift) < min_int(
                        trunc_signed, trunc_narrow, trunc_out_bits
                    ) or (sum_max >> shift) > max_int(trunc_signed, trunc_narrow, trunc_out_bits)
                    round_with_thresholds = shift_clamps or (
                        rounding_mode != b"FLOOR" and trunc_in_bits != trunc_out_bits
                    )

                    # Insert scale nodes, QuantAvgPool2d node and required tensors
                    scale = model.get_initializer(t_node.input[1])
                    # for Trunc v2 update input scale by receptive field
                    scale = (scale * k_s * k_s).astype(scale.dtype)
                    scale_div_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        None,
                    )
                    graph.value_info.append(scale_div_tensor)
                    model.set_initializer(scale_div_tensor.name, scale)

                    act_scale_div_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        None,
                    )
                    graph.value_info.append(act_scale_div_tensor)

                    scale_div_node = helper.make_node(
                        "Div",
                        [node.input[0], scale_div_tensor.name],
                        [act_scale_div_tensor.name],
                    )
                    graph.node.insert(running_node_index, scale_div_node)
                    running_node_index += 1

                    act_scale_mul_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        None,
                    )
                    graph.value_info.append(act_scale_mul_tensor)
                    QuantAvgPool2d_node = helper.make_node(
                        "QuantAvgPool2d",
                        [act_scale_div_tensor.name],
                        [act_scale_mul_tensor.name],
                        domain="qonnx.custom_op.general",
                        stride=stride,
                        kernel=k_s,
                        ibits=ibits,
                        obits=trunc_out_bits,
                        signed=int(signed),
                        data_layout=data_layout,
                    )
                    if round_with_thresholds:
                        # Output the full window sum (no shift) and round it with a
                        # MultiThreshold. The sum is sized for its actual range, and at
                        # least the QuantAvgPool2d accumulator so its shift is zero.
                        pool_inst = getCustomOp(QuantAvgPool2d_node)
                        sum_bits = max(
                            int(sum_max).bit_length(), int(-sum_min - 1).bit_length()
                        ) + int(signed)
                        sum_bits = max(sum_bits, pool_inst.get_accum_size())
                        if sum_bits > MAX_TRUNC_THRESHOLD_SUM_BITS:
                            raise ValueError(
                                f"The Trunc node (rounding_mode {rounding_mode.decode()}) "
                                f"can not be implemented as a right shift and thresholding "
                                f"is only supported for window sums of up to "
                                f"{MAX_TRUNC_THRESHOLD_SUM_BITS} bits (float32 thresholds), "
                                f"but the sum has {sum_bits} bits."
                            )
                        pool_inst.set_nodeattr("obits", sum_bits)
                    graph.node.insert(running_node_index, QuantAvgPool2d_node)
                    running_node_index += 1

                    if round_with_thresholds:
                        sum_dt = DataType[("INT" if signed else "UINT") + str(sum_bits)]
                        thresholds, min_level = _get_trunc_v2_thresholds(model, t_node, sum_dt)
                        thres_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(thres_tensor)
                        model.set_initializer(thres_tensor.name, thresholds)

                        act_thres_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,
                        )
                        graph.value_info.append(act_thres_tensor)
                        out_dt = ("INT" if min_level < 0 else "UINT") + str(trunc_out_bits)
                        thres_node = helper.make_node(
                            "MultiThreshold",
                            [act_scale_mul_tensor.name, thres_tensor.name],
                            [act_thres_tensor.name],
                            domain="qonnx.custom_op.general",
                            out_dtype=out_dt,
                            out_bias=float(min_level),
                            data_layout=data_layout,
                        )
                        graph.node.insert(running_node_index, thres_node)
                        running_node_index += 1
                        act_scale_mul_tensor = act_thres_tensor

                    scale_mul_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        None,
                    )
                    graph.value_info.append(scale_mul_tensor)
                    model.set_initializer(scale_mul_tensor.name, out_scale)

                    scale_mul_node = helper.make_node(
                        "Mul",
                        [act_scale_mul_tensor.name, scale_mul_tensor.name],
                        [t_node.output[0]],
                    )
                    graph.node.insert(running_node_index, scale_mul_node)
                    running_node_index += 1

                    # Remove old nodes
                    graph.node.remove(node)
                    graph.node.remove(t_node)

                    # Recompute shapes and datatypes
                    model = model.transform(InferShapes())
                    model = model.transform(InferDataTypes())

                    return model, True

        return model, False
