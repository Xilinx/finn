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
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name


def _get_signed_from_upstream(model, trunc_node):
    """
    Find out what the sign of the input to the trunc node is,
    by looking at the upstream nodes.
    """
    node = trunc_node
    # Check if the input of this node already has a FINN datatype
    signed = None
    inp_dt = model.get_tensor_datatype(node.input[0])
    if inp_dt is not None and inp_dt is not DataType["FLOAT32"]:
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
            if out_dt is not None and out_dt is not DataType["FLOAT32"]:
                signed = out_dt.signed()
                break
            # Special cases where the node has an internal or intrinsic datatype.
            if next_node.op_type == "MultiThreshold":
                mt_inst = getCustomOp(next_node)
                out_dt = DataType[mt_inst.get_nodeattr("out_dtype")]
                if out_dt is not None and out_dt is not DataType["FLOAT32"]:
                    signed = out_dt.signed()
                    break
            if next_node.op_type == "BipolarQuant":
                signed = True
                break
            if next_node.op_type == "Quant":
                q_inst = getCustomOp(next_node)
                out_dt = q_inst.get_integer_datatype(model)
                if out_dt is not None and out_dt is not DataType["FLOAT32"]:
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
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "AveragePool":
                mul_node = model.find_direct_successors(n)
                if (
                    mul_node is not None
                    and len(mul_node) == 1
                    and mul_node[0].op_type == "Mul"
                ):
                    mul_node = mul_node[0]
                    t_node = model.find_direct_successors(mul_node)
                    if (
                        t_node is not None
                        and len(t_node) == 1
                        and t_node[0].op_type == "Trunc"
                    ):
                        t_node = t_node[0]
                        running_node_index = node_ind
                        # Check node for compatibility
                        # Avg pooling node
                        k_s = get_by_name(n.attribute, "kernel_shape")
                        if k_s is None or len(k_s.ints) != 2 or len(set(k_s.ints)) != 1:
                            raise ValueError(
                                "FINN only supports average pooling with "
                                "2D square kernels."
                            )
                        k_s = k_s.ints[0]

                        pads = get_by_name(n.attribute, "pads")
                        if (
                            pads is None
                            or len(set(pads.ints)) != 1
                            or pads.ints[0] != 0
                        ):
                            raise ValueError(
                                "FINN dosn't support padding for average pooling."
                            )

                        stride = get_by_name(n.attribute, "strides")
                        if (
                            stride is None
                            or len(stride.ints) != 2
                            or len(set(stride.ints)) != 1
                        ):
                            raise ValueError(
                                "FINN only supports 2D strides with equal values in "
                                "each direction."
                            )
                        stride = stride.ints[0]

                        # Mul node
                        mul_val = model.get_initializer(mul_node.input[1])
                        if (
                            mul_val is None
                            or len(mul_val.shape) != 0
                            or mul_val != k_s * k_s
                        ):
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
                        if rounding_mode is None or rounding_mode.s != b"FLOOR":
                            raise ValueError(
                                "The Trunc node must have the rounding_mode "
                                "set to 'FLOOR'."
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
                        trunc_out_bits = model.get_initializer(
                            t_node.input[4]
                        ).flatten()
                        if (
                            len(trunc_in_bits.shape) != 1
                            or len(trunc_out_bits.shape) != 1
                        ):
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
                        ibits = math.floor(
                            math.log(2 ** trunc_in_bits / (k_s * k_s), 2)
                        )
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
                            domain="finn.custom_op.general",
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
