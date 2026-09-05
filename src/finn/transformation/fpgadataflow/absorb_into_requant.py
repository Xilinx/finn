# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation


class AbsorbElementwiseOpsIntoRequant(Transformation):
    """Absorb preceding ElementwiseMul and/or ElementwiseAdd into Requant.

    Detects patterns where ElementwiseMul and/or ElementwiseAdd operations
    (with const operand) precede a Requant node and absorbs them into the
    Requant's scale and bias parameters.

    Requant computes: clip(round(x * scale + bias), min, max)

    Supported patterns:
    - ElementwiseMul -> Requant: absorbed into scale
    - ElementwiseAdd -> Requant: absorbed into bias
    - ElementwiseMul -> ElementwiseAdd -> Requant: scale and bias

    This transformation is run at the end of step_convert_to_hw after all HW
    layers have been inferred.

    The absorbed operations are removed from the graph and their constant
    parameters are folded into the Requant's scale/bias initializers.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for node in list(graph.node):  # Use list() to allow modification during iteration
            if node.op_type != "Requant":
                continue

            node_inst = getCustomOp(node)

            # Get current scale and bias
            current_scale = node_inst.get_scale(model)
            current_bias = node_inst.get_bias(model)

            # Find predecessor
            requant_input = node.input[0]
            predecessor = model.find_producer(requant_input)
            if predecessor is None:
                continue

            absorbed_scale = None
            absorbed_bias = None
            new_input = None
            nodes_to_remove = []

            # Pattern: ElementwiseMul -> Requant
            if predecessor.op_type == "ElementwiseMul":
                mul_inst = getCustomOp(predecessor)
                mul_const, mul_input = self._get_const_and_input(model, predecessor, mul_inst)
                if mul_const is None:
                    continue

                absorbed_scale = mul_const.flatten()
                new_input = mul_input
                nodes_to_remove.append(predecessor)

            # Pattern: ElementwiseAdd -> Requant
            # Also check for: ElementwiseMul -> ElementwiseAdd -> Requant
            elif predecessor.op_type == "ElementwiseAdd":
                add_inst = getCustomOp(predecessor)
                add_const, add_input = self._get_const_and_input(model, predecessor, add_inst)
                if add_const is None:
                    continue

                absorbed_bias = add_const.flatten()
                new_input = add_input
                nodes_to_remove.append(predecessor)

                # Check for ElementwiseMul -> ElementwiseAdd -> Requant
                add_pred = model.find_producer(new_input)
                if add_pred is not None and add_pred.op_type == "ElementwiseMul":
                    mul_inst = getCustomOp(add_pred)
                    mul_const, mul_input = self._get_const_and_input(model, add_pred, mul_inst)
                    if mul_const is not None:
                        # Check that ElementwiseMul has only one consumer
                        mul_consumers = model.find_consumers(add_pred.output[0])
                        if len(mul_consumers) == 1:
                            # output = input * mul + add
                            absorbed_scale = mul_const.flatten()
                            new_input = mul_input
                            nodes_to_remove.append(add_pred)

            else:
                # Not a supported predecessor pattern
                continue

            if new_input is None:
                continue

            # Check that nodes to remove have only one consumer
            can_remove = True
            for n in nodes_to_remove:
                consumers = model.find_consumers(n.output[0])
                if len(consumers) != 1:
                    can_remove = False
                    break
            if not can_remove:
                continue

            # Compute new scale and bias
            # Requant computes: clip(round(x * scale + bias), min, max)
            # Use numpy broadcasting - result will be per-channel only if any input is per-channel
            if absorbed_scale is not None:
                new_scale = (current_scale * absorbed_scale).astype(np.float32)
            else:
                new_scale = current_scale

            if absorbed_bias is not None:
                # bias is applied after scale, so absorbed_bias needs to be scaled
                new_bias = (current_bias + absorbed_bias * current_scale).astype(np.float32)
            else:
                new_bias = current_bias

            # Update Requant initializers
            scale_name = node.input[1]
            bias_name = node.input[2]
            model.set_initializer(scale_name, new_scale.astype(np.float32))
            model.set_initializer(bias_name, new_bias.astype(np.float32))

            # Update Requant input and input datatype
            node.input[0] = new_input
            new_input_dtype = model.get_tensor_datatype(new_input)
            if new_input_dtype is not None:
                node_inst.set_nodeattr("inputDataType", new_input_dtype.name)

            # Remove absorbed nodes
            for n in nodes_to_remove:
                graph.node.remove(n)

            graph_modified = True

        return (model, graph_modified)

    def _get_const_and_input(self, model, node, node_inst):
        """Get constant operand and streaming input from ElementwiseBinary node.

        Returns:
            Tuple of (const_value, input_name) or (None, None) if no const operand.
        """
        lhs_style = node_inst.get_nodeattr("lhs_style")
        rhs_style = node_inst.get_nodeattr("rhs_style")

        if lhs_style == "const" and rhs_style == "input":
            const_value = model.get_initializer(node.input[0])
            input_name = node.input[1]
        elif rhs_style == "const" and lhs_style == "input":
            const_value = model.get_initializer(node.input[1])
            input_name = node.input[0]
        else:
            # Both are streaming or both are const - not supported
            return None, None

        return const_value, input_name
