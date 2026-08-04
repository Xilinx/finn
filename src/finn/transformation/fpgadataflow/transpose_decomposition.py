############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
import numpy as np
from collections import deque
from onnx import helper
from operator import itemgetter
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import List, Optional, Tuple


def shuffle_perfect_loopnest_coeffs(shape: tuple[int], perm: tuple[int]) -> tuple[int]:
    """
    Given an input shape and permutation matrix calculate the
    coefficients for the perfect loop nest for HLS generation.
    """
    adjusted_shape = list(shape) + [1]
    input_coeffs = [np.prod(adjusted_shape[i + 1 :]) for i in range(len(shape))]
    out_coeffs = [input_coeffs[i] for i in perm]
    return tuple(out_coeffs)


def apply_inner_shuffle_operation(
    perm: List[int], shape: List[int] = None, simd: int = 1
) -> List[int]:
    """
    Apply inner_shuffle operation: swap the last two positions
    (..., a, b) -> (..., b, a)
    """
    if len(perm) < 2:
        return perm[:]

    # Check SIMD constraint for inner_shuffle: SIMD must divide the final
    # innermost dimension (J)
    if shape is not None and simd > 1:
        j_dim = shape[-1]  # Final innermost dimension
        if j_dim % simd != 0:
            return perm[:]  # Return unchanged if constraint not satisfied

    result = perm[:]
    result[-2], result[-1] = result[-1], result[-2]
    return result


def apply_outer_shuffle_operation(
    perm: List[int], i: int, j: int, shape: List[int] = None, simd: int = 1
) -> Optional[List[int]]:
    """
    Apply outer_shuffle operation: swap positions i and j
    Constraint: cannot move the very last dimension
    """
    n = len(perm)
    if n < 2:
        return None

    # Check constraints for outer_shuffle operation - cannot move the very
    # last dimension
    if i == n - 1 or j == n - 1:
        return None

    if i == j or i < 0 or j < 0 or i >= n or j >= n:
        return None

    # Check SIMD constraint for outer_shuffle: SIMD must divide the
    # innermost dimension that cannot move
    if shape is not None and simd > 1:
        innermost_dim = shape[-1]  # The innermost dimension that cannot move
        if innermost_dim % simd != 0:
            return None

    result = perm[:]
    result[i], result[j] = result[j], result[i]
    return result


def get_all_possible_moves(
    perm: List[int], shape: List[int] = None, simd: int = 1
) -> List[Tuple[List[int], str, Optional[Tuple[int, int]]]]:
    """
    Get all possible moves from current permutation.
    Returns list of (new_permutation, operation_type, operation_params) tuples.

    Each outer_shuffle move represents a single pairwise swap that doesn't
    involve the last dimension. Complex permutations are built by chaining
    multiple such operations.

    operation_type is either 'inner_shuffle' or 'outer_shuffle'
    operation_params is None for inner_shuffle, (i, j) for outer_shuffle
    """
    moves = []
    n = len(perm)

    # Try inner_shuffle operation
    new_perm = apply_inner_shuffle_operation(perm, shape, simd)
    if new_perm != perm:
        moves.append((new_perm, "inner_shuffle", None))

    # Try all valid outer_shuffle operations
    for i in range(n):
        for j in range(i + 1, n):
            new_perm = apply_outer_shuffle_operation(perm, i, j, shape, simd)
            if new_perm is not None and new_perm != perm:
                moves.append((new_perm, "outer_shuffle", (i, j)))

    return moves


def is_valid_hardware_permutation(perm_array: List[int]) -> bool:
    """
    Check if a permutation array represents a valid hardware operation.
    Valid operations are:
    - inner_shuffle: swap last two elements
    - outer_shuffle: any permutation that doesn't move the last element
    """
    n = len(perm_array)
    if n < 2:
        return True

    identity = list(range(n))
    if perm_array == identity:
        return True  # Identity is always valid

    expected_inner_shuffle = identity[:]
    expected_inner_shuffle[-2], expected_inner_shuffle[-1] = (
        expected_inner_shuffle[-1],
        expected_inner_shuffle[-2],
    )
    if perm_array == expected_inner_shuffle:
        return True

    diff_count = sum(1 for i in range(n) if perm_array[i] != identity[i])
    if diff_count == 2:  # Simple two-element swap (one type of outer_shuffle)
        diff_positions = [i for i in range(n) if perm_array[i] != identity[i]]
        if len(diff_positions) == 2:
            pos1, pos2 = diff_positions
            if pos1 != n - 1 and pos2 != n - 1:
                if perm_array[pos1] == pos2 and perm_array[pos2] == pos1:
                    return True

    return False


def find_minimal_operation_sequence(
    start_perm: List[int],
    target_perm: List[int],
    shape: List[int] = None,
    simd: int = 1,
) -> Optional[List[Tuple[str, Optional[Tuple[int, int]]]]]:
    """
    Find minimal sequence of operations to transform start_perm into target_perm.
    Uses BFS to find shortest path, ensuring all intermediate permutations
    are hardware-valid.
    Returns list of (operation_type, operation_params) tuples.


    TODO: We want this to be cost based and include a buffer size cost model.
    """
    if start_perm == target_perm:
        return []

    queue = deque([(start_perm, [])])
    visited = {tuple(start_perm)}

    while queue:
        current_perm, operations = queue.popleft()

        for next_perm, op_type, op_params in get_all_possible_moves(current_perm, shape, simd):
            test_operations = operations + [(op_type, op_params)]
            test_perms = convert_operations_to_permutations(
                list(range(len(start_perm))), test_operations, shape, simd
            )

            if not is_valid_hardware_permutation(test_perms[-1]):
                continue

            if next_perm == target_perm:
                return test_operations

            next_tuple = tuple(next_perm)
            if next_tuple not in visited:
                visited.add(next_tuple)
                queue.append((next_perm, test_operations))

    return None


def convert_operations_to_permutations(
    start_perm: List[int],
    operations: List[Tuple[str, Optional[Tuple[int, int]]]],
    shape: List[int] = None,
    simd: int = 1,
) -> List[List[int]]:
    """
    Convert a sequence of operations to a list of permutation arrays.
    Each permutation represents the transformation for that step.
    """
    current_perm = start_perm[:]
    permutations = []

    for op_type, op_params in operations:
        if op_type == "inner_shuffle":
            new_perm = apply_inner_shuffle_operation(current_perm, shape, simd)
        elif op_type == "outer_shuffle" and op_params is not None:
            i, j = op_params
            new_perm = apply_outer_shuffle_operation(current_perm, i, j, shape, simd)
            if new_perm is None:
                raise RuntimeError(f"Invalid outer_shuffle operation: ({i}, {j})")
        else:
            raise RuntimeError(f"Unknown operation: {op_type}")

        perm_array = [0] * len(current_perm)
        for new_idx, val in enumerate(new_perm):
            old_idx = current_perm.index(val)
            perm_array[new_idx] = old_idx

        permutations.append(perm_array)
        current_perm = new_perm

    return permutations


def can_be_single_operation(
    target_perm: List[int], shape: List[int] = None, simd: int = 1
) -> Optional[Tuple[str, Optional[Tuple[int, int]]]]:
    """
    Check if the target permutation can be achieved with a single operation.
    i.e. no decomposition is required.
    Returns (operation_type, operation_params) or None if not possible.
    """
    n = len(target_perm)
    start_perm = list(range(n))

    if target_perm == start_perm:
        return None

    # Check if it's a simple inner_shuffle operation (swap last two)
    if n >= 2:
        expected_inner_shuffle = apply_inner_shuffle_operation(start_perm, shape, simd)
        if target_perm == expected_inner_shuffle:
            return ("inner_shuffle", None)

    # Check if it's a simple outer_shuffle operation (any permutation
    # that doesn't move the last position)
    for i in range(n):
        for j in range(i + 1, n):
            expected_outer_shuffle = apply_outer_shuffle_operation(start_perm, i, j, shape, simd)
            if expected_outer_shuffle is not None and target_perm == expected_outer_shuffle:
                return ("outer_shuffle", (i, j))

    return None


def decompose_transpose_with_constraints(
    target_perm: List[int], shape: List[int] = None, simd: int = 1
) -> Tuple[List[List[int]], List[str]]:
    """
    Decompose a target permutation into a sequence of hardware-constrained
    operations.

    inner_shuffle: swaps the last two dimensions
    outer_shuffle: can implement any permutation that doesn't move the last
                   dimension (may require multiple steps)

    Returns (permutations, operation_types).
    - permutations: list of permutation arrays for each step
    - operation_types: list of operation types ('inner_shuffle' or
      'outer_shuffle') for each step
    """
    n = len(target_perm)
    start_perm = list(range(n))

    if target_perm == start_perm:
        return [], []  # Identity permutation

    # First check if this can be done with a single operation
    single_op = can_be_single_operation(target_perm, shape, simd)
    if single_op is not None:
        op_type, op_params = single_op
        # Create the permutation array for this single operation
        permutations = convert_operations_to_permutations(start_perm, [single_op], shape, simd)
        return permutations, [op_type]

    # If not a single operation, find minimal sequence using BFS
    operations = find_minimal_operation_sequence(start_perm, target_perm, shape, simd)

    if operations is None:
        raise RuntimeError(f"No solution found for permutation: {target_perm}")

    if len(operations) == 0:
        return [], []  # Identity permutation

    # Convert operations to permutation arrays
    permutations = convert_operations_to_permutations(start_perm, operations, shape, simd)
    operation_types = [op[0] for op in operations]

    return permutations, operation_types


class ShuffleDecomposition(Transformation):
    """
    Transformation that decomposes Shuffle nodes into
    a chain of Shuffle ops that can map to InnerShuffle
    and OuterShuffle nodes.
    """

    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self._name_counter = 0

    def _unique(self, base):
        self._name_counter += 1
        return f"{base}_{self._name_counter}"

    def get_perm(self, node) -> List[int]:
        for a in node.attribute:
            if a.name == "perm":
                return list(a.ints)
        raise RuntimeError("Unable to determine the permutations from the Transpose node")

    def apply(self, model):
        g = model.graph
        original_nodes = list(g.node)

        for node in original_nodes:
            if node.op_type != "Shuffle":
                continue

            perm = self.get_perm(node)
            f_inst = getCustomOp(node)
            orig_in_shape = f_inst.get_nodeattr("in_shape")
            in_shape = orig_in_shape
            transpose_in_shape = f_inst.get_nodeattr("transpose_in_shape")
            simd = f_inst.get_nodeattr("SIMD")

            try:
                P_list, operation_types = decompose_transpose_with_constraints(
                    perm, transpose_in_shape, simd
                )
                if len(P_list) == 0:
                    print("\tNo swaps necessary (identity permutation).")
                    continue
            except RuntimeError as e:
                print(f"\tSkipping node {node.name}: {e}")
                continue
            orig_input = list(node.input)
            orig_output = list(node.output)

            if len(orig_input) != 1 or len(orig_output) != 1:
                # Transpose usually has one input and one output; if not,
                # skip replacement
                # TODO: Should this raise an exception? Probably need to be handled.
                print(f"\tSkipping node {node.name}: unexpected number of inputs/outputs.")
                continue

            prev_tensor = orig_input[0]
            new_nodes = []
            orig_out_shape = f_inst.get_nodeattr("out_shape")

            # Create decomposed transposes using hardware-constrained operations
            for step_idx, (P, op_type) in enumerate(zip(P_list, operation_types), start=1):
                step_name = self._unique(f"{node.name}_{op_type}_step{step_idx}")
                out_shape = itemgetter(*P)(transpose_in_shape)
                if step_idx < len(P_list):
                    out_tensor = self._unique(f"{node.output[0]}_step{step_idx}")
                    out_reshaped = out_shape
                else:
                    out_tensor = orig_output[0]
                    out_reshaped = orig_out_shape

                if step_idx == 1:
                    in_shape = orig_in_shape
                else:
                    in_shape = transpose_in_shape

                perm_attr = helper.make_attribute("perm", P)
                transpose_node = helper.make_node(
                    op_type="Shuffle",
                    domain="finn.custom_op.fpgadataflow",
                    inputs=[prev_tensor],
                    outputs=[out_tensor],
                    in_shape=in_shape,
                    transpose_in_shape=transpose_in_shape,
                    transpose_out_shape=out_shape,
                    out_shape=out_reshaped,
                    SIMD=f_inst.get_nodeattr("SIMD"),
                    data_type=f_inst.get_nodeattr("data_type"),
                    name=step_name,
                    original_node_name=node.name,  # Track original shuffle name
                    original_simd=f_inst.get_nodeattr("SIMD"),  # Track original SIMD
                )
                transpose_node.attribute.extend([perm_attr])
                new_nodes.append(transpose_node)
                prev_tensor = out_tensor
                transpose_in_shape = out_shape

            for nnode in new_nodes:
                g.node.append(nnode)

            try:
                g.node.remove(node)
            except ValueError:
                for idx, gn in enumerate(list(g.node)):
                    if gn.name == node.name:
                        del g.node[idx]
                        break

        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        return model, False


def _is_inner_shuffle(perm, shape):
    """
    Check if the permutation represents a streaming InnerShuffle case.
    A streaming InnerShuffle is only possible when only the last two dimensions
    are swapped, regardless of how many outer dimensions there are.
    """
    if len(perm) < 2 or len(shape) < 2:
        return False

    # Check if last two dimensions are swapped while others stay in order
    expected_perm = list(range(len(perm) - 2)) + [len(perm) - 1, len(perm) - 2]
    return perm == expected_perm


class InferInnerOuterShuffles(Transformation):
    """
    Infers Inner and Outer Shuffles from Shuffle operators.
    This should run after the ShuffleDecomposition transformation.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Shuffle":  # should we also check for fpgadataflow here?
                to_remove = [node]
                new_in_tensor = node.input[0]
                new_out_tensor = node.output[0]  # What if a transpose is going to multiple sinks?
                f_inst = getCustomOp(node)
                in_shape = f_inst.get_nodeattr("in_shape")
                in_reshaped = f_inst.get_nodeattr("transpose_in_shape")
                out_shape = f_inst.get_nodeattr("out_shape")
                out_reshaped = f_inst.get_nodeattr("transpose_out_shape")
                data_type = f_inst.get_nodeattr("data_type")
                perm = f_inst.get_nodeattr("perm")
                simd = f_inst.get_nodeattr("SIMD")

                if _is_inner_shuffle(perm, in_shape):
                    # Get original node name if it exists, otherwise use current node name
                    try:
                        original_name = f_inst.get_nodeattr("original_node_name") or node.name
                        original_simd = f_inst.get_nodeattr("original_simd") or simd
                    except (AttributeError, KeyError):
                        original_name = node.name
                        original_simd = simd
                    new_node = helper.make_node(
                        "InnerShuffle",
                        [new_in_tensor],
                        [new_out_tensor],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        in_shape=in_shape,
                        data_type=data_type,
                        perm=perm,
                        name=f"InnerShuffle_{node.name}",
                        original_node_name=original_name,  # Preserve original shuffle name
                        original_simd=original_simd,  # Preserve original SIMD
                        SIMD=simd,
                        I=in_shape[-2],  # Second to last dim
                        J=in_shape[-1],  # Last dim
                    )
                else:
                    # Get original node name if it exists, otherwise use current node name
                    try:
                        original_name = f_inst.get_nodeattr("original_node_name") or node.name
                        original_simd = f_inst.get_nodeattr("original_simd") or simd
                    except (AttributeError, KeyError):
                        original_name = node.name
                        original_simd = simd
                    new_node = helper.make_node(
                        "OuterShuffle",
                        [new_in_tensor],
                        [new_out_tensor],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        in_shape=in_shape,
                        transpose_in_shape=in_reshaped,
                        perm=perm,
                        out_shape=out_shape,
                        transpose_out_shape=out_reshaped,
                        data_type=data_type,
                        name=f"OuterShuffle_{node.name}",
                        original_node_name=original_name,  # Preserve original shuffle name
                        original_simd=original_simd,  # Preserve original SIMD
                        loop_coeffs=shuffle_perfect_loopnest_coeffs(shape=in_reshaped, perm=perm),
                        SIMD=simd,
                        NumChannels=in_reshaped[-1],
                        cpp_interface="hls_vector",
                        hls_style="freerunning",
                    )
                graph.node.insert(node_ind, new_node)

                for i in to_remove:
                    graph.node.remove(i)
                    graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)
