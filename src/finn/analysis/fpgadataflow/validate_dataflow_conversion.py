# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Analysis pass to validate that model has been properly converted to fpgadataflow layers."""

from finn.util.fpgadataflow import is_fpgadataflow_node


def validate_dataflow_conversion(model):
    """Validate that model has been properly converted to dataflow layers.

    Checks that either:
    1. All layers are fpgadataflow layers (ideal case), OR
    2. Fpgadataflow layers form a contiguous block in the middle of the model,
       with only non-dataflow layers on the outside (partition case)

    Returns a dictionary with validation results:
    - 'valid': bool indicating if validation passed
    - 'message': str with validation status message
    - 'unconverted_layers': list of (index, name, op_type) tuples for non-dataflow layers
    - 'dataflow_block': tuple (first_index, last_index) if dataflow forms a block, else None

    Example usage in transformation:
        result = model.analysis(validate_dataflow_conversion)
        if not result['valid']:
            raise AssertionError(result['message'])
    """
    nodes = model.graph.node
    fpgadataflow_nodes = []
    non_fpgadataflow_nodes = []

    for i, node in enumerate(nodes):
        if is_fpgadataflow_node(node):
            fpgadataflow_nodes.append((i, node.name, node.op_type))
        else:
            non_fpgadataflow_nodes.append((i, node.name, node.op_type))

    # Case 1: All nodes are fpgadataflow (ideal)
    if len(non_fpgadataflow_nodes) == 0:
        return {
            "valid": True,
            "message": "Dataflow conversion validation: All layers are fpgadataflow layers",
            "unconverted_layers": [],
            "dataflow_block": None,
        }

    # Case 2: Check if fpgadataflow nodes form contiguous block
    if len(fpgadataflow_nodes) > 0:
        dataflow_indices = [i for i, _, _ in fpgadataflow_nodes]
        first_dataflow = min(dataflow_indices)
        last_dataflow = max(dataflow_indices)

        # Check all indices between first and last are dataflow
        for i in range(first_dataflow, last_dataflow + 1):
            node = nodes[i]
            if not is_fpgadataflow_node(node):
                # Found non-dataflow layer inside dataflow block
                unconverted_str = "\n".join(
                    [
                        f"  [{idx}] {name} (op_type: {op})"
                        for idx, name, op in non_fpgadataflow_nodes
                    ]
                )
                return {
                    "valid": False,
                    "message": (
                        "Non-contiguous dataflow block detected.\n"
                        f"Layer '{node.name}' (op_type: {node.op_type}) at position {i} "
                        "is not a fpgadataflow layer but is between dataflow layers.\n"
                        f"Dataflow block spans positions {first_dataflow} to {last_dataflow}.\n"
                        f"Unconverted layers:\n{unconverted_str}"
                    ),
                    "unconverted_layers": non_fpgadataflow_nodes,
                    "dataflow_block": (first_dataflow, last_dataflow),
                }

        # Valid: fpgadataflow block in middle
        return {
            "valid": True,
            "message": (
                "Dataflow conversion validation: Fpgadataflow layers form contiguous block "
                f"(positions {first_dataflow}-{last_dataflow})"
            ),
            "unconverted_layers": non_fpgadataflow_nodes,
            "dataflow_block": (first_dataflow, last_dataflow),
        }

    # Case 3: No fpgadataflow layers at all
    unconverted_str = "\n".join(
        [f"  [{idx}] {name} (op_type: {op})" for idx, name, op in non_fpgadataflow_nodes]
    )
    return {
        "valid": False,
        "message": (
            "No fpgadataflow layers found in model.\n"
            f"All layers remain unconverted:\n{unconverted_str}"
        ),
        "unconverted_layers": non_fpgadataflow_nodes,
        "dataflow_block": None,
    }
