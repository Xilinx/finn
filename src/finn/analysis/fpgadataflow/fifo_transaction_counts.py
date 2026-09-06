# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import onnx
from qonnx.custom_op.registry import getCustomOp


def _has_fifos(model):
    if any(n.op_type.startswith("StreamingFIFO") for n in model.graph.node):
        return True
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                if _has_fifos(model.make_subgraph_modelwrapper(attr.g)):
                    return True
    return False


def fifo_transaction_counts(model, apply_to_subgraphs=False):
    """Return the expected number of stream transactions each FIFO should carry.

    Requires a model that already has FIFOs inserted. Returns a dict mapping FIFO
    name -> expected transaction count, where the count is the per-frame
    transaction count (np.prod(folded_shape[:-1])) multiplied by the iteration
    count of any enclosing FINNLoop.

    With apply_to_subgraphs=True the pass descends into subgraphs (e.g. FINNLoop
    bodies). FIFOs are keyed by their node name directly (body FIFOs should already
    be uniquely named via GiveUniqueNodeNames with the FINNLoop prefix).
    """
    assert _has_fifos(model), (
        "fifo_transaction_counts requires a model with FIFOs inserted "
        "(run InsertFIFO / set_fifo_depths first)"
    )
    return _count(model, apply_to_subgraphs)


def _count(model, apply_to_subgraphs):
    ret = {}
    for node in model.graph.node:
        if node.op_type.startswith("StreamingFIFO"):
            folded_shape = getCustomOp(node).get_nodeattr("folded_shape")
            per_frame = int(np.prod(folded_shape[:-1])) if len(folded_shape) > 1 else 1
            ret[node.name] = per_frame
    if apply_to_subgraphs:
        for node in model.graph.node:
            # FINNLoop re-streams its body once per iteration; generic subgraphs run once
            multiplier = (
                getCustomOp(node).get_nodeattr("iteration") if node.op_type == "FINNLoop" else 1
            )
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph = model.make_subgraph_modelwrapper(attr.g)
                    for k, v in _count(subgraph, apply_to_subgraphs=True).items():
                        ret[k] = v * multiplier
    return ret
