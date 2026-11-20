# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The one build step a prepared (pre-rolled) TinyDeiT model needs on top of the stock flow."""

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.assign_ddr_weight_offsets import (
    AssignMemoryOffset,
)
from finn.transformation.fpgadataflow.loop_rolling import make_upstream_lazy


def step_assign_ddr_weight_offsets_for_loops(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Place FINNLoop weights and intermediate frames in DDR (PYNQ shell).

    FINN's own step_assign_ddr_weight_offsets only acts when cfg.mlo is set, but cfg.mlo
    must stay off here: the model is rolled into a FINNLoop beforehand and cfg.mlo=True would
    ask the builder to roll it again.

    Inject after: step_assign_ddr_weight_offsets (it runs before step_hw_codegen, which bakes
    the offsets into the generated RTL).
    """
    for node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getCustomOp(node)
        if loop_inst.get_nodeattr("mem_type") == "":
            loop_inst.set_nodeattr("mem_type", cfg._resolve_mem_type())
    return model.transform(AssignMemoryOffset())


def step_make_loop_inputs_lazy(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Set Eager=0 on every padding/pool node upstream of a FINNLoop.

    LoopRolling does this when it rolls the loop, but the model is rolled beforehand (cfg.mlo
    stays off), so that never runs here. An eager Pad1D in front of the loop emits the next
    frame's leading padding without waiting for input; the loop's input mux takes it for the
    start of a new frame and deadlocks (the throughput test with one frame then never ends).

    Inject after: step_minimize_bit_width (any step before step_hw_codegen works).
    """
    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        make_upstream_lazy(model, loop_node)
    return model


def step_round_loop_thresholds(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Round/clip the per-iteration thresholds of FINNLoop bodies that work on integer inputs.

    RoundAndClipThresholds only sees thresholds that are initializers of the (sub)graph it runs
    on. When the thresholds differ between the rolled layers they become *loop parameters*: the
    body only has a graph input, the stacked ``[iteration, ...]`` values live as initializer of
    the FINNLoop node in the parent graph. Such thresholds stay non-integer, which the RTL
    Thresholding rejects for integer inputs (here: the one behind the attention-times-V MVAU).

    Inject after: step_minimize_bit_width (body input datatypes are final by then).
    """
    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getCustomOp(loop_node)
        body = loop_inst.get_nodeattr("body")
        changed = False
        for i, param in enumerate(loop_node.input[1:]):
            body_tensor = body.graph.input[i + 1].name
            consumer = body.find_consumer(body_tensor)
            if consumer is None or not consumer.op_type.startswith("Thresholding"):
                continue
            if consumer.input[1] != body_tensor:
                continue
            idt = body.get_tensor_datatype(consumer.input[0])
            if not idt.is_integer():
                continue
            thresholds = model.get_initializer(param)
            assert thresholds is not None, f"no loop parameter values for {param}"
            rounded = np.clip(np.ceil(thresholds), idt.min(), idt.max() + 1).astype(np.float32)
            lo, hi = int(rounded.min()), int(rounded.max())
            tdt = DataType.get_smallest_possible(min(lo, -hi - 1) if lo < 0 else hi)
            model.set_initializer(param, rounded)
            model.set_tensor_datatype(param, tdt)
            body.set_tensor_datatype(body_tensor, tdt)
            getCustomOp(consumer).set_nodeattr("weightDataType", tdt.name)
            print(
                f"{loop_node.name}/{consumer.name}: rounded loop thresholds for {idt} input -> "
                f"{tdt} (range {lo}..{hi})"
            )
            changed = True
        if changed:
            loop_inst.set_nodeattr("body", body.graph)
    return model
