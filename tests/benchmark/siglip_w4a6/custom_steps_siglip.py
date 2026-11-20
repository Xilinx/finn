# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The one build step a prepared (pre-rolled) SigLIP model needs on top of the stock flow."""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.assign_ddr_weight_offsets import (
    AssignMemoryOffset,
)

URAM_FIFO_NAMES = ("StreamingFIFO_rtl_28", "StreamingFIFO_rtl_29")
URAM_FIFO_STREAM_WIDTH = 2048


def step_force_uram_attention_fifos(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Force ``ram_style=ultra`` (URAM) on the two wide attention FIFOs in every FINNLoop body.

    FIFO names only exist once loop-body FIFO sizing has run, and the RAM style is baked into the
    FIFO's Verilog by PrepareIP, so this has to run after step_loop_body_set_fifo_depths and
    before step_hw_codegen.

    Inject before: step_hw_codegen.
    """
    forced = []
    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getCustomOp(loop_node)
        body = loop_inst.get_nodeattr("body")
        by_name = {n.name: n for n in body.graph.node}
        for short in URAM_FIFO_NAMES:
            full = f"{loop_node.name}_{short}"
            assert full in by_name, (
                f"{full} not found in the {loop_node.name} body: loop-body FIFO numbering changed, "
                "re-check which FIFOs are the wide attention streams before forcing URAM"
            )
            fifo = getCustomOp(by_name[full])
            width = fifo.get_instream_width()
            assert width == URAM_FIFO_STREAM_WIDTH, (
                f"{full} is {width} bits wide, expected {URAM_FIFO_STREAM_WIDTH}: "
                "FIFO numbering changed, refusing to force URAM on the wrong FIFO"
            )
            fifo.set_nodeattr("ram_style", "ultra")
            forced.append((full, width, fifo.get_nodeattr("depth")))
        # write the modified body back so the change lands in the serialised FINNLoop node
        loop_inst.set_nodeattr("body", body.graph)
    for full, width, depth in forced:
        print(
            f"step_force_uram_attention_fifos: {full} width={width} depth={depth} "
            "-> ram_style=ultra"
        )
    return model


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
