# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Custom build steps for the pre-rolled SigLIP model."""

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.assign_ddr_weight_offsets import (
    AssignMemoryOffset,
)


def step_remove_patch_embed_shuffle_roundtrip(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Patch embedding: Shuffle(NHWC->NCHW) -> Mul -> Add -> Shuffle -> Add.

    The two Shuffles compose to a plain reshape, so the stream order is unchanged. Drop them and
    run Mul/Add/Add channel-last, i.e. directly in the producer's (1, 14, 14, 768) layout.
    """
    shuffle_a, shuffle_b = model.get_nodes_by_op_type("Shuffle")
    mul = model.find_consumer(shuffle_a.output[0])
    add = model.find_consumer(mul.output[0])
    pos_add = model.find_consumer(shuffle_b.output[0])
    layout = list(getCustomOp(shuffle_a).get_nodeattr("in_shape"))

    stale = [shuffle_a.output[0], add.output[0]]
    mul.input[0] = shuffle_a.input[0]
    add.output[0] = shuffle_b.output[0]
    model.graph.node.remove(shuffle_a)
    model.graph.node.remove(shuffle_b)
    for name in stale:
        model.graph.value_info.remove(model.get_tensor_valueinfo(name))

    for node in (mul, add):
        model.set_initializer(node.input[1], model.get_initializer(node.input[1]).reshape(-1))
        getCustomOp(node).set_nodeattr("rhs_shape", layout[-1:])
    model.set_initializer(pos_add.input[1], model.get_initializer(pos_add.input[1]).reshape(layout))
    getCustomOp(pos_add).set_nodeattr("rhs_shape", layout)
    for node in (mul, add, pos_add):
        inst = getCustomOp(node)
        inst.set_nodeattr("lhs_shape", layout)
        inst.set_nodeattr("out_shape", layout)
        model.set_tensor_shape(node.output[0], layout)
    return model.transform(InferShapes())


def step_threshold_before_attention_shuffles(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Attention Q/K/V: rewrite ``Shuffle -> Thresholding`` as ``Thresholding -> Shuffle``. The
    Thresholding uses one threshold row for all channels, so it commutes with the Shuffle
    and the FIFOs/DWCs around it carry INT6 instead of FLOAT32.
    """
    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop = getCustomOp(loop_node)
        body = loop.get_nodeattr("body")
        for shuffle in body.get_nodes_by_op_type("Shuffle"):
            thr = body.find_consumer(shuffle.output[0])
            shuf_op, thr_op = getCustomOp(shuffle), getCustomOp(thr)
            new_dt = DataType[thr_op.get_nodeattr("outputDataType")]
            # the last Shuffle (head merge) already carries INT6: leave it
            if new_dt.bitwidth() >= DataType[shuf_op.get_nodeattr("data_type")].bitwidth():
                continue

            # x -> Thresholding -> mid -> Shuffle -> y, reusing the tensor between the two nodes
            in_shape = list(shuf_op.get_nodeattr("in_shape"))
            mid = shuffle.output[0]
            thr.input[0] = shuffle.input[0]
            shuffle.output[0] = thr.output[0]
            thr.output[0] = mid
            shuffle.input[0] = mid
            body.set_tensor_shape(mid, in_shape)
            body.set_tensor_datatype(mid, new_dt)
            thr_op.set_nodeattr("NumChannels", in_shape[-1])
            thr_op.set_nodeattr("numInputVectors", in_shape[:-1])
            shuf_op.set_nodeattr("data_type", new_dt.name)
        body.cleanup()  # topological sort
        loop.set_nodeattr("body", body.graph)
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
