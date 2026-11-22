# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Analysis pass checking that connected hardware streams have equal widths and datatypes."""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.floorplan import get_attached_iodma
from finn.util.fpgadataflow import is_fpgadataflow_node


def _is_untyped_stream_node(model: ModelWrapper, node) -> bool:
    """IODMAs and the DWCs InsertIODMA attaches to them move raw bytes (UINT8 or the
    weight type) whatever the datatype of the stream, and TLastMarker has no datatype;
    only the width of their streams is meaningful."""
    return (
        node.op_type in ("IODMA_hls", "TLastMarker_rtl")
        or get_attached_iodma(model, node) is not None
    )


def stream_compatibility(model: ModelWrapper) -> dict:
    """Checks every stream between two hardware nodes: the padded output stream width
    of the producer must equal the padded input stream width of the consumer, and the
    output datatype of the producer the input datatype of the consumer.

    Streams of width 0 (not exposed as a stream) are skipped, and so are the
    datatypes of streams from or to an IODMA, the DWC attached to it, or a
    TLastMarker.

    Returns a dictionary with the key "mismatches": a list of one message per
    incompatible stream, naming both nodes, their stream indices, widths and
    datatypes. The list is empty if all streams are compatible.
    """
    mismatches = []
    for consumer in model.graph.node:
        if not is_fpgadataflow_node(consumer):
            continue
        for in_ind, tensor in enumerate(consumer.input):
            producer = model.find_producer(tensor)
            if producer is None or not is_fpgadataflow_node(producer):
                continue
            out_ind = list(producer.output).index(tensor)
            producer_inst = getCustomOp(producer)
            consumer_inst = getCustomOp(consumer)
            out_width = producer_inst.get_outstream_width_padded(out_ind)
            in_width = consumer_inst.get_instream_width_padded(in_ind)
            if out_width == 0 or in_width == 0:
                continue
            if _is_untyped_stream_node(model, producer) or _is_untyped_stream_node(model, consumer):
                odt_name = idt_name = None
            else:
                odt_name = producer_inst.get_output_datatype(out_ind).name
                idt_name = consumer_inst.get_input_datatype(in_ind).name
            if out_width != in_width or odt_name != idt_name:
                mismatches.append(
                    f"{producer.name} output {out_ind} ({out_width} bits, {odt_name}) -> "
                    f"{consumer.name} input {in_ind} ({in_width} bits, {idt_name}) "
                    f"on tensor {tensor}"
                )
    return {"mismatches": mismatches}
