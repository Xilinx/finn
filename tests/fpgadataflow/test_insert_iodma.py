# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA

DT = DataType["UINT4"]


def _make_concat(inputs, output, channels):
    return helper.make_node(
        "StreamingConcat",
        inputs,
        [output],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=2,
        ChannelsPerStream=channels,
        inputDataTypes=[DT.name] * len(channels),
        numInputVectors=[1],
    )


def _make_dup(inp, outputs, channels):
    return helper.make_node(
        "DuplicateStreams",
        [inp],
        outputs,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=channels,
        PE=2,
        NumOutputStreams=len(outputs),
        inputDataType=DT.name,
        numInputVectors=[1],
    )


def _make_model(nodes, inputs, outputs):
    """Builds a model from nodes with UINT4 tensors of shape [1, C], C given per graph I/O."""
    graph = helper.make_graph(
        nodes=nodes,
        name="iodma_graph",
        inputs=[helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, c]) for n, c in inputs],
        outputs=[helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, c]) for n, c in outputs],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="iodma-model"))
    for node in model.graph.node:
        for tensor in list(node.input) + list(node.output):
            model.set_tensor_datatype(tensor, DT)
    return model


def _upstream_iodma(model, tensor):
    """Follows the producers of tensor through DWCs to the IODMA feeding it."""
    node = model.find_producer(tensor)
    while node.op_type.startswith("StreamingDataWidthConverter"):
        node = model.find_producer(node.input[0])
    return node


def _downstream_iodma(model, tensor):
    """Follows the consumers of tensor through DWCs to the IODMA it feeds."""
    node = model.find_consumer(tensor)
    while node.op_type.startswith("StreamingDataWidthConverter"):
        node = model.find_consumer(node.output[0])
    return node


def _transfer_bits(dma_node):
    dma_inst = getCustomOp(dma_node)
    return dma_inst.get_nodeattr("streamWidth") * int(
        np.prod(dma_inst.get_nodeattr("numInputVectors"))
    )


@pytest.mark.fpgadataflow
def test_insert_iodma_multi_input_multi_output():
    # in0, in1 -> Concat -> cat -> Dup1 -> (mid, out_b); mid -> Dup2 -> (out_a, out_c):
    # graph inputs at stream index 0 and 1 of the Concat, graph outputs at stream
    # index 1 of Dup1 (whose stream 0 stays internal) and 0 and 1 of Dup2
    nodes = [
        _make_concat(["in0", "in1"], "cat", [4, 8]),
        _make_dup("cat", ["mid", "out_b"], 12),
        _make_dup("mid", ["out_a", "out_c"], 12),
    ]
    model = _make_model(
        nodes, [("in0", 4), ("in1", 8)], [("out_a", 12), ("out_b", 12), ("out_c", 12)]
    )

    # the first application inserts all IODMAs, the second finds nothing left to do
    # (before the fix, out_b stayed attached to Dup1 and the pass looped forever)
    model, modified = InsertIODMA().apply(model)
    assert modified
    model, modified = InsertIODMA().apply(model)
    assert not modified

    assert len(model.get_nodes_by_op_type("IODMA_hls")) == 5
    concat = model.get_nodes_by_op_type("StreamingConcat")[0]
    dup1, dup2 = model.get_nodes_by_op_type("DuplicateStreams")
    if dup1.input[0] != concat.output[0]:
        dup1, dup2 = dup2, dup1
    assert dup2.input[0] == dup1.output[0], "internal stream of Dup1 must stay connected"

    # every graph input reaches its own Concat stream through its own IODMA,
    # sized for that stream (4 and 8 UINT4 channels)
    for ind, (graph_in, bits) in enumerate([("in0", 16), ("in1", 32)]):
        dma = _upstream_iodma(model, concat.input[ind])
        assert dma.input[0] == graph_in
        assert _transfer_bits(dma) == bits

    # every graph output is written by an IODMA fed from the right node stream
    for producer, ind, graph_out in [(dup2, 0, "out_a"), (dup1, 1, "out_b"), (dup2, 1, "out_c")]:
        dma = _downstream_iodma(model, producer.output[ind])
        assert dma.output[0] == graph_out
        assert model.find_producer(graph_out) == dma
        assert _transfer_bits(dma) == 48


@pytest.mark.fpgadataflow
def test_insert_iodma_rejects_fanout():
    # graph input feeding two nodes
    nodes = [_make_dup("inp", ["out0", "out1"], 4), _make_dup("inp", ["out2", "out3"], 4)]
    model = _make_model(nodes, [("inp", 4)], [(n, 4) for n in ["out0", "out1", "out2", "out3"]])
    with pytest.raises(ValueError, match="graph input inp must feed exactly one"):
        model.transform(InsertIODMA())

    # graph output that is also consumed inside the graph
    nodes = [_make_dup("inp", ["out0", "out1"], 4), _make_dup("out0", ["out2", "out3"], 4)]
    model = _make_model(nodes, [("inp", 4)], [(n, 4) for n in ["out0", "out1", "out2", "out3"]])
    with pytest.raises(ValueError, match="graph output out0 .* is also consumed"):
        model.transform(InsertIODMA())
