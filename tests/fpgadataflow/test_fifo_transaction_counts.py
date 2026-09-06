# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from functools import partial
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.fifo_transaction_counts import fifo_transaction_counts
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO


def make_multi_fclayer_model(ch, wdt, adt, nnodes):
    """Build a chain of MVAU_hls nodes (no activation) directly as HLS layers,
    mirroring tests/fpgadataflow/test_set_folding.py."""
    W = np.random.randint(wdt.min(), wdt.max() + 1, size=(ch, ch)).astype(np.float32)

    tensors = [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch])]
    for i in range(1, nnodes):
        tensors.append(helper.make_tensor_value_info("inter_" + str(i), TensorProto.FLOAT, [1, ch]))
    tensors.append(helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch]))

    nodes = []
    for i in range(nnodes):
        nodes.append(
            helper.make_node(
                "MVAU_hls",
                [tensors[i].name, "weights_" + str(i)],
                [tensors[i + 1].name],
                domain="finn.custom_op.fpgadataflow.hls",
                backend="fpgadataflow",
                MW=ch,
                MH=ch,
                SIMD=1,
                PE=1,
                inputDataType=adt.name,
                weightDataType=wdt.name,
                outputDataType=adt.name,
                ActVal=0,
                binaryXnorMode=0,
                noActivation=1,
            )
        )

    graph = helper.make_graph(
        nodes=nodes, name="fclayer_graph", inputs=[tensors[0]], outputs=[tensors[-1]]
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="fclayer-model"))
    model.set_tensor_datatype("inp", adt)
    model.set_tensor_datatype("outp", adt)
    for i in range(1, nnodes + 1):
        if tensors[i].name != "outp":
            model.graph.value_info.append(tensors[i])
        model.set_initializer("weights_" + str(i - 1), W)
        model.set_tensor_datatype("weights_" + str(i - 1), wdt)
    return model


@pytest.mark.fpgadataflow
def test_fifo_transaction_counts_inserted_fifos():
    # build a chain of hw layers (CH=128, PE=SIMD=1) and let InsertFIFO place FIFOs
    # in between the layers as well as at the graph input and output
    ch = 128
    nnodes = 5
    model = make_multi_fclayer_model(ch, DataType["INT4"], DataType["INT4"], nnodes)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(GiveUniqueNodeNames())

    result = model.analysis(fifo_transaction_counts)

    # InsertFIFO adds one FIFO between each pair of layers plus one at the graph
    # input and one at the output: (nnodes - 1) + 2 FIFOs
    assert len(result) == (nnodes - 1) + 2
    # with PE=SIMD=1 every FIFO carries one transaction per element, i.e. CH per frame
    assert all(v == ch for v in result.values())


@pytest.mark.fpgadataflow
def test_fifo_transaction_counts_finnloop_iteration():
    ch = 64
    nnodes = 4
    iteration = 3
    dtype = DataType["INT4"]
    body = make_multi_fclayer_model(ch, dtype, dtype, nnodes)

    top_in = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch])
    top_out = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch])
    loop_node = helper.make_node(
        "FINNLoop",
        ["inp"],
        ["outp"],
        name="FINNLoop_0",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        body=body.graph,
        iteration=iteration,
        inputDataType=dtype.name,
        outputDataType=dtype.name,
    )
    top = helper.make_graph(nodes=[loop_node], name="top", inputs=[top_in], outputs=[top_out])
    model = ModelWrapper(qonnx_make_model(top, producer_name="loop-model"))

    # insert FIFOs in the body subgraph (and around the loop at the top level)
    model = model.transform(InsertFIFO(create_shallow_fifos=True), apply_to_subgraphs=True)

    model = model.transform(GiveUniqueNodeNames())
    for ln in model.get_nodes_by_op_type("FINNLoop"):
        inst = getCustomOp(ln)
        loop_body = inst.get_nodeattr("body")
        loop_body = loop_body.transform(GiveUniqueNodeNames(prefix=ln.name + "_"))
        inst.set_nodeattr("body", loop_body.graph)

    # without descending, only the top-level FIFOs around the loop are visible
    top_only = model.analysis(fifo_transaction_counts)
    # descending multiplies body FIFO counts by the loop iteration count
    result = model.analysis(partial(fifo_transaction_counts, apply_to_subgraphs=True))

    # top level: an input and an output FIFO around the loop, counted once (×1)
    assert sorted(top_only.values()) == [ch, ch]
    # body: (nnodes - 1) internal + input + output FIFOs, each counted ×iteration,
    # plus the two top-level FIFOs (×1)
    n_body_fifos = (nnodes - 1) + 2
    assert sorted(result.values()) == sorted([ch] * 2 + [ch * iteration] * n_body_fifos)


@pytest.mark.fpgadataflow
def test_fifo_transaction_counts_requires_fifos():
    # a model without any FIFOs must raise a clear assertion
    model = make_multi_fclayer_model(64, DataType["INT4"], DataType["INT4"], 3)
    model = model.transform(GiveUniqueNodeNames())
    with pytest.raises(AssertionError):
        model.analysis(fifo_transaction_counts)
