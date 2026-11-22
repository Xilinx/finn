# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.analysis.fpgadataflow.stream_compatibility import stream_compatibility
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import pynq_part_map


def _make_dup_concat_model(concat_simd=2, concat_idts=("UINT4", "UINT4")):
    """inp -> DuplicateStreams (PE=2, UINT4) -> (a, b) -> StreamingConcat -> outp."""
    dup = helper.make_node(
        "DuplicateStreams",
        ["inp"],
        ["a", "b"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=4,
        PE=2,
        NumOutputStreams=2,
        inputDataType="UINT4",
        numInputVectors=[1],
    )
    concat = helper.make_node(
        "StreamingConcat",
        ["a", "b"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=concat_simd,
        ChannelsPerStream=[4, 4],
        inputDataTypes=list(concat_idts),
        numInputVectors=[1],
    )
    graph = helper.make_graph(
        nodes=[dup, concat],
        name="stream_graph",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 8])],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="stream-model"))
    model = model.transform(GiveUniqueNodeNames())
    for tensor in ["inp", "a", "b", "outp"]:
        model.set_tensor_datatype(tensor, DataType["UINT4"])
    return model


@pytest.mark.fpgadataflow
def test_stream_compatibility_passes_matching_streams():
    model = _make_dup_concat_model()
    assert model.analysis(stream_compatibility)["mismatches"] == []


@pytest.mark.fpgadataflow
def test_stream_compatibility_reports_width_mismatch():
    # SIMD=4 makes both Concat inputs 16 bits wide, the Dup outputs are 8 bits
    model = _make_dup_concat_model(concat_simd=4)
    mismatches = model.analysis(stream_compatibility)["mismatches"]
    assert mismatches == [
        "DuplicateStreams_0 output 0 (8 bits, UINT4) -> "
        "StreamingConcat_0 input 0 (16 bits, UINT4) on tensor a",
        "DuplicateStreams_0 output 1 (8 bits, UINT4) -> "
        "StreamingConcat_0 input 1 (16 bits, UINT4) on tensor b",
    ]
    # CreateStitchedIP checks before it starts Vivado
    with pytest.raises(ValueError, match="incompatible stream widths or datatypes"):
        CreateStitchedIP("xczu3eg-sbva484-1-e", 5).apply(model)


@pytest.mark.fpgadataflow
def test_stream_compatibility_reports_datatype_mismatch():
    # same width, but the second Concat input expects INT4
    model = _make_dup_concat_model(concat_idts=("UINT4", "INT4"))
    mismatches = model.analysis(stream_compatibility)["mismatches"]
    assert mismatches == [
        "DuplicateStreams_0 output 1 (8 bits, UINT4) -> "
        "StreamingConcat_0 input 1 (8 bits, INT4) on tensor b"
    ]


def _make_mvau(inp, outp, wname, mem_mode):
    return helper.make_node(
        "MVAU",
        [inp, wname],
        [outp],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=8,
        MH=8,
        SIMD=2,
        PE=4,
        inputDataType="INT3",
        weightDataType="INT2",
        outputDataType="INT3",
        ActVal=0,
        noActivation=1,
        mem_mode=mem_mode,
    )


@pytest.mark.fpgadataflow
def test_stream_compatibility_passes_partitioned_flow():
    # the partitions of MakePYNQProject: IODMAs with byte streams and attached
    # DWCs, a weight IODMA, generic DWCs (PE=4 -> SIMD=2) and FIFOs; the stream
    # widths of INT3 are not byte-aligned; a TLastMarker on the compute partition
    graph = helper.make_graph(
        nodes=[
            _make_mvau("inp", "mid", "w0", "internal_decoupled"),
            _make_mvau("mid", "outp", "w1", "external"),
        ],
        name="flow_graph",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 8])],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 8])],
        value_info=[helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 8])],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="flow-model"))
    for tensor in ["inp", "mid", "outp"]:
        model.set_tensor_datatype(tensor, DataType["INT3"])
    for wname in ["w0", "w1"]:
        model.set_tensor_datatype(wname, DataType["INT2"])
        model.set_initializer(wname, gen_finn_dt_tensor(DataType["INT2"], (8, 8)))
    fpga_part = pynq_part_map["AUP-ZU3_8GB"]
    model = model.transform(SpecializeLayers(fpga_part))
    model = model.transform(InferDataLayouts())
    for trn in [
        InsertIODMA(64),
        InsertDWC(),
        SpecializeLayers(fpga_part),
        Floorplan(),
        CreateDataflowPartition(),
    ]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

    op_types = set()
    for sdp in model.graph.node:
        kernel = ModelWrapper(getCustomOp(sdp).get_nodeattr("model"))
        kernel = kernel.transform(InsertFIFO(create_shallow_fifos=True))
        if not any(n.op_type == "IODMA_hls" for n in kernel.graph.node):
            # the compute partition, as stitched by the ipstitch tests
            kernel = kernel.transform(InsertTLastMarker())
        kernel = kernel.transform(SpecializeLayers(fpga_part))
        kernel = kernel.transform(GiveUniqueNodeNames())
        assert kernel.analysis(stream_compatibility)["mismatches"] == []
        op_types.update(n.op_type for n in kernel.graph.node)
    assert {"IODMA_hls", "StreamingFIFO_rtl", "TLastMarker_rtl"} <= op_types
    assert any(t.startswith("StreamingDataWidthConverter") for t in op_types)
