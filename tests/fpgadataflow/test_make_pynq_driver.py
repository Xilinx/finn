# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.make_driver import MakePYNQDriver
from finn.util.basic import robust_rmtree


def _make_mvau(inp, outp, wname, mem_mode):
    return helper.make_node(
        "MVAU_hls",
        [inp, wname],
        [outp],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=8,
        MH=8,
        SIMD=2,
        PE=2,
        inputDataType="INT8",
        weightDataType="INT2",
        outputDataType="INT8",
        ActVal=0,
        noActivation=1,
        mem_mode=mem_mode,
        runtime_writeable_weights=1,
    )


def _make_two_mvau_model():
    """Two runtime-writable MVAUs, the first with on-chip, the second with external weights."""
    nodes = [
        _make_mvau("inp", "mid", "w0", "internal_decoupled"),
        _make_mvau("mid", "outp", "w1", "external"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="driver_graph",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 8])],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 8])],
        value_info=[helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 8])],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="driver-model"))
    for tensor in ["inp", "mid", "outp"]:
        model.set_tensor_datatype(tensor, DataType["INT8"])
    for wname in ["w0", "w1"]:
        model.set_tensor_datatype(wname, DataType["INT2"])
        model.set_initializer(wname, gen_finn_dt_tensor(DataType["INT2"], (8, 8)))
    return model


@pytest.mark.fpgadataflow
def test_make_pynq_driver_external_runtime_writeable_weights():
    # the flow of MakePYNQProject up to the partitioning, without IP generation
    model = _make_two_mvau_model()
    model = model.transform(InferDataLayouts())
    for trn in [InsertIODMA(32), InsertDWC(), Floorplan(), CreateDataflowPartition()]:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

    # MakeZYNQProject names the partition instances; the node names suffice here
    for node in model.graph.node:
        getCustomOp(node).set_nodeattr("instance_name", node.name)

    # before the fix, reading the (moved) initializer of the external MVAU crashed here
    model = model.transform(MakePYNQDriver("pynq-iodma"))
    driver_dir = model.get_metadata_prop("pynq_driver_dir")
    rt_dir = os.path.join(driver_dir, "weights", "runtime_weights")
    files = sorted(os.listdir(rt_dir))

    # the decoupled MVAU gets its AXI-lite .dat file, the external MVAU only the .npy
    # of its weight IODMA
    dat_files = [f for f in files if f.endswith(".dat")]
    npy_files = [f for f in files if f.endswith(".npy")]
    assert len(dat_files) == 1, files
    assert "MVAU" in dat_files[0] and dat_files[0].split("_")[1] == "0", dat_files
    assert len(npy_files) == 1, files
    assert npy_files[0][: -len(".npy")] in [n.name for n in model.graph.node]
    robust_rmtree(driver_dir)
