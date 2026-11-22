# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


def _make_model():
    """Two DuplicateStreams_hls nodes, both with two outputs (graph outputs)."""
    nodes = [
        helper.make_node(
            "DuplicateStreams_hls",
            [inp],
            [inp + "_0", inp + "_1"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            NumChannels=4,
            PE=2,
            NumOutputStreams=2,
            inputDataType="INT4",
            numInputVectors=[1],
        )
        for inp in ["a", "b"]
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="cppsim_graph",
        inputs=[helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 4]) for n in "ab"],
        outputs=[
            helper.make_tensor_value_info(n + s, TensorProto.FLOAT, [1, 4])
            for n in "ab"
            for s in ["_0", "_1"]
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="cppsim-model"))
    for n in "ab":
        model.set_tensor_datatype(n, DataType["INT4"])
    return model.transform(GiveUniqueNodeNames())


def _exe_mtimes(model):
    return [
        os.stat(getCustomOp(n).get_nodeattr("executable_path")).st_mtime_ns
        for n in model.graph.node
    ]


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_compile_cppsim_skips_up_to_date_nodes():
    model = _make_model()
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    built = _exe_mtimes(model)

    # compiling again, also after regenerating the same code, keeps the executables
    model = model.transform(CompileCppSim())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    assert _exe_mtimes(model) == built

    # a change of the generated code of one node rebuilds only that node
    getCustomOp(model.graph.node[1]).set_nodeattr("PE", 4)
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    rebuilt = _exe_mtimes(model)
    assert rebuilt[0] == built[0]
    assert rebuilt[1] != built[1]

    # the executables, linked against the shared cnpy object, still work
    model = model.transform(SetExecMode("cppsim"))
    inp = {n: gen_finn_dt_tensor(DataType["INT4"], (1, 4)) for n in "ab"}
    out = oxe.execute_onnx(model, inp)
    for n in "ab":
        for s in ["_0", "_1"]:
            assert np.array_equal(out[n + s], inp[n])
