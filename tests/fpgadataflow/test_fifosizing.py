# Copyright (c) 2022 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import pytest

import json
import numpy as np
import torch
from brevitas.export import export_qonnx
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveCharacteristic,
    DeriveFIFOSizes,
    _find_minimum_phase_shift,
)
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    SplitLargeFIFOs,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.test import get_trained_network_and_ishape


def fetch_test_model(topology, wbits=2, abits=2):
    tmp_output_dir = make_build_dir("build_fifosizing_%s_" % topology)
    (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    chkpt_name = tmp_output_dir + "/model.onnx"
    export_qonnx(model, torch.randn(ishape), chkpt_name)
    return tmp_output_dir


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize("method", ["largefifo_rtlsim", "characterize"])
@pytest.mark.parametrize("topology", ["tfc", "cnv"])
def test_fifosizing_linear(method, topology):
    tmp_output_dir = fetch_test_model(topology)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        auto_fifo_depths=True,
        auto_fifo_strategy=method,
        target_fps=10000 if topology == "tfc" else 1000,
        synth_clk_period_ns=10.0,
        board="AUP-ZU3_8GB",
        rtlsim_batch_size=100 if topology == "tfc" else 2,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)
    with open(tmp_output_dir + "/report/estimate_network_performance.json") as f:
        est_data = json.load(f)
    with open(tmp_output_dir + "/report/rtlsim_performance.json") as f:
        sim_data = json.load(f)
    assert sim_data["completed_output_frames"] == cfg.rtlsim_batch_size
    assert sim_data["interval_valid"] == 1
    assert sim_data["interval_is_steady_state"] is True
    assert sim_data["steady_state_frames"] == cfg.rtlsim_batch_size - 1
    assert sim_data["steady_state_cycles"] > 0
    assert sim_data["stable_throughput_valid"] is True
    expected_stable_throughput = (
        sim_data["steady_state_frames"]
        * 1.0e9
        / (cfg.synth_clk_period_ns * sim_data["steady_state_cycles"])
    )
    assert sim_data["stable_throughput[images/s]"] == pytest.approx(expected_stable_throughput)
    assert (
        float(sim_data["stable_throughput[images/s]"]) / float(est_data["estimated_throughput_fps"])
        > 0.9
    )
    # now run the same build using the generated folding and FIFO config
    tmp_output_dir_cmp = fetch_test_model(topology)
    cfg_cmp = cfg
    cfg_cmp.output_dir = tmp_output_dir_cmp
    cfg_cmp.auto_fifo_depths = False
    cfg_cmp.target_fps = None
    cfg_cmp.generate_outputs = [build_cfg.DataflowOutputType.STITCHED_IP]
    cfg_cmp.folding_config_file = tmp_output_dir + "/final_hw_config.json"
    build.build_dataflow_cfg(tmp_output_dir_cmp + "/model.onnx", cfg_cmp)

    model0 = ModelWrapper(tmp_output_dir + "/intermediate_models/step_create_stitched_ip.onnx")
    model1 = ModelWrapper(tmp_output_dir_cmp + "/intermediate_models/step_create_stitched_ip.onnx")

    assert len(model0.graph.node) == len(model1.graph.node)
    for i in range(len(model0.graph.node)):
        node0 = model0.graph.node[i]
        node1 = model1.graph.node[i]
        assert node0.op_type == node1.op_type
        if node0.op_type == "StreamingFIFO":
            node0_inst = getCustomOp(node0)
            node1_inst = getCustomOp(node1)
            assert node0_inst.get_nodeattr("depth") == node1_inst.get_nodeattr("depth")

    robust_rmtree(tmp_output_dir)
    robust_rmtree(tmp_output_dir_cmp)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_fifosizing_multi_io():
    # construct small onnx graph with addstreams, followed by duplicate streams
    # to have test model with multiple inputs and multiple outputs
    model = make_multi_io_modelwrapper(2, 2, DataType["INT4"])
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InsertAndSetFIFODepths("xc7z020clg400-1", 5))
    fifos = model.get_nodes_by_op_type("StreamingFIFO_rtl")
    assert len(fifos) > 1, "No FIFOs inserted"


def test_characterization_phase_shift_binary_search_matches_linear_scan():
    rng = np.random.default_rng(0)
    for period in range(1, 65):
        prod_chrc = np.cumsum(rng.integers(0, 3, size=2 * period))
        cons_chrc = np.cumsum(rng.integers(0, 3, size=2 * period))
        expected = period - 1
        for candidate in range(period):
            if (prod_chrc[candidate:period] >= cons_chrc[: period - candidate]).all():
                expected = candidate
                break
        assert _find_minimum_phase_shift(prod_chrc, cons_chrc, period) == expected


def test_characterization_can_skip_named_node(monkeypatch):
    model = make_multi_io_modelwrapper(2, 2, DataType["INT4"])
    node = model.graph.node[0]

    def fail_if_called(_node):
        raise AssertionError("skipped node must not be characterized")

    monkeypatch.setattr(
        "finn.transformation.fpgadataflow.derive_characteristic.registry.getCustomOp",
        fail_if_called,
    )
    returned, changed = DeriveCharacteristic(
        period=4,
        skip_node_names={node.name},
    ).applyNodeLocal(node)

    assert returned is node
    assert changed is False


def test_oversized_vivado_axis_fifo_stays_rtl():
    model = make_multi_io_modelwrapper(300, 300, DataType["INT8"])
    producer = getCustomOp(model.graph.node[0])
    consumer = getCustomOp(model.graph.node[1])
    producer.set_nodeattr("outFIFODepths", [784])
    consumer.set_nodeattr("inFIFODepths", [784])

    model = model.transform(InsertFIFO(max_qsrl_depth=256))
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    fifos = model.get_nodes_by_op_type("StreamingFIFO_rtl")
    assert len(fifos) == 1
    assert getCustomOp(fifos[0]).get_nodeattr("impl_style") == "rtl"

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SplitLargeFIFOs())
    fifos = model.get_nodes_by_op_type("StreamingFIFO_rtl")
    assert [getCustomOp(fifo).get_nodeattr("depth") for fifo in fifos] == [
        256,
        256,
        256,
        16,
    ]
    assert all(getCustomOp(fifo).get_nodeattr("impl_style") == "rtl" for fifo in fifos)


def test_characterization_fifosizing_uses_matching_consumer_input(tmp_path):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 1])
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT, [1, 1])
    branch = helper.make_tensor_value_info("branch", TensorProto.FLOAT, [1, 1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])
    fork = helper.make_node(
        "DuplicateStreams_rtl",
        ["inp"],
        ["skip", "branch"],
        name="fork",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=1,
        NumOutputStreams=2,
        PE=1,
        inputDataType="INT4",
        numInputVectors=[1],
        outFIFODepths=[2, 2],
    )
    join = helper.make_node(
        "ElementwiseAdd_rtl",
        ["skip", "branch"],
        ["out"],
        name="join",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        lhs_shape=[1, 1],
        rhs_shape=[1, 1],
        out_shape=[1, 1],
        lhs_dtype="INT4",
        rhs_dtype="INT4",
        out_dtype="INT4",
        lhs_style="input",
        rhs_style="input",
        PE=1,
        inFIFODepths=[2, 2],
    )
    graph = helper.make_graph(
        [fork, join], "residual", [inp], [out], value_info=[skip, branch]
    )
    model = ModelWrapper(qonnx_make_model(graph))

    period = 4
    producer_chrc = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    consumer_chrc = np.asarray(
        [producer_chrc, [0, 0, 0, 1, 1, 1, 1, 2]], dtype=np.int32
    )
    fork_node = model.get_nodes_by_op_type("DuplicateStreams_rtl")[0]
    join_node = model.get_nodes_by_op_type("ElementwiseAdd_rtl")[0]
    for node, chrc_in, chrc_out in [
        (fork_node, None, np.stack([producer_chrc, producer_chrc])),
        (join_node, consumer_chrc, producer_chrc.reshape(1, -1)),
    ]:
        inst = getCustomOp(node)
        inst.set_nodeattr("io_chrc_period", period)
        if chrc_in is not None:
            path = tmp_path / f"{node.name}_io_chrc_in.npy"
            np.save(path, chrc_in)
            inst.set_nodeattr("io_chrc_in_file", str(path))
        if node is fork_node:
            path = tmp_path / f"{node.name}_io_chrc_out.npz"
            np.savez_compressed(path, io_chrc=chrc_out)
        else:
            path = tmp_path / f"{node.name}_io_chrc_out.npy"
            np.save(path, chrc_out)
        inst.set_nodeattr("io_chrc_out_file", str(path))

    model = model.transform(DeriveFIFOSizes())

    fork_inst = getCustomOp(model.get_nodes_by_op_type("DuplicateStreams_rtl")[0])
    assert fork_inst.get_nodeattr("outFIFODepths") == [0, 3]


def test_characterization_fifosizing_honors_output_override():
    model = make_multi_io_modelwrapper(2, 2, DataType["INT4"])
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    model = model.transform(GiveUniqueNodeNames())
    producer = model.graph.node[0]
    producer_inst = getCustomOp(producer)

    transformation = DeriveFIFOSizes(
        output_fifo_depth_overrides={
            producer.name: {i: i + 2 for i in range(len(producer.output))}
        }
    )
    transformation.ref_input_model = model
    returned, changed = transformation.applyNodeLocal(producer)

    assert returned is producer
    assert changed is False
    assert producer_inst.get_nodeattr("outFIFODepths") == [2]


def make_multi_io_modelwrapper(ch, pe, idt):
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, ch])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, ch])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, ch])
    out0 = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1, ch])
    out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, ch])

    addstreams_node = helper.make_node(
        "ElementwiseAdd",
        ["in0", "in1"],
        ["mid"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        lhs_shape=[1, ch],
        rhs_shape=[1, ch],
        out_shape=[1, ch],
        lhs_dtype=idt.name,
        rhs_dtype=idt.name,
        out_dtype=idt.name,
        lhs_style="input",
        rhs_style="input",
        PE=pe,
        inFIFODepths=[2, 2],
    )
    duplicate_node = helper.make_node(
        "DuplicateStreams",
        ["mid"],
        ["out0", "out1"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=ch,
        NumOutputStreams=2,
        PE=pe,
        inputDataType=idt.name,
        numInputVectors=[1],
        outFIFODepths=[2, 2],
    )
    graph = helper.make_graph(
        nodes=[addstreams_node, duplicate_node],
        name="graph",
        inputs=[in0, in1],
        outputs=[out0, out1],
        value_info=[mid],
    )

    model = qonnx_make_model(graph, producer_name="multi-io-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("in0", idt)
    model.set_tensor_datatype("in1", idt)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model
