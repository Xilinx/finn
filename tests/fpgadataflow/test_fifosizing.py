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
import shutil
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
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir
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
        board="Pynq-Z1",
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

    shutil.rmtree(tmp_output_dir)
    shutil.rmtree(tmp_output_dir_cmp)


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


def make_multi_io_modelwrapper(ch, pe, idt):
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, ch])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, ch])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, ch])
    out0 = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1, ch])
    out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, ch])

    addstreams_node = helper.make_node(
        "AddStreams",
        ["in0", "in1"],
        ["mid"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=ch,
        PE=pe,
        inputDataTypes=[idt.name, idt.name],
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
