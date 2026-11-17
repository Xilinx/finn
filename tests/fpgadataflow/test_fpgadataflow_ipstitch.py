# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
# * Neither the name of FINN nor the names of its
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

import numpy as np
import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.alveo_build import PrepareForLinking, VitisLink
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan, get_attached_iodma
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.make_pynq_proj import PYNQBuild
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.util.basic import (
    make_build_dir,
    pynq_part_map,
    robust_rmtree,
    vitis_default_platform,
    vitis_part_map,
)
from finn.util.test import load_test_checkpoint_or_skip
from finn.util.vivado import parse_ooc_synth_results

test_pynq_board = "AUP-ZU3_8GB"
test_fpga_part = pynq_part_map[test_pynq_board]

ip_stitch_model_dir = os.environ["FINN_BUILD_DIR"]


def create_one_fc_model(mem_mode="internal_embedded"):
    # create a model with a MatrixVectorActivation instance with no activation
    # the wider range of the full accumulator makes debugging a bit easier
    wdt = DataType["INT2"]
    idt = DataType["INT32"]
    odt = DataType["INT32"]
    m = 4
    no_act = 1

    actval = 0
    simd = 4
    pe = 4

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "MVAU_hls",
        ["inp", "w0"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        noActivation=no_act,
        mem_mode=mem_mode,
    )

    graph = helper.make_graph(nodes=[fc0], name="fclayer_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("w0", wdt)

    # generate weights
    w0 = np.eye(m, dtype=np.float32)
    model.set_initializer("w0", w0)

    model = model.transform(CreateDataflowPartition())
    return model


def create_two_fc_model(mem_mode="internal_decoupled"):
    # create a model with two MatrixVectorActivation instances
    wdt = DataType["INT2"]
    idt = DataType["INT32"]
    odt = DataType["INT32"]
    m = 4
    actval = 0
    no_act = 1

    pe = 2
    simd = 2

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "MVAU_hls",
        ["inp", "w0"],
        ["mid"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        noActivation=no_act,
        mem_mode=mem_mode,
    )

    fc1 = helper.make_node(
        "MVAU_hls",
        ["mid", "w1"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        noActivation=no_act,
        mem_mode=mem_mode,
    )

    graph = helper.make_graph(
        nodes=[fc0, fc1],
        name="fclayer_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[mid],
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("mid", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("w0", wdt)
    model.set_tensor_datatype("w1", wdt)

    # generate weights
    w0 = np.eye(m, dtype=np.float32)
    w1 = np.eye(m, dtype=np.float32)
    model.set_initializer("w0", w0)
    model.set_initializer("w1", w1)

    model = model.transform(CreateDataflowPartition())
    return model


# gen_model -> do_stitch -> rtlsim hand a checkpoint between separate tests via
# load_test_checkpoint_or_skip. grouping each mem_mode chain keeps it in one
# shard (and, with workers > 1, loadgroup keeps it on one worker), so each
# step's output is on disk before the next test reads it. the shared table
# below keeps the three tests' group names in lockstep.
MEM_MODE_PARAMS = [
    pytest.param(m, marks=pytest.mark.xdist_group(name=f"ipstitch_{m}"))
    for m in ("internal_embedded", "internal_decoupled")
]


@pytest.mark.parametrize("mem_mode", MEM_MODE_PARAMS)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_gen_model(mem_mode):
    model = create_one_fc_model(mem_mode)
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, 5))
    model = model.transform(HLSSynthIP())
    assert model.graph.node[0].op_type == "MVAU_hls"
    assert model.graph.node[-1].op_type == "TLastMarker_rtl"
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model_%s.onnx" % mem_mode)


@pytest.mark.parametrize("mem_mode", MEM_MODE_PARAMS)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_ipstitch_do_stitch(mem_mode):
    model = load_test_checkpoint_or_skip(
        ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_gen_model_%s.onnx" % mem_mode
    )
    # Run CreateStitchedIP with run_pnr=True to also get OOC synthesis results
    model = model.transform(CreateStitchedIP(test_fpga_part, 5, run_pnr=True))

    # Check IP stitching outputs
    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    assert vivado_stitch_proj_dir is not None
    assert os.path.isdir(vivado_stitch_proj_dir)
    assert os.path.isfile(vivado_stitch_proj_dir + "/ip/component.xml")
    vivado_stitch_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
    assert vivado_stitch_vlnv is not None
    assert vivado_stitch_vlnv == "xilinx_finn:finn:finn_design:1.0"

    # Check OOC synthesis results
    ret = parse_ooc_synth_results(vivado_stitch_proj_dir)
    assert ret is not None
    # example expected output: (details may differ based on Vivado version etc)
    # {'LUT': 708, 'FF': 1516, 'DSP': 0, 'BRAM_18K': 0, 'BRAM_36K': 0,
    # 'WNS': 0.152, 'fmax_mhz': 206.27}
    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    assert ret.get("BRAM_18K", 0) == 0
    assert ret.get("BRAM_36K", 0) == 0
    assert ret["fmax_mhz"] > 100

    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch_%s.onnx" % mem_mode)


@pytest.mark.parametrize("mem_mode", MEM_MODE_PARAMS)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_rtlsim(mem_mode):
    model = load_test_checkpoint_or_skip(
        ip_stitch_model_dir + "/test_fpgadataflow_ip_stitch_%s.onnx" % mem_mode
    )
    model.set_metadata_prop("rtlsim_trace", "whole_trace.wdb")
    model.set_metadata_prop("exec_mode", "rtlsim")
    idt = model.get_tensor_datatype("inp")
    ishape = model.get_tensor_shape("inp")
    x = gen_finn_dt_tensor(idt, ishape)
    # x = np.zeros(ishape, dtype=np.float32)
    # x = np.asarray([[-2, -1, 0, 1]], dtype=np.float32)
    rtlsim_res = execute_onnx(model, {"inp": x})["outp"]
    assert (rtlsim_res == x).all()


@pytest.mark.fpgadataflow
def test_fpgadataflow_ipstitch_iodma_floorplan():
    model = create_one_fc_model()
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(InferDataLayouts())
    model = model.transform(InsertIODMA())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(Floorplan())
    # Verify that IODMAs and compute nodes get distinct partition_ids.
    # After DWC externalization, the graph may have more than 3 nodes
    # (DWC nodes inserted when intfWidth != streamWidth), so we look up
    # nodes by op_type rather than by positional index.
    iodma_partitions = set()
    compute_partitions = set()
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        pid = node_inst.get_nodeattr("partition_id")
        if node.op_type == "IODMA_hls":
            iodma_partitions.add(pid)
        elif node.op_type in ("MVAU_hls", "MVAU_rtl"):
            compute_partitions.add(pid)
    # Each IODMA (input + output) gets its own partition
    assert len(iodma_partitions) == 2, f"Expected 2 IODMA partitions, got {iodma_partitions}"
    # Compute node(s) get a separate partition from the IODMAs
    assert len(compute_partitions) >= 1, f"Expected compute partition, got {compute_partitions}"
    assert iodma_partitions.isdisjoint(
        compute_partitions
    ), f"IODMA partitions {iodma_partitions} overlap with compute partitions {compute_partitions}"
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_iodma_floorplan.onnx")


def create_iodma_dwc_model():
    # single MVAU with external weights whose input, weight and output stream
    # widths all differ from the DMA interface width (max_intfwidth=32):
    # input 2 x INT8 = 16 bit, weights 2 x 2 x INT2 = 8 bit, output 2 x INT32 = 64 bit
    wdt = DataType["INT2"]
    idt = DataType["INT8"]
    odt = DataType["INT32"]
    m = 8
    pe = 2
    simd = 2

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "MVAU_hls",
        ["inp", "w0"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=0,
        noActivation=1,
        mem_mode="external",
    )

    graph = helper.make_graph(nodes=[fc0], name="fclayer_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="fclayer-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("w0", wdt)
    model.set_initializer("w0", gen_finn_dt_tensor(wdt, (m, m)))
    return model


@pytest.mark.fpgadataflow
def test_fpgadataflow_ipstitch_iodma_dwc_floorplan():
    model = create_iodma_dwc_model()
    model = model.transform(InferDataLayouts())
    model = model.transform(InsertIODMA())
    model = model.transform(GiveUniqueNodeNames())

    # every IODMA is straight-through, width conversion happens in a DWC next to it
    dma_nodes = model.get_nodes_by_op_type("IODMA_hls")
    assert len(dma_nodes) == 3, "Expected input, output and weight IODMA"
    for dma in dma_nodes:
        dma_inst = getCustomOp(dma)
        assert dma_inst.get_nodeattr("intfWidth") == dma_inst.get_nodeattr("streamWidth")
    dwc_nodes = [n for n in model.graph.node if n.op_type.startswith("StreamingDataWidthConverter")]
    assert len(dwc_nodes) == 3, "Expected one DWC per IODMA"
    attached = {get_attached_iodma(model, n).name for n in dwc_nodes}
    assert attached == {n.name for n in dma_nodes}, "Each IODMA needs its own attached DWC"
    # the MVAU is fed by DWCs on both inputs and feeds a DWC on its output
    mvau = model.get_nodes_by_op_type("MVAU_hls")[0]
    for in_tensor in mvau.input:
        assert model.find_producer(in_tensor) in dwc_nodes
    assert model.find_consumer(mvau.output[0]) in dwc_nodes

    # each DWC shares the partition (and SLR) of its IODMA, the MVAU is separate
    model = model.transform(Floorplan())
    # transform() returns a copy, so look the nodes up again
    dwc_nodes = [n for n in model.graph.node if n.op_type.startswith("StreamingDataWidthConverter")]
    mvau = model.get_nodes_by_op_type("MVAU_hls")[0]
    dma_partitions = set()
    for dwc in dwc_nodes:
        dwc_inst = getCustomOp(dwc)
        dma_inst = getCustomOp(get_attached_iodma(model, dwc))
        assert dwc_inst.get_nodeattr("partition_id") == dma_inst.get_nodeattr("partition_id")
        assert dwc_inst.get_nodeattr("slr") == dma_inst.get_nodeattr("slr")
        dma_partitions.add(dma_inst.get_nodeattr("partition_id"))
    assert len(dma_partitions) == 3, "Each IODMA needs its own partition"
    assert getCustomOp(mvau).get_nodeattr("partition_id") not in dma_partitions

    # the compute partition holds only the MVAU, i.e. starts and ends with a compute node;
    # DMA partitions keep the IODMA first (input, weights) or last (output) as the
    # driver generation expects
    parent = model.transform(CreateDataflowPartition())
    sdp_nodes = parent.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 4
    partition_ops = [
        [n.op_type for n in ModelWrapper(getCustomOp(sdp).get_nodeattr("model")).graph.node]
        for sdp in sdp_nodes
    ]
    assert sorted(partition_ops) == sorted(
        [
            ["IODMA_hls", "StreamingDataWidthConverter"],
            ["IODMA_hls", "StreamingDataWidthConverter"],
            ["MVAU_hls"],
            ["StreamingDataWidthConverter", "IODMA_hls"],
        ]
    )


# board
@pytest.mark.parametrize("board", ["U55C"])
# clock period
@pytest.mark.parametrize("period_ns", [5])
# override mem_mode to external
@pytest.mark.parametrize("extw", [True, False])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.vitis
def test_fpgadataflow_ipstitch_vitis_end2end(board, period_ns, extw):
    if "VITIS_PATH" not in os.environ:
        pytest.skip("VITIS_PATH not set")
    test_dir = make_build_dir("test_fpgadataflow_ipstitch_vitis_")
    platform = vitis_default_platform[board]
    fpga_part = vitis_part_map[board]
    model = create_two_fc_model("external" if extw else "internal_decoupled")
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpga_part, period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareForLinking(fpga_part, period_ns, "vitis-xrt"))
    model = model.transform(VitisLink(platform, period_ns))
    model.save(os.path.join(test_dir, "test_fpgadataflow_ipstitch_vitis.onnx"))
    assert model.get_metadata_prop("platform") == "vitis-xrt"
    assert os.path.isdir(model.get_metadata_prop("vitis_link_proj"))
    assert os.path.isfile(model.get_metadata_prop("bitfile"))
    robust_rmtree(test_dir)


# board
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_zynqbuild_end2end(board):
    model = create_two_fc_model()
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    # bitfile using PYNQBuild
    model = model.transform(PYNQBuild(board, 10))
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_customzynq.onnx")

    bitfile_name = model.get_metadata_prop("bitfile")
    assert bitfile_name is not None
    assert os.path.isfile(bitfile_name)
