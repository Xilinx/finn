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
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow import create_stitched_ip
from finn.transformation.fpgadataflow.alveo_build import PrepareForLinking, VitisLink
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import (
    CreateStitchedIP,
    append_missing_finnloop_rtlsim_sources,
)
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.util.basic import (
    getHWCustomOp,
    make_build_dir,
    pynq_part_map,
    robust_rmtree,
    vitis_default_platform,
    vitis_part_map,
)
from finn.util.test import load_test_checkpoint_or_skip
from finn.util.vivado import parse_ooc_synth_results

test_pynq_board = "Pynq-Z1"
test_fpga_part = pynq_part_map[test_pynq_board]

ip_stitch_model_dir = os.environ["FINN_BUILD_DIR"]


class FakeHWCustomOp:
    def __init__(self, code_gen_dir):
        self.code_gen_dir = code_gen_dir

    def get_nodeattr(self, name):
        assert name == "code_gen_dir_ipgen"
        return self.code_gen_dir


class FakeNode:
    def __init__(self, op_type, code_gen_dir=None):
        self.op_type = op_type
        self.code_gen_dir = code_gen_dir


class FakeGraph:
    def __init__(self, nodes):
        self.node = nodes


class FakeModel:
    def __init__(self, nodes):
        self.graph = FakeGraph(nodes)


@pytest.mark.fpgadataflow
def test_ipstitch_appends_missing_finnloop_rtlsim_sources(tmp_path, monkeypatch):
    top_list = tmp_path / "all_verilog_srcs.txt"
    existing_top = tmp_path / "top_existing.v"
    existing_duplicate_basename = tmp_path / "same_name.sv"
    existing_top.write_text("// top\n")
    existing_duplicate_basename.write_text("// shared source\n")
    top_list.write_text(str(existing_top) + "\n" + str(existing_duplicate_basename) + "\n")

    loop_a = tmp_path / "loop_a"
    loop_b = tmp_path / "loop_b"
    loop_a.mkdir()
    loop_b.mkdir()

    loop_a_new_v = loop_a / "nested_a.v"
    loop_a_new_sv = loop_a / "nested_a_extra.SV"
    loop_a_skip_txt = loop_a / "ignore.txt"
    loop_a_dup_basename = loop_a / "same_name.sv"
    for source_path in [
        loop_a_new_v,
        loop_a_new_sv,
        loop_a_skip_txt,
    ]:
        source_path.write_text("// loop a\n")
    loop_a_dup_basename.write_text("// shared source\n")
    (loop_a / "all_verilog_srcs.txt").write_text(
        "\n".join(
            map(
                str,
                [
                    loop_a_new_v,
                    loop_a_new_sv,
                    loop_a_skip_txt,
                    loop_a_dup_basename,
                ],
            )
        )
        + "\n"
    )

    loop_b_new_vhd = loop_b / "nested_b.vhd"
    loop_b_dup_basename = loop_b / "nested_a.v"
    for source_path in [loop_b_new_vhd, loop_b_dup_basename]:
        source_path.write_text("// loop b\n")
    loop_b_dup_basename.write_text("// loop a\n")
    (loop_b / "all_verilog_srcs.txt").write_text(
        "\n".join(map(str, [loop_b_new_vhd, loop_b_dup_basename])) + "\n"
    )

    model = FakeModel(
        [
            FakeNode("MVAU_rtl", str(loop_a)),
            FakeNode("FINNLoop", str(loop_a)),
            FakeNode("FINNLoop", str(loop_b)),
            FakeNode("FINNLoop", str(tmp_path / "missing_loop")),
        ]
    )
    monkeypatch.setattr(
        create_stitched_ip,
        "getHWCustomOp",
        lambda node, model: FakeHWCustomOp(node.code_gen_dir),
    )

    append_missing_finnloop_rtlsim_sources(model, str(top_list))
    assert top_list.read_text().splitlines() == [
        str(existing_top),
        str(existing_duplicate_basename),
        str(loop_a_new_v),
        str(loop_a_new_sv),
        str(loop_b_new_vhd),
    ]

    append_missing_finnloop_rtlsim_sources(model, str(tmp_path / "missing_top.txt"))


@pytest.mark.fpgadataflow
def test_ipstitch_rejects_conflicting_nested_rtlsim_basename(tmp_path, monkeypatch):
    top_source = tmp_path / "top" / "same_name.v"
    nested_source = tmp_path / "loop" / "same_name.v"
    top_source.parent.mkdir()
    nested_source.parent.mkdir()
    top_source.write_text("module same_name(input [7:0] in); endmodule\n")
    nested_source.write_text("module same_name(input [15:0] in); endmodule\n")
    top_list = tmp_path / "all_verilog_srcs.txt"
    top_list.write_text(str(top_source) + "\n")
    (nested_source.parent / "all_verilog_srcs.txt").write_text(str(nested_source) + "\n")
    model = FakeModel([FakeNode("FINNLoop", str(nested_source.parent))])
    monkeypatch.setattr(
        create_stitched_ip,
        "getHWCustomOp",
        lambda node, model: FakeHWCustomOp(node.code_gen_dir),
    )

    with pytest.raises(RuntimeError, match="Conflicting stitched-RTL sources"):
        append_missing_finnloop_rtlsim_sources(model, str(top_list))


def create_one_fc_model(mem_mode="internal_embedded"):
    # create a model with a MatrixVectorActivation instance with no activation
    # the wider range of the full accumulator makes debugging a bit easier
    wdt = DataType["INT2"]
    idt = DataType["INT32"]
    odt = DataType["INT32"]
    m = 4
    no_act = 1
    binary_xnor_mode = 0
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
        binaryXnorMode=binary_xnor_mode,
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
    binary_xnor_mode = 0
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
        binaryXnorMode=binary_xnor_mode,
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
        binaryXnorMode=binary_xnor_mode,
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
        sdp_node = getHWCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, 5))
    model = model.transform(HLSSynthIP())
    assert model.graph.node[0].op_type == "MVAU_hls"
    assert model.graph.node[-1].op_type == "TLastMarker_hls"
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
        sdp_node = getHWCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    model = model.transform(InferDataLayouts())
    model = model.transform(InsertIODMA())
    model = model.transform(Floorplan())
    assert getHWCustomOp(model.graph.node[0]).get_nodeattr("partition_id") == 0
    assert getHWCustomOp(model.graph.node[1]).get_nodeattr("partition_id") == 2
    assert getHWCustomOp(model.graph.node[2]).get_nodeattr("partition_id") == 1
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_iodma_floorplan.onnx")


# board
@pytest.mark.parametrize("board", ["U250"])
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
        sdp_node = getHWCustomOp(model.graph.node[0])
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
@pytest.mark.parametrize("board", ["Pynq-Z1"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_ipstitch_zynqbuild_end2end(board):
    model = create_two_fc_model()
    if model.graph.node[0].op_type == "StreamingDataflowPartition":
        sdp_node = getHWCustomOp(model.graph.node[0])
        assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
        assert os.path.isfile(sdp_node.get_nodeattr("model"))
        model = load_test_checkpoint_or_skip(sdp_node.get_nodeattr("model"))
    # bitfile using ZynqBuild
    model = model.transform(ZynqBuild(board, 10))
    model.save(ip_stitch_model_dir + "/test_fpgadataflow_ipstitch_customzynq.onnx")

    bitfile_name = model.get_metadata_prop("bitfile")
    assert bitfile_name is not None
    assert os.path.isfile(bitfile_name)
