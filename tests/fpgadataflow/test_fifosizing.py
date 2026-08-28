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
import os
import re
import shutil
import subprocess
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
from finn.util.basic import make_build_dir, robust_rmtree, which
from finn.util.test import get_trained_network_and_ishape

FPGAPART = "xc7z020clg400-1"

# Residual model for the maxcount overflow tests
IDT = DataType["INT4"]
CH = 8
PE = 8
# Folded tensor size, the natural insertion depth
NVEC = [1, 8, 8]
SHAPE = NVEC + [CH]
FULL_W = CH * IDT.bitwidth()
NARROW_W = IDT.bitwidth()

# Tiny insertion depth keeps the wrap point at 8
MAX_DEPTH = 4
WRAP_AT = 2 ** int(MAX_DEPTH).bit_length()
# Width-converter pairs delaying the long branch
NPAIR = 2


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


def run_xsim(tmpdir, top):
    """Runs one fifo_gauge_overflow_tb top under xsim."""
    rtllib_fifo_hdl = os.environ["FINN_ROOT"] + "/finn-rtllib/fifo/hdl"
    for src in ["fifo_gauge.sv", "fifo_gauge_overflow_tb.sv"]:
        shutil.copy(os.path.join(rtllib_fifo_hdl, src), tmpdir)
    out = ""
    for args in (
        ["xvlog", "-sv", "fifo_gauge.sv", "fifo_gauge_overflow_tb.sv"],
        # fifo_gauge.sv carries no timescale, the testbench does
        ["xelab", top, "-debug", "off", "-s", "sim", "--timescale", "1ns/1ps"],
        ["xsim", "sim", "-runall"],
    ):
        proc = subprocess.run(
            args, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8"
        )
        out += proc.stdout
        if proc.returncode != 0:
            return (proc.returncode, out)
    # xsim exits 0 even on $fatal, so scan transcript
    return (1 if "Fatal:" in out else 0, out)


def make_residual_modelwrapper():
    """Fork -> (short branch | long branch with buffers) -> join."""
    nodes = []
    vi = []
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, SHAPE)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, SHAPE)

    nodes.append(
        helper.make_node(
            "DuplicateStreams",
            ["inp"],
            ["short", "long0"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=CH,
            NumOutputStreams=2,
            PE=PE,
            inputDataType=IDT.name,
            numInputVectors=NVEC,
            outFIFODepths=[2, 2],
        )
    )
    vi += [helper.make_tensor_value_info(n, TensorProto.FLOAT, SHAPE) for n in ("short", "long0")]

    # Narrow leg drains slowly, backing up short branch
    cur = "long0"
    for i in range(NPAIR):
        narrow, wide = "lnarrow%d" % i, "long%d" % (i + 1)
        nodes.append(
            helper.make_node(
                "StreamingDataWidthConverter",
                [cur],
                [narrow],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                shape=SHAPE,
                inWidth=FULL_W,
                outWidth=NARROW_W,
                dataType=IDT.name,
            )
        )
        nodes.append(
            helper.make_node(
                "StreamingDataWidthConverter",
                [narrow],
                [wide],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                shape=SHAPE,
                inWidth=NARROW_W,
                outWidth=FULL_W,
                dataType=IDT.name,
            )
        )
        vi += [helper.make_tensor_value_info(n, TensorProto.FLOAT, SHAPE) for n in (narrow, wide)]
        cur = wide

    nodes.append(
        helper.make_node(
            "ElementwiseAdd",
            ["short", cur],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            lhs_shape=SHAPE,
            rhs_shape=SHAPE,
            out_shape=SHAPE,
            lhs_dtype=IDT.name,
            rhs_dtype=IDT.name,
            out_dtype=IDT.name,
            lhs_style="input",
            rhs_style="input",
            PE=PE,
            inFIFODepths=[2, 2],
        )
    )

    graph = helper.make_graph(nodes, "residual", [inp], [outp], value_info=vi)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="residual-maxcount"))
    for tname in ["inp", "outp"] + [x.name for x in vi]:
        model.set_tensor_datatype(tname, IDT)
    model = model.transform(InferShapes())
    return model.transform(InferDataTypes())


def gauge_maxfill_per_fifo(log_dir):
    """Reads untruncated MaxCount from each gauge log."""
    fills = {}
    for fname in os.listdir(log_dir):
        match = re.search(r"MaxFill:\s*(\d+)", open(os.path.join(log_dir, fname)).read())
        if match:
            fills[os.path.splitext(fname)[0]] = int(match.group(1))
    return fills


@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_fifo_gauge_maxcount_overflow_is_fatal():
    """Checks MaxCount > COUNT_WIDTH capacity is fatal."""
    if which("xsim") is None:
        pytest.skip("xsim not available")
    tmpdir = make_build_dir("test_fifo_maxcount_overflow_")
    (rc, out) = run_xsim(tmpdir, "fifo_gauge_overflow_fires_tb")
    assert rc != 0, "Overflowing the maxcount port did not abort the simulation:\n" + out
    assert "NO_OVERFLOW_DETECTED" not in out, "Fill ran to completion without tripping the guard"
    assert "COUNT_WIDTH" in out, "Simulation failed, but not with the overflow diagnostic:\n" + out


@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_fifo_gauge_maxcount_no_false_positive():
    """Checks MaxCount < COUNT_WIDTH capacity is quiet.

    Guards `Q.size >= 2**COUNT_WIDTH`, which overflows to 0 at COUNT_WIDTH=32.
    """
    if which("xsim") is None:
        pytest.skip("xsim not available")
    tmpdir = make_build_dir("test_fifo_maxcount_no_false_positive_")
    (rc, out) = run_xsim(tmpdir, "fifo_gauge_overflow_quiet_tb")
    assert rc == 0, "Guard fired on a fill well within the maxcount port's range:\n" + out
    assert "NO_FALSE_POSITIVE maxcount=19999" in out, "Unexpected gauge behaviour:\n" + out


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("depth", [2, 32, 1024, 150526])
def test_small_fifo_counter_overflow(depth):
    """Checks COUNT_WIDTH is not derived from insertion_depth."""
    shape = [1, 32]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    fifo_node = helper.make_node(
        "StreamingFIFO_rtl",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        depth=depth,
        folded_shape=shape,
        normal_shape=shape,
        dataType="INT4",
        impl_style="rtl",
    )
    graph = helper.make_graph(nodes=[fifo_node], name="fifo_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="fifo-model"))
    model.set_tensor_datatype("inp", DataType["INT4"])
    model.set_tensor_datatype("outp", DataType["INT4"])

    inst = getCustomOp(model.graph.node[0])
    code_gen_dir = make_build_dir("test_fifo_count_width_")
    inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
    inst.generate_hdl(model, FPGAPART, 10.0)

    with open(os.path.join(code_gen_dir, inst.get_verilog_top_module_name() + ".v")) as f:
        gen = f.read()
    # Gauge must represent fills above nominal depth
    assert ".COUNT_WIDTH(32)" in gen, "count/maxcount ports were sized from the depth"


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_fifosizing_residual_matches_gauge_maxfill():
    """Checks sized depth matches MaxCount > insertion_depth."""
    log_dir = make_build_dir("test_maxcount_sizing_logs_")
    model = make_residual_modelwrapper()
    model = model.transform(SpecializeLayers(FPGAPART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(
        InsertAndSetFIFODepths(FPGAPART, max_depth=MAX_DEPTH, debug_log_dir=log_dir)
    )

    observed = gauge_maxfill_per_fifo(log_dir)
    assert observed, "No gauge logs were produced, so nothing was actually measured"

    # Guard against passing for the wrong reason
    peak = max(observed.values())
    assert peak >= WRAP_AT, (
        "Residual skew was too small to exercise the truncation: peak observed "
        "fill %d, need >= %d. The test model no longer stresses the sizing path." % (peak, WRAP_AT)
    )

    sized = {}
    for node in model.get_nodes_by_op_type("StreamingFIFO_rtl"):
        sized[node.name] = getCustomOp(node).get_nodeattr("depth")

    mismatches = []
    for name, fill in observed.items():
        if name not in sized:
            continue
        # optimize_depth() floors small depths at 32, and a
        # truncated readout always lands below true fill
        if fill > 32 and sized[name] != fill:
            mismatches.append("%s: gauge observed %d, FIFO sized to %d" % (name, fill, sized[name]))

    assert not mismatches, (
        "FIFO depth does not match the occupancy the gauge measured, which means "
        "maxcount was truncated on readout:\n  " + "\n  ".join(mismatches)
    )
