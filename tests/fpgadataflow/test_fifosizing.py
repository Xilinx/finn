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
from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    check_fifo_gauge_overflow,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.test import get_trained_network_and_ishape

FPGAPART = "xc7z020clg400-1"


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


def make_residual_modelwrapper():
    """Fork -> (short branch | long branch with buffers) -> join."""
    idt = DataType["INT4"]
    ch = 8
    pe = 8
    nvec = [1, 8, 8]  # folded tensor size, the natural insertion depth
    shape = nvec + [ch]
    full_w = ch * idt.bitwidth()
    narrow_w = idt.bitwidth()
    npair = 2  # width-converter pairs delaying the long branch

    nodes = []
    vi = []
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    nodes.append(
        helper.make_node(
            "DuplicateStreams",
            ["inp"],
            ["short", "long0"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            NumChannels=ch,
            NumOutputStreams=2,
            PE=pe,
            inputDataType=idt.name,
            numInputVectors=nvec,
            outFIFODepths=[2, 2],
        )
    )
    vi += [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in ("short", "long0")]

    # Narrow leg drains slowly, backing up short branch
    cur = "long0"
    for i in range(npair):
        narrow, wide = "lnarrow%d" % i, "long%d" % (i + 1)
        nodes.append(
            helper.make_node(
                "StreamingDataWidthConverter",
                [cur],
                [narrow],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                shape=shape,
                inWidth=full_w,
                outWidth=narrow_w,
                dataType=idt.name,
            )
        )
        nodes.append(
            helper.make_node(
                "StreamingDataWidthConverter",
                [narrow],
                [wide],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                shape=shape,
                inWidth=narrow_w,
                outWidth=full_w,
                dataType=idt.name,
            )
        )
        vi += [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in (narrow, wide)]
        cur = wide

    nodes.append(
        helper.make_node(
            "ElementwiseAdd",
            ["short", cur],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            lhs_shape=shape,
            rhs_shape=shape,
            out_shape=shape,
            lhs_dtype=idt.name,
            rhs_dtype=idt.name,
            out_dtype=idt.name,
            lhs_style="input",
            rhs_style="input",
            PE=pe,
            inFIFODepths=[2, 2],
        )
    )

    graph = helper.make_graph(nodes, "residual", [inp], [outp], value_info=vi)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="residual-maxcount"))
    for tname in ["inp", "outp"] + [x.name for x in vi]:
        model.set_tensor_datatype(tname, idt)
    model = model.transform(InferShapes())
    return model.transform(InferDataTypes())


def gauge_maxfill_per_fifo(log_dir):
    """Reads the peak MaxCount from each gauge log."""
    fills = {}
    for fname in os.listdir(log_dir):
        match = re.search(r"MaxFill:\s*(\d+)", open(os.path.join(log_dir, fname)).read())
        if match:
            fills[os.path.splitext(fname)[0]] = int(match.group(1))
    return fills


@pytest.mark.fpgadataflow
def test_fifo_gauge_overflow_sentinel_matches_rtl():
    """Ties the RTL gauge counter, the wrapper carrier and the Python sentinel together.

    The gauge counter width, the depth reader's overflow check and the wrapper port
    width are three separate definitions that must stay consistent; a silent mismatch
    is exactly the class of bug this guard exists to catch. This parses the two HDL
    sources so any drift trips the test instead of passing quietly.
    """
    hdl_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/fifo/hdl"
    with open(os.path.join(hdl_dir, "fifo_gauge.sv")) as f:
        gauge = f.read()
    with open(os.path.join(hdl_dir, "fifo_template.v")) as f:
        wrapper = f.read()

    # fifo_gauge.sv reports occupancy on a fixed-width counter and flags overflow when
    # it wraps from the all-ones value ('1). "int unsigned" is 32-bit by definition, so
    # the sentinel the gauge holds on overflow is 2^32-1.
    assert re.search(r"\bint\s+unsigned\s+MaxCount\b", gauge), (
        "fifo_gauge.sv no longer declares MaxCount as int unsigned; "
        "check_fifo_gauge_overflow assumes a 32-bit counter"
    )
    assert re.search(r"Count\s*!=\s*'1", gauge), (
        "fifo_gauge.sv no longer flags overflow at the all-ones sentinel; "
        "revisit check_fifo_gauge_overflow"
    )
    gauge_bits = 32  # SystemVerilog int unsigned
    sentinel = 2**gauge_bits - 1

    # The depth reader must reject exactly that sentinel and nothing below it
    check_fifo_gauge_overflow("fifo_ok", sentinel - 1)
    with pytest.raises(RuntimeError, match="overflowed"):
        check_fifo_gauge_overflow("fifo_bad", sentinel)

    # The wrapper only carries the gauge reading out to sizing, so its port width must
    # not be narrower than the gauge counter, else maxcount truncates on readout
    match = re.search(r"parameter\s+COUNT_WIDTH\s*=\s*(\d+)", wrapper)
    assert match, "fifo_template.v no longer defines a COUNT_WIDTH parameter"
    assert int(match.group(1)) >= gauge_bits, (
        "fifo_template.v COUNT_WIDTH (%s) is narrower than the gauge's %d-bit counter; "
        "maxcount would truncate on readout" % (match.group(1), gauge_bits)
    )


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_fifosizing_residual_matches_gauge_maxfill():
    """Checks the sized depth matches the peak occupancy the gauge measured."""
    # Tiny insertion depth so the residual skew backs FIFOs up well beyond it
    max_depth = 4
    # Peak fill the model must produce for the sizing comparison to be meaningful
    min_peak_fill = 8
    log_dir = make_build_dir("test_maxcount_sizing_logs_")
    model = make_residual_modelwrapper()
    model = model.transform(SpecializeLayers(FPGAPART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(
        InsertAndSetFIFODepths(FPGAPART, max_depth=max_depth, debug_log_dir=log_dir)
    )

    observed = gauge_maxfill_per_fifo(log_dir)
    assert observed, "No gauge logs were produced, so nothing was actually measured"

    # Guard against passing for the wrong reason
    peak = max(observed.values())
    assert peak >= min_peak_fill, (
        "Residual skew was too small to stress the sizing path: peak observed "
        "fill %d, need >= %d." % (peak, min_peak_fill)
    )

    sized = {}
    for node in model.get_nodes_by_op_type("StreamingFIFO_rtl"):
        sized[node.name] = getCustomOp(node).get_nodeattr("depth")

    mismatches = []
    for name, fill in observed.items():
        if name not in sized:
            continue
        # optimize_depth() floors small depths at 32; above that the sized depth
        # should equal the fill the gauge measured
        if fill > 32 and sized[name] != fill:
            mismatches.append("%s: gauge observed %d, FIFO sized to %d" % (name, fill, sized[name]))

    assert (
        not mismatches
    ), "FIFO depth does not match the occupancy the gauge measured:\n  " + "\n  ".join(mismatches)
