# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FIFO sizing of elementwise joins: an elementwise binary operation whose two
operands are both streams, fed by a fork.
"""

import pytest

import json
import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveTokenAccessVectors,
    is_stream_join,
)
from finn.transformation.fpgadataflow.fifo_sizing.derive_fifo_sizes import (
    join_window_floors,
)
from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

FPGA_PART = "xczu3eg-sbva484-1-e"
CLK_NS = 5.0
IDT = DataType["INT4"]
ODT = DataType["INT8"]


def make_fork_join_model(op_type, ch, pe, n_vec, rhs_style="input"):
    """``inp -> DuplicateStreams -> (a, b)``, ``b`` through a 1:8 and an 8:1
    width converter, then ``op_type(a, b)``. With ``rhs_style="const"`` the
    second operand is a constant and the node has a single streamed input."""
    shape = [1, n_vec, ch]
    tensors = ["inp", "a", "b", "b_wide", "b_narrow", "outp"]
    nodes = [
        helper.make_node(
            "DuplicateStreams_hls",
            ["inp"],
            ["a", "b"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            NumChannels=ch,
            PE=pe,
            NumOutputStreams=2,
            inputDataType=IDT.name,
            numInputVectors=[1, n_vec],
            # as the conversion to HW layers creates it
            outFIFODepths=[2, 2],
        )
    ]
    width = pe * IDT.bitwidth()
    for inw, outw, src, dst in [
        (width, 8 * width, "b", "b_wide"),
        (8 * width, width, "b_wide", "b_narrow"),
    ]:
        nodes.append(
            helper.make_node(
                "StreamingDataWidthConverter_hls",
                [src],
                [dst],
                domain="finn.custom_op.fpgadataflow.hls",
                backend="fpgadataflow",
                shape=shape,
                inWidth=inw,
                outWidth=outw,
                dataType=IDT.name,
            )
        )
    nodes.append(
        helper.make_node(
            op_type + "_hls",
            ["a", "b_narrow"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            lhs_shape=shape,
            rhs_shape=shape,
            out_shape=shape,
            lhs_dtype=IDT.name,
            rhs_dtype=IDT.name,
            out_dtype=ODT.name,
            lhs_style="input",
            rhs_style=rhs_style,
            PE=pe,
        )
    )
    vi = {t: helper.make_tensor_value_info(t, TensorProto.FLOAT, shape) for t in tensors}
    graph = helper.make_graph(
        nodes,
        "fork_join",
        [vi["inp"]],
        [vi["outp"]],
        value_info=[v for t, v in vi.items() if t not in ("inp", "outp")],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="fork-join"))
    for t in tensors:
        model.set_tensor_datatype(t, ODT if t == "outp" else IDT)
    # the sizer keys its bookkeeping by node name, as the builder leaves them
    return model.transform(GiveUniqueNodeNames())


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseMul", "ElementwiseSub"])
def test_elementwise_join_is_not_characterized_and_gets_floor(op_type):
    """A two-stream elementwise operation is a join: no TAV, transparent in the
    chained pass, and a join window floor on its shortcut input. Needs no IP:
    every other node of the graph has a tree model."""
    model = make_fork_join_model(op_type, ch=16, pe=2, n_vec=8)
    join = model.get_nodes_by_op_type(op_type + "_hls")[0]
    assert is_stream_join(join)

    model = model.transform(DeriveTokenAccessVectors(model, 0, "tree_model", None, None))
    join = model.get_nodes_by_op_type(op_type + "_hls")[0]
    assert getCustomOp(join).get_nodeattr("io_chrc_in") == ""

    hw_nodes = [n for n in model.graph.node if is_hls_node(n) or is_rtl_node(n)]
    floors = join_window_floors(model, hw_nodes)
    # the shortcut has to buffer what the fork writes while the 1:8 converter
    # in the long branch collects a word
    assert floors.get(join.input[0], 0) > 2


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseMul"])
def test_elementwise_with_constant_operand_is_not_a_join(op_type):
    model = make_fork_join_model(op_type, ch=16, pe=2, n_vec=8, rhs_style="const")
    assert not is_stream_join(model.get_nodes_by_op_type(op_type + "_hls")[0])


def fifo_depths(model):
    """``{"<producer>.out<i> -> <consumer>.in<j>": depth}`` for every FIFO."""
    depths = {}
    for node in model.graph.node:
        if not node.op_type.startswith("StreamingFIFO"):
            continue
        prod = model.find_producer(node.input[0])
        cons = model.find_consumer(node.output[0])
        src = (
            "input"
            if prod is None
            else "%s.out%d"
            % (
                prod.name,
                list(prod.output).index(node.input[0]),
            )
        )
        dst = (
            "output"
            if cons is None
            else "%s.in%d"
            % (
                cons.name,
                list(cons.input).index(node.output[0]),
            )
        )
        depths["%s -> %s" % (src, dst)] = getCustomOp(node).get_nodeattr("depth")
    return depths


def build_and_measure(model_file, build_dir, strategy):
    """Build to stitched IP with ``strategy``; returns (throughput, depths)."""
    output_dir = os.path.join(build_dir, strategy)
    folding = os.path.join(build_dir, "folding.json")
    with open(folding, "w") as f:
        json.dump({"Defaults": {}}, f)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        fpga_part=FPGA_PART,
        synth_clk_period_ns=CLK_NS,
        # the model already is the hardware graph: start where the convert phase ends
        steps=["phase_optimize_hardware", "phase_build_hardware", "phase_generate_outputs"],
        folding_config_file=folding,
        auto_fifo_depths=True,
        auto_fifo_strategy=strategy,
        rtlsim_batch_size=8,
        stitched_ip_gen_dcp=False,
        verify_steps=[],
        enable_build_pdb_debug=False,
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )
    assert (
        build.build_dataflow_cfg(model_file, cfg) == 0
    ), "%s build failed, see %s/build_dataflow.log" % (strategy, output_dir)
    with open(os.path.join(output_dir, "report", "rtlsim_performance.json")) as f:
        perf = json.load(f)
    assert perf["completed_output_frames"] == cfg.rtlsim_batch_size, (
        "%s: rtlsim did not complete all frames (deadlock?)" % strategy
    )
    assert perf["stable_throughput_valid"], "%s: no steady-state throughput" % strategy
    stitched = ModelWrapper(
        os.path.join(output_dir, "intermediate_models", "step_create_stitched_ip.onnx")
    )
    return float(perf["stable_throughput[images/s]"]), fifo_depths(stitched)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseMul", "ElementwiseSub"])
def test_fifosizing_elementwise_join(op_type):
    """Fork -> (shortcut, delayed branch) -> two-stream elementwise join, built
    with largefifo_rtlsim as the reference and with both heuristic strategies.
    Every build has to complete: heuristic_rtlsim used to stall characterizing
    any join but ElementwiseAdd. The heuristics should also reach the reference's
    stitched-IP throughput, which they do not yet (see the xfail below)."""
    build_dir = make_build_dir("fifosizing_elementwise_join_")
    model_file = os.path.join(build_dir, "model.onnx")
    make_fork_join_model(op_type, ch=16, pe=2, n_vec=8).save(model_file)

    results = {}
    for strategy in ["largefifo_rtlsim", "heuristic_analytical", "heuristic_rtlsim"]:
        fps, depths = build_and_measure(model_file, build_dir, strategy)
        results[strategy] = {"fps": fps, "depths": depths}

    summary_dir = os.path.join(os.environ["FINN_BUILD_DIR"], "fifosizing_elementwise_join")
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, op_type + ".json"), "w") as f:
        json.dump(dict(results, build_dir=build_dir), f, indent=2)

    ref = results["largefifo_rtlsim"]["fps"]
    ratios = {s: results[s]["fps"] / ref for s in ("heuristic_analytical", "heuristic_rtlsim")}
    low = {s: r for s, r in ratios.items() if r < 0.98}
    if low:
        # Known issue: the shortcut gets the join window floor (the 1:8
        # converter's window plus margin, 10) but the converters' pipeline
        # latency is not covered, while largefifo_rtlsim measures 15. All
        # elementwise joins, ElementwiseAdd included, reach about 59 %.
        pytest.xfail(
            "shortcut under-sized: %s of the reference throughput, depths %s vs %s"
            % (
                ", ".join("%s %.1f%%" % (s, 100 * r) for s, r in low.items()),
                results["heuristic_analytical"]["depths"],
                results["largefifo_rtlsim"]["depths"],
            )
        )
    robust_rmtree(build_dir)
