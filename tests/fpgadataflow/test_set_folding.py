# Copyright (c) 2020, Xilinx
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

import copy
import json
import math
import numpy as np
import os
from functools import partial
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.builder.build_dataflow_steps as build_steps
from finn.analysis.fpgadataflow.dataflow_performance import folding_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_recursive,
)
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.set_folding import (
    ResourceAwareFoldingPPO,
    SetFolding,
    divisors,
    resource_capacity,
)
from finn.util.basic import make_build_dir, pynq_part_map, vitis_part_map
from finn.util.platforms import platforms
from finn.util.test import load_test_checkpoint_or_skip


def test_resource_capacity_vck190():
    expected = {
        "LUT": 899840.0,
        "FF": 1799680.0,
        "BRAM_18K": 1934.0,
        "URAM": 463.0,
        "DSP": 1968.0,
    }
    assert resource_capacity("VCK190", None) == expected
    assert resource_capacity(None, "xcvc1902-vsva2197-2MP-e-S") == expected


def make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes):
    W = np.random.randint(wdt.min(), wdt.max() + 1, size=(ch, ch))
    W = W.astype(np.float32)

    T = np.random.randint(tdt.min(), tdt.max() + 1, size=(ch, 2 ** adt.bitwidth() - 1))
    T = T.astype(np.float32)

    tensors = []
    tensors.append(helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch]))
    for i in range(1, nnodes):
        inter = helper.make_tensor_value_info("inter_" + str(i), TensorProto.FLOAT, [1, ch])
        tensors.append(inter)
    tensors.append(helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch]))

    FCLayer_nodes = []
    for i in range(nnodes):
        pe = 1
        simd = 1
        FCLayer_nodes += [
            helper.make_node(
                "MVAU_hls",
                [tensors[i].name, "weights_" + str(i), "thresh_" + str(i)],
                [tensors[i + 1].name],
                domain="finn.custom_op.fpgadataflow.hls",
                backend="fpgadataflow",
                MW=ch,
                MH=ch,
                SIMD=simd,
                PE=pe,
                inputDataType=adt.name,
                weightDataType=wdt.name,
                outputDataType=adt.name,
                ActVal=0,
                binaryXnorMode=0,
                noActivation=0,
            )
        ]

    graph = helper.make_graph(
        nodes=FCLayer_nodes,
        name="fclayer_graph",
        inputs=[tensors[0]],
        outputs=[tensors[-1]],
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", adt)
    model.set_tensor_datatype("outp", adt)

    for i in range(1, nnodes + 1):
        if tensors[i].name != "outp":
            model.graph.value_info.append(tensors[i])
        model.set_initializer("weights_" + str(i - 1), W)
        model.set_initializer("thresh_" + str(i - 1), T)
        model.set_tensor_datatype("weights_" + str(i - 1), wdt)
        model.set_tensor_datatype("thresh_" + str(i - 1), tdt)

    return model


def make_single_fclayer_model(mw, mh, wdt, adt, tdt):
    W = np.random.randint(wdt.min(), wdt.max() + 1, size=(mw, mh)).astype(np.float32)
    T = np.random.randint(tdt.min(), tdt.max() + 1, size=(mh, 2 ** adt.bitwidth() - 1))
    T = T.astype(np.float32)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    mvau_node = helper.make_node(
        "MVAU_hls",
        ["inp", "weights", "thresh"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType=adt.name,
        weightDataType=wdt.name,
        outputDataType=adt.name,
        ActVal=0,
        binaryXnorMode=0,
        noActivation=0,
    )

    graph = helper.make_graph(
        nodes=[mvau_node],
        name="single_fclayer_graph",
        inputs=[inp],
        outputs=[outp],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="single-fclayer-model"))
    model.set_initializer("weights", W)
    model.set_initializer("thresh", T)
    model.set_tensor_datatype("inp", adt)
    model.set_tensor_datatype("outp", adt)
    model.set_tensor_datatype("weights", wdt)
    model.set_tensor_datatype("thresh", tdt)
    return model


def make_single_fclayer_loop_model(mw, mh, wdt, adt, tdt, iteration=12):
    body = make_single_fclayer_model(mw, mh, wdt, adt, tdt)
    inp = helper.make_tensor_value_info("loop_inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("loop_outp", TensorProto.FLOAT, [1, mh])
    loop_node = helper.make_node(
        "FINNLoop",
        ["loop_inp"],
        ["loop_outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        body=body.graph,
        iteration=iteration,
        inputDataType=adt.name,
        outputDataType=adt.name,
    )
    graph = helper.make_graph([loop_node], "single_fclayer_loop", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="single-fclayer-loop"))
    model.set_tensor_datatype("loop_inp", adt)
    model.set_tensor_datatype("loop_outp", adt)
    return model


def make_outer_shuffle_model():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 197, 192])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 3, 197, 64])
    node = helper.make_node(
        "OuterShuffle_hls",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        data_type="INT4",
        in_shape=[1, 197, 192],
        out_shape=[1, 3, 197, 64],
        transpose_in_shape=[1, 197, 3, 64],
        transpose_out_shape=[1, 3, 197, 64],
        loop_coeffs=[37824, 64, 192, 1],
        perm=[0, 2, 1, 3],
        SIMD=1,
        NumChannels=64,
    )
    graph = helper.make_graph([node], "outer_shuffle_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="outer-shuffle-model"))
    model.set_tensor_datatype("inp", DataType["INT4"])
    model.set_tensor_datatype("outp", DataType["INT4"])
    return model


@pytest.mark.fpgadataflow
def test_finnloop_throughput_accounts_for_reused_body_work():
    model = make_single_fclayer_loop_model(
        16, 16, DataType["INT4"], DataType["INT2"], DataType["INT16"], iteration=12
    )
    body_cycles = 16 * 16

    performance = folding_performance(model)

    assert performance["max_cycles"] == 12 * (body_cycles + 40)
    assert performance["max_cycles"] > body_cycles * 12


@pytest.mark.fpgadataflow
def test_resource_aware_folding_optimizes_loop_body_with_one_budget():
    model = make_single_fclayer_loop_model(
        16, 16, DataType["INT4"], DataType["INT2"], DataType["INT16"], iteration=12
    )
    model = model.transform(GiveUniqueNodeNames())
    optimizer = ResourceAwareFoldingPPO(
        model,
        600,
        vitis_part_map["U200"],
        board="U200",
        episodes=0,
        rollout_steps=0,
    )

    assert any(knob.scope_path for knob in optimizer.knobs)
    model = optimizer.optimize()

    performance = folding_performance(model)
    resources = res_estimation_recursive(model, vitis_part_map["U200"])
    body = getCustomOp(model.get_nodes_by_op_type("FINNLoop")[0]).get_nodeattr("body")
    body_mvau = getCustomOp(body.get_nodes_by_op_type("MVAU_hls")[0])
    assert performance["max_cycles"] <= 600
    assert body_mvau.get_nodeattr("PE") * body_mvau.get_nodeattr("SIMD") > 1
    assert any(name.endswith("MVAU_hls_0") and "/" in name for name in resources)


@pytest.mark.fpgadataflow
def test_resource_aware_outer_shuffle_only_advertises_legal_simd():
    model = make_outer_shuffle_model().transform(GiveUniqueNodeNames())
    optimizer = ResourceAwareFoldingPPO(
        model,
        100_000,
        "xcvc1902-vsva2197-2MP-e-S",
        board="VCK190",
        episodes=0,
        rollout_steps=0,
    )
    simd_knob = next(knob for knob in optimizer.knobs if knob.attr_name == "SIMD")

    assert simd_knob.values == tuple(divisors(64))
    assert 96 not in simd_knob.values
    assert 192 not in simd_knob.values
    for simd in simd_knob.values:
        node_inst = getCustomOp(model.graph.node[0])
        node_inst.set_nodeattr("SIMD", simd)
        node_inst = getCustomOp(model.graph.node[0])
        assert node_inst.get_nodeattr("SIMD") == simd
        node_inst.get_folded_input_shape()
        node_inst.get_folded_output_shape()


@pytest.mark.fpgadataflow
def test_outer_shuffle_throughput_includes_recurring_frame_boundary_bubble():
    model = make_outer_shuffle_model().transform(GiveUniqueNodeNames())
    node_inst = getCustomOp(model.graph.node[0])
    transactions = int(np.prod(node_inst.get_folded_output_shape()[:-1]))
    optimizer = ResourceAwareFoldingPPO(
        model,
        100_000,
        "xcvc1902-vsva2197-2MP-e-S",
        board="VCK190",
        episodes=0,
        rollout_steps=0,
    )

    performance = folding_performance(model)
    optimizer_performance = optimizer._scope_throughput_performance()

    assert node_inst.get_exp_cycles() > transactions
    assert performance["max_cycles"] == node_inst.get_exp_cycles()
    assert optimizer_performance["max_cycles"] == performance["max_cycles"]


@pytest.mark.fpgadataflow
def test_outer_shuffle_boundary_bubbles_accumulate_along_stream_path():
    model = make_outer_shuffle_model()
    first_shuffle = model.graph.node[0]
    second_shuffle = copy.deepcopy(first_shuffle)
    first_shuffle.output[0] = "middle"
    second_shuffle.input[0] = "middle"
    model.graph.node.append(second_shuffle)
    model = model.transform(GiveUniqueNodeNames())

    first_inst = getCustomOp(model.graph.node[0])
    transactions = int(np.prod(first_inst.get_folded_output_shape()[:-1]))
    boundary_cycles = first_inst.get_exp_cycles() - transactions
    optimizer = ResourceAwareFoldingPPO(
        model,
        100_000,
        "xcvc1902-vsva2197-2MP-e-S",
        board="VCK190",
        episodes=0,
        rollout_steps=0,
    )

    performance = folding_performance(model)
    optimizer_performance = optimizer._scope_throughput_performance()

    assert boundary_cycles > 0
    expected_cycles = transactions + boundary_cycles + (boundary_cycles + 1) // 2
    assert performance["max_cycles"] == expected_cycles
    assert optimizer_performance["max_cycles"] == performance["max_cycles"]


@pytest.mark.fpgadataflow
def test_resource_aware_folding_uniquifies_unnamed_root_nodes():
    model = make_multi_fclayer_model(16, DataType["INT4"], DataType["INT2"], DataType["INT16"], 2)
    for node in model.graph.node:
        node.name = ""

    optimizer = ResourceAwareFoldingPPO(
        model,
        1_000,
        vitis_part_map["U200"],
        board="U200",
        episodes=0,
        rollout_steps=0,
    )

    names = [node.name for node in optimizer.scopes[()].graph.node]
    assert all(names)
    assert len(names) == len(set(names))


@pytest.mark.fpgadataflow
def test_builder_resource_aware_folding_owns_and_reports_loop_body():
    model = make_single_fclayer_loop_model(
        16, 16, DataType["INT4"], DataType["INT2"], DataType["INT16"], iteration=12
    )
    output_dir = make_build_dir("build_resource_aware_loop_")
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        board="U200",
        target_fps=100_000,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    )

    model = build_steps.step_target_fps_parallelization(model, cfg)

    performance = folding_performance(model)
    body = getCustomOp(model.get_nodes_by_op_type("FINNLoop")[0]).get_nodeattr("body")
    body_mvau = getCustomOp(body.get_nodes_by_op_type("MVAU_hls")[0])
    assert performance["max_cycles"] <= cfg._resolve_cycles_per_frame()
    assert body_mvau.get_nodeattr("PE") * body_mvau.get_nodeattr("SIMD") > 1

    with open(output_dir + "/auto_folding_config.json") as f:
        folding_config = json.load(f)
    body_configs = {
        name: attrs
        for name, attrs in folding_config.items()
        if "FINNLoop_0_body" in name and "MVAU_hls" in name
    }
    assert len(body_configs) == 1
    assert next(iter(body_configs.values()))["PE"] * next(iter(body_configs.values()))["SIMD"] > 1

    model = build_steps.step_generate_estimate_reports(model, cfg)
    with open(output_dir + "/report/estimate_layer_resources.json") as f:
        resource_report = json.load(f)
    with open(output_dir + "/report/estimate_network_performance.json") as f:
        performance_report = json.load(f)
    nested_resources = {
        name: resources
        for name, resources in resource_report.items()
        if name != "total" and "/" in name
    }
    assert nested_resources
    for resource_name in ["LUT", "BRAM_18K", "URAM", "DSP"]:
        expected_total = sum(
            float(resources.get(resource_name, 0.0))
            for name, resources in resource_report.items()
            if name != "total"
        )
        assert resource_report["total"][resource_name] == expected_total
    assert performance_report["max_cycles"] == performance["max_cycles"]
    assert performance_report["estimated_throughput_fps"] >= cfg.target_fps


@pytest.mark.parametrize("mlo", [False, True])
def test_rtlsim_performance_compares_folding_estimate_for_all_models(tmp_path, monkeypatch, mlo):
    if mlo:
        model = make_single_fclayer_loop_model(
            16,
            16,
            DataType["INT4"],
            DataType["INT2"],
            DataType["INT16"],
            iteration=12,
        )
    else:
        model = make_single_fclayer_model(
            16, 16, DataType["INT4"], DataType["INT2"], DataType["INT16"]
        )
    model = model.transform(GiveUniqueNodeNames())
    estimated_cycles = folding_performance(model)["max_cycles"]
    if mlo:
        loop_inst = getCustomOp(model.get_nodes_by_op_type("FINNLoop")[0])
        body_estimated_cycles = folding_performance(loop_inst.get_nodeattr("body"))["max_cycles"]
        measured_cycles = body_estimated_cycles + 6
        expected_effective_cycles = loop_inst.get_nodeattr("iteration") * (measured_cycles + 40)
    else:
        measured_cycles = estimated_cycles + 6
        expected_effective_cycles = measured_cycles

    def fake_xsi_fifosim(model, batch_size, max_iters, behav):
        assert not model.get_nodes_by_op_type("FINNLoop")
        assert batch_size == (5 if mlo else 2)
        assert max_iters > 0
        assert behav is False
        steady_state_frames = batch_size - 1
        return {
            "cycles": 50_000 + steady_state_frames * measured_cycles,
            "latency_cycles": 50_000,
            "interval_cycles": measured_cycles,
            "interval_valid": 1,
            "completed_output_frames": batch_size,
            "steady_state_frames": steady_state_frames,
            "steady_state_cycles": steady_state_frames * measured_cycles,
            "TIMEOUT": 0,
            "UNFINISHED_INS": 0,
            "UNFINISHED_OUTS": 0,
        }

    monkeypatch.setattr(build_steps, "xsi_fifosim", fake_xsi_fifosim)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=str(tmp_path),
        synth_clk_period_ns=5.0,
        rtlsim_batch_size=4 if mlo else 2,
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )

    build_steps.step_measure_rtlsim_performance(model, cfg)

    with open(tmp_path / "report/rtlsim_performance.json") as report_file:
        report = json.load(report_file)
    assert report["interval_is_steady_state"] is True
    assert report["estimated_interval_cycles"] == estimated_cycles
    assert report["estimate_vs_rtlsim_cycles"] == estimated_cycles - expected_effective_cycles
    if mlo:
        assert report["measurement_scope"] == "finnloop_body_capacity"
        assert report["body_interval_cycles"] == measured_cycles
        assert report["interval_cycles"] == expected_effective_cycles
        assert report["effective_interval_cycles"] == expected_effective_cycles
        assert report["estimate_comparison_interval_source"] == "effective_interval_cycles"
    else:
        assert report["interval_cycles"] == measured_cycles
        assert "effective_interval_cycles" not in report
        assert report["estimate_comparison_interval_source"] == "interval_cycles"


@pytest.mark.fpgadataflow
def test_rtl_mvau_pumped_compute_halves_physical_simd_dsps():
    model = make_single_fclayer_model(12, 12, DataType["INT4"], DataType["INT2"], DataType["INT16"])
    node = model.graph.node[0]
    node.op_type = "MVAU_rtl"
    node.domain = "finn.custom_op.fpgadataflow.rtl"
    inst = getCustomOp(node)
    inst.set_nodeattr("PE", 6)
    inst.set_nodeattr("SIMD", 12)
    inst.set_nodeattr("pumpedCompute", 0)
    unpumped_dsps = inst.dsp_estimation("xcvc1902-vsva2197-2MP-e-S")
    inst.set_nodeattr("pumpedCompute", 1)
    pumped_dsps = inst.dsp_estimation("xcvc1902-vsva2197-2MP-e-S")

    assert unpumped_dsps == 24
    assert pumped_dsps == 12


@pytest.mark.fpgadataflow
def test_resource_aware_rtl_mvau_simd_is_not_limited_by_hls_weight_width():
    model = make_single_fclayer_model(
        192, 12, DataType["INT4"], DataType["INT2"], DataType["INT16"]
    )
    node = model.graph.node[0]
    node.op_type = "MVAU_rtl"
    node.domain = "finn.custom_op.fpgadataflow.rtl"
    model = model.transform(GiveUniqueNodeNames())
    optimizer = ResourceAwareFoldingPPO(
        model,
        10**6,
        "xcvc1902-vsva2197-2MP-e-S",
        board="VCK190",
        mvau_wwidth_max=36,
        episodes=0,
        rollout_steps=0,
    )
    simd_knob = next(knob for knob in optimizer.knobs if knob.attr_name == "SIMD")

    assert 192 in simd_knob.values


@pytest.mark.fpgadataflow
def test_resource_minimizer_can_jump_over_worse_intermediate_ram_style():
    model = make_single_fclayer_model(
        192, 1000, DataType["INT8"], DataType["INT8"], DataType["INT32"]
    )
    node = model.graph.node[0]
    node.op_type = "MVAU_rtl"
    node.domain = "finn.custom_op.fpgadataflow.rtl"
    model = model.transform(GiveUniqueNodeNames())
    optimizer = ResourceAwareFoldingPPO(
        model,
        100_000,
        "xcvc1902-vsva2197-2MP-e-S",
        board="VCK190",
        episodes=0,
        rollout_steps=0,
    )
    balanced = optimizer._balanced_target_start()
    ram_index = next(
        index for index, knob in enumerate(optimizer.knobs) if knob.attr_name == "ram_style"
    )
    ram_knob = optimizer.knobs[ram_index]
    auto_indices = list(balanced["indices"])
    auto_indices[ram_index] = ram_knob.values.index("auto")
    auto_eval = optimizer._evaluate_indices(tuple(auto_indices))
    distributed_indices = list(auto_eval["indices"])
    distributed_indices[ram_index] = ram_knob.values.index("distributed")
    distributed_eval = optimizer._evaluate_indices(tuple(distributed_indices))

    assert auto_eval["meets_target"]
    assert distributed_eval["meets_target"]
    assert distributed_eval["resource_score"] < auto_eval["resource_score"]

    optimizer.resource_limit = (
        auto_eval["resource_pressure"] + distributed_eval["resource_pressure"]
    ) / 2
    optimizer.eval_cache.clear()
    auto_eval = optimizer._evaluate_indices(tuple(auto_indices))
    distributed_eval = optimizer._evaluate_indices(tuple(distributed_indices))
    assert not auto_eval["valid"]
    assert distributed_eval["valid"]

    repaired = optimizer._repair_meeting_capacity(auto_eval)
    assert repaired["valid"]
    assert repaired["meets_target"]

    minimized = optimizer._minimize_meeting_resources(auto_eval)
    optimizer._set_indices(minimized["indices"])
    assert minimized == auto_eval

    minimized = optimizer._minimize_meeting_resources(repaired)
    optimizer._set_indices(minimized["indices"])
    assert minimized["meets_target"] and minimized["valid"]
    assert minimized["resource_score"] <= repaired["resource_score"]
    assert getCustomOp(model.graph.node[0]).get_nodeattr("ram_style") == "distributed"


# desired frames per second
@pytest.mark.parametrize("target_fps", [30, 10**5, 10**7])
# target chip or board
@pytest.mark.parametrize("platform", ["Pynq-Z1", "Ultra96", "U200"])
@pytest.mark.fpgadataflow
def test_set_folding(target_fps, platform):
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 5)

    model = model.transform(GiveUniqueNodeNames())
    parent_model = model.transform(CreateDataflowPartition())
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    clk_ns = 5
    target_cycles_per_frame = int((10**9 / clk_ns) / target_fps)
    dataflow_model = dataflow_model.transform(SetFolding(target_cycles_per_frame))

    exp_cycles_dict = dataflow_model.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame = max(exp_cycles_dict.values())

    min_cycles = dict()
    min_cycles["Pynq-Z1"] = 128
    min_cycles["Ultra96"] = 64
    min_cycles["U200"] = 1

    assert achieved_cycles_per_frame <= max(
        min_cycles[platform], target_cycles_per_frame
    ), "Folding target not met"


@pytest.mark.parametrize("resource_aware", [False, True])
@pytest.mark.fpgadataflow
def test_set_folding_hls_mvau_uses_legal_minimum_simd(resource_aware):
    mw = 1025
    model = make_single_fclayer_model(mw, 8, DataType["INT4"], DataType["INT2"], DataType["INT16"])
    model = model.transform(GiveUniqueNodeNames())

    kwargs = {}
    if resource_aware:
        kwargs = {
            "fpgapart": pynq_part_map["Pynq-Z1"],
            "board": "Pynq-Z1",
            "ppo_episodes": 0,
            "ppo_rollout_steps": 0,
        }
    model = model.transform(SetFolding(10**9, **kwargs))

    mvau_node = model.get_nodes_by_op_type("MVAU_hls")[0]
    simd = getCustomOp(mvau_node).get_nodeattr("SIMD")
    assert simd >= math.ceil(mw / 1024)
    assert mw % simd == 0


@pytest.mark.fpgadataflow
def test_resource_aware_set_folding_min_resource_for_target():
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 1)
    model = model.transform(GiveUniqueNodeNames())

    target_cycles_per_frame = 512
    model = model.transform(
        SetFolding(
            target_cycles_per_frame,
            fpgapart=pynq_part_map["Pynq-Z1"],
            board="Pynq-Z1",
            ppo_episodes=1,
            ppo_rollout_steps=4,
        )
    )

    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame = max(exp_cycles_dict.values())
    assert achieved_cycles_per_frame == target_cycles_per_frame


@pytest.mark.fpgadataflow
def test_resource_aware_set_folding_max_throughput_when_target_too_high():
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 1)
    model = model.transform(GiveUniqueNodeNames())

    model = model.transform(
        SetFolding(
            1,
            fpgapart=vitis_part_map["U200"],
            board="U200",
            ppo_episodes=1,
            ppo_rollout_steps=4,
        )
    )

    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame = max(exp_cycles_dict.values())
    assert achieved_cycles_per_frame == 16


@pytest.mark.fpgadataflow
def test_resource_aware_set_folding_respects_board_budget_when_target_too_high():
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 5)
    model = model.transform(GiveUniqueNodeNames())

    model = model.transform(
        SetFolding(
            1,
            fpgapart=pynq_part_map["Pynq-Z1"],
            board="Pynq-Z1",
            ppo_episodes=1,
            ppo_rollout_steps=4,
        )
    )

    resource_estimates = model.analysis(partial(res_estimation, fpgapart=pynq_part_map["Pynq-Z1"]))
    total_lut = sum(node_res["LUT"] for node_res in resource_estimates.values())
    total_bram = sum(node_res["BRAM_18K"] for node_res in resource_estimates.values())
    total_dsp = sum(node_res["DSP"] for node_res in resource_estimates.values())
    board_resources = platforms["Pynq-Z1"]().compute_resources[0]

    assert total_lut <= board_resources[0]
    assert total_bram <= board_resources[2]
    assert total_dsp <= board_resources[4]
    assert (
        max(
            total_lut / board_resources[0],
            total_bram / board_resources[2],
            total_dsp / board_resources[4],
        )
        > 0.95
    )


@pytest.mark.fpgadataflow
def test_resource_aware_target_fps_real_tfc_estimate_only():
    tmp_output_dir = make_build_dir("build_resource_aware_tfc_")
    model_file = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/build_dataflow/model.onnx"
    specialize_config = (
        os.environ["FINN_ROOT"] + "/src/finn/qnn-data/build_dataflow/specialize_layers_config.json"
    )
    target_fps = 100000

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        synth_clk_period_ns=10.0,
        board="Pynq-Z1",
        target_fps=target_fps,
        mvau_wwidth_max=10000,
        specialize_layers_config_file=specialize_config,
        steps=build_cfg.estimate_only_dataflow_steps,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        save_intermediate_models=True,
    )
    build.build_dataflow_cfg(model_file, cfg)

    with open(tmp_output_dir + "/report/estimate_network_performance.json") as f:
        perf = json.load(f)
    with open(tmp_output_dir + "/report/estimate_layer_resources.json") as f:
        resources = json.load(f)["total"]

    board_resources = platforms["Pynq-Z1"]().compute_resources[0]
    assert perf["estimated_throughput_fps"] >= target_fps
    assert resources["LUT"] <= board_resources[0]
    assert resources["BRAM_18K"] <= board_resources[2]
    assert resources["DSP"] <= board_resources[4]


@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.skipif(
    os.environ.get("FINN_RUN_REAL_TFC_RTLSIM") != "1",
    reason="set FINN_RUN_REAL_TFC_RTLSIM=1 to run the real TFC rtlsim/OOC validation",
)
def test_resource_aware_target_fps_real_tfc_rtlsim_ooc():
    from finn import xsi  # noqa: PLC0415

    if not xsi.is_available():
        pytest.skip("finn_xsi is required for stitched-IP rtlsim")

    tmp_output_dir = make_build_dir("build_resource_aware_tfc_rtlsim_ooc_")
    model_file = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/build_dataflow/model.onnx"
    specialize_config = (
        os.environ["FINN_ROOT"] + "/src/finn/qnn-data/build_dataflow/specialize_layers_config.json"
    )
    target_fps = 100000
    steps = [
        "step_qonnx_to_finn",
        "step_tidy_up",
        "step_streamline",
        "step_convert_to_hw",
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        synth_clk_period_ns=10.0,
        board="Pynq-Z1",
        target_fps=target_fps,
        mvau_wwidth_max=10000,
        specialize_layers_config_file=specialize_config,
        steps=steps,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.OOC_SYNTH,
        ],
        auto_fifo_depths=True,
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod.LARGEFIFO_RTLSIM,
        fifosim_n_inferences=2,
        rtlsim_batch_size=100,
        save_intermediate_models=True,
    )
    build.build_dataflow_cfg(model_file, cfg)

    with open(tmp_output_dir + "/report/estimate_network_performance.json") as f:
        estimate_perf = json.load(f)
    with open(tmp_output_dir + "/report/estimate_layer_resources.json") as f:
        estimate_resources = json.load(f)["total"]
    with open(tmp_output_dir + "/report/estimate_layer_resources_post_fifo.json") as f:
        post_fifo_estimate_resources = json.load(f)["total"]
    with open(tmp_output_dir + "/report/rtlsim_performance.json") as f:
        rtlsim_perf = json.load(f)
    with open(tmp_output_dir + "/report/ooc_synth_and_timing.json") as f:
        ooc = json.load(f)

    assert estimate_perf["estimated_throughput_fps"] >= target_fps
    assert rtlsim_perf["stable_throughput[images/s]"] >= target_fps
    assert rtlsim_perf["TIMEOUT"] == 0
    assert rtlsim_perf["UNFINISHED_INS"] == 0
    assert rtlsim_perf["UNFINISHED_OUTS"] == 0

    est_fps = estimate_perf["estimated_throughput_fps"]
    rtlsim_fps = rtlsim_perf["stable_throughput[images/s]"]
    assert abs(est_fps - rtlsim_fps) / est_fps < 0.05

    board_resources = platforms["Pynq-Z1"]().compute_resources[0]
    ooc_bram18 = 2 * ooc["BRAM_36K"] + ooc["BRAM_18K"]
    assert ooc["WNS"] >= 0.0
    assert ooc["estimated_throughput_fps"] >= target_fps
    assert ooc["LUT"] <= board_resources[0]
    assert ooc_bram18 <= board_resources[2]
    assert ooc["DSP"] <= board_resources[4]

    assert estimate_resources["LUT"] <= board_resources[0]
    assert post_fifo_estimate_resources["LUT"] <= board_resources[0]
    assert post_fifo_estimate_resources["BRAM_18K"] <= board_resources[2]
    assert post_fifo_estimate_resources["DSP"] <= board_resources[4]
    assert post_fifo_estimate_resources["LUT"] >= estimate_resources["LUT"]
    assert post_fifo_estimate_resources["BRAM_18K"] >= estimate_resources["BRAM_18K"]
    assert ooc["LUT"] <= 1.5 * post_fifo_estimate_resources["LUT"]
    assert ooc_bram18 <= 1.5 * post_fifo_estimate_resources["BRAM_18K"]
