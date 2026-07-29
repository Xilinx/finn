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

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.util.basic import part_map
from finn.util.platforms import DEFAULT_RES_LIMITS, platforms
from finn.util.test import load_test_checkpoint_or_skip


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


# desired frames per second
@pytest.mark.parametrize("target_fps", [30, 10**5, 10**7])
# target chip or board
@pytest.mark.parametrize("platform", ["Pynq-Z1", "Ultra96", "U200"])
@pytest.mark.parametrize("style", ["naive", "optimizer"])
@pytest.mark.fpgadataflow
def test_set_folding(target_fps, platform, style):
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 5)

    model = model.transform(GiveUniqueNodeNames())
    parent_model = model.transform(CreateDataflowPartition())
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    clk_ns = 5
    target_cycles_per_frame = int((10**9 / clk_ns) / target_fps)
    dataflow_model = dataflow_model.transform(
        SetFolding(target_cycles_per_frame, platform=platform, style=style, folding_effort=200)
    )

    exp_cycles_dict = dataflow_model.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame = max(exp_cycles_dict.values())

    min_cycles = dict()
    min_cycles["Pynq-Z1"] = 128
    min_cycles["Ultra96"] = 64
    min_cycles["U200"] = 1

    assert achieved_cycles_per_frame <= max(
        min_cycles[platform], target_cycles_per_frame
    ), "Folding target not met"


def _optimizer_fold(nnodes=5, **kwargs):
    """Fold an nnodes-deep MVAU model with the optimizer style (defaults suited
    to fast, deterministic unit tests)."""
    model = make_multi_fclayer_model(
        128, DataType["INT4"], DataType["INT2"], DataType["INT16"], nnodes
    )
    model = model.transform(GiveUniqueNodeNames())
    kwargs.setdefault("platform", "U250")
    kwargs.setdefault("style", "optimizer")
    kwargs.setdefault("folding_effort", 50)
    kwargs.setdefault("seed", 0)
    return model.transform(SetFolding(**kwargs))


@pytest.mark.fpgadataflow
def test_set_folding_maximize_throughput():
    # target_cycles_per_frame=None => maximize throughput within the resource
    # budget (no user-supplied ceiling). On a large device (U250) the optimizer
    # must actually fold -- i.e. run strictly faster than the fully unfolded MVAU
    # stack (128*128 cycles) -- while keeping resource usage within budget.
    unfolded = 128 * 128
    model = _optimizer_fold(target_cycles_per_frame=None)
    achieved = max(model.analysis(exp_cycles_per_layer).values())
    assert 0 < achieved < unfolded
    budget_lut = DEFAULT_RES_LIMITS[0] * sum(
        r["LUT"] for r in platforms["U250"](1).resource_count_dict.values()
    )
    total_lut = sum(
        getCustomOp(n).node_res_estimation(part_map["U250"]).get("LUT", 0) for n in model.graph.node
    )
    assert total_lut <= budget_lut


@pytest.mark.fpgadataflow
def test_set_folding_naive_requires_target():
    # the naive folder has no maximize-throughput mode, so a None target (which
    # means "maximize" for the optimizer) must be rejected with a clear error
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 2)
    model = model.transform(GiveUniqueNodeNames())
    with pytest.raises(ValueError):
        model.transform(SetFolding(target_cycles_per_frame=None, style="naive"))


@pytest.mark.parametrize("style", ["optimized", "Optimizer", "greedy", ""])
def test_set_folding_rejects_unknown_style(style):
    # An unrecognized style used to fall through to the optimizer, so a
    # misspelling ("optimized" for "optimizer") silently ran a folder the caller
    # never asked for and the build still succeeded.
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 2)
    model = model.transform(GiveUniqueNodeNames())
    with pytest.raises(ValueError, match="unknown SetFolding style"):
        model.transform(SetFolding(target_cycles_per_frame=1000, style=style))


@pytest.mark.parametrize("prefer_memory", ["bram", "uram"])
@pytest.mark.fpgadataflow
def test_set_folding_prefer_memory(prefer_memory):
    # the always-on resource-type pass must still meet the throughput target and
    # must never pick the mis-estimated "distributed" ram_style (excluded so the
    # BRAM<->URAM trade stays honest)
    model = _optimizer_fold(target_cycles_per_frame=2000, prefer_memory=prefer_memory)
    for node in model.graph.node:
        inst = getCustomOp(node)
        if "ram_style" in inst.get_nodeattr_types():
            assert inst.get_nodeattr("ram_style") != "distributed"
    assert max(model.analysis(exp_cycles_per_layer).values()) <= 2000


@pytest.mark.xfail(
    strict=True,
    reason="FINN's node_res_estimation is currently blind to resType on RTL "
    "MVAU/VVAU, so the optimizer's LUT<->DSP trade is a no-op there. Remove this "
    "xfail once the estimator scores resType (the optimizer already tunes it "
    "wherever the estimate responds).",
)
@pytest.mark.fpgadataflow
def test_resType_changes_rtl_mvau_estimate():
    # a folded RTL MVAU *should* estimate differently for resType "lut" vs "dsp";
    # today it does not, which is exactly why the optimizer skips this inert lever
    ch = 256
    W = np.random.randint(-8, 8, size=(ch, ch)).astype(np.float32)
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch])
    node = helper.make_node(
        "MVAU_rtl",
        ["inp", "w"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=ch,
        MH=ch,
        SIMD=16,
        PE=16,
        inputDataType="INT4",
        weightDataType="INT4",
        outputDataType="INT16",
        noActivation=1,
    )
    model = ModelWrapper(qonnx_make_model(helper.make_graph([node], "g", [inp], [outp])))
    model.set_initializer("w", W)
    model.set_tensor_datatype("w", DataType["INT4"])
    model.set_tensor_datatype("inp", DataType["INT4"])
    model = model.transform(GiveUniqueNodeNames())
    inst = getCustomOp(model.graph.node[0])
    part = part_map["U250"]
    inst.set_nodeattr("resType", "lut")
    lut_est = inst.node_res_estimation(part)
    inst.set_nodeattr("resType", "dsp")
    dsp_est = inst.node_res_estimation(part)
    assert lut_est != dsp_est, "resType does not affect the RTL MVAU estimate"
