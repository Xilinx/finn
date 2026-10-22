# Copyright (C) 2026, Paderborn University
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

"""Selection and code generation of the distributed-arithmetic RTL MVU
(MVAU_rtl with mem_mode=internal_embedded), without Vivado."""

import pytest

import math
import numpy as np
import os
from functools import partial
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import (
    SpecializeDAMVAU,
    SpecializeLayers,
)

PART = "xczu28dr-ffvg1517-2-e"


def make_embedded_mvau_model(idt="UINT4", wdt="INT4", mw=16, mh=16, preferred="rtl", act=False):
    """Single MVAU (SIMD=PE=1) with static weights and mem_mode=internal_embedded,
    optionally with fused thresholds; bit widths minimized as
    step_minimize_bit_width_initial does before specialization."""
    idt, wdt = DataType[idt], DataType[wdt]
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    inputs = ["inp", "weights"]
    odt, act_val = DataType["INT32"], 0
    if act:
        inputs.append("thresh")
        odt, act_val = DataType["INT4"], DataType["INT4"].min()
    node = helper.make_node(
        "MVAU",
        inputs,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=act_val,
        binaryXnorMode=0,
        noActivation=0 if act else 1,
    )
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    graph = helper.make_graph([node], "mvau_da", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-da"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", W)
    if act:
        T = np.sort(np.random.randint(-64, 64, (mh, 15)).astype(np.float32), axis=1)
        model.set_tensor_datatype("thresh", DataType["INT32"])
        model.set_initializer("thresh", T)
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("mem_mode", "internal_embedded")
    inst.set_nodeattr("resType", "lut")
    inst.set_nodeattr("preferred_impl_style", preferred)
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    return model.transform(InferDataTypes())


@pytest.mark.fpgadataflow
def test_specialize_mvau_rtl_da_opt_in():
    model = make_embedded_mvau_model(preferred="rtl")
    model = model.transform(SpecializeLayers(PART))
    assert model.graph.node[0].op_type == "MVAU_rtl"
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("mem_mode") == "internal_embedded"
    assert inst._is_da_mode()


@pytest.mark.parametrize("preferred", ["hls", ""])
@pytest.mark.fpgadataflow
def test_specialize_mvau_embedded_stays_hls_without_opt_in(preferred):
    model = make_embedded_mvau_model(preferred=preferred)
    model = model.transform(SpecializeLayers(PART))
    assert model.graph.node[0].op_type == "MVAU_hls"
    assert getCustomOp(model.graph.node[0]).get_nodeattr("mem_mode") == "internal_embedded"


@pytest.mark.parametrize(
    "attr, value, reason",
    [
        ("binaryXnorMode", 1, "binaryXnorMode"),
        ("runtime_writeable_weights", 1, "runtime_writeable_weights"),
        ("resType", "dsp", "LUT-only"),
        ("TH", 2, "tiling"),
        ("pumpedMemory", 1, "clock pumping"),
    ],
)
@pytest.mark.fpgadataflow
def test_specialize_mvau_rtl_da_rejects(attr, value, reason):
    model = make_embedded_mvau_model(preferred="rtl")
    getCustomOp(model.graph.node[0]).set_nodeattr(attr, value)
    with pytest.raises(ValueError, match=reason):
        model.transform(SpecializeLayers(PART))


@pytest.mark.fpgadataflow
def test_specialize_mvau_rtl_da_rejects_embedded_thresholds():
    model = make_embedded_mvau_model(preferred="rtl", act=True)
    with pytest.raises(ValueError, match="standalone thresholds"):
        model.transform(SpecializeLayers(PART))


@pytest.mark.fpgadataflow
def test_mvau_rtl_da_codegen_rejects_partial_folding():
    """SIMD=MW and PE=MH is only checked at code generation time."""
    model = make_embedded_mvau_model(preferred="rtl")
    model = model.transform(SpecializeLayers(PART))
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", 8)
    inst.set_nodeattr("PE", 16)
    with pytest.raises(ValueError, match="SIMD=MW"):
        model.transform(PrepareIP(PART, 5))
    # fully unrolled: code generation succeeds and reports the solution
    inst.set_nodeattr("SIMD", 16)
    model = model.transform(PrepareIP(PART, 5))
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("da_cost") > 0
    assert inst.get_nodeattr("da_latency_cycles") >= 2
    assert inst.lut_estimation(PART) > 0
    assert inst.dsp_estimation(PART) == 0
    files = inst.get_rtl_file_list(abspath=True)
    assert all(os.path.isfile(f) for f in files), files
    assert any(f.endswith("mvu_da_axi.sv") for f in files)
    assert any(f.endswith("_da_core_wrapper.v") for f in files)


@pytest.mark.fpgadataflow
def test_mvau_rtl_da_estimate_runs_solver():
    """The estimate reports run before code generation; the analysis pass runs the
    solver so that the LUT estimate is based on the adder graph."""
    model = make_embedded_mvau_model(preferred="rtl")
    model = model.transform(SpecializeLayers(PART))
    model = model.transform(GiveUniqueNodeNames())
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", 16)
    inst.set_nodeattr("PE", 16)
    fallback = inst.lut_estimation(PART)
    res = model.analysis(partial(res_estimation, fpgapart=PART))
    inst = getCustomOp(model.graph.node[0])
    da_cost = inst.get_nodeattr("da_cost")
    assert da_cost > 0
    queue = 16 * inst.get_output_datatype().bitwidth()
    assert res["MVAU_rtl_0"]["LUT"] == math.ceil(da_cost) + queue
    assert res["MVAU_rtl_0"]["LUT"] != fallback
    assert res["MVAU_rtl_0"]["DSP"] == 0
    # partially folded: the solver is skipped and the fallback estimate remains
    model2 = make_embedded_mvau_model(preferred="rtl")
    model2 = model2.transform(SpecializeLayers(PART))
    model2 = model2.transform(GiveUniqueNodeNames())
    res2 = model2.analysis(partial(res_estimation, fpgapart=PART))
    assert getCustomOp(model2.graph.node[0]).get_nodeattr("da_cost") == 0
    assert res2["MVAU_rtl_0"]["LUT"] > 0


@pytest.mark.parametrize("act", [False, True])
@pytest.mark.fpgadataflow
def test_specialize_da_mvau_after_folding(act):
    """enable_da_mvau (SpecializeDAMVAU): only MVAUs that the folding already
    unrolls fully become the DA core; the folding is never changed, fused
    thresholds and explicit DSP resources keep the regular implementation."""
    model = make_embedded_mvau_model(preferred="", act=act)
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("mem_mode", "internal_decoupled")
    inst.set_nodeattr("resType", "auto")
    model = model.transform(SpecializeLayers(PART))
    assert getCustomOp(model.graph.node[0]).get_nodeattr("mem_mode") == "internal_decoupled"
    # partially folded: untouched
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", 8)
    inst.set_nodeattr("PE", 16)
    folded = model.transform(SpecializeDAMVAU())
    p_inst = getCustomOp(folded.graph.node[0])
    assert p_inst.get_nodeattr("mem_mode") == "internal_decoupled"
    assert (p_inst.get_nodeattr("SIMD"), p_inst.get_nodeattr("PE")) == (8, 16)
    # fully unrolled: DA core unless thresholds are fused
    inst.set_nodeattr("SIMD", 16)
    full = model.transform(SpecializeDAMVAU())
    node = full.graph.node[0]
    f_inst = getCustomOp(node)
    assert (f_inst.get_nodeattr("SIMD"), f_inst.get_nodeattr("PE")) == (16, 16)
    if act:
        assert f_inst.get_nodeattr("mem_mode") == "internal_decoupled"
    else:
        assert node.op_type == "MVAU_rtl"
        assert f_inst.get_nodeattr("mem_mode") == "internal_embedded"
        assert f_inst.get_nodeattr("resType") == "lut"
        assert f_inst._is_da_mode()
        assert node.name == model.graph.node[0].name
        # explicit DSP resources are respected
        inst.set_nodeattr("resType", "dsp")
        dsp = model.transform(SpecializeDAMVAU())
        assert getCustomOp(dsp.graph.node[0]).get_nodeattr("mem_mode") == "internal_decoupled"
