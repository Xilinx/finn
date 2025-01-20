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
import qonnx.custom_op.general.xnorpopcount as xp
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.derive_characteristic import DeriveCharacteristic
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T=None, tdt=None):
    mw = W.shape[0]
    mh = W.shape[1]
    assert mh % pe == 0
    assert mw % simd == 0

    # there are two ways to implement bipolar weights and inputs for
    # MatrixVectorActivation:
    # - specify their datatypes as such
    # - specify their datatypes as BINARY as use binaryXnorMode
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # we'll internally convert weights/inputs to binary and specify the
        # datatypes as such, and also set the binaryXnorMode attribute to 1
        export_wdt = DataType["BINARY"]
        export_idt = DataType["BINARY"]
        binary_xnor_mode = 1
    else:
        export_wdt = wdt
        export_idt = idt
        binary_xnor_mode = 0

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType["BIPOLAR"]:
            actval = 0
        else:
            actval = odt.min()
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
        actval = 0
        no_act = 1
    FCLayer_node = helper.make_node(
        "MVAU",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=export_idt.name,
        weightDataType=export_wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    if binary_xnor_mode:
        # convert bipolar to binary
        model.set_initializer("weights", (W + 1) / 2)
    else:
        model.set_initializer("weights", W)
    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)
    return model


def make_single_matmul_modelwrapper(ifm, ofm, idt, wdt, W):
    matmul_node = helper.make_node("MatMul", ["ifm", "weights"], ["ofm"])
    graph = helper.make_graph(nodes=[matmul_node], name="matmul_graph", inputs=[ifm], outputs=[ofm])

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("ifm", idt)
    model.set_tensor_datatype("weights", wdt)
    model.set_tensor_datatype(
        "ofm", DataType["INT32"]
    )  # At this step, the MatMul layer does not optimize the bit-width of the output datatype
    model.set_initializer("weights", W)
    # model.set_tensor_layout("ifm", DataLayout.NHWC)

    return model


def prepare_inputs(input_tensor, idt, wdt, inp_name="inp"):
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert bipolar to binary
        return {inp_name: (input_tensor + 1) / 2}
    else:
        return {inp_name: input_tensor}


# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType["BIPOLAR"], DataType["INT4"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["BIPOLAR"], DataType["INT4"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["BIPOLAR"], DataType["INT4"]])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2, 1])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1, 2, 1])
# HLS matrix width (input features)
@pytest.mark.parametrize("mw", [16])
# HLS matrix height (output features)
@pytest.mark.parametrize("mh", [16])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_mvau_hwop(idt, wdt, act, nf, sf, mw, mh):
    if nf == -1:
        nf = mh
    if sf == -1:
        sf = mw
    pe = mh // nf
    simd = mw // sf
    assert mh % pe == 0
    assert mw % sf == 0
    # generate weights
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    # generate input data
    x = gen_finn_dt_tensor(idt, (1, mw))
    if act is None:
        # no activation, produce accumulators
        T = None
        tdt = None
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            odt = DataType["UINT32"]
        else:
            odt = DataType["INT32"]
    else:
        odt = act
        (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
        # provide non-decreasing thresholds
        T = np.sort(T, axis=1)
        # generate thresholds for activation
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            tdt = DataType["UINT32"]
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType["INT32"]
    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    # prepare input data
    input_dict = prepare_inputs(x, idt, wdt)
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert inputs to binary and use xnorpopcountmatmul
        y = xp.xnorpopcountmatmul((x + 1) / 2, (W + 1) / 2)
    else:
        y = np.matmul(x, W)
    if T is not None:
        # y = multithreshold(y, T)
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            # y = 2 * y - 1
            y = multithreshold(y, T, 2, -1)
        else:
            # signed offset
            # y += act.min()
            y = multithreshold(y, T, 1, act.min())
    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all(), "cppsim hw-op failed"


# mem_mode: internal_embedded or internal_decoupled
@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled", "external"])
# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType["BIPOLAR"], DataType["INT4"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["BIPOLAR"], DataType["INT4"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["BIPOLAR"], DataType["INT4"]])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2, 1])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1, 2, 1])
# HLS matrix width (input features)
@pytest.mark.parametrize("mw", [16])
# HLS matrix height (output features)
@pytest.mark.parametrize("mh", [16])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_mvau_cppsim(mem_mode, idt, wdt, act, nf, sf, mw, mh):
    if nf == -1:
        nf = mh
    if sf == -1:
        sf = mw
    pe = mh // nf
    simd = mw // sf
    assert mh % pe == 0
    assert mw % sf == 0
    # generate weights
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    # generate input data
    x = gen_finn_dt_tensor(idt, (1, mw))
    if act is None:
        # no activation, produce accumulators
        T = None
        tdt = None
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            odt = DataType["UINT32"]
        else:
            odt = DataType["INT32"]
    else:
        odt = act
        (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
        # provide non-decreasing thresholds
        T = np.sort(T, axis=1)
        # generate thresholds for activation
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            tdt = DataType["UINT32"]
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType["INT32"]
    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    model = model.transform(GiveUniqueNodeNames())
    for node in model.graph.node:
        # lookup op_type in registry of CustomOps
        inst = getCustomOp(node)
        inst.set_nodeattr("mem_mode", mem_mode)
        # Note: only HLS-based MVAU layers execute CPPsim
        inst.set_nodeattr("preferred_impl_style", "hls")
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    # prepare input data
    input_dict = prepare_inputs(x, idt, wdt)
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert inputs to binary and use xnorpopcountmatmul
        y = xp.xnorpopcountmatmul((x + 1) / 2, (W + 1) / 2)
    else:
        y = np.matmul(x, W)
    if T is not None:
        y = multithreshold(y, T)
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y = 2 * y - 1
        else:
            # signed offset
            y += act.min()
    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all(), "cppsim hls-op failed"


# mem_mode: internal_embedded or internal_decoupled
@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled", "external"])
# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType["BIPOLAR"], DataType["INT4"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["BIPOLAR"], DataType["INT4"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["BIPOLAR"], DataType["INT4"]])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2, 1])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1, 2, 1])
# HLS matrix width (input features)
@pytest.mark.parametrize("mw", [16])
# HLS matrix height (output features)
@pytest.mark.parametrize("mh", [16])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_mvau_rtlsim(mem_mode, idt, wdt, act, nf, sf, mw, mh):
    if nf == -1:
        nf = mh
    if sf == -1:
        sf = mw
    pe = mh // nf
    simd = mw // sf
    assert mh % pe == 0
    assert mw % sf == 0
    # generate weights
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    # generate input data
    x = gen_finn_dt_tensor(idt, (1, mw))
    if act is None:
        # no activation, produce accumulators
        T = None
        tdt = None
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            odt = DataType["UINT32"]
        else:
            odt = DataType["INT32"]
    else:
        odt = act
        (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
        # provide non-decreasing thresholds
        T = np.sort(T, axis=1)
        # generate thresholds for activation
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            tdt = DataType["UINT32"]
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType["INT32"]
    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    for node in model.graph.node:
        # lookup op_type in registry of CustomOps
        inst = getCustomOp(node)
        inst.set_nodeattr("mem_mode", mem_mode)
        inst.set_nodeattr("preferred_impl_style", "hls")

    # prepare input data
    input_dict = prepare_inputs(x, idt, wdt)
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert inputs to binary and use xnorpopcountmatmul
        y = xp.xnorpopcountmatmul((x + 1) / 2, (W + 1) / 2)
    else:
        y = np.matmul(x, W)
    if T is not None:
        y = multithreshold(y, T)
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y = 2 * y - 1
        else:
            # signed offset
            y += act.min()
    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)
    # TODO split up into several dependent tests -- need to check how this
    # works for parametrized tests...
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced.reshape(y_expected.shape) == y_expected).all(), "rtlsim failed"

    hls_synt_res_est = model.analysis(hls_synth_res_estimation)
    assert "MVAU_hls_0" in hls_synt_res_est

    node = model.get_nodes_by_op_type("MVAU_hls")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
    assert exp_cycles != 0


# mem_mode: internal_embedded or internal_decoupled
@pytest.mark.parametrize("mem_mode", ["internal_decoupled"])
# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType["INT4"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["INT4"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT4"]])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1])
# HLS matrix width (input features)
@pytest.mark.parametrize("mw", [128])
# HLS matrix height (output features)
@pytest.mark.parametrize("mh", [128])
# Backend
@pytest.mark.parametrize("preferred_impl_style", ["hls", "rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_mvau_large_depth_decoupled_mode_rtlsim(
    mem_mode, idt, wdt, act, nf, sf, mw, mh, preferred_impl_style
):
    if preferred_impl_style == "rtl" and act is not None:
        pytest.skip("RTL-MVAU doesn't support const mem mode or embedded activations")
    if nf == -1:
        nf = mh
    if sf == -1:
        sf = mw
    pe = mh // nf
    simd = mw // sf
    assert mh % pe == 0
    assert mw % sf == 0
    # generate weights
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    # generate input data
    x = gen_finn_dt_tensor(idt, (1, mw))
    if act is None:
        # no activation, produce accumulators
        T = None
        tdt = None
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            odt = DataType["UINT32"]
        else:
            odt = DataType["INT32"]
    else:
        odt = act
        (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
        # provide non-decreasing thresholds
        T = np.sort(T, axis=1)
        # generate thresholds for activation
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            tdt = DataType["UINT32"]
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType["INT32"]
    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    for node in model.graph.node:
        # lookup op_type in registry of CustomOps
        inst = getCustomOp(node)
        inst.set_nodeattr("mem_mode", mem_mode)
        inst.set_nodeattr("resType", "auto")
        inst.set_nodeattr("preferred_impl_style", preferred_impl_style)

    # prepare input data
    input_dict = prepare_inputs(x, idt, wdt)
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert inputs to binary and use xnorpopcountmatmul
        y = xp.xnorpopcountmatmul((x + 1) / 2, (W + 1) / 2)
    else:
        y = np.matmul(x, W)
    if T is not None:
        y = multithreshold(y, T)
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y = 2 * y - 1
        else:
            # signed offset
            y += act.min()
    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)
    # TODO split up into several dependent tests -- need to check how this
    # works for parametrized tests...
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced.reshape(y_expected.shape) == y_expected).all(), "rtlsim failed"

    hls_synt_res_est = model.analysis(hls_synth_res_estimation)
    if preferred_impl_style == "hls":
        assert "MVAU_hls_0" in hls_synt_res_est

    if preferred_impl_style == "hls":
        node = model.get_nodes_by_op_type("MVAU_hls")[0]
    else:
        node = model.get_nodes_by_op_type("MVAU_rtl")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
    assert exp_cycles != 0


# mem_mode: internal_embedded or internal_decoupled
@pytest.mark.parametrize("mem_mode", ["internal_decoupled", "internal_embedded"])
# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType["INT4"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["INT4"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT4"]])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [8])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [8])
# HLS matrix width (input features)
@pytest.mark.parametrize("mw", [32])
# HLS matrix height (output features)
@pytest.mark.parametrize("mh", [32])
# Backend
@pytest.mark.parametrize("preferred_impl_style", ["hls", "rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_mvau_fifocharacterize_rtlsim(
    mem_mode, idt, wdt, act, nf, sf, mw, mh, preferred_impl_style
):
    if preferred_impl_style == "rtl" and (mem_mode == "internal_embedded" or act is not None):
        pytest.skip("RTL-MVAU doesn't support const mem mode or embedded activations")
    if nf == -1:
        nf = mh
    if sf == -1:
        sf = mw
    pe = mh // nf
    simd = mw // sf
    assert mh % pe == 0
    assert mw % sf == 0
    # generate weights
    W = gen_finn_dt_tensor(wdt, (mw, mh))

    # no activation, produce accumulators
    T = None
    tdt = None
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        odt = DataType["UINT32"]
    else:
        odt = DataType["INT32"]

    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    for node in model.graph.node:
        # lookup op_type in registry of CustomOps
        inst = getCustomOp(node)
        inst.set_nodeattr("mem_mode", mem_mode)
        inst.set_nodeattr("resType", "auto")
        inst.set_nodeattr("preferred_impl_style", preferred_impl_style)
    total_fold = nf * sf
    exp_total_cycles = total_fold + 10
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    model = model.transform(DeriveCharacteristic(exp_total_cycles))
    node_inst = getCustomOp(model.graph.node[0])
    period_attr = node_inst.get_nodeattr("io_chrc_period")
    assert period_attr == exp_total_cycles
    chrc_in = node_inst.get_nodeattr("io_chrc_in")
    chrc_out = node_inst.get_nodeattr("io_chrc_out")
    assert chrc_in.shape == (1, 2 * exp_total_cycles)
    assert chrc_out.shape == (1, 2 * exp_total_cycles)
    # total number of transactions == 2*SF
    assert chrc_in[0, -1] == 2 * sf
    # all outputs should be produced within the exp n of cycles
    assert chrc_out[0, exp_total_cycles] == nf


@pytest.mark.parametrize("mh", [18])
@pytest.mark.parametrize("mw", [128])
@pytest.mark.parametrize("pe", [1, 9, 18])
@pytest.mark.parametrize("simd", [1, 64, 128])
@pytest.mark.parametrize("idt", [DataType["UINT4"], DataType["UINT8"]])
@pytest.mark.parametrize("wdt", [DataType["INT4"], DataType["INT8"]])
@pytest.mark.parametrize(
    "part", ["xcvc1902-vsva2197-2MP-e-S", "xcku3p-ffva676-1-e", "xc7z020clg400-1"]
)
@pytest.mark.parametrize("clk_ns", [1.66, 4])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_rtl_mvau(mh, mw, pe, simd, idt, wdt, part, clk_ns):
    if part != "xcvc1902-vsva2197-2MP-e-S" and clk_ns != 1.66:
        pytest.skip(
            """Skip test for varying clk for devices other than Versal,
            since this variable only affects DSP58s"""
        )

    # Create test input vector (produced by SWG)
    ofm_shape = (3, 3)
    ofm_h, ofm_w = ofm_shape
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, ofm_h, ofm_w, mw])
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, (1, ofm_h, ofm_w, mh))
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    # if 7 series, force weights to narrow range
    if part == "xc7z020clg400-1":
        W = np.clip(W, wdt.min() + 1, wdt.max())
    model = make_single_matmul_modelwrapper(ifm, ofm, idt, wdt, W)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Create MatMul & obtain golden reference output
    A = gen_finn_dt_tensor(
        model.get_tensor_datatype("global_in"), model.get_tensor_shape("global_in")
    )
    input_dict = prepare_inputs(A, idt, wdt, inp_name="global_in")

    # Execute ONNX model
    output_matmul = oxe.execute_onnx(model, input_dict)["global_out"]

    # Create MVAU (HLS)
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(GiveUniqueNodeNames())

    # Apply convert-to-rtl step
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())

    # Apply folding (i.e. specify to use DSPs)
    folding_config = {
        "Defaults": {},
        "MVAU_rtl_0": {
            "PE": pe,
            "SIMD": simd,
            "resType": "dsp",
        },
    }
    model = model.transform(ApplyConfig(folding_config))
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    # make sure the changed datatypes are propagated through the network
    model = model.transform(InferDataTypes())

    # Run CPPsim
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    output_mvau_hls = oxe.execute_onnx(model, input_dict)["global_out"]
    assert (
        output_matmul == output_mvau_hls
    ).all(), "Output of ONNX model not matching output of node-by-node CPPsim!"

    # Run node-by-node RTLsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    output_mvau_rtl = oxe.execute_onnx(model, input_dict)["global_out"]
    assert (
        output_matmul == output_mvau_rtl
    ).all(), "Output of ONNX model not matching output of node-by-node RTLsim!"

    # Run stitched-ip RTLsim
    model = model.transform(InsertAndSetFIFODepths(part, clk_ns))
    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(part, clk_ns))

    model.set_metadata_prop("rtlsim_so", "")
    model.set_metadata_prop("exec_mode", "rtlsim")
    output_mvau_rtl_stitch = oxe.execute_onnx(model, input_dict)["global_out"]

    assert (
        output_matmul == output_mvau_rtl_stitch
    ).all(), "Output of ONNX model not matching output of stitched-IP RTL model!"
