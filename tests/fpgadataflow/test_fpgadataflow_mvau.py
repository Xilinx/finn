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
import qonnx.custom_op.general.xnorpopcount as xp
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.derive_characteristic import DeriveCharacteristic
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from qonnx.transformation.general import ApplyConfig, GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.transformation.infer_shapes import InferShapes
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
        "MatrixVectorActivation",
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


def prepare_inputs(input_tensor, idt, wdt):
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert bipolar to binary
        return {"inp": (input_tensor + 1) / 2}
    else:
        return {"inp": input_tensor}


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
def test_fpgadataflow_fclayer_hwop(idt, wdt, act, nf, sf, mw, mh):
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


# mem_mode: const or decoupled
@pytest.mark.parametrize("mem_mode", ["const", "decoupled", "external"])
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
def test_fpgadataflow_fclayer_hlsop_cppsim(mem_mode, idt, wdt, act, nf, sf, mw, mh):
    if idt == DataType["BIPOLAR"] and wdt != DataType["BIPOLAR"] or idt != DataType["BIPOLAR"] and wdt == DataType["BIPOLAR"]:
        pytest.skip("Bipolar activations/weights only supported in MVU if both operands are bipolar")
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
    model = model.transform(SpecializeLayers())
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


# mem_mode: const or decoupled
@pytest.mark.parametrize("mem_mode", ["const", "decoupled", "external"])
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
# Backend
@pytest.mark.parametrize("backend", ["rtl", "hls"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_fclayer_rtlsim(mem_mode, idt, wdt, act, nf, sf, mw, mh, backend):
    if backend == "rtl" and act is not None:
        pytest.skip("RTL MVU doesn't support embedded thresholding functionality.")
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
        inst.set_nodeattr("preferred_impl_style", backend)

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
    model = model.transform(SpecializeLayers())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced.reshape(y_expected.shape) == y_expected).all(), "rtlsim failed"

    hls_synt_res_est = model.analysis(hls_synth_res_estimation)
    if backend == "hls":
        assert "MatrixVectorActivation_hls_0" in hls_synt_res_est
    else:
        assert "MatrixVectorActivation_rtl_0" in hls_synt_res_est

    node = model.get_nodes_by_op_type("MatrixVectorActivation")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
    assert exp_cycles != 0


# mem_mode: const or decoupled
@pytest.mark.parametrize("mem_mode", ["decoupled"])
# activation: None or DataType
@pytest.mark.parametrize("act", [DataType["INT4"]])
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
@pytest.mark.parametrize("backend", ["rtl", "hls"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_fclayer_large_depth_decoupled_mode_rtlsim(
    mem_mode, idt, wdt, act, nf, sf, mw, mh, backend
):
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
    model = model.transform(SpecializeLayers())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced.reshape(y_expected.shape) == y_expected).all(), "rtlsim failed"

    hls_synt_res_est = model.analysis(hls_synth_res_estimation)
    if backend == "hls":
        assert "MatrixVectorActivation_hls_0" in hls_synt_res_est
    else:
        assert "MatrixVectorActivation_rtl_0" in hls_synt_res_est

    node = model.get_nodes_by_op_type("MatrixVectorActivation")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
    assert exp_cycles != 0


# mem_mode: const or decoupled
@pytest.mark.parametrize("mem_mode", ["decoupled", "const"])
# activation: None or DataType
@pytest.mark.parametrize("act", [DataType["INT4"]])
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
@pytest.mark.parametrize("backend", ["rtl", "hls"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fclayer_fifocharacterize_rtlsim(mem_mode, idt, wdt, act, nf, sf, mw, mh, backend):
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
    total_fold = nf * sf
    exp_total_cycles = total_fold + 10
    model = model.transform(SpecializeLayers())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
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
    # first sf cycles should read input continuously
    assert (chrc_in[0, :sf] == range(1, sf + 1)).all()
    # all outputs should be produced within the exp n of cycles
    assert chrc_out[0, exp_total_cycles] == nf
