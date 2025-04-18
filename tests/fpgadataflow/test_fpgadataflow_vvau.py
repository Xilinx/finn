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
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
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


def _infer_sparse_weight_tensor(W_conv, k_h, k_w, channels):
    W_sparse = np.zeros((channels, channels, k_h, k_w), dtype=np.float32)
    for ch in range(channels):
        W_sparse[ch][ch] = W_conv[ch][0]
    W_conv = W_sparse.astype(np.float32)
    W_matmul = W_conv.transpose(0, 2, 3, 1)
    W_matmul = W_matmul.reshape(channels, channels * k_h * k_w)
    W_matmul = W_matmul.T

    return W_matmul


def _calculate_dot_prod_range(dt_a, dt_b, len):
    """Returns the (min,max) values a dot product between two (un)signed vectors of
    types dt_a and dt_b of len elements can take."""
    min_prod = 2**30
    max_prod = -(2**30)
    for a_val in [dt_a.min(), dt_a.max()]:
        for b_val in [dt_b.min(), dt_b.max()]:
            prod = a_val * b_val * len
            if prod < min_prod:
                min_prod = prod
            if prod > max_prod:
                max_prod = prod
    return (min_prod, max_prod)


def _make_single_vvau_modelwrapper(
    W,
    pe,
    simd,
    k_h,
    k_w,
    channels,
    dim_h,
    dim_w,
    wdt,
    idt,
    odt,
    T=None,
    tdt=None,
    mem_mode="internal_embedded",
):
    in_shape = [1, dim_h, dim_w, k_h * k_w * channels]  # [N, H, W, K*K*CH]
    out_shape = [
        1,
        dim_h,
        dim_w,
        channels,
    ]  # [N, H, W, OFM_CH] (OFM_CH=IFM_CH because depthwise convolution)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType["BIPOLAR"]:
            actval = 0
        else:
            actval = odt.min()
    else:
        no_act = 1
        node_inp_list = ["inp", "weights"]
        actval = 0

    VVAU_node = helper.make_node(
        "VVAU",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=pe,
        SIMD=simd,
        Dim=[dim_h, dim_w],
        Channels=channels,
        Kernel=[k_h, k_w],
        resType="lut",
        ActVal=actval,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        noActivation=no_act,
        mem_mode=mem_mode,
    )

    graph = helper.make_graph(nodes=[VVAU_node], name="vvau_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="vvau-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)

    model.set_initializer("weights", W)
    model.set_tensor_shape("weights", (channels, 1, k_h, k_w))

    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


# input datatype
@pytest.mark.parametrize("idt", [DataType["BIPOLAR"], DataType["UINT4"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["BIPOLAR"], DataType["UINT4"]])
# activation: None or DataType
@pytest.mark.parametrize("act", [DataType["BIPOLAR"], DataType["UINT4"], None])
# PE
@pytest.mark.parametrize("pe", [1, 3, 6])
# SIMD
@pytest.mark.parametrize("simd", [1, 9])
# Input image shape
@pytest.mark.parametrize("dim_h", [10])
@pytest.mark.parametrize("dim_w", [10, 1])
# Kernel shape
@pytest.mark.parametrize("k_h", [3])
@pytest.mark.parametrize("k_w", [3, 1])
# Number of input and output channels
@pytest.mark.parametrize("channels", [3, 6])
# memory mode
@pytest.mark.parametrize("mem_mode", ["internal_embedded", "internal_decoupled"])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_vvau(
    idt, wdt, act, pe, simd, dim_h, dim_w, k_h, k_w, channels, mem_mode, exec_mode
):
    if dim_w == 1 and k_w != 1:
        pytest.skip("1D image requires 1D kernel, skipping.")

    if channels % pe != 0:
        pytest.skip("Requirement Channels divisable by PE is violated.")

    if (k_h * k_w) % simd != 0:
        pytest.skip("Requirement kernel (k_h * k_w) divisable by SIMD is violated.")

    # Generate weights in expected shape for ONNX and HLS node
    W = gen_finn_dt_tensor(wdt, (channels, 1, k_h, k_w))  # shape: [channels, 1, k, k]
    W_onnx = _infer_sparse_weight_tensor(W, k_h, k_w, channels)  # shape: [k*k*channels, channels]

    # Generate inputs in expected format for ONNX and HLS node
    x = gen_finn_dt_tensor(idt, (1, dim_h, dim_w, k_h * k_w * channels))
    x_vvau = x.reshape(1, dim_h, dim_w, k_h * k_w, channels // pe, pe)
    x_vvau = x_vvau.transpose(0, 1, 2, 4, 3, 5)
    x_vvau = x_vvau.reshape(1, dim_h, dim_w, channels * k_h * k_w)

    if act is None:
        T = None
        tdt = None
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            odt = DataType["UINT32"]
        else:
            odt = DataType["INT32"]
    else:
        odt = act
        (min_v, max_v) = _calculate_dot_prod_range(idt, wdt, k_h * k_w)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min_v, max_v - 1, (channels, n_steps)).astype(np.float32)
        T = np.sort(T, axis=1)
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            tdt = DataType["UINT32"]
            # bias thresholds to be positive
            T = np.ceil((T + (k_h * k_w)) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType["INT32"]

    model = _make_single_vvau_modelwrapper(
        W, pe, simd, k_h, k_w, channels, dim_h, dim_w, wdt, idt, odt, T, tdt, mem_mode
    )
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    input_dict = prepare_inputs(x_vvau)
    y_hwop = oxe.execute_onnx(model, input_dict)["global_out"]
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))

    if exec_mode == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode in test_fpgadataflow_vvau")

    # Calculate output
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # Simulate XNOR-popcount matrix multiplication, see
        # qonnx.custom_op.general.xnorpopcount (not usable due to sparse W)
        y_expected = np.matmul(x, W_onnx)
        y_expected = (y_expected + (k_h * k_w)) / 2
    else:
        y_expected = np.matmul(x, W_onnx)  # Y is in [N, H, W, C] format

    if T is not None:
        # Reshape Y, as multithreshold expects Y to be in [N, C, H, W] format
        y_expected = np.transpose(y_expected, (0, 3, 1, 2))
        y_expected = multithreshold(y_expected, T)
        y_expected = np.transpose(y_expected, (0, 2, 3, 1))
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y_expected = 2 * y_expected - 1
        else:
            # signed offset
            y_expected += act.min()

    y_produced = oxe.execute_onnx(model, input_dict, return_full_exec_context=False)["global_out"]

    assert (y_hwop == y_expected).all(), "VVAU HW-op mismatches with golden output!"
    assert (y_produced == y_expected).all(), "VVAU specialized-op mismatches with golden output!"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("VVAU_hls")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10, rtol=1.1)
        assert exp_cycles != 0

        # if rtlsim and internal_decoupled mode is selected, also run stitched IP rtlsim
        if mem_mode == "internal_decoupled":
            model = model.transform(InsertAndSetFIFODepths("xczu7ev-ffvc1156-2-e", 5))
            model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5))
            model = model.transform(HLSSynthIP())
            model = model.transform(CreateStitchedIP("xczu7ev-ffvc1156-2-e", 5))

            model.set_metadata_prop("exec_mode", "rtlsim")
            y_expected = oxe.execute_onnx(model, input_dict)["global_out"]

            assert (
                y_produced == y_expected
            ).all(), "Output of ONNX model not matching output of stitched-IP RTL model!"


def make_single_dw_conv_modelwrapper(conv_config, idt, wdt):
    kernel_size, in_feature_dim, in_chn = conv_config
    stride = 1
    pad = 0

    out_feature_dim = compute_conv_output_dim(in_feature_dim, kernel_size, stride, pad)
    group = out_chn = in_chn

    conv_param_shape = [out_chn, 1, kernel_size, kernel_size]
    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    output_shape = [1, out_chn, out_feature_dim, out_feature_dim]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = group
    conv_config["kernel_shape"] = [kernel_size, kernel_size]
    conv_config["pads"] = [pad, pad, pad, pad]
    conv_config["strides"] = [stride, stride]

    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, input_shape)
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, output_shape)
    weights = [helper.make_tensor_value_info("weights", TensorProto.FLOAT, conv_param_shape)]

    modelproto = qonnx_make_model(
        helper.make_graph(
            name="conv_test",
            inputs=[ifm],
            outputs=[ofm],
            value_info=weights,
            nodes=[helper.make_node("Conv", ["ifm", "weights"], ["ofm"], **conv_config)],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("ifm", idt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", gen_finn_dt_tensor(wdt, conv_param_shape))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def prepare_inputs(input_tensor):
    return {"global_in": input_tensor}


# kernel size (square)
@pytest.mark.parametrize("kernel_size", [3])
# IFM size (square)
@pytest.mark.parametrize("in_feature_dim", [5])
# input channels
@pytest.mark.parametrize("in_chn", [4])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT8"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["INT6"]])
# targeted board
@pytest.mark.parametrize("part", ["xcvm1802-vsvd1760-2MP-e-S"])
# pe
@pytest.mark.parametrize("pe", [1, 2, 4])
# simd
@pytest.mark.parametrize("simd", [1, 3, 9])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_vvau_rtl(kernel_size, in_feature_dim, in_chn, idt, wdt, part, pe, simd):
    # Create depthwise-separable convolution
    conv_config = (kernel_size, in_feature_dim, in_chn)
    model = make_single_dw_conv_modelwrapper(conv_config, idt, wdt)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Obtain golden reference output
    golden_in = gen_finn_dt_tensor(
        model.get_tensor_datatype("global_in"), model.get_tensor_shape("global_in")
    )
    input_dict = prepare_inputs(golden_in)
    golden_out = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)["global_out"]

    # Convert to HLS custom-op first
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    output_vvau_hw = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)[
        "global_out"
    ]
    assert (
        golden_out == output_vvau_hw
    ).all(), "Output of ONNX model not matching output of HW-ops!"

    # Obtain second reference from HLS-based VVAU layer
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())

    # Apply folding (i.e. specify to use DSPs)
    folding_config = {
        "Defaults": {},
        "ConvolutionInputGenerator_rtl_0": {
            "SIMD": pe,
            "parallel_window": 1,
        },
        "VVAU_rtl_0": {
            "PE": pe,
            "SIMD": simd,
            "resType": "dsp",
        },
    }
    model = model.transform(ApplyConfig(folding_config))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    # make sure the changed datatypes are propagated through the network
    model = model.transform(InferDataTypes())

    # Run CPPsim
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    output_vvau_cppsim = oxe.execute_onnx(model, input_dict)["global_out"]
    assert (
        golden_out == output_vvau_cppsim
    ).all(), "Output of ONNX model not matching output of node-by-node CPPsim!"

    # Run node-by-node RTLsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(part, 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    output_vvau_rtlsim = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)[
        "global_out"
    ]

    assert (
        golden_out == output_vvau_rtlsim
    ).all(), "Output of ONNX model not matching output of specialized HW-ops!"

    # Stitched-IP RTLsim
    model = model.transform(CreateDataflowPartition())
    partition_model_path = getCustomOp(
        model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    ).get_nodeattr("model")
    partitioned_model = ModelWrapper(partition_model_path)
    # FIFOs needed for stitched-ip RTLsim, DWC needed for VVU operating on SIMD parallelism
    partitioned_model = partitioned_model.transform(InsertAndSetFIFODepths(part, 5))
    partitioned_model = partitioned_model.transform(PrepareIP(part, 5))
    partitioned_model = partitioned_model.transform(HLSSynthIP())
    partitioned_model = partitioned_model.transform(CreateStitchedIP(part, 5))
    # set top-level prop for stitched-ip rtlsim and launch
    partitioned_model.set_metadata_prop("exec_mode", "rtlsim")
    # transpose input since we're now simulating HW layers (NCHW --> NHWC)
    input_dict["global_in"] = np.transpose(input_dict["global_in"], (0, 2, 3, 1))
    output_vvau_stitched = oxe.execute_onnx(
        partitioned_model, input_dict, return_full_exec_context=True
    )["global_out"]
    # tranpose hardware-generated outputs NHWC -> NCHW to be comparable
    output_vvau_stitched = output_vvau_stitched.transpose(0, 3, 1, 2)

    assert (
        golden_out == output_vvau_stitched
    ).all(), "Output of ONNX model not matching output of stitched-IP RTL model!"
