# Copyright (c) 2020-2022, Xilinx
# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn import xsi as finnxsi
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.custom_op.fpgadataflow.convolutioninputgenerator import swg_default_tree
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import get_watchdog_timeout_cycles
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.test import tree_model_test


def make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw):
    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    ofm_dim_h, ofm_dim_w = ofm_dim

    odt = idt
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_dim_h, ifm_dim_w, ifm_ch])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, k_h * k_w * ifm_ch]
    )

    im2col_node = helper.make_node(
        "Im2Col",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.general",
        stride=[stride_h, stride_w],
        kernel_size=[k_h, k_w],
        input_shape=str((1, ifm_dim_h, ifm_dim_w, ifm_ch)),
        dilations=[dilation_h, dilation_w],
        pad_amount=[0, 0, 0, 0],
        pad_value=0,
        depthwise=dw,
    )
    graph = helper.make_graph(
        nodes=[im2col_node], name="im2col_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# input datatype
@pytest.mark.parametrize("idt", [DataType["INT2"], DataType["UINT4"]])
# kernel size
@pytest.mark.parametrize("k", [[2, 2], [3, 3], [1, 5]])
# input dimension
@pytest.mark.parametrize("ifm_dim", [[8, 8], [1, 21]])
# input channels
@pytest.mark.parametrize("ifm_ch", [2, 4])
# Stride
@pytest.mark.parametrize("stride", [[1, 1], [2, 2], [2, 1]])
# Dilation
@pytest.mark.parametrize("dilation", [[1, 1], [2, 2], [2, 1]])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# input channel parallelism ("SIMD")
@pytest.mark.parametrize("simd", [1, 2, 4])
# depthwise
@pytest.mark.parametrize("dw", [0, 1])
# parallel_window enable (MMV_out = M*K)
@pytest.mark.parametrize("parallel_window", [0, 1])
# in/out MMV ("M")
@pytest.mark.parametrize("m", [1])
# Flip dimensions
@pytest.mark.parametrize("flip", [False])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_slidingwindow(
    idt,
    k,
    ifm_dim,
    ifm_ch,
    stride,
    dilation,
    exec_mode,
    simd,
    dw,
    parallel_window,
    m,
    flip,
):
    if flip:
        if (
            ifm_dim[0] == ifm_dim[1]
            and k[0] == k[1]
            and stride[0] == stride[1]
            and dilation[0] == dilation[1]
        ):
            pytest.skip("Dimension flip would have no effect")
        k = k[::-1]
        ifm_dim = ifm_dim[::-1]
        stride = stride[::-1]
        dilation = dilation[::-1]

    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    kernel_width = (k_w - 1) * dilation_w + 1  # incl. dilation
    kernel_height = (k_h - 1) * dilation_h + 1  # incl. dilation

    if simd > ifm_ch:
        pytest.skip("SIMD cannot be larger than number of input channels")
    if ifm_ch % simd != 0:
        pytest.skip("SIMD must divide number of input channels")
    if kernel_height > ifm_dim_h or stride_h > ifm_dim_h:
        pytest.skip("Illegal convolution configuration: kernel or stride > FM dimension")
    if kernel_width > ifm_dim_w or stride_w > ifm_dim_w:
        pytest.skip("Illegal convolution configuration: kernel or stride > FM dimension")
    if (k_h == 1 and dilation_h != 1) or (k_w == 1 and dilation_w != 1):
        pytest.skip("Illegal convolution configuration: dilation for unitary kernel dim")
    if ((stride_h > k_h) or (stride_w > k_w)) and not (parallel_window or (k_h == 1 and k_w == 1)):
        pytest.skip("Not all combinations for stride > k edge case supported in default mode")
    if parallel_window and simd != ifm_ch and not (dw or (k_h == 1 and k_w == 1)):
        pytest.skip("Parallel window requires SIMD=C for non-depthwise case")

    ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
    ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
    ofm_dim = [ofm_dim_h, ofm_dim_w]

    x = gen_finn_dt_tensor(idt, (1, ifm_dim_h, ifm_dim_w, ifm_ch))
    # prepare input data
    input_dict = prepare_inputs(x)
    model = make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw)
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    model = model.transform(to_hw.InferConvInpGen())
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    # set simd
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", simd)
    optype = model.graph.node[0].op_type
    if optype == "ConvolutionInputGenerator_rtl":
        inst.set_nodeattr("parallel_window", parallel_window)
        inst.set_nodeattr("M", m)

    if exec_mode == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode in test_fpgadataflow_slidingwindow")

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    if dw == 0:
        assert (y_produced == y_expected).all()
    else:
        y_expected = y_expected.reshape(1, ofm_dim_h, ofm_dim_w, k_h * k_w, ifm_ch // simd, simd)
        y_expected = y_expected.transpose(0, 1, 2, 4, 3, 5)
        y_expected = y_expected.reshape(1, ofm_dim_h, ofm_dim_w, ifm_ch * k_h * k_w)
        assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        nodes = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
        node = nodes[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10, rtol=1.1)
        assert exp_cycles != 0


# One configuration per regime the schedule distinguishes, rather than the
# cross-product: the generator's tiers (1x1, the depthwise k=2 s=2 shape, the
# state machine everything else runs through), SIMD folded and unfolded, stride
# and dilation on their own and together, the parallel-window variant, and a
# single-channel feature map.
@pytest.mark.parametrize(
    "k,ifm_ch,stride,dilation,simd,dw,parallel_window",
    [
        ([1, 1], 10, [1, 1], [1, 1], 1, 0, 0),
        ([1, 1], 10, [1, 1], [1, 1], 10, 0, 0),
        ([1, 1], 1, [1, 1], [1, 1], 1, 0, 0),
        ([2, 2], 10, [1, 1], [1, 1], 1, 0, 0),
        ([2, 2], 10, [1, 1], [1, 1], 10, 0, 0),
        ([2, 2], 1, [1, 1], [1, 1], 1, 0, 0),
        ([2, 2], 10, [2, 2], [1, 1], 1, 0, 0),
        ([2, 2], 10, [1, 1], [2, 2], 1, 0, 0),
        ([2, 2], 10, [2, 2], [2, 2], 1, 0, 0),
        ([2, 2], 10, [1, 1], [1, 1], 1, 1, 0),
        ([2, 2], 10, [2, 2], [1, 1], 1, 1, 0),
        ([2, 2], 10, [1, 1], [1, 1], 10, 0, 1),
    ],
)
@pytest.mark.parametrize("idt", [DataType["INT2"]])
@pytest.mark.parametrize("ifm_dim", [[10, 6]])
# in/out MMV ("M")
@pytest.mark.parametrize("m", [1])
# Flip dimensions
@pytest.mark.parametrize("flip", [False])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_slidingwindow(
    idt,
    k,
    ifm_dim,
    ifm_ch,
    stride,
    dilation,
    simd,
    dw,
    parallel_window,
    m,
    flip,
):
    if flip:
        if (
            ifm_dim[0] == ifm_dim[1]
            and k[0] == k[1]
            and stride[0] == stride[1]
            and dilation[0] == dilation[1]
        ):
            pytest.skip("Dimension flip would have no effect")
        k = k[::-1]
        ifm_dim = ifm_dim[::-1]
        stride = stride[::-1]
        dilation = dilation[::-1]

    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    kernel_width = (k_w - 1) * dilation_w + 1  # incl. dilation
    kernel_height = (k_h - 1) * dilation_h + 1  # incl. dilation

    if simd > ifm_ch:
        pytest.skip("SIMD cannot be larger than number of input channels")
    if ifm_ch % simd != 0:
        pytest.skip("SIMD must divide number of input channels")
    if kernel_height > ifm_dim_h or stride_h > ifm_dim_h:
        pytest.skip("Illegal convolution configuration: kernel or stride > FM dimension")
    if kernel_width > ifm_dim_w or stride_w > ifm_dim_w:
        pytest.skip("Illegal convolution configuration: kernel or stride > FM dimension")
    if (k_h == 1 and dilation_h != 1) or (k_w == 1 and dilation_w != 1):
        pytest.skip("Illegal convolution configuration: dilation for unitary kernel dim")
    if ((stride_h > k_h) or (stride_w > k_w)) and not (parallel_window or (k_h == 1 and k_w == 1)):
        pytest.skip("Not all combinations for stride > k edge case supported in default mode")
    if parallel_window and simd != ifm_ch and not (dw or (k_h == 1 and k_w == 1)):
        pytest.skip("Parallel window requires SIMD=C for non-depthwise case")

    ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
    ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
    ofm_dim = [ofm_dim_h, ofm_dim_w]

    model = make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw)

    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    # set simd
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", simd)
    optype = model.graph.node[0].op_type
    if optype == "ConvolutionInputGenerator_rtl":
        inst.set_nodeattr("parallel_window", parallel_window)
        inst.set_nodeattr("M", m)

    node_details = (
        "ConvolutionInputGenerator",
        ifm_dim,
        k,
        stride,
        dilation,
        ifm_ch,
        simd,
        dw,
        parallel_window,
        idt,
        ofm_dim,
        "hls",
    )
    part = "xc7z020clg400-1"
    target_clk_ns = 4
    # TAV tolerances are fractions -- of the tokens moved and of the frame
    # length -- with a floor for the fixed part of the error (wind-up, adder
    # tree depth, the cycle rtlsim's period rounds away). The floors are the
    # flat budgets this test used before the split, so nothing that passed
    # then fails now; the fractions are what govern once the frame is long
    # enough to exceed them.
    max_allowed_volume_frac = 0.2
    volume_const = 5000
    max_allowed_length_frac = 0.2
    length_const = 5000

    # ``swg_default_tree`` executes the RTL sliding-window FSM rather than
    # approximating it, so wherever it applies the token access vector is a
    # pure function of the generated parameters and has to match rtlsim
    # exactly. Only the configurations it declines -- HLS impl style,
    # dynamic_mode, a SIMD that does not divide, an FSM that does not settle --
    # fall back to the approximation and keep a tolerance.
    if swg_default_tree(getCustomOp(model.graph.node[0])) is not None:
        max_allowed_volume_frac = 0.0
        volume_const = 0
        max_allowed_length_frac = 0.0
        length_const = 0

    assert tree_model_test(
        model,
        node_details,
        part,
        target_clk_ns,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    ), "characterized TAV does not match RTLsim'd one!"


# input datatype
@pytest.mark.parametrize("idt", [DataType["INT2"]])
# kernel size
# @pytest.mark.parametrize("k", [[2, 2], [3, 3], [1, 5]])
@pytest.mark.parametrize("k", [[7, 7]])
# input dimension
# @pytest.mark.parametrize("ifm_dim", [[8, 8], [1, 21]])
@pytest.mark.parametrize("ifm_dim", [[7, 7]])
# input channels
# @pytest.mark.parametrize("ifm_ch", [2, 4])
@pytest.mark.parametrize("ifm_ch", [1024])
# Stride
# @pytest.mark.parametrize("stride", [[1, 1]])
@pytest.mark.parametrize("stride", [[1, 1]])
# Dilation
# @pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
# input channel parallelism ("SIMD")
@pytest.mark.parametrize("simd", [1])
# depthwise
@pytest.mark.parametrize("dw", [1])
# parallel_window enable (MMV_out = M*K)
@pytest.mark.parametrize("parallel_window", [0])
# in/out MMV ("M")
@pytest.mark.parametrize("m", [1])
# Flip dimensions
@pytest.mark.parametrize("flip", [False])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_slidingwindow_mobilenet(
    idt,
    k,
    ifm_dim,
    ifm_ch,
    stride,
    dilation,
    simd,
    dw,
    parallel_window,
    m,
    flip,
):
    if flip:
        if (
            ifm_dim[0] == ifm_dim[1]
            and k[0] == k[1]
            and stride[0] == stride[1]
            and dilation[0] == dilation[1]
        ):
            pytest.skip("Dimension flip would have no effect")
        k = k[::-1]
        ifm_dim = ifm_dim[::-1]
        stride = stride[::-1]
        dilation = dilation[::-1]

    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    kernel_width = (k_w - 1) * dilation_w + 1  # incl. dilation
    kernel_height = (k_h - 1) * dilation_h + 1  # incl. dilation

    if simd > ifm_ch:
        pytest.skip("SIMD cannot be larger than number of input channels")
    if ifm_ch % simd != 0:
        pytest.skip("SIMD must divide number of input channels")
    if kernel_height > ifm_dim_h or stride_h > ifm_dim_h:
        pytest.skip("Illegal convolution configuration: kernel or stride > FM dimension")
    if kernel_width > ifm_dim_w or stride_w > ifm_dim_w:
        pytest.skip("Illegal convolution configuration: kernel or stride > FM dimension")
    if (k_h == 1 and dilation_h != 1) or (k_w == 1 and dilation_w != 1):
        pytest.skip("Illegal convolution configuration: dilation for unitary kernel dim")
    if ((stride_h > k_h) or (stride_w > k_w)) and not (parallel_window or (k_h == 1 and k_w == 1)):
        pytest.skip("Not all combinations for stride > k edge case supported in default mode")
    if parallel_window and simd != ifm_ch and not (dw or (k_h == 1 and k_w == 1)):
        pytest.skip("Parallel window requires SIMD=C for non-depthwise case")

    ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
    ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
    ofm_dim = [ofm_dim_h, ofm_dim_w]

    model = make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw)

    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    # set simd
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", simd)
    optype = model.graph.node[0].op_type
    if optype == "ConvolutionInputGenerator_rtl":
        inst.set_nodeattr("parallel_window", parallel_window)
        inst.set_nodeattr("M", m)

    node_details = (
        "ConvolutionInputGenerator",
        ifm_dim,
        k,
        stride,
        dilation,
        ifm_ch,
        simd,
        dw,
        parallel_window,
        idt,
        ofm_dim,
        "hls",
    )
    part = "xc7z020clg400-1"
    target_clk_ns = 4
    # TAV tolerances are fractions -- of the tokens moved and of the frame
    # length -- with a floor for the fixed part of the error (wind-up, adder
    # tree depth, the cycle rtlsim's period rounds away). The floors are the
    # flat budgets this test used before the split, so nothing that passed
    # then fails now; the fractions are what govern once the frame is long
    # enough to exceed them.
    max_allowed_volume_frac = 0.2
    volume_const = 2140
    max_allowed_length_frac = 0.2
    length_const = 2140

    # ``swg_default_tree`` executes the RTL sliding-window FSM rather than
    # approximating it, so wherever it applies the token access vector is a
    # pure function of the generated parameters and has to match rtlsim
    # exactly. Only the configurations it declines -- HLS impl style,
    # dynamic_mode, a SIMD that does not divide, an FSM that does not settle --
    # fall back to the approximation and keep a tolerance.
    if swg_default_tree(getCustomOp(model.graph.node[0])) is not None:
        max_allowed_volume_frac = 0.0
        volume_const = 0
        max_allowed_length_frac = 0.0
        length_const = 0

    assert tree_model_test(
        model,
        node_details,
        part,
        target_clk_ns,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    ), "characterized TAV does not match RTLsim'd one!"


def _run_swg_depthwise_throttle_check(k, ifm_dim, ifm_ch, simd, throttle):
    """Build a depthwise ConvolutionInputGenerator_rtl (reusing the file's
    im2col model builder), drive its input stream with the given throttle
    profile via XSI rtlsim, and assert it neither deadlocks (watchdog) nor
    mismatches the reordered im2col reference."""
    fpga_part = "xc7z020clg400-1"
    clk_ns = 5
    stride = [1, 1]
    dilation = [1, 1]
    idt = DataType["INT4"]
    k_h, k_w = k
    ofm_dim_h = compute_conv_output_dim(ifm_dim[0], k_h, stride[0], 0, dilation[0])
    ofm_dim_w = compute_conv_output_dim(ifm_dim[1], k_w, stride[1], 0, dilation[1])
    ofm_dim = [ofm_dim_h, ofm_dim_w]

    # build the im2col model once and take the reference before converting to HW
    model = make_single_im2col_modelwrapper(
        k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw=1
    )
    x = gen_finn_dt_tensor(idt, (1, ifm_dim[0], ifm_dim[1], ifm_ch))
    y_ref = oxe.execute_onnx(model, prepare_inputs(x))["outp"]
    # reorder im2col output into the SWG depthwise channel/window layout
    y_ref = y_ref.reshape(1, ofm_dim_h, ofm_dim_w, k_h * k_w, ifm_ch // simd, simd)
    y_ref = y_ref.transpose(0, 1, 2, 4, 3, 5)
    y_ref = y_ref.reshape(1, ofm_dim_h, ofm_dim_w, ifm_ch * k_h * k_w)

    model = model.transform(to_hw.InferConvInpGen())
    # force the RTL variant so we exercise the swg_template_default.sv datapath
    getCustomOp(model.graph.node[0]).set_nodeattr("preferred_impl_style", "rtl")
    model = model.transform(SpecializeLayers(fpga_part))
    assert model.graph.node[0].op_type == "ConvolutionInputGenerator_rtl"
    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", simd)
    model = model.transform(GiveUniqueNodeNames())

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    inst = getCustomOp(model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")[0])
    in_dt, in_w, in_folded = (
        inst.get_input_datatype(0),
        inst.get_instream_width(0),
        inst.get_folded_input_shape(0),
    )
    out_dt, out_w, out_folded = (
        inst.get_output_datatype(0),
        inst.get_outstream_width(0),
        inst.get_folded_output_shape(0),
    )
    num_out = inst.get_number_output_values()

    packed_in = npy_to_rtlsim_input(np.asarray(x, dtype=np.float32).reshape(in_folded), in_dt, in_w)
    hex_in = map(lambda v: f"{v:0x}", packed_in)

    sim = inst.get_rtlsim()
    liveness = get_watchdog_timeout_cycles(inst.get_exp_cycles())
    try:
        inst.reset_rtlsim(sim)
        sim.stream_input("in0_V", hex_in, throttle=throttle)
        out_buf = sim.collect_output(
            "out0_V", num_out, watchdog=sim.create_watchdog("out0_V timeout", liveness)
        )
        cfg = "k=%s, ifm_ch=%s, simd=%s, throttle=%s" % (k, ifm_ch, simd, throttle)
        assert not sim.run(), "rtlsim watchdog timed out -> SWG deadlock (%s)" % cfg
        packed_out = [int(v, base=16) for v in out_buf]
    finally:
        inst.close_rtlsim(sim)

    got = rtlsim_output_to_npy(packed_out, None, out_dt, out_folded, out_w, out_dt.bitwidth())
    got = np.asarray(got, dtype=np.float32).reshape(inst.get_normal_output_shape(0))

    assert (got.reshape(y_ref.shape) == y_ref).all(), "SWG output mismatch (%s)" % cfg


# throttle=(count_txns, wait_cycles): a back-to-back producer masks the bug,
# a bursty producer feeding no faster than the window rate exposes the deadlock.
@pytest.mark.parametrize("throttle", [(float("inf"), 0), (1, 32)], ids=["backtoback", "bursty"])
# kernel size (4x4 over 4x4 -> single window = the exact last-window edge case)
@pytest.mark.parametrize("k", [[2, 2], [3, 3], [4, 4]])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_swg_depthwise_slow_producer_deadlock(k, throttle, monkeypatch):
    """Regression test for the depthwise SWG deadlock (PR #1698).

    In depthwise mode, ``First_elem_next_window`` overflowed its register at the
    first fetch of a feature map's last window: advanced by ``TAIL_INCR_LAST`` it
    exceeded ``LAST_READ_ELEM``, wrapped negative through ``$signed()`` in
    ``read_cmd`` and permanently blocked the last input read. The stall only
    appears when the producer feeds input no faster than the SWG's window rate,
    so a back-to-back stream (``throttle=(inf, 0)``) passes even on the buggy RTL
    while a bursty stream (``throttle=(1, 32)``) deadlocks. Post-fix both must
    complete and return the reference im2col output.
    """
    if not finnxsi.is_available():
        pytest.skip("finn_xsi (XSI rtlsim) not available")
    # keep the default watchdog bound so a genuine deadlock aborts quickly
    monkeypatch.delenv("LIVENESS_THRESHOLD", raising=False)
    _run_swg_depthwise_throttle_check(k, [4, 4], ifm_ch=4, simd=1, throttle=throttle)


# Wider channels / SIMD>1
# (e.g. 4x4 depthwise over 4x4 with 512 channels, plus 3x3 / 2x2 variants).
@pytest.mark.parametrize("throttle", [(float("inf"), 0), (1, 16)], ids=["backtoback", "bursty"])
@pytest.mark.parametrize(
    "k, ifm_dim, ifm_ch, simd",
    [
        ([4, 4], [4, 4], 512, 64),
        ([4, 4], [4, 4], 512, 1),  # SIMD=1 (max channel_factor -> largest overflow)
        ([3, 3], [4, 4], 256, 32),
        ([2, 2], [4, 4], 128, 4),
    ],
    ids=["k4_ch512_simd64", "k4_ch512_simd1", "k3_ch256_simd32", "k2_ch128_simd4"],
)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_swg_depthwise_slow_producer_deadlock_wide(
    k, ifm_dim, ifm_ch, simd, throttle, monkeypatch
):
    """Wide-channel / SIMD>1 coverage of the depthwise SWG deadlock (PR #1698),
    matching the author's xsim sweep. Same trigger as the light test: the bursty
    producer exposes the ``First_elem_next_window`` overflow, back-to-back masks
    it."""
    if not finnxsi.is_available():
        pytest.skip("finn_xsi (XSI rtlsim) not available")
    monkeypatch.delenv("LIVENESS_THRESHOLD", raising=False)
    _run_swg_depthwise_throttle_check(k, ifm_dim, ifm_ch, simd, throttle)
