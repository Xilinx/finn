# Copyright (c) 2022, Advanced Micro Devices, Inc.
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
import numpy as np
import onnx.parser as oprs
import os
from onnx import TensorProto, helper
from pyverilator.util.axi_utils import axilite_write, reset_rtlsim
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import (
    LowerConvsToMatMul,
    _auto_pad_to_explicit_padding,
)
from qonnx.util.basic import gen_finn_dt_tensor, get_by_name, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.onnx_exec import execute_onnx
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.util.basic import pyverilate_get_liveness_threshold_cycles


def create_conv_model(idim_h, idim_w, ifm, k, stride, ofm, idt, wdt, pad_mode, depthwise):
    np.random.seed(0)
    group = ifm if depthwise else 1
    group_str = str(group)
    ishp = (1, ifm, idim_h, idim_w)
    pad_0 = _auto_pad_to_explicit_padding(pad_mode, idim_h, idim_w, k, k, stride, stride, 2)
    int_dim_h = compute_conv_output_dim(idim_h, k, stride, total_pad=pad_0[0] + pad_0[2])
    int_dim_w = compute_conv_output_dim(idim_w, k, stride, total_pad=pad_0[1] + pad_0[3])

    pad_1 = _auto_pad_to_explicit_padding(pad_mode, int_dim_h, int_dim_w, k, k, stride, stride, 2)
    odim_h = compute_conv_output_dim(int_dim_h, k, stride, total_pad=pad_1[0] + pad_1[2])
    odim_w = compute_conv_output_dim(int_dim_w, k, stride, total_pad=pad_1[1] + pad_1[3])
    oshp = (1, ifm, odim_h, odim_w) if depthwise else (1, ofm, odim_h, odim_w)
    wshp = (ifm, 1, k, k) if depthwise else (ofm, ifm, k, k)
    wshp_1 = (ifm, 1, k, k) if depthwise else (ofm, ofm, k, k)
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    wshp_str = str(list(wshp))
    wshp_1_str = str(list(wshp_1))
    kshp_str = str([k, k])
    pad_0_str = str(list(pad_0))
    pad_1_str = str(list(pad_1))
    stride_str = str([stride, stride])
    dil_str = str([1, 1])

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    <
        float{wshp_str} param_c0_weight,
        float{wshp_1_str} param_c1_weight
    >
    {{
        conv0 = Conv<
                dilations={dil_str},group={group_str},kernel_shape={kshp_str},pads={pad_0_str},
                strides={stride_str}
            >(in0, param_c0_weight)
        out0 = Conv<
                dilations={dil_str},group={group_str},kernel_shape={kshp_str},pads={pad_1_str},
                strides={stride_str}
            >(conv0, param_c1_weight)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model.set_tensor_datatype("in0", idt)
    model.set_tensor_datatype("param_c0_weight", wdt)
    model.set_tensor_datatype("param_c1_weight", wdt)
    model.set_initializer("param_c0_weight", gen_finn_dt_tensor(wdt, wshp))
    model.set_initializer("param_c1_weight", gen_finn_dt_tensor(wdt, wshp_1))
    return model


def update_conv_model_dims(model, idim_new_h, idim_new_w):
    cnode = model.get_nodes_by_op_type("Conv")[0]
    k, _ = get_by_name(cnode.attribute, "kernel_shape").ints
    stride, _ = get_by_name(cnode.attribute, "strides").ints
    ishp = model.get_tensor_shape("in0")
    n, ci, _, _ = ishp
    n, co, _, _ = model.get_tensor_shape("out0")
    int_dim_h = compute_conv_output_dim(idim_new_h, k, stride)
    int_dim_w = compute_conv_output_dim(idim_new_w, k, stride)
    odim_h = compute_conv_output_dim(int_dim_h, k, stride)
    odim_w = compute_conv_output_dim(int_dim_w, k, stride)
    model.set_tensor_shape("in0", (n, ci, idim_new_h, idim_new_w))
    model.set_tensor_shape("out0", (n, co, odim_h, odim_w))
    # remove all existing shapes
    del model.graph.value_info[:]
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


# Helper function to update tensor dimensions manually because shape inference
# does not work on FINN nodes (they assume well-defined tensor shapes).
def update_tensor_dim(model, tensor_name, new_hw):
    shape = model.get_tensor_shape(tensor_name)
    shape[1] = new_hw[0]
    shape[2] = new_hw[1]
    model.set_tensor_shape(tensor_name, shape)


# Helper function that delivers the hook to program the SWG via AXI-Lite
def config_hook(configs):
    if configs is None:
        return None

    def write_swg_config(sim):
        reset_rtlsim(sim)
        for axi_name, config in configs:
            # Write config registers to the SWG/FMPadding dict
            # defines (addr, value) tuples
            for config_entry in config.values():
                axilite_write(sim, config_entry[0], config_entry[1], basename=axi_name)
        reset_rtlsim(sim)

    return write_swg_config


cfg0 = {
    "idims": [(32, 32), (8, 8)],
    "ifm": 64,
    "k": 3,
    "stride": 1,
    "ofm": 64,
    "depthwise": True,
    "pad_mode": "SAME_UPPER",
    # run synthesis for one configuration
    # this helped expose a bug in enum decls previously
    # (which config the synth runs on does not matter)
    "do_synth": True,
}
cfg1 = {
    "idims": [(32, 16), (16, 8)],
    "ifm": 4,
    "k": 4,
    "stride": 1,
    "ofm": 8,
    "depthwise": False,
    "pad_mode": "SAME_UPPER",
    "do_synth": False,
}
cfg2 = {
    "idims": [(64, 128), (2, 4)],
    "ifm": 64,
    "k": 3,
    "stride": 1,
    "ofm": 64,
    "depthwise": True,
    "pad_mode": "SAME_UPPER",
    "do_synth": False,
}


@pytest.mark.parametrize("cfg", [cfg0, cfg1, cfg2])
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_fpgadataflow_conv_dynamic(cfg):
    do_synth = cfg["do_synth"]
    pad_mode = cfg["pad_mode"]
    depthwise = cfg["depthwise"]
    idims = cfg["idims"]
    ifm = cfg["ifm"]
    k = cfg["k"]
    stride = cfg["stride"]
    ofm = cfg["ofm"]
    idt = DataType["UINT4"]
    wdt = DataType["INT2"]
    exp_cfgs = []
    largest_model = None
    for idim in idims:
        idim_h, idim_w = idim
        ishp = (1, ifm, idim_h, idim_w)
        np.random.seed(0)
        inp = gen_finn_dt_tensor(idt, ishp)
        model = create_conv_model(
            idim_h, idim_w, ifm, k, stride, ofm, idt, wdt, pad_mode, depthwise
        )
        _, _, int_dim_h, int_dim_w = model.get_tensor_shape("conv0")
        _, _, odim_h, odim_w = model.get_tensor_shape("out0")
        pad0 = get_by_name(model.graph.node[0].attribute, "pads").ints
        pad1 = get_by_name(model.graph.node[1].attribute, "pads").ints
        if idim == max(idims):
            # use largest model for hardware conversion
            largest_model = copy.deepcopy(model)
        golden = execute_onnx(model, {"in0": inp})["out0"]
        exp_cfg = (
            (idim_h, idim_w),
            (int_dim_h, int_dim_w),
            (odim_h, odim_w),
            pad0,
            pad1,
            inp,
            golden,
        )
        exp_cfgs.append(exp_cfg)

    # convert to hardware and prepare simulation
    model = largest_model.transform(LowerConvsToMatMul())
    model = model.transform(to_hls.InferConvInpGen(use_rtl_variant=True))
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode="decoupled"))
    model = model.transform(to_hls.InferVectorVectorActivation())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    parent_model = model.transform(CreateDataflowPartition())
    sdp_inst = getCustomOp(parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0])
    model = ModelWrapper(sdp_inst.get_nodeattr("model"))
    assert len(model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")) == 2
    if pad_mode == "VALID":
        assert len(model.get_nodes_by_op_type("FMPadding_rtl")) == 0
    else:
        assert len(model.get_nodes_by_op_type("FMPadding_rtl")) == 2
    dyn_nodes = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
    dyn_nodes += model.get_nodes_by_op_type("FMPadding_rtl")
    for swg_node in dyn_nodes:
        getCustomOp(swg_node).set_nodeattr("SIMD", 4)
        getCustomOp(swg_node).set_nodeattr("dynamic_mode", 1)
        getCustomOp(swg_node).set_nodeattr("inFIFODepths", [16])
        getCustomOp(swg_node).set_nodeattr("outFIFODepths", [16])
    comp_nodes = model.get_nodes_by_op_type("MatrixVectorActivation")
    comp_nodes += model.get_nodes_by_op_type("VectorVectorActivation")
    for comp_node in comp_nodes:
        if depthwise:
            getCustomOp(comp_node).set_nodeattr("PE", 4)
        else:
            getCustomOp(comp_node).set_nodeattr("SIMD", 4)
            getCustomOp(comp_node).set_nodeattr("PE", 4)
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP("xc7z020clg400-1", 5, vitis=do_synth))
    model.set_metadata_prop("exec_mode", "rtlsim")

    # loop through experiment configurations
    for exp_cfg in exp_cfgs:
        (
            (idim_h, idim_w),
            (int_dim_h, int_dim_w),
            (odim_h, odim_w),
            pad0,
            pad1,
            inp,
            golden,
        ) = exp_cfg
        conv0_idim_h = idim_h + pad0[0] + pad0[2]
        conv0_idim_w = idim_w + pad0[1] + pad0[3]
        conv1_idim_h = int_dim_h + pad1[0] + pad1[2]
        conv1_idim_w = int_dim_w + pad1[1] + pad1[3]
        # get config for the new dimensions
        swg_nodes = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
        swg0 = getCustomOp(swg_nodes[0])
        update_tensor_dim(model, swg0.onnx_node.input[0], (conv0_idim_h, conv0_idim_w))
        update_tensor_dim(model, swg0.onnx_node.output[0], (int_dim_h, int_dim_w))
        swg_config0 = swg0.get_dynamic_config((conv0_idim_h, conv0_idim_w))
        swg1 = getCustomOp(swg_nodes[1])
        update_tensor_dim(model, swg1.onnx_node.input[0], (conv1_idim_h, conv1_idim_w))
        update_tensor_dim(model, swg1.onnx_node.output[0], (odim_h, odim_w))
        swg_config1 = swg1.get_dynamic_config((conv1_idim_h, conv1_idim_w))
        if pad_mode != "VALID":
            pad_nodes = model.get_nodes_by_op_type("FMPadding_rtl")
            padder0 = getCustomOp(pad_nodes[0])
            update_tensor_dim(model, padder0.onnx_node.input[0], (idim_h, idim_w))
            update_tensor_dim(model, padder0.onnx_node.output[0], (conv0_idim_h, conv0_idim_w))
            pad_config0 = padder0.get_dynamic_config((idim_h, idim_w), pad0)
            padder1 = getCustomOp(pad_nodes[1])
            update_tensor_dim(model, padder1.onnx_node.input[0], (int_dim_h, int_dim_w))
            update_tensor_dim(model, padder1.onnx_node.output[0], (conv1_idim_h, conv1_idim_w))
            pad_config1 = padder1.get_dynamic_config((int_dim_h, int_dim_w), pad1)
            configs = [
                ("s_axilite_0_", pad_config0),
                ("s_axilite_1_", swg_config0),
                ("s_axilite_2_", pad_config1),
                ("s_axilite_3_", swg_config1),
            ]
        else:
            configs = [("s_axilite_0_", swg_config0), ("s_axilite_1_", swg_config1)]
        # adjust folded shapes for I/O FIFOs
        # (since rtlsim_exec uses folded shape info to fold global i/o tensors)
        first_node = getCustomOp(model.graph.node[0])
        first_node_shp = list(first_node.get_folded_input_shape())
        first_node_shp[1] = idim_h
        first_node_shp[2] = idim_w
        first_node.set_nodeattr("folded_shape", first_node_shp)
        update_tensor_dim(model, first_node.onnx_node.input[0], (idim_h, idim_w))
        last_node = getCustomOp(model.graph.node[-1])
        last_node_shp = list(last_node.get_folded_output_shape())
        last_node_shp[1] = odim_h
        last_node_shp[2] = odim_w
        update_tensor_dim(model, last_node.onnx_node.output[0], (odim_h, odim_w))
        last_node.set_nodeattr("folded_shape", last_node_shp)
        ctx = {"global_in": inp.transpose(0, 2, 3, 1)}
        liveness_prev = pyverilate_get_liveness_threshold_cycles()
        os.environ["LIVENESS_THRESHOLD"] = "100000"
        rtlsim_exec(model, ctx, pre_hook=config_hook(configs))
        os.environ["LIVENESS_THRESHOLD"] = str(liveness_prev)
        ret = ctx["global_out"].transpose(0, 3, 1, 2)
        assert np.isclose(golden, ret).all()


def make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt):
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
    )
    graph = helper.make_graph(
        nodes=[im2col_node], name="im2col_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def make_single_slidingwindow_modelwrapper(
    k, ifm_ch, ifm_dim, ofm_dim, simd, m, parallel_window, stride, dilation, idt, dw=0
):
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

    SlidingWindow_node = helper.make_node(
        "ConvolutionInputGenerator_rtl",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ConvKernelDim=[k_h, k_w],
        IFMChannels=ifm_ch,
        IFMDim=[ifm_dim_h, ifm_dim_w],
        OFMDim=[ofm_dim_h, ofm_dim_w],
        SIMD=simd,
        M=m,
        parallel_window=parallel_window,
        Stride=[stride_h, stride_w],
        Dilation=[dilation_h, dilation_w],
        inputDataType=idt.name,
        outputDataType=odt.name,
        depthwise=dw,
        dynamic_mode=1,
    )
    graph = helper.make_graph(
        nodes=[SlidingWindow_node],
        name="slidingwindow_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="slidingwindow-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# input datatype
@pytest.mark.parametrize("idt", [DataType["UINT4"]])
# kernel size
@pytest.mark.parametrize("k", [[3, 3]])
# input dimension
@pytest.mark.parametrize("ifm_dim_series", [[[32, 32], [16, 16], [8, 8]]])
# input channels
@pytest.mark.parametrize("ifm_ch", [6])
# Stride
@pytest.mark.parametrize("stride", [[1, 1]])
# Dilation
@pytest.mark.parametrize("dilation", [[1, 1]])
# depthwise
@pytest.mark.parametrize("dw", [0, 1])
# input channel parallelism ("SIMD")
@pytest.mark.parametrize("simd", [2, 6])
# parallel_window enable (MMV_out = M*K)
@pytest.mark.parametrize("parallel_window", [0])
# in/out MMV ("M")
@pytest.mark.parametrize("m", [1])
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_fpgadataflow_slidingwindow_rtl_dynamic(
    idt, k, ifm_dim_series, ifm_ch, stride, dilation, dw, simd, m, parallel_window
):
    # Begin test by generating RTL SWG normally for the first FM of the series.
    # The following FM dimensions must be equal or smaller than the initial
    # dimensions (in terms of required buffer depth).
    ifm_dim = ifm_dim_series[0]

    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
    ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
    ofm_dim = [ofm_dim_h, ofm_dim_w]
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
    if (k_h == 1 and (stride_h != 1 or dilation_h != 1)) or (
        k_w == 1 and (stride_w != 1 or dilation_w != 1)
    ):
        pytest.skip(
            """Illegal convolution configuration:
            stride or dilation defined for unitary kernel dim"""
        )
    if k_h == 1 and k_w == 1 and simd != ifm_ch:
        pytest.skip("1x1 Kernel only supported in parallel mode (SIMD=C)")
    if parallel_window and simd != ifm_ch:
        pytest.skip("Parallel window requires SIMD=C")

    model = make_single_slidingwindow_modelwrapper(
        k=k,
        ifm_ch=ifm_ch,
        ifm_dim=ifm_dim,
        ofm_dim=ofm_dim,
        simd=simd,
        m=m,
        parallel_window=parallel_window,
        stride=stride,
        dilation=dilation,
        idt=idt,
        dw=dw,
    )

    # Simulate using stitched-ip-rtlsim so we can use existing infrastructure
    # that supports hook functions to re-program configuration before rtlsim
    model = model.transform(InsertFIFO(True))  # required for proper simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP("xc7z020clg400-1", 5))
    model.set_metadata_prop("exec_mode", "rtlsim")

    # Simulate 1 FM for each dimension in the series
    for i, ifm_dim in enumerate(ifm_dim_series):
        ifm_dim_h, ifm_dim_w = ifm_dim
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
        ofm_dim = [ofm_dim_h, ofm_dim_w]

        configs = None
        if i > 0:  # skip re-programming for initial FM dimension
            # Necessary update of node and tensor attributes to make rtlsim work:
            swg_node = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")[0]
            swg_inst = getCustomOp(swg_node)
            update_tensor_dim(model, swg_node.input[0], ifm_dim)
            update_tensor_dim(model, swg_node.output[0], ofm_dim)

            # Generate config, also overwrites IFMDim/OFMDim attributes:
            config = swg_inst.get_dynamic_config(ifm_dim)
            configs = [("s_axilite_0_", config)]

            # Also update FIFO nodes and corresponding tensors
            fifo_node = model.get_nodes_by_op_type("StreamingFIFO")[0]
            fifo_inst = getCustomOp(fifo_node)
            shape = fifo_inst.get_nodeattr("folded_shape")
            shape[1] = ifm_dim_h
            shape[2] = ifm_dim_w
            fifo_inst.set_nodeattr("folded_shape", shape)
            update_tensor_dim(model, fifo_node.input[0], ifm_dim)

            fifo_node = model.get_nodes_by_op_type("StreamingFIFO")[1]
            fifo_inst = getCustomOp(fifo_node)
            shape = fifo_inst.get_nodeattr("folded_shape")
            shape[1] = ofm_dim_h
            shape[2] = ofm_dim_w
            fifo_inst.set_nodeattr("folded_shape", shape)
            update_tensor_dim(model, fifo_node.output[0], ofm_dim)

        # Run rtlsim on stitched-ip
        x = gen_finn_dt_tensor(idt, (1, ifm_dim_h, ifm_dim_w, ifm_ch))
        context = prepare_inputs(x)
        rtlsim_exec(model, context, pre_hook=config_hook(configs))
        y_produced = context["outp"]

        # Generate golden result
        golden = make_single_im2col_modelwrapper(
            k=k,
            ifm_ch=ifm_ch,
            ifm_dim=ifm_dim,
            ofm_dim=ofm_dim,
            stride=stride,
            dilation=dilation,
            idt=idt,
        )
        input_dict = prepare_inputs(x)
        y_expected = oxe.execute_onnx(golden, input_dict)["outp"]

        # Check result
        if dw == 0:
            assert (y_produced == y_expected).all()
        else:
            y_expected = y_expected.reshape(
                1, ofm_dim_h, ofm_dim_w, k_h * k_w, ifm_ch // simd, simd
            )
            y_expected = y_expected.transpose(0, 1, 2, 4, 3, 5)
            y_expected = y_expected.reshape(1, ofm_dim_h, ofm_dim_w, ifm_ch * k_h * k_w)
            assert (y_produced == y_expected).all()
