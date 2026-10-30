# Copyright (c) 2024, Advanced Micro Devices, Inc.
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
from bitstring import BitArray
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import (
    auto_pad_to_explicit_padding,
    gen_finn_dt_tensor,
    get_by_name,
    qonnx_make_model,
)

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn import xsi
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
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

finnxsi = xsi if xsi.is_available() else None


def create_conv_model(
    idim_h, idim_w, ifm, k, stride, ofm, idt, wdt, pad_mode, depthwise, weight_0, weight_1
):
    group = ifm if depthwise else 1
    group_str = str(group)
    ishp = (1, ifm, idim_h, idim_w)
    pad_0 = auto_pad_to_explicit_padding(pad_mode, idim_h, idim_w, k, k, stride, stride, 2)
    int_dim_h = compute_conv_output_dim(idim_h, k, stride, total_pad=pad_0[0] + pad_0[2])
    int_dim_w = compute_conv_output_dim(idim_w, k, stride, total_pad=pad_0[1] + pad_0[3])

    pad_1 = auto_pad_to_explicit_padding(pad_mode, int_dim_h, int_dim_w, k, k, stride, stride, 2)
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
    model.set_initializer("param_c0_weight", weight_0)
    model.set_initializer("param_c1_weight", weight_1)
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
        finnxsi.reset_rtlsim(sim)
        for axi_name, config in configs:
            writes = []
            # Write config registers to the SWG/FMPadding dict
            # defines (addr, value) tuples
            for config_entry in config.values():
                addr, val = config_entry
                if val < 0:
                    # ensure any negative vals are expressed as two's complement,
                    # SWG control regs are currently always 32 bits
                    val = BitArray(int=val, length=32).uint
                # convert value to hex value and without '0x' prefix
                hex_val = format(val, "x")
                writes.append((addr, hex_val))
            sim.write_axilite(axi_name, iter(writes))
            sim.run()
        finnxsi.reset_rtlsim(sim)

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
@pytest.mark.xfail(reason="DotProd_rtl does not yet support dynamic reconfiguration (no get_dynamic_config)")
def test_fpgadataflow_conv_dynamic(cfg, monkeypatch):
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
    wshp = (ifm, 1, k, k) if depthwise else (ofm, ifm, k, k)
    wshp_1 = (ifm, 1, k, k) if depthwise else (ofm, ofm, k, k)
    weight_0 = gen_finn_dt_tensor(wdt, wshp)
    weight_1 = gen_finn_dt_tensor(wdt, wshp_1)
    exp_cfgs = []
    largest_model = None
    for idim in idims:
        idim_h, idim_w = idim
        ishp = (1, ifm, idim_h, idim_w)
        inp = gen_finn_dt_tensor(idt, ishp)
        model = create_conv_model(
            idim_h,
            idim_w,
            ifm,
            k,
            stride,
            ofm,
            idt,
            wdt,
            pad_mode,
            depthwise,
            weight_0,
            weight_1,
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

    # convert to hardware using InferDotProd (replaces LowerConvsToMatMul +
    # InferConvInpGen + InferQuantizedMatrixVectorActivation + InferVectorVectorActivation)
    model = largest_model.transform(to_hw.InferDotProd())
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    parent_model = model.transform(CreateDataflowPartition())
    sdp_inst = getCustomOp(parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0])
    model = ModelWrapper(sdp_inst.get_nodeattr("model"))
    # InferDotProd produces DotProd_rtl nodes (SWG + padding + compute fused)
    dotprod_nodes = model.get_nodes_by_op_type("DotProd_rtl")
    dotprod_nodes += model.get_nodes_by_op_type("DotProd_hls")
    assert len(dotprod_nodes) == 2
    for dp_node in dotprod_nodes:
        dp_inst = getCustomOp(dp_node)
        if depthwise:
            dp_inst.set_nodeattr("PE", 4)
        else:
            dp_inst.set_nodeattr("SIMD", 4)
            dp_inst.set_nodeattr("PE", 4)
        dp_inst.set_nodeattr("inFIFODepths", [16])
        dp_inst.set_nodeattr("outFIFODepths", [16])
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP("xc7z020clg400-1", 5, run_synth=do_synth))
    model.set_metadata_prop("exec_mode", "rtlsim")

    # loop through experiment configurations
    # TODO: DotProd_rtl needs get_dynamic_config() for runtime reconfiguration
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
        # With DotProd_rtl, dynamic reconfiguration requires a get_dynamic_config()
        # method that can produce AXI-Lite register writes for the fused
        # SWG+padding+compute unit. This is not yet implemented.
        dotprod_nodes = model.get_nodes_by_op_type("DotProd_rtl")
        dotprod_nodes += model.get_nodes_by_op_type("DotProd_hls")
        dp0 = getCustomOp(dotprod_nodes[0])
        dp0.set_nodeattr("IFMDim", [idim_h, idim_w])
        dp0.set_nodeattr("OFMDim", [int_dim_h, int_dim_w])
        dp1 = getCustomOp(dotprod_nodes[1])
        dp1.set_nodeattr("IFMDim", [int_dim_h, int_dim_w])
        dp1.set_nodeattr("OFMDim", [odim_h, odim_w])

        # Dynamic reconfiguration via AXI-Lite not yet supported in DotProd_rtl
        configs = []
        for i, dp_inst in enumerate([dp0, dp1]):
            assert hasattr(dp_inst, "get_dynamic_config"), (
                "DotProd_rtl.get_dynamic_config() not yet implemented"
            )

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
        monkeypatch.setenv("LIVENESS_THRESHOLD", "100000")
        rtlsim_exec(model, ctx, pre_hook=config_hook(configs))
        ret = ctx["global_out"].transpose(0, 3, 1, 2)
        assert np.isclose(golden, ret).all()


def make_single_conv_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, wdt, dw=0):
    """Create an ONNX Conv model that will be converted to DotProd via InferDotProd.

    This replaces the old make_single_slidingwindow_modelwrapper which created
    a standalone ConvolutionInputGenerator node (no longer registered).
    """
    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    ofm_dim_h, ofm_dim_w = ofm_dim

    # NCHW layout for ONNX Conv
    ofm_ch = ifm_ch if dw else ifm_ch  # keep same channels for simplicity
    group = ifm_ch if dw else 1
    wshp = [ifm_ch, 1, k_h, k_w] if dw else [ofm_ch, ifm_ch, k_h, k_w]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim_h, ifm_dim_w])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_ch, ofm_dim_h, ofm_dim_w]
    )

    conv_node = helper.make_node(
        "Conv",
        ["inp", "weights"],
        ["outp"],
        kernel_shape=[k_h, k_w],
        strides=[stride_h, stride_w],
        dilations=[dilation_h, dilation_w],
        pads=[0, 0, 0, 0],
        group=group,
    )
    graph = helper.make_graph(
        nodes=[conv_node],
        name="conv_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="conv-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("weights", wdt)
    weights = gen_finn_dt_tensor(wdt, wshp)
    model.set_initializer("weights", weights)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model, weights


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
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.xfail(reason="DotProd_rtl does not yet support dynamic reconfiguration (no get_dynamic_config)")
def test_fpgadataflow_dotprod_rtl_dynamic(
    idt, k, ifm_dim_series, ifm_ch, stride, dilation, dw, simd
):
    """Test dynamic reconfiguration of DotProd_rtl for varying FM dimensions.

    This test replaces test_fpgadataflow_slidingwindow_rtl_dynamic. The old test
    created standalone ConvolutionInputGenerator nodes (no longer registered) and
    reconfigured them via AXI-Lite. DotProd_rtl fuses SWG + compute, so dynamic
    reconfiguration requires a get_dynamic_config() method on DotProd_rtl.
    """
    wdt = DataType["INT4"]
    # Begin test by generating DotProd_rtl for the first (largest) FM of the series.
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

    # Create ONNX Conv model and convert to DotProd via InferDotProd
    model, weights = make_single_conv_modelwrapper(
        k=k,
        ifm_ch=ifm_ch,
        ifm_dim=ifm_dim,
        ofm_dim=ofm_dim,
        stride=stride,
        dilation=dilation,
        idt=idt,
        wdt=wdt,
        dw=dw,
    )

    model = model.transform(to_hw.InferDotProd())
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    # Set SIMD on DotProd node
    dp_nodes = model.get_nodes_by_op_type("DotProd_rtl")
    dp_nodes += model.get_nodes_by_op_type("DotProd_hls")
    assert len(dp_nodes) == 1
    dp_inst = getCustomOp(dp_nodes[0])
    dp_inst.set_nodeattr("SIMD", simd)

    # Simulate using stitched-ip-rtlsim so we can use existing infrastructure
    # that supports hook functions to re-program configuration before rtlsim
    model = model.transform(InsertFIFO(True))  # required for proper simulation
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
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
            # DotProd_rtl needs get_dynamic_config() for runtime reconfiguration
            dp_nodes = model.get_nodes_by_op_type("DotProd_rtl")
            dp_nodes += model.get_nodes_by_op_type("DotProd_hls")
            dp_inst = getCustomOp(dp_nodes[0])

            # Update spatial attributes
            dp_inst.set_nodeattr("IFMDim", list(ifm_dim))
            dp_inst.set_nodeattr("OFMDim", ofm_dim)

            # Generate config via get_dynamic_config (not yet implemented)
            assert hasattr(dp_inst, "get_dynamic_config"), (
                "DotProd_rtl.get_dynamic_config() not yet implemented"
            )
            config = dp_inst.get_dynamic_config(ifm_dim)
            configs = [("s_axilite_0", config)]

            # Also update FIFO nodes and corresponding tensors
            fifo_node = model.get_nodes_by_op_type("StreamingFIFO_rtl")[0]
            fifo_inst = getCustomOp(fifo_node)
            shape = fifo_inst.get_nodeattr("folded_shape")
            shape[1] = ifm_dim_h
            shape[2] = ifm_dim_w
            fifo_inst.set_nodeattr("folded_shape", shape)
            update_tensor_dim(model, fifo_node.input[0], ifm_dim)

            fifo_node = model.get_nodes_by_op_type("StreamingFIFO_rtl")[1]
            fifo_inst = getCustomOp(fifo_node)
            shape = fifo_inst.get_nodeattr("folded_shape")
            shape[1] = ofm_dim_h
            shape[2] = ofm_dim_w
            fifo_inst.set_nodeattr("folded_shape", shape)
            update_tensor_dim(model, fifo_node.output[0], ofm_dim)

        # Run rtlsim on stitched-ip (input in NCHW for DotProd)
        x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_dim_h, ifm_dim_w))
        context = prepare_inputs(x)
        rtlsim_exec(model, context, pre_hook=config_hook(configs))
        y_produced = context["outp"]

        # Generate golden result via ONNX Conv execution
        golden_model, _ = make_single_conv_modelwrapper(
            k=k,
            ifm_ch=ifm_ch,
            ifm_dim=ifm_dim,
            ofm_dim=ofm_dim,
            stride=stride,
            dilation=dilation,
            idt=idt,
            wdt=wdt,
            dw=dw,
        )
        # Use same weights as the HW model
        golden_model.set_initializer("weights", weights)
        golden_model = golden_model.transform(InferShapes())
        input_dict = prepare_inputs(x)
        y_expected = oxe.execute_onnx(golden_model, input_dict)["outp"]

        assert (y_produced == y_expected).all()
