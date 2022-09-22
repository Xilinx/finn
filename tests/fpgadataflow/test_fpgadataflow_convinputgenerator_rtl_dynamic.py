# Copyright (c) 2022, Xilinx
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

from onnx import TensorProto, helper
from pyverilator.util.axi_utils import axilite_write, reset_rtlsim
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP


def make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt):
    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    ofm_dim_h, ofm_dim_w = ofm_dim

    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim_h, ifm_dim_w, ifm_ch]
    )
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

    model = helper.make_model(graph, producer_name="im2col-model")
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
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim_h, ifm_dim_w, ifm_ch]
    )
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
    )
    graph = helper.make_graph(
        nodes=[SlidingWindow_node],
        name="slidingwindow_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="slidingwindow-model")
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
        pytest.skip(
            "Illegal convolution configuration: kernel or stride > FM dimension"
        )
    if kernel_width > ifm_dim_w or stride_w > ifm_dim_w:
        pytest.skip(
            "Illegal convolution configuration: kernel or stride > FM dimension"
        )
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

    # Helper function that delivers the hook to program the SWG via AXI-Lite
    def config_hook(config):
        if config is None:
            return None

        def write_swg_config(sim):
            axi_name = "s_axi_cfg_0_"
            # Write config registers to the SWG, dict defines (addr, value) tuples
            for config_entry in config.values():
                axilite_write(sim, config_entry[0], config_entry[1], basename=axi_name)
            axilite_write(
                sim, 0, 1, basename=axi_name
            )  # 1. set cfg_valid flag (>= 1 cycle)
            reset_rtlsim(sim)  # 2. reset SWG (>= 1 cycle)
            axilite_write(
                sim, 0, 0, basename=axi_name
            )  # 3. unset cfg_valid flag (not required)

        return write_swg_config

    # Helper function to update tensor dimensions manually because shape inference
    # does not work on FINN nodes (they assume well-defined tensor shapes).
    def update_tensor_dim(model, tensor_name, new_hw):
        shape = model.get_tensor_shape(tensor_name)
        shape[1] = new_hw[0]
        shape[2] = new_hw[1]
        model.set_tensor_shape(tensor_name, shape)

    # Simulate 1 FM for each dimension in the series
    for i, ifm_dim in enumerate(ifm_dim_series):
        ifm_dim_h, ifm_dim_w = ifm_dim
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
        ofm_dim = [ofm_dim_h, ofm_dim_w]

        config = None
        if i > 0:  # skip re-programming for initial FM dimension
            # Necessary update of node and tensor attributes to make rtlsim work:
            swg_node = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")[0]
            swg_inst = getCustomOp(swg_node)
            update_tensor_dim(model, swg_node.input[0], ifm_dim)
            update_tensor_dim(model, swg_node.output[0], ofm_dim)

            # Generate config, also overwrites IFMDim/OFMDim attributes:
            config = swg_inst.get_dynamic_config(ifm_dim)

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
        rtlsim_exec(model, context, pre_hook=config_hook(config))
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
