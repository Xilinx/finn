import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext


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

    model = helper.make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def make_single_slidingwindow_modelwrapper(
    type,
    k,
    ifm_ch,
    ifm_dim,
    ofm_dim,
    simd,
    m,
    parallel_window,
    stride,
    dilation,
    idt,
    dw=0,
    ram_style="auto",
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
        type,
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
        ram_style=ram_style,
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

    # DEBUG
    # swg_node = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")[0]
    # swg_inst = getCustomOp(swg_node)
    # swg_inst.set_nodeattr("rtlsim_trace", "/workspace/finn/finn-rtllib/swg/swg_test_trace.vcd")

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


def bench_rtl_swg(params, task_id, run_id, results_dir):
    # Read params
    idt = params["idt"]
    k = params["k"]
    ifm_dim = params["ifm_dim"]
    ifm_ch = params["ifm_ch"]
    stride = params["stride"]
    dilation = params["dilation"]
    dw = params["dw"]
    simd = params["simd"]
    m = params["m"]
    parallel_window = params["parallel_window"]
    flip = params["flip"]
    ram_style = params["ram_style"]

    only_estimates = params["only_estimates"]
    skip_rtlsim = params["skip_rtlsim"]
    skip_synth = params["skip_synth"]
    synthesize_hls_comparison = params["synthesize_hls_comparison"]

    output_dict = {}

    # convert string to FINN DataType
    idt = DataType[idt]

    if flip:
        if (
            ifm_dim[0] == ifm_dim[1]
            and k[0] == k[1]
            and stride[0] == stride[1]
            and dilation[0] == dilation[1]
        ):
            return
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

    # inter-dependent test parameters
    if simd == "ifm_ch":
        simd = ifm_ch

    # skip conditions
    if simd > ifm_ch:
        return
    if ifm_ch % simd != 0:
        return
    if kernel_width > ifm_dim_w or stride_w > ifm_dim_w:
        return
    if kernel_height > ifm_dim_h or stride_h > ifm_dim_h:
        return
    if (k_h == 1 and (stride_h != 1 or dilation_h != 1)) or (
        k_w == 1 and (stride_w != 1 or dilation_w != 1)
    ):
        return
    if k_h == 1 and k_w == 1 and simd != ifm_ch:
        return
    if parallel_window and simd != ifm_ch:
        return
    if not parallel_window and m > 1:
        return

    ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
    ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
    ofm_dim = [ofm_dim_h, ofm_dim_w]

    x = gen_finn_dt_tensor(idt, (1, ifm_dim_h, ifm_dim_w, ifm_ch))
    model = make_single_slidingwindow_modelwrapper(
        type="ConvolutionInputGenerator_rtl",
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
        ram_style=ram_style,
    )

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    if not only_estimates:
        model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5))
        model = model.transform(PrepareRTLSim())

    node = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")[0]
    inst = getCustomOp(node)

    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    exp_res_dict = model.analysis(res_estimation)
    exp_res = exp_res_dict[node.name]

    output_dict["est_Cycles"] = exp_cycles
    output_dict["est_LUT"] = exp_res["LUT"]
    output_dict["est_BRAM"] = exp_res["BRAM_18K"] * 0.5
    output_dict["est_URAM"] = exp_res["URAM"]

    if only_estimates:
        return output_dict

    if not skip_rtlsim:
        # prepare input data
        input_dict = prepare_inputs(x)
        # execute model
        oxe.execute_onnx(model, input_dict)["outp"]

        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        output_dict["Cycles"] = cycles_rtlsim
        print("RTLSIM cycles: %d" % cycles_rtlsim)

    if not skip_synth:
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP("xczu7ev-ffvc1156-2-e", 5))
        model = model.transform(SynthOutOfContext(part="xczu7ev-ffvc1156-2-e", clk_period_ns=5))
        ooc_res_dict = eval(model.get_metadata_prop("res_total_ooc_synth"))
        output_dict["LUT"] = ooc_res_dict["LUT"]
        output_dict["BRAM"] = ooc_res_dict["BRAM_18K"] * 0.5 + ooc_res_dict["BRAM_36K"]
        output_dict["URAM"] = ooc_res_dict["URAM"]
        output_dict["WNS"] = ooc_res_dict["WNS"]
        output_dict["Fmax"] = ooc_res_dict["fmax_mhz"]

    ###############################################################
    # HLS COMPARISON:
    if synthesize_hls_comparison:
        output_dict["HLS_compatible"] = "yes"

        is_square = True
        props_to_check = [k, ifm_dim, ofm_dim, stride, dilation]
        for prop in props_to_check:
            is_square = prop[0] == prop[1]
            if not is_square:
                is_square = False

        if not is_square or dilation[0] != 1 or dilation[1] != 1:
            # try 1D HLS ConvInpGen

            # rectangular case not supported
            if ifm_dim[0] == 1:
                if ofm_dim[0] != 1 or k[0] != 1 or stride[0] != 1 or dilation[0] != 1:
                    output_dict["HLS_compatible"] = "no"
            elif ifm_dim[1] == 1:
                if ofm_dim[1] != 1 or k[1] != 1 or stride[1] != 1 or dilation[1] != 1:
                    output_dict["HLS_compatible"] = "no"
            else:
                output_dict["HLS_compatible"] = "no"

            # unsupported parallelization
            if m > 1:
                output_dict["HLS_compatible"] = "no"
            if parallel_window > 0:
                fully_unfolded = simd == ifm_ch
                non_dws = dw == 0
                no_stride = stride_h == 1 and stride_w == 1
                no_dilation = dilation_h == 1 and dilation_w == 1
                supported_ram_style = ram_style in ["auto", "distributed"]
                if not (
                    fully_unfolded and non_dws and no_stride and no_dilation and supported_ram_style
                ):
                    output_dict["HLS_compatible"] = "no"

            # unsupported hyperparams
            if (dilation_h > 1 or dilation_w > 1) and (stride_h > 1 or stride_w > 1):
                output_dict["HLS_compatible"] = "no"
            if (dilation_h > 1 or dilation_w > 1) and dw == 0:
                output_dict["HLS_compatible"] = "no"

            model = make_single_slidingwindow_modelwrapper(
                type="ConvolutionInputGenerator1D",
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
                ram_style=ram_style,
            )
        else:
            # try 2D HLS ConvInpGen

            # unsupported parallelization
            if m > 1 or parallel_window > 0:
                output_dict["HLS_compatible"] = "no"

            model = make_single_slidingwindow_modelwrapper(
                type="ConvolutionInputGenerator",
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
                ram_style=ram_style,
            )

        if output_dict["HLS_compatible"] == "no":
            return output_dict

        # perform usual RTLSIM steps
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

        # extract first results (estimates)
        node_ = model.get_nodes_by_op_type("ConvolutionInputGenerator")
        if len(node_) == 0:
            node_ = model.get_nodes_by_op_type("ConvolutionInputGenerator1D")
        node = node_[0]
        inst = getCustomOp(node)

        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        output_dict["HLS_FINN_est_Cycles"] = exp_cycles

        exp_res_dict = model.analysis(res_estimation)
        exp_res = exp_res_dict[node.name]
        output_dict["HLS_FINN_est_LUT"] = exp_res["LUT"]
        output_dict["HLS_FINN_est_BRAM"] = exp_res["BRAM_18K"] * 0.5
        output_dict["HLS_FINN_est_URAM"] = exp_res["URAM"]

        exp_res_dict_hls = model.analysis(hls_synth_res_estimation)
        exp_res_hls = exp_res_dict_hls[node.name]
        output_dict["HLS_HLS_est_LUT"] = int(exp_res_hls["LUT"])
        output_dict["HLS_HLS_est_BRAM"] = int(exp_res_hls["BRAM_18K"]) * 0.5
        output_dict["HLS_HLS_est_URAM"] = int(exp_res_hls["URAM"])

        # perform rtlsim (for cycle measurement)
        if not skip_rtlsim:
            input_dict = prepare_inputs(x)
            oxe.execute_onnx(model, input_dict)["outp"]
            cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
            output_dict["HLS_Cycles"] = cycles_rtlsim

        # perform ooc synthesis (for resource/slack measurement)
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP("xczu7ev-ffvc1156-2-e", 5))
        model = model.transform(SynthOutOfContext(part="xczu7ev-ffvc1156-2-e", clk_period_ns=5))
        ooc_res_dict = eval(model.get_metadata_prop("res_total_ooc_synth"))
        output_dict["HLS_LUT"] = ooc_res_dict["LUT"]
        output_dict["HLS_BRAM"] = ooc_res_dict["BRAM_18K"] * 0.5 + ooc_res_dict["BRAM_36K"]
        output_dict["HLS_URAM"] = ooc_res_dict["URAM"]
        output_dict["HLS_WNS"] = ooc_res_dict["WNS"]
        output_dict["HLS_Fmax"] = ooc_res_dict["fmax_mhz"]

    return output_dict
