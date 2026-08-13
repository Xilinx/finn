import pytest

import json
import numpy as np
import os
import re
from dataclasses import replace
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
from finn.custom_op.fpgadataflow.rtl.finn_loop import _get_stream_tap_adjacency
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import getHWCustomOp, make_build_dir

verif_steps = [
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

fpga_part = "xcvc1902-vsva2197-2MP-e-S"
clk_ns = 5


def test_stream_tap_adjacency_includes_hls_mvau_fork():
    """HLS MVAUs in a Q/K/V fork must receive the forwarded loop index."""
    activation = create_tensor_info("activation", [1, 8])
    weights = [create_tensor_info(f"weights_{name}", [8, 8]) for name in ("q", "k", "v")]
    outputs = [create_tensor_info(f"output_{name}", [1, 8]) for name in ("q", "k", "v")]
    nodes = [
        helper.make_node(
            "MVAU_hls",
            ["activation", f"weights_{name}"],
            [f"output_{name}"],
            name=f"MVAU_hls_{name}",
            domain="finn.custom_op.fpgadataflow.hls",
            mlo_max_iter=12,
        )
        for name in ("q", "k", "v")
    ]
    graph = helper.make_graph(nodes, "qkv", [activation] + weights, outputs)
    loop_body = ModelWrapper(qonnx_make_model(graph, producer_name="qkv-loop-body"))

    _, pruned_adjacency = _get_stream_tap_adjacency(loop_body)

    assert pruned_adjacency["__INPUT0__"] == [
        "MVAU_hls_q",
        "MVAU_hls_k",
        "MVAU_hls_v",
    ]


def generate_random_threshold_values(data_type, num_input_channels, num_steps):
    if data_type.is_integer():
        return np.random.randint(
            data_type.min(),
            data_type.max() + 1,
            (num_input_channels, num_steps),
        ).astype(np.float32)
    else:
        return (np.random.randn(num_input_channels, num_steps) * 1000).astype(
            data_type.to_numpy_dt()
        )


def create_tensor_info(name, shape, proto=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, proto, shape)


def create_threshold(name, shape):
    return create_tensor_info(name, shape)


def create_node(node_type, inputs, outputs, name, extra_params={}):
    base_params = {
        "domain": "finn.custom_op.fpgadataflow.rtl"
        if "rtl" in node_type
        else "finn.custom_op.fpgadataflow.hls",
        "backend": "fpgadataflow",
        "numInputVectors": list((1, 3, 3)),
        "name": name,
    }
    return helper.make_node(node_type, inputs, outputs, **{**base_params, **extra_params})


def make_loop_modelwrapper(
    mw,
    mh,
    dtype=DataType["INT8"],
    elemwise_optype="ElementwiseMul_hls",
    rhs_shape=[1],
    eltw_param_dtype="INT8",
    name_suffix="",
    mvau_pe=2,
    mvau_simd=2,
    mvau_th=1,
    helper_pe=2,
    weight_bitwidth=None,
):
    is_float = eltw_param_dtype == "FLOAT32"

    # Output dtype of adding two `dtype` values needs one extra bit
    add_out_dtype = DataType[f"INT{dtype.bitwidth() + 1}"]

    # weights default to the activation dtype, but can use a separate (e.g. wider)
    # width to exercise the fetch-weights DDR path independently of the data path
    wdtype = DataType[f"INT{weight_bitwidth}"] if weight_bitwidth is not None else dtype

    # Determine elementwise output dtype
    # HLS elementwise outputs FLOAT32 if parameter is FLOAT32, otherwise INT32
    if is_float:
        elemwise_output_dtype = DataType["FLOAT32"]
        thresholding_input_dtype = DataType["FLOAT32"]
    else:
        elemwise_output_dtype = DataType["INT32"]
        thresholding_input_dtype = DataType["INT32"]

    W0 = gen_finn_dt_tensor(wdtype, (mw, mh))
    W1 = gen_finn_dt_tensor(wdtype, (mw, mh))
    W2 = gen_finn_dt_tensor(wdtype, (mh, mh))
    T0 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    T1 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    T2 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    T3_dtype = dtype
    T3 = np.sort(
        generate_random_threshold_values(T3_dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    # RTL elementwise requires matching bitwidths for int/int path
    actual_eltw_param_dtype = (
        add_out_dtype.name
        if (eltw_param_dtype != "FLOAT32" and "rtl" in elemwise_optype)
        else eltw_param_dtype
    )
    EltwParam = gen_finn_dt_tensor(DataType[actual_eltw_param_dtype], rhs_shape)

    tensor_shapes = {
        f"ifm{name_suffix}": [1, 3, 3, mw],
        f"weights{name_suffix}": [mw, mh],
        f"weights2{name_suffix}": [mh, mh],
    }
    output_shapes = {f"mm{name_suffix}": [1, 3, 3, mh], f"ofm{name_suffix}": (1, 3, 3, mh)}

    tensor_infos = {k: create_tensor_info(k, v) for k, v in tensor_shapes.items()}
    thresholds = [
        create_threshold(f"thresh{i}{name_suffix}", (1, dtype.get_num_possible_values() - 1))
        for i in range(4)
    ]

    nodes = [
        create_node(
            "DuplicateStreams_hls",
            [f"ifm{name_suffix}"],
            [f"ifm_1{name_suffix}", f"ifm_2{name_suffix}"],
            f"DuplicateStreams_hls_0{name_suffix}",
            {
                "NumChannels": mh,
                "NumOutputStreams": 2,
                "PE": helper_pe,
                "inputDataType": dtype.name,
                "outFIFODepths": [2, 2],
                "cpp_interface": "hls_vector",
                "hls_style": "freerunning",
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"ifm_1{name_suffix}", f"weights0{name_suffix}"],
            [f"mm0_out{name_suffix}"],
            f"MVAU_rtl_0{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": mvau_simd,
                "PE": mvau_pe,
                "TH": mvau_th,
                "inputDataType": dtype.name,
                "weightDataType": wdtype.name,
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm0_out{name_suffix}", f"thresh0{name_suffix}"],
            [f"mt0_out{name_suffix}"],
            f"Thresholding_rtl_0{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"mt0_out{name_suffix}", f"weights1{name_suffix}"],
            [f"mm1_out{name_suffix}"],
            f"MVAU_rtl_1{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": mvau_simd,
                "PE": mvau_pe,
                "TH": mvau_th,
                "inputDataType": dtype.name,
                "weightDataType": wdtype.name,
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm1_out{name_suffix}", f"thresh1{name_suffix}"],
            [f"mt1_out{name_suffix}"],
            f"Thresholding_rtl_1{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"ifm_2{name_suffix}", f"weights2{name_suffix}"],
            [f"mm2_out{name_suffix}"],
            f"MVAU_rtl_2{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": mvau_simd,
                "PE": mvau_pe,
                "TH": mvau_th,
                "inputDataType": dtype.name,
                "weightDataType": wdtype.name,
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm2_out{name_suffix}", f"thresh2{name_suffix}"],
            [f"mt2_out{name_suffix}"],
            f"Thresholding_rtl_2{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "ElementwiseAdd_hls",
            [f"mt2_out{name_suffix}", f"mt1_out{name_suffix}"],
            [f"ofm{name_suffix}"],
            f"ElementwiseAdd_hls_0{name_suffix}",
            {
                "lhs_shape": [1, 3, 3, mh],
                "rhs_shape": [1, 3, 3, mh],
                "out_shape": [1, 3, 3, mh],
                "lhs_dtype": dtype.name,
                "rhs_dtype": dtype.name,
                "out_dtype": add_out_dtype.name,
                "lhs_style": "input",
                "rhs_style": "input",
                "PE": helper_pe,
            },
        ),
        create_node(
            elemwise_optype,
            [f"ofm{name_suffix}", f"mul_param{name_suffix}"],
            [f"ofm_ew{name_suffix}"],
            f"ElementwiseOp{'_rtl' if 'rtl' in elemwise_optype else '_hls'}_0{name_suffix}",
            {
                "lhs_shape": [1, 3, 3, mh],
                "rhs_shape": rhs_shape,
                "out_shape": [1, 3, 3, mh],
                "lhs_dtype": add_out_dtype.name,
                # RTL elementwise requires matching bitwidths for int/int path
                "rhs_dtype": add_out_dtype.name
                if (eltw_param_dtype != "FLOAT32" and "rtl" in elemwise_optype)
                else eltw_param_dtype,
                "out_dtype": elemwise_output_dtype.name,
            },
        ),
    ]

    # Add RTL elementwise node after HLS elementwise when parameter is FLOAT32
    if is_float:
        # Use the same operation type as HLS but with _rtl suffix
        rtl_optype = elemwise_optype.replace("_hls", "_rtl")
        nodes.append(
            create_node(
                rtl_optype,
                [f"ofm_ew{name_suffix}", f"mul_param_rtl{name_suffix}"],
                [f"ofm_ew_rtl{name_suffix}"],
                f"ElementwiseOp_rtl_1{name_suffix}",
                {
                    "lhs_shape": [1, 3, 3, mh],
                    "rhs_shape": rhs_shape,
                    "out_shape": [1, 3, 3, mh],
                    "lhs_dtype": "FLOAT32",
                    "rhs_dtype": "FLOAT32",
                    "out_dtype": "FLOAT32",
                },
            )
        )
        thresholding_input_tensor = f"ofm_ew_rtl{name_suffix}"
    else:
        thresholding_input_tensor = f"ofm_ew{name_suffix}"

    nodes.append(
        create_node(
            "Thresholding_rtl",
            [thresholding_input_tensor, f"thresh3{name_suffix}"],
            [f"ofm_final{name_suffix}"],
            f"Thresholding_rtl4{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "numSteps": dtype.get_num_possible_values() - 1,
                "inputDataType": thresholding_input_dtype.name,
                "weightDataType": thresholding_input_dtype.name,
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
            },
        ),
    )

    # Build value_info list
    value_info_list = [
        create_tensor_info(name, output_shapes[f"mm{name_suffix}"])
        for name in [
            f"mm0_out{name_suffix}",
            f"mm1_out{name_suffix}",
            f"mm2_out{name_suffix}",
            f"ifm_1{name_suffix}",
            f"ifm_2{name_suffix}",
        ]
    ] + [
        create_tensor_info(name, output_shapes[f"ofm{name_suffix}"])
        for name in [
            f"mt0_out{name_suffix}",
            f"mt1_out{name_suffix}",
            f"mt2_out{name_suffix}",
            f"ofm{name_suffix}",
            f"ofm_ew{name_suffix}",
        ]
    ]

    # Add RTL elementwise output tensor to value_info if FLOAT32
    if is_float:
        value_info_list.append(
            create_tensor_info(f"ofm_ew_rtl{name_suffix}", output_shapes[f"ofm{name_suffix}"])
        )

    loop_body = helper.make_graph(
        nodes=nodes,
        name=f"matmul_graph{name_suffix}",
        inputs=[tensor_infos[f"ifm{name_suffix}"]] + thresholds,
        outputs=[create_tensor_info(f"ofm_final{name_suffix}", output_shapes[f"ofm{name_suffix}"])],
        value_info=value_info_list,
    )

    loop_body_model = qonnx_make_model(loop_body, producer_name=f"loop-body-model{name_suffix}")
    loop_body_model = ModelWrapper(loop_body_model)

    # Set initializers using generated values
    loop_body_model.set_initializer(f"weights0{name_suffix}", W0)
    loop_body_model.set_initializer(f"weights1{name_suffix}", W1)
    loop_body_model.set_initializer(f"weights2{name_suffix}", W2)
    loop_body_model.set_initializer(f"thresh0{name_suffix}", T0)
    loop_body_model.set_initializer(f"thresh1{name_suffix}", T1)
    loop_body_model.set_initializer(f"thresh2{name_suffix}", T2)
    loop_body_model.set_initializer(f"thresh3{name_suffix}", T3)
    loop_body_model.set_initializer(f"mul_param{name_suffix}", EltwParam)

    # Add RTL elementwise parameter when FLOAT32
    if is_float:
        EltwParamRtl = gen_finn_dt_tensor(DataType["FLOAT32"], rhs_shape)
        loop_body_model.set_initializer(f"mul_param_rtl{name_suffix}", EltwParamRtl)

    # Set tensor datatypes
    tensors = [
        f"weights0{name_suffix}",
        f"weights1{name_suffix}",
        f"weights2{name_suffix}",
        f"thresh0{name_suffix}",
        f"thresh1{name_suffix}",
        f"thresh2{name_suffix}",
        f"ifm{name_suffix}",
        f"ofm_final{name_suffix}",
    ]
    for tensor in tensors:
        loop_body_model.set_tensor_datatype(tensor, dtype)

    # weights may use a different (wider) datatype than the activations
    for w in (f"weights0{name_suffix}", f"weights1{name_suffix}", f"weights2{name_suffix}"):
        loop_body_model.set_tensor_datatype(w, wdtype)

    loop_body_model.set_tensor_datatype(f"thresh3{name_suffix}", T3_dtype)
    loop_body_model.set_tensor_datatype(
        f"mul_param{name_suffix}", DataType[actual_eltw_param_dtype]
    )

    # Set RTL elementwise parameter datatype when FLOAT32
    if is_float:
        loop_body_model.set_tensor_datatype(f"mul_param_rtl{name_suffix}", DataType["FLOAT32"])

    return loop_body_model


def make_single_mvau_loop_body(
    mw,
    mh,
    dtype=DataType["INT8"],
    name_suffix="",
    mvau_pe=2,
    mvau_simd=2,
    mvau_th=1,
    helper_pe=2,
):
    """Create a minimal loop body with just MVAU_rtl -> Thresholding_rtl."""

    W0 = gen_finn_dt_tensor(dtype, (mw, mh))
    T0 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )

    nodes = [
        create_node(
            "MVAU_rtl",
            [f"ifm{name_suffix}", f"weights0{name_suffix}"],
            [f"mm0_out{name_suffix}"],
            f"MVAU_rtl_0{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": mvau_simd,
                "PE": mvau_pe,
                "TH": mvau_th,
                "inputDataType": "INT8",
                "weightDataType": "INT8",
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm0_out{name_suffix}", f"thresh0{name_suffix}"],
            [f"ofm{name_suffix}"],
            f"Thresholding_rtl_0{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
    ]

    loop_body = helper.make_graph(
        nodes=nodes,
        name=f"single_mvau_graph{name_suffix}",
        inputs=[
            create_tensor_info(f"ifm{name_suffix}", [1, 3, 3, mw]),
            create_threshold(f"thresh0{name_suffix}", (1, dtype.get_num_possible_values() - 1)),
        ],
        outputs=[create_tensor_info(f"ofm{name_suffix}", (1, 3, 3, mh))],
        value_info=[
            create_tensor_info(f"mm0_out{name_suffix}", [1, 3, 3, mh]),
        ],
    )

    loop_body_model = qonnx_make_model(loop_body, producer_name=f"single-mvau-body{name_suffix}")
    loop_body_model = ModelWrapper(loop_body_model)

    loop_body_model.set_initializer(f"weights0{name_suffix}", W0)
    loop_body_model.set_initializer(f"thresh0{name_suffix}", T0)

    for tensor in [
        f"weights0{name_suffix}",
        f"thresh0{name_suffix}",
        f"ifm{name_suffix}",
        f"ofm{name_suffix}",
    ]:
        loop_body_model.set_tensor_datatype(tensor, dtype)

    return loop_body_model


def create_chained_loop_bodies(
    mw,
    mh,
    num_copies,
    elemwise_optype="ElementwiseMul_hls",
    rhs_shape=[1],
    eltw_param_dtype="INT8",
    dtype=DataType["INT8"],
    mvau_pe=2,
    mvau_simd=2,
    mvau_th=1,
    helper_pe=2,
    weight_bitwidth=None,
):
    loop_body_models = []

    # Create multiple instances of the loop body with unique name_suffix
    for i in range(num_copies):
        name_suffix = f"_{i}"
        loop_body_model = make_loop_modelwrapper(
            mw=mw,
            mh=mh,
            dtype=dtype,
            elemwise_optype=elemwise_optype,
            rhs_shape=rhs_shape,
            eltw_param_dtype=eltw_param_dtype,
            name_suffix=name_suffix,
            mvau_pe=mvau_pe,
            mvau_simd=mvau_simd,
            mvau_th=mvau_th,
            helper_pe=helper_pe,
            weight_bitwidth=weight_bitwidth,
        )
        loop_body_models.append(loop_body_model)

    return loop_body_models


@pytest.mark.fpgadataflow
def test_finnloop_exposes_body_double_pumped_clock():
    loop_body = make_loop_modelwrapper(16, 16)
    loop_body.set_metadata_prop(
        "vivado_stitch_ifnames",
        str(
            {
                "clk": ["ap_clk"],
                "clk2x": ["ap_clk2x"],
                "rst": ["ap_rst_n"],
                "s_axis": [],
                "m_axis": [],
                "aximm": [],
                "axilite": [],
            }
        ),
    )
    loop_node = helper.make_node(
        "FINNLoop",
        ["top_in"],
        ["top_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="rtl",
        name="FINNLoop_0",
        body=loop_body.graph,
        inputDataType="INT8",
        outputDataType="INT8",
        iteration=1,
    )
    graph = helper.make_graph(
        [loop_node],
        "double_pumped_loop",
        [create_tensor_info("top_in", [1, 3, 3, 16])],
        [create_tensor_info("top_out", [1, 3, 3, 16])],
    )
    model = ModelWrapper(qonnx_make_model(graph))

    loop_inst = getHWCustomOp(loop_node, model)
    assert loop_inst.get_verilog_top_module_intf_names()["clk2x"] == ["ap_clk2x"]
    assert CreateStitchedIP(fpga_part, clk_ns).is_double_pumped(loop_node, model)


@pytest.mark.fpgadataflow
def test_finnloop_mux_prioritizes_intermediate_frames():
    mux_path = Path(__file__).resolve().parents[2] / "finn-rtllib/mlo/infrastructure/mux.sv"
    mux_source = mux_path.read_text()
    control_start = mux_source.index("always_comb begin: DP_CTRL")
    control_end = mux_source.index("Q_srl #(", control_start)
    control = mux_source[control_start:control_end]

    # If a new batch input and recirculated frame are both ready, the feedback
    # frame must run first. Otherwise batch input can starve loop progress.
    assert control.index("if(s_idx_tvalid)") < control.index("else if(idx_fs_tvalid)")


@pytest.mark.fpgadataflow
def test_mlo_write_dma_boundary_helpers_are_continuous_nets():
    dma_path = (
        Path(__file__).resolve().parents[2]
        / "finn-rtllib/cdma/cdma_u/axi_dma_wr_u.sv"
    )
    dma_source = dma_path.read_text()

    # These values depend on live descriptor state. Declaring them as logic
    # with an initializer evaluates them only once at time zero in SystemVerilog,
    # causing later unaligned frames to overrun a 4 KiB boundary and underflow
    # op_word_count_reg.
    assert (
        "uwire [AXI_ADDR_WIDTH-1:0] addr_plus_max_burst = "
        "addr_reg + AXI_MAX_BURST_SIZE;"
    ) in dma_source
    assert (
        "uwire [AXI_ADDR_WIDTH-1:0] addr_plus_count = "
        "addr_reg + op_word_count_reg;"
    ) in dma_source


# MVAU folding as a jointly-valid tuple (dim, mvau_pe, mvau_simd, mvau_th, helper_pe).
# TH=1 selects the standard MVAU; TH>1 selects the tiled MVAU (Versal DSP58).
# The dimensions must satisfy the tiling constraints: MW % SIMD == 0, MH % PE == 0
# and (PE * SIMD) % TH == 0, so pe/simd/th cannot be stacked independently.
@pytest.mark.parametrize(
    "mvau_cfg",
    [
        (16, 2, 2, 1, 2),
        (12, 6, 3, 3, 6),
    ],
)
# iteration count, number of models chained together
@pytest.mark.parametrize("iteration", [3])
# elementwise operation
@pytest.mark.parametrize("elemwise_optype", ["ElementwiseMul_hls", "ElementwiseAdd_rtl"])
# elementwise shape
@pytest.mark.parametrize("rhs_shape", [[1], [16]])
# eltwise param dtype
@pytest.mark.parametrize("eltw_param_dtype", ["INT8", "FLOAT32"])
# tail node
@pytest.mark.parametrize("tail_node", [False, True])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_finnloop_end2end_mlo(
    mvau_cfg, iteration, elemwise_optype, rhs_shape, eltw_param_dtype, tail_node
):
    dim, mvau_pe, mvau_simd, mvau_th, helper_pe = mvau_cfg
    # The tiled MVAU (TH>1) is only exercised on selected elementwise configs to
    # avoid a combinatorial explosion of long Vivado builds. rhs_shape is pinned to
    # [1] since [16] is incompatible with the tiled config's dim. Within that, we
    # cover INT8/no-tail (canonical), FLOAT32/no-tail (float path) and INT8/tail
    # (tail-node integration), skipping the redundant FLOAT32+tail combination.
    if mvau_th > 1 and not (
        elemwise_optype == "ElementwiseMul_hls"
        and rhs_shape == [1]
        and not (eltw_param_dtype == "FLOAT32" and tail_node)
    ):
        pytest.skip("Tiled MVAU only exercised on selected elementwise configs")
    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    year, minor = int(match.group(1)), int(match.group(2))
    if (year, minor) < (2024, 2):
        pytest.skip("""At least Vivado version 2024.2 needed for MLO.""")
    loop_body_models = create_chained_loop_bodies(
        dim,
        dim,
        iteration,
        elemwise_optype,
        rhs_shape,
        eltw_param_dtype,
        mvau_pe=mvau_pe,
        mvau_simd=mvau_simd,
        mvau_th=mvau_th,
        helper_pe=helper_pe,
    )
    nodes_per_body = len(loop_body_models[0].graph.node)
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        model = model.transform(MergeONNXModels(m))

    if tail_node:
        tail_outp = create_tensor_info("tail_outp", [1, 3, 3, dim])
        tr_node = create_node(
            "ElementwiseAdd_hls",
            [model.graph.output[0].name, "tail_add"],
            ["tail_outp"],
            "Add_tail",
            {
                "lhs_shape": [1, 3, 3, dim],
                "rhs_shape": [1],
                "out_shape": [1, 3, 3, dim],
                "lhs_dtype": "INT8",
                "rhs_dtype": "INT8",
                "out_dtype": "INT9",
            },
        )
        model.graph.node.insert(len(model.graph.node), tr_node)
        model.graph.value_info.append(model.graph.output[0])
        model.graph.output.pop(0)
        model.graph.output.append(tail_outp)
        AddtailParam = gen_finn_dt_tensor(DataType["INT8"], [1])
        model.set_initializer("tail_add", AddtailParam)
        model.set_tensor_datatype("tail_add", DataType["INT8"])

    # cleanup
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Generate reference output
    input_dtype = DataType["INT8"]
    x = gen_finn_dt_tensor(input_dtype, (1, 3, 3, dim))
    model_ref = model.transform(PrepareCppSim())
    model_ref = model_ref.transform(CompileCppSim())
    model_ref = model_ref.transform(SetExecMode("cppsim"))
    io_dict = {model_ref.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model_ref, io_dict)
    y_ref = y_dict[model_ref.graph.output[0].name]

    tmp_output_dir = make_build_dir("build_mlo")

    np.save(tmp_output_dir + "/input.npy", x)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)

    model.save(tmp_output_dir + "/mlo_model.onnx")

    # Use phase-based pipeline
    # Steps are adjusted because test model already has HLS and RTL layers
    steps = [
        "step_create_dataflow_partition",  # Fine-grained (model already specialized)
        "phase_convert_to_hardware",  # Phase (includes loop rolling)
        "phase_optimize_hardware",  # Phase (includes folding, bit-width, reports)
        "phase_build_hardware",  # Phase (includes codegen, ipgen, FIFOs)
        "phase_generate_outputs",  # Phase (only stitched IP requested, so no full synth)
    ]

    # debug_fifo forces behavioral verification and per-FIFO log capture, which
    # noticeably extends the flow. Only exercise it on a single canonical config.
    run_fifo_debug = (
        mvau_cfg == (16, 2, 2, 1, 2)
        and elemwise_optype == "ElementwiseMul_hls"
        and rhs_shape == [1]
        and eltw_param_dtype == "INT8"
        and not tail_node
    )

    # Exercise multi-frame stitched-MLO performance measurement once. This
    # path uses the ideal AXI-MM models for loop storage and MVAU weights.
    run_rtlsim_performance = (
        mvau_cfg == (16, 2, 2, 1, 2)
        and elemwise_optype == "ElementwiseAdd_rtl"
        and rhs_shape == [1]
        and eltw_param_dtype == "INT8"
        and not tail_node
    )
    generate_outputs = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
    ]
    if run_rtlsim_performance:
        generate_outputs.append(build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE)

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        synth_clk_period_ns=10.0,
        board="V80",
        rtlsim_batch_size=2 if run_rtlsim_performance else 100,
        standalone_thresholds=True,
        mlo=True,
        loop_body_hierarchy=[["", "layers.0"]],
        loop_body_range=(model.graph.node[0], model.graph.node[nodes_per_body - 1]),
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        verify_save_full_context=True,  # Enable per-iteration context saving
        debug_fifo=run_fifo_debug,  # snapshot per-FIFO sizing logs (tagged per loop body)
        generate_outputs=generate_outputs,
    )
    build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

    # check if expected files are there
    assert os.path.isfile(tmp_output_dir + "/loop-body-template.onnx")
    report_dir = tmp_output_dir + "/report"
    assert os.path.isfile(report_dir + "/estimate_layer_config_alternatives_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_config_alternatives.json")
    assert os.path.isfile(report_dir + "/estimate_layer_cycles_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_cycles.json")
    assert os.path.isfile(report_dir + "/estimate_layer_resources_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_resources.json")
    assert os.path.isfile(report_dir + "/op_and_param_counts_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/op_and_param_counts.json")
    assert os.path.isfile(tmp_output_dir + "/stitched_ip/ip/component.xml")
    if run_rtlsim_performance:
        with open(report_dir + "/rtlsim_performance.json") as f:
            rtlsim_perf = json.load(f)
        assert rtlsim_perf["measurement_scope"] == "stitched_mlo"
        assert rtlsim_perf["external_memory_model"] == "ideal_axi_mm"
        assert rtlsim_perf["external_memory_model_is_ideal"] is True
        assert rtlsim_perf["performance_interpretation"] == "ideal_memory_upper_bound"
        assert rtlsim_perf["N"] == 2
        assert rtlsim_perf["completed_output_frames"] == 2
        assert rtlsim_perf["interval_valid"] == 1
        assert rtlsim_perf["steady_state_frames"] == 1
        assert rtlsim_perf["steady_state_cycles"] > 0
        assert rtlsim_perf["stable_throughput_valid"] is True

    verif_dir = tmp_output_dir + "/verification_output"
    # With verify_save_full_context=True, cppsim and node_by_node_rtlsim save as .npz
    # stitched_ip_rtlsim with MLO uses rtlsim_pre_hook so it saves as .npy
    assert os.path.isfile(
        verif_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"

    # Verify that the per-iteration context file was created for FINNLoop
    iteration_context_files = [
        f for f in os.listdir(verif_dir) if f.startswith("iteration_context_")
    ]
    assert len(iteration_context_files) > 0, f"No iteration context files found in {verif_dir}"

    # Load and verify the iteration context file has expected structure
    ctx_file = os.path.join(verif_dir, iteration_context_files[0])
    ctx_data = np.load(ctx_file)
    iter_keys = [k for k in ctx_data.files if k.startswith("iter_")]
    assert len(iter_keys) > 0, "No iteration keys found in context file"

    # Verify we have contexts for all iterations
    iter_indices = set()
    for key in iter_keys:
        parts = key.split("_", 2)
        if len(parts) >= 2:
            iter_indices.add(int(parts[1]))
    assert (
        len(iter_indices) == iteration
    ), f"Expected {iteration} iterations in context, found {len(iter_indices)}"

    # debug_fifo snapshots per-FIFO sizing logs. For MLO the loop-body FIFO sizing
    # tags each log with its enclosing FINNLoop name and stores them under a subdir
    # named after that loop, so verify the per-loop logs landed there.
    if run_fifo_debug:
        loop_fifo_debug_dir = tmp_output_dir + "/debug/fifo_logs/fifo_sizing/FINNLoop_0"
        assert os.path.isdir(
            loop_fifo_debug_dir
        ), f"missing per-loop fifo debug dir {loop_fifo_debug_dir}"
        loop_fifo_logs = [f for f in os.listdir(loop_fifo_debug_dir) if f.endswith(".log")]
        assert len(loop_fifo_logs) > 0, f"no per-FIFO debug logs in {loop_fifo_debug_dir}"
        assert all(
            f.startswith("FINNLoop_0_") for f in loop_fifo_logs
        ), f"per-loop fifo logs not tagged with loop context: {loop_fifo_logs}"

    # also run dcp generation for a subset of the test parameters
    # this extends the test run time quite a lot
    # so only do for 2 of the scenarios

    if (
        elemwise_optype == "ElementwiseMul_hls"
        and rhs_shape == [1]
        and eltw_param_dtype == "FLOAT32"
    ):
        # launch another build just to test dcp generation
        cfg = replace(
            cfg,
            start_step="phase_generate_outputs",
            stitched_ip_gen_dcp=True,
            verify_steps=[],
        )
        build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

        # check if stitched IP dcp is there
        assert os.path.isfile(
            tmp_output_dir + "/stitched_ip/finn_design.dcp"
        ), f"Check vivado.log in {tmp_output_dir}/stitched_ip"


@pytest.mark.parametrize(
    "dim, simd, pe, bitwidth, weight_bitwidth",
    [(16, 1, 1, 8, 8), (8, 8, 4, 3, 3)],
)
# iteration count, number of models chained together
@pytest.mark.parametrize("iteration", [3])
# elementwise operation
@pytest.mark.parametrize("elemwise_optype", ["ElementwiseAdd_hls"])
# elementwise shape
@pytest.mark.parametrize("rhs_shape", [[1]])
# tail node
@pytest.mark.parametrize("tail_node", [True])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_finnloop_end2end_mlo_ddr(
    dim,
    simd,
    pe,
    iteration,
    elemwise_optype,
    rhs_shape,
    bitwidth,
    weight_bitwidth,
    tail_node,
    request,
):
    # End-to-end MLO+DDR flow parametrized by data/weight bitwidth and MVAU folding.
    data_dtype = DataType[f"INT{bitwidth}"]
    eltw_param_dtype = data_dtype.name
    # output dtype of adding two `data_dtype` values needs one extra bit
    add_out_dtype = DataType[f"INT{data_dtype.bitwidth() + 1}"]

    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    year, minor = int(match.group(1)), int(match.group(2))
    if (year, minor) < (2024, 2):
        pytest.skip("""At least Vivado version 2024.2 needed for MLO.""")
    loop_body_models = create_chained_loop_bodies(
        dim,
        dim,
        iteration,
        elemwise_optype,
        rhs_shape,
        eltw_param_dtype,
        dtype=data_dtype,
        mvau_simd=simd,
        mvau_pe=pe,
        weight_bitwidth=weight_bitwidth,
    )
    nodes_per_body = len(loop_body_models[0].graph.node)
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        model = model.transform(MergeONNXModels(m))

    if tail_node:
        tail_outp = create_tensor_info("tail_outp", [1, 3, 3, dim])
        tr_node = create_node(
            "ElementwiseAdd_hls",
            [model.graph.output[0].name, "tail_add"],
            ["tail_outp"],
            "Add_tail",
            {
                "lhs_shape": [1, 3, 3, dim],
                "rhs_shape": [1],
                "out_shape": [1, 3, 3, dim],
                "lhs_dtype": data_dtype.name,
                "rhs_dtype": data_dtype.name,
                "out_dtype": add_out_dtype.name,
            },
        )
        model.graph.node.insert(len(model.graph.node), tr_node)
        model.graph.value_info.append(model.graph.output[0])
        model.graph.output.pop(0)
        model.graph.output.append(tail_outp)
        AddtailParam = gen_finn_dt_tensor(data_dtype, [1])
        model.set_initializer("tail_add", AddtailParam)
        model.set_tensor_datatype("tail_add", data_dtype)

    # cleanup
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Generate reference output
    input_dtype = data_dtype
    x = gen_finn_dt_tensor(input_dtype, (1, 3, 3, dim))
    model_ref = model.transform(PrepareCppSim())
    model_ref = model_ref.transform(CompileCppSim())
    model_ref = model_ref.transform(SetExecMode("cppsim"))
    io_dict = {model_ref.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model_ref, io_dict)
    y_ref = y_dict[model_ref.graph.output[0].name]

    test_id = re.sub(r"[^0-9A-Za-z_]+", "_", request.node.name)
    tmp_output_dir = make_build_dir(f"build_mlo_{test_id}_")

    batch_size = 16
    np.save(tmp_output_dir + "/input.npy", np.broadcast_to(x, (batch_size, 3, 3, dim)))
    np.save(
        tmp_output_dir + "/expected_output.npy",
        np.broadcast_to(y_ref, (batch_size, 3, 3, dim)),
    )

    model.save(tmp_output_dir + "/mlo_model.onnx")

    # Use phase-based pipeline
    # Steps are adjusted because test model already has HLS and RTL layers
    steps = [
        "step_create_dataflow_partition",  # Fine-grained (model already specialized)
        "phase_convert_to_hardware",  # Phase (includes loop rolling)
        "phase_optimize_hardware",  # Phase (includes folding, bit-width, reports)
        "phase_build_hardware",  # Phase (includes codegen, ipgen, FIFOs)
        "phase_generate_outputs",  # Phase (stitched IP, bitfile synth, driver, deployment)
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        synth_clk_period_ns=10.0,
        board="AUP-ZU3_8GB",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        rtlsim_batch_size=100,
        standalone_thresholds=True,
        mlo=True,
        fifosim_save_waveform=True,
        loop_body_hierarchy=[["", "layers.0"]],
        loop_body_range=(model.graph.node[0], model.graph.node[nodes_per_body - 1]),
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        verify_save_full_context=True,  # Enable per-iteration context saving
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

    # check if expected files are there
    assert os.path.isfile(tmp_output_dir + "/loop-body-template.onnx")
    report_dir = tmp_output_dir + "/report"
    assert os.path.isfile(report_dir + "/estimate_layer_config_alternatives_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_config_alternatives.json")
    assert os.path.isfile(report_dir + "/estimate_layer_cycles_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_cycles.json")
    assert os.path.isfile(report_dir + "/estimate_layer_resources_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/estimate_layer_resources.json")
    assert os.path.isfile(report_dir + "/op_and_param_counts_FINNLoop_0.json")
    assert os.path.isfile(report_dir + "/op_and_param_counts.json")
    assert os.path.isfile(tmp_output_dir + "/stitched_ip/ip/component.xml")

    verif_dir = tmp_output_dir + "/verification_output"
    # With verify_save_full_context=True, cppsim and node_by_node_rtlsim save as .npz
    # stitched_ip_rtlsim with MLO uses rtlsim_pre_hook so it saves as .npy
    assert os.path.isfile(
        verif_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"

    # Verify that the per-iteration context file was created for FINNLoop
    iteration_context_files = [
        f for f in os.listdir(verif_dir) if f.startswith("iteration_context_")
    ]
    assert len(iteration_context_files) > 0, f"No iteration context files found in {verif_dir}"

    # Load and verify the iteration context file has expected structure
    ctx_file = os.path.join(verif_dir, iteration_context_files[0])
    ctx_data = np.load(ctx_file)
    iter_keys = [k for k in ctx_data.files if k.startswith("iter_")]
    assert len(iter_keys) > 0, "No iteration keys found in context file"

    # Verify we have contexts for all iterations
    iter_indices = set()
    for key in iter_keys:
        parts = key.split("_", 2)
        if len(parts) >= 2:
            iter_indices.add(int(parts[1]))
    assert (
        len(iter_indices) == iteration
    ), f"Expected {iteration} iterations in context, found {len(iter_indices)}"
