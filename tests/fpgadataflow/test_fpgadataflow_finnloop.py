import pytest

import numpy as np
import os
import re
from dataclasses import replace
from functools import partial
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.fifo_transaction_counts import fifo_transaction_counts
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import make_build_dir
from finn.util.rtlsim import fifo_log_is_consistent, read_fifo_log_snapshot

verif_steps = [
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

fpga_part = "xcvc1902-vsva2197-2MP-e-S"
clk_ns = 5


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
    # Requant scale/bias (per-tensor, magnitudes in safe range)
    # Scale must be >= ~0.125 (exponent >= -3) for INT8 output due to fixed-point constraints
    requant_scale = np.random.uniform(0.5, 2.0, [1]).astype(np.float32)
    requant_bias = np.random.uniform(-4.0, 4.0, [1]).astype(np.float32)
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
    # Both paths use a final Requant_rtl: 3 thresholds + scale/bias
    thresholds = [
        create_threshold(f"thresh{i}{name_suffix}", (1, dtype.get_num_possible_values() - 1))
        for i in range(3)
    ]
    requant_inputs = [
        helper.make_tensor_value_info(f"scale{name_suffix}", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info(f"bias{name_suffix}", TensorProto.FLOAT, [1]),
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

    # Final layer is Requant_rtl for both paths. On the float path the input is
    # FLOAT32 (requantf.sv / Versal); on the int path it is INT32.
    nodes.append(
        create_node(
            "Requant_rtl",
            [thresholding_input_tensor, f"scale{name_suffix}", f"bias{name_suffix}"],
            [f"ofm_final{name_suffix}"],
            f"Requant_rtl_0{name_suffix}",
            {
                "NumChannels": mh,
                "PE": helper_pe,
                "inputDataType": thresholding_input_dtype.name,
                "outputDataType": dtype.name,
                "narrow": 0,
                "numInputVectors": [1, 3, 3],
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
        inputs=[tensor_infos[f"ifm{name_suffix}"]] + thresholds + requant_inputs,
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
    loop_body_model.set_initializer(f"mul_param{name_suffix}", EltwParam)

    # Final Requant scale/bias (both paths)
    loop_body_model.set_initializer(f"scale{name_suffix}", requant_scale)
    loop_body_model.set_initializer(f"bias{name_suffix}", requant_bias)
    # Float path also has an extra RTL elementwise mul before the Requant
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

    loop_body_model.set_tensor_datatype(
        f"mul_param{name_suffix}", DataType[actual_eltw_param_dtype]
    )

    # Final Requant scale/bias datatypes (both paths)
    loop_body_model.set_tensor_datatype(f"scale{name_suffix}", DataType["FLOAT32"])
    loop_body_model.set_tensor_datatype(f"bias{name_suffix}", DataType["FLOAT32"])
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


def assert_finnloop_cycle_estimate(build_dir, x, rtol=0.35, atol=50):
    """Compare FINNLoop.get_exp_cycles() against the measured rtlsim cycle count.

    Reuses the loop-body IP already built by an MLO end-to-end build (no extra
    synthesis): loads the node-by-node rtlsim child model saved by the build,
    re-runs rtlsim on the single FINNLoop node (which populates its
    ``cycles_rtlsim`` nodeattr, see HWCustomOp.rtlsim_multi_io) and asserts the
    estimate is close to the measured value, mirroring the FMPadding cycle check.
    """
    child_fn = build_dir + "/intermediate_models/verify_node_by_node_rtlsim.onnx"
    assert os.path.isfile(child_fn), f"missing node-by-node rtlsim model {child_fn}"
    # child was saved by the build with exec_mode=rtlsim and rtlsim prepared; the
    # loop params are baked as initializers, so only the activation input is needed
    child = ModelWrapper(child_fn)
    loop_nodes = child.get_nodes_by_op_type("FINNLoop")
    assert len(loop_nodes) == 1, f"expected exactly one FINNLoop, got {len(loop_nodes)}"
    inst = getCustomOp(loop_nodes[0])
    in_name = child.graph.input[0].name
    ishape = tuple(child.get_tensor_shape(in_name))
    x_single = x.reshape((1,) + ishape[1:])
    oxe.execute_onnx(child, {in_name: x_single})
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles = inst.get_exp_cycles()
    assert exp_cycles != 0
    # The FINNLoop estimate is a conservative liveness bound and tends to slightly
    # over-count (measured ~2928 vs ~2310 rtlsim, i.e. ~27% high, for the canonical
    # config), which is the safe direction for the watchdog. A relative tolerance is
    # used because the per-iteration overhead heuristic makes the error scale with
    # the iteration count.
    assert np.isclose(exp_cycles, cycles_rtlsim, rtol=rtol, atol=atol), (
        f"FINNLoop cycle estimate {exp_cycles} not close to rtlsim {cycles_rtlsim} "
        f"(rtol={rtol}, atol={atol})"
    )


def _read_fifo_log_dir(log_dir, expect_verbose=True):
    """Parse one debug_fifo snapshot directory, checking its format and consistency."""
    assert os.path.isdir(log_dir), f"missing fifo debug dir {log_dir}"
    logs = read_fifo_log_snapshot(log_dir)
    assert logs, f"no non-empty per-FIFO debug logs in {log_dir}"
    empty = [
        f
        for f in sorted(os.listdir(log_dir))
        if f.endswith(".log") and os.path.getsize(os.path.join(log_dir, f)) == 0
    ]
    assert empty == [], f"empty FIFO logs in {log_dir}: {empty}"
    wrong_format = {
        n: log["verbose"] for n, log in logs.items() if log["verbose"] != expect_verbose
    }
    assert wrong_format == {}, (
        f"built with fifo_log_verbose={expect_verbose} but these logs say "
        f"otherwise: {wrong_format}"
    )
    for log in logs.values():
        fifo_log_is_consistent(log)
    return logs


def _logs_by_instance(logs, scope):
    """Re-key a snapshot by the FIFO instance name the gauge itself reported."""
    by_inst = {}
    for fname, log in logs.items():
        # e.g. "..._wrapper.finn_design_i.FINNLoop_0.<...>.<node name>.inst.fifo"
        inst = log["name"].split(".")[-3]
        assert inst not in by_inst, (
            f"{scope}: {fname}.log and {by_inst[inst]['path']} both report on " f"instance {inst}"
        )
        by_inst[inst] = log
    return by_inst


def assert_mlo_fifo_logs(build_dir, loop_name):
    """Check the debug_fifo snapshots of an MLO build against its final graph."""
    log_root = build_dir + "/debug/fifo_logs"
    body_prefix = loop_name + "_"

    sizing_logs = _read_fifo_log_dir(f"{log_root}/fifo_sizing/{loop_name}")
    stitched_body_logs = _read_fifo_log_dir(f"{log_root}/stitched_ip_rtlsim/{loop_name}")
    for scope, logs in [("fifo_sizing", sizing_logs), ("stitched_ip_rtlsim", stitched_body_logs)]:
        untagged = sorted(n for n in logs if not n.startswith(body_prefix))
        assert not untagged, (
            f"{scope}/{loop_name}: log(s) {untagged} are in the loop snapshot but "
            f"are not tagged with that loop context"
        )

    final_model = ModelWrapper(build_dir + "/intermediate_models/step_create_stitched_ip.onnx")
    all_counts = final_model.analysis(partial(fifo_transaction_counts, apply_to_subgraphs=True))
    body_counts = {k: v for k, v in all_counts.items() if k.startswith(body_prefix)}
    main_counts = {k: v for k, v in all_counts.items() if not k.startswith(body_prefix)}
    assert body_counts, "expected FIFOs inside the loop body"
    # the non_mlo_nodes head/tail layers are what put FIFOs in the main graph
    assert main_counts, "expected top-level FIFOs around the FINNLoop"

    # The stitched-ip rtlsim runs the whole design once, so every FIFO in the final
    # graph must have reported, in the scope it belongs to and only there.
    body_by_inst = _logs_by_instance(stitched_body_logs, f"stitched_ip_rtlsim/{loop_name}")
    _assert_logs_cover_fifos(body_by_inst, body_counts, f"stitched_ip_rtlsim/{loop_name}")
    main_logs = _read_fifo_log_dir(f"{log_root}/stitched_ip_rtlsim/main")
    main_by_inst = _logs_by_instance(main_logs, "stitched_ip_rtlsim/main")
    _assert_logs_cover_fifos(main_by_inst, main_counts, "stitched_ip_rtlsim/main")
    body_in_main = sorted(n for n in main_by_inst if n.startswith(body_prefix))
    assert not body_in_main, (
        f"stitched_ip_rtlsim/main: loop-body log(s) {body_in_main} leaked into the "
        f"top-level snapshot"
    )

    assert len(sizing_logs) >= len(body_counts), (
        f"fifo_sizing/{loop_name}: {len(sizing_logs)} log(s) for a body that ends "
        f"up with {len(body_counts)} FIFO(s)"
    )

    for name, log in body_by_inst.items():
        assert log["in"] == body_counts[name], "%s: logged in=%d, expected %d" % (
            log["path"],
            log["in"],
            body_counts[name],
        )
    for name, log in main_by_inst.items():
        assert log["in"] == main_counts[name], "%s: logged in=%d, expected %d" % (
            log["path"],
            log["in"],
            main_counts[name],
        )


def _assert_logs_cover_fifos(logs, expected_counts, scope):
    """Assert a snapshot holds exactly one log per expected FIFO -- no gaps, no strays."""
    missing = sorted(set(expected_counts) - set(logs))
    extra = sorted(set(logs) - set(expected_counts))
    assert not missing, f"{scope}: no log written for FIFO(s) {missing}"
    assert not extra, f"{scope}: log(s) {extra} do not correspond to a FIFO in this scope"


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
# insert non-MLO head/tail nodes (and a non-HW parent node) around the FINNLoop
@pytest.mark.parametrize("non_mlo_nodes", [False, True])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_finnloop_end2end_mlo(
    mvau_cfg, iteration, elemwise_optype, rhs_shape, eltw_param_dtype, non_mlo_nodes
):
    dim, mvau_pe, mvau_simd, mvau_th, helper_pe = mvau_cfg
    # The tiled MVAU (TH>1) is only exercised on selected elementwise configs to
    # avoid a combinatorial explosion of long Vivado builds. rhs_shape is pinned to
    # [1] since [16] is incompatible with the tiled config's dim. Within that, we
    # cover INT8/no-extra-nodes (canonical), FLOAT32/no-extra-nodes (float path) and
    # INT8/extra-nodes (head/tail/parent integration), skipping the redundant
    # FLOAT32+extra-nodes combination.
    if mvau_th > 1 and not (
        elemwise_optype == "ElementwiseMul_hls"
        and rhs_shape == [1]
        and not (eltw_param_dtype == "FLOAT32" and non_mlo_nodes)
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

    if non_mlo_nodes:
        # tail node on the output side
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

        # head node on the input side, symmetric with the tail node, kept inside the
        # partition ahead of the FINNLoop. A zero-crop preserves shape and datatype so
        # the loop-carried datatype (loop input dtype == loop output dtype) is intact.
        loop_in_name = model.graph.input[0].name
        first_body_node = model.find_consumer(loop_in_name)
        head_node = create_node(
            "Crop_hls",
            [loop_in_name],
            ["head_out"],
            "Crop_head",
            {
                "DataType": "INT8",
                "ImgDim": [3, 3],
                "NumChannels": dim,
                "CropNorth": 0,
                "CropSouth": 0,
                "CropWest": 0,
                "CropEast": 0,
                "SIMD": 1,
                "numInputVectors": [1],
                "cpp_interface": "hls_vector",
                "hls_style": "freerunning",
            },
        )
        for i, inp in enumerate(first_body_node.input):
            if inp == loop_in_name:
                first_body_node.input[i] = "head_out"
        model.graph.node.insert(0, head_node)
        model.graph.value_info.append(create_tensor_info("head_out", [1, 3, 3, dim]))
        model.set_tensor_datatype("head_out", DataType["INT8"])

    # A non-HW parent node (single config): prepended ahead of the head node so it
    # lands in dataflow_parent.onnx, outside the SDP, exercising parent-routed
    # stitched-IP verification. Shape-preserving Transpose over the two size-3 axes
    # keeps (1, 3, 3, dim) but permutes values, so the hardware must reproduce it.
    insert_parent_node = (
        non_mlo_nodes
        and mvau_cfg == (16, 2, 2, 1, 2)
        and elemwise_optype == "ElementwiseMul_hls"
        and rhs_shape == [1]
        and eltw_param_dtype == "INT8"
    )
    if insert_parent_node:
        parent_in = model.graph.input[0].name
        parent_consumer = model.find_consumer(parent_in)
        parent_tr = helper.make_node(
            "Transpose",
            [parent_in],
            ["parent_transpose_out"],
            name="Parent_Transpose",
            perm=[0, 2, 1, 3],
        )
        for i, inp in enumerate(parent_consumer.input):
            if inp == parent_in:
                parent_consumer.input[i] = "parent_transpose_out"
        model.graph.node.insert(0, parent_tr)
        model.graph.value_info.append(create_tensor_info("parent_transpose_out", [1, 3, 3, dim]))
        model.set_tensor_datatype("parent_transpose_out", DataType["INT8"])

    # number of nodes prepended ahead of the first loop body (head + parent node)
    n_prepended = int(non_mlo_nodes) + int(insert_parent_node)

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

    # Use phase-based pipeline. phase_convert_to_hardware already partitions
    # internally, so step_create_dataflow_partition must not be listed separately:
    # a second partition would re-derive dataflow_parent.onnx off the child and
    # drop the non-HW parent nodes.
    steps = [
        "phase_convert_to_hardware",  # Phase (includes partition + loop rolling)
        "phase_optimize_hardware",  # Phase (includes folding, bit-width, reports)
        "phase_build_hardware",  # Phase (includes codegen, ipgen, FIFOs)
        "phase_generate_outputs",  # Phase (only stitched IP requested, so no full synth)
    ]

    # The single canonical parameter combination, used to gate the two expensive
    # extra checks below so they each run exactly once per non_mlo_nodes value.
    canonical_cfg = (
        mvau_cfg == (16, 2, 2, 1, 2)
        and elemwise_optype == "ElementwiseMul_hls"
        and rhs_shape == [1]
        and eltw_param_dtype == "INT8"
    )

    run_fifo_debug = canonical_cfg and non_mlo_nodes

    # Cycle-count verification stays on the plain FINNLoop, where the measured
    # rtlsim cycles are the loop's alone and not inflated by head/tail nodes.
    run_cycle_check = canonical_cfg and not non_mlo_nodes

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        synth_clk_period_ns=10.0,
        board="V80",
        rtlsim_batch_size=100,
        standalone_thresholds=True,
        mlo=True,
        loop_body_hierarchy=[["", "layers.0"]],
        loop_body_range=(
            model.graph.node[n_prepended],
            model.graph.node[n_prepended + nodes_per_body - 1],
        ),
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        verify_save_full_context=True,  # Enable per-iteration context saving
        debug_fifo=run_fifo_debug,  # snapshot per-FIFO sizing logs (tagged per loop body)
        fifo_log_verbose=True,
        # MLO pins folding via mvau_pe/mvau_simd on the nodes at creation time, so the
        # folding_missing check (which assumes creation-time PE=1/SIMD=1) is a false
        # positive here; target_fps would instead override the deliberate folding.
        mute_config_assertions=True,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

    if insert_parent_node:
        # The prepended non-HW node must stay outside the StreamingDataflowPartition,
        # i.e. it lands in dataflow_parent.onnx. This ensures stitched_ip_rtlsim
        # verification actually routes the input through the parent graph.
        parent_model = ModelWrapper(tmp_output_dir + "/intermediate_models/dataflow_parent.onnx")
        non_sdp_nodes = [
            n for n in parent_model.graph.node if n.op_type != "StreamingDataflowPartition"
        ]
        assert len(non_sdp_nodes) > 0, "Parent model has no non-SDP node to run"

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
    # With verify_save_full_context=True, all verification steps save the full
    # context as .npz. MLO stitched_ip_rtlsim now routes through the parent model
    # (need_parent=True), so it also saves the full context as .npz.
    assert os.path.isfile(
        verif_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"

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

    # Cycle-count verification for the FINNLoop: compare get_exp_cycles() against the
    # measured rtlsim cycles (FMPadding-style). Gated to the single canonical config so
    # it reuses the already-built loop-body IP with just one extra rtlsim run.
    if run_cycle_check:
        assert_finnloop_cycle_estimate(tmp_output_dir, x)

    if run_fifo_debug:
        assert_mlo_fifo_logs(tmp_output_dir, "FINNLoop_0")

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
    [
        # Coverage matrix over {folding} x {256-divisibility of the element widths}.
        (16, 1, 1, 8, 8),  # unfolded, divisor (8|256): baseline PASS
        (8, 8, 4, 4, 4),  # folded, divisor (4|256, DMA_PE=64): guards word-aligned image
        #   stays byte-identical for divisors at the real folding
        (8, 8, 4, 3, 3),  # folded, non-divisor (3 wasted bits/word): exercises the
        #   DMA-word-aligned fix on both the activation and weight paths
    ],
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

    np.save(tmp_output_dir + "/input.npy", x)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)

    model.save(tmp_output_dir + "/mlo_model.onnx")

    # Use phase-based pipeline. phase_convert_to_hardware already partitions
    # internally, so step_create_dataflow_partition must not be listed separately.
    steps = [
        "phase_convert_to_hardware",  # Phase (includes partition + loop rolling)
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
        # MLO pins folding via mvau_pe/mvau_simd on the nodes at creation time, so the
        # folding_missing check (which assumes creation-time PE=1/SIMD=1) is a false
        # positive here; target_fps would instead override the deliberate folding.
        mute_config_assertions=True,
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
    # With verify_save_full_context=True, all verification steps save the full
    # context as .npz. MLO stitched_ip_rtlsim now routes through the parent model
    # (need_parent=True), so it also saves the full context as .npz.
    assert os.path.isfile(
        verif_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npz"
    ), f"Check npz files in {verif_dir}"

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
