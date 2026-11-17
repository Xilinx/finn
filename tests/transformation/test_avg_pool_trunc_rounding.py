# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN


def get_trunc_in_bits(act_bits, kernel):
    # Brevitas sizes the Trunc input for the largest unsigned window sum
    return int(np.ceil(np.log2((2**act_bits - 1) * kernel**2)))


def make_avgpool_trunc_model(kernel, act_bits, signed, trunc_out_bits, rounding_mode, n_windows):
    """Quant -> AveragePool -> Trunc (v2), with one kernel x kernel window per sum."""
    act_scale = np.float32(0.25)
    # Trunc v2 sees the window average: input scale = act scale / window size
    trunc_scale = np.float32(act_scale / kernel**2)
    trunc_in_bits = get_trunc_in_bits(act_bits, kernel)
    trunc_out_scale = np.float32(trunc_scale * 2 ** (trunc_in_bits - trunc_out_bits))

    ishape = [1, 1, kernel, kernel * n_windows]
    oshape = [1, 1, 1, n_windows]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)
    nodes = [
        helper.make_node(
            "Quant",
            ["inp", "act_scale", "zeropt", "act_bits"],
            ["act"],
            domain="qonnx.custom_op.general",
            signed=int(signed),
            narrow=0,
            rounding_mode="ROUND",
        ),
        helper.make_node(
            "AveragePool",
            ["act"],
            ["avg"],
            kernel_shape=[kernel, kernel],
            strides=[kernel, kernel],
            pads=[0, 0, 0, 0],
        ),
        helper.make_node(
            "Trunc",
            [
                "avg",
                "trunc_scale",
                "zeropt",
                "trunc_in_bits",
                "trunc_out_scale",
                "trunc_out_bits",
            ],
            ["outp"],
            domain="qonnx.custom_op.general",
            rounding_mode=rounding_mode,
            signed=int(signed),
            narrow=0,
        ),
    ]
    graph = helper.make_graph(nodes, "avgpool_trunc", [inp], [outp])
    model = qonnx_make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("qonnx.custom_op.general", 2),
        ],
    )
    model = ModelWrapper(model)
    for name, value in [
        ("act_scale", act_scale),
        ("zeropt", np.float32(0)),
        ("act_bits", np.float32(act_bits)),
        ("trunc_scale", trunc_scale),
        ("trunc_in_bits", np.float32(trunc_in_bits)),
        ("trunc_out_scale", trunc_out_scale),
        ("trunc_out_bits", np.float32(trunc_out_bits)),
    ]:
        model.set_initializer(name, np.asarray(value, dtype=np.float32))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model, act_scale, trunc_out_scale


def make_all_window_sums(kernel, act_bits, signed):
    """Integer activations whose kernel x kernel windows cover every possible sum."""
    n = kernel**2
    lo, hi = (-(2 ** (act_bits - 1)), 2 ** (act_bits - 1) - 1) if signed else (0, 2**act_bits - 1)
    sums = np.arange(n * lo, n * hi + 1)
    # split each sum into n values that differ by at most one
    base = np.floor_divide(sums, n)
    rem = sums - n * base
    windows = base[:, None] + (np.arange(n)[None, :] < rem[:, None])
    # (n_windows, kernel, kernel) -> (1, 1, kernel, kernel * n_windows)
    windows = windows.reshape(-1, kernel, kernel).transpose(1, 0, 2)
    return windows.reshape(1, 1, kernel, -1).astype(np.float32), len(sums)


@pytest.mark.transform
@pytest.mark.parametrize("rounding_mode", ["FLOOR", "ROUND", "CEIL"])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("kernel", [2, 3, 4])
@pytest.mark.parametrize("act_bits", [3, 8])
# number of bits the Trunc drops from the window sum
@pytest.mark.parametrize("dropped_bits", [0, 1, 3])
def test_avg_pool_trunc_rounding(rounding_mode, signed, kernel, act_bits, dropped_bits):
    trunc_out_bits = get_trunc_in_bits(act_bits, kernel) - dropped_bits
    act_int, n_windows = make_all_window_sums(kernel, act_bits, signed)
    model, act_scale, out_scale = make_avgpool_trunc_model(
        kernel, act_bits, signed, trunc_out_bits, rounding_mode, n_windows
    )
    idict = {"inp": act_int * act_scale}
    expected = oxe.execute_onnx(model, idict)["outp"]

    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    assert len(model.get_nodes_by_op_type("Trunc")) == 0
    pool = model.get_nodes_by_op_type("QuantAvgPool2d")
    assert len(pool) == 1
    # FLOOR is the right shift of the QuantAvgPool2d, other rounding modes are
    # applied by a MultiThreshold on the window sum if the Trunc drops bits. The
    # MultiThreshold is also used if the Trunc clamps, which happens for signed
    # inputs whose window sum exceeds the Trunc range.
    if signed:
        sum_min = -(2 ** (act_bits - 1)) * kernel**2
        sum_max = (2 ** (act_bits - 1) - 1) * kernel**2
        out_min, out_max = -(2 ** (trunc_out_bits - 1)), 2 ** (trunc_out_bits - 1) - 1
    else:
        sum_min, sum_max = 0, (2**act_bits - 1) * kernel**2
        out_min, out_max = 0, 2**trunc_out_bits - 1
    clamps = (sum_min >> dropped_bits) < out_min or (sum_max >> dropped_bits) > out_max
    pool_consumer = model.find_consumer(pool[0].output[0])
    if clamps or (rounding_mode != "FLOOR" and dropped_bits > 0):
        assert pool_consumer.op_type == "MultiThreshold"
    else:
        assert pool_consumer.op_type == "Mul"

    produced = oxe.execute_onnx(model, idict)[model.graph.output[0].name]
    # Both outputs are level * out_scale, compare the integer levels exactly
    # instead of floats with a tolerance
    assert np.array_equal(np.round(produced / out_scale), np.round(expected / out_scale))
