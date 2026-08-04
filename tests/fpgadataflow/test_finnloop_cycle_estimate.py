# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from onnx import TensorProto, helper
from qonnx.custom_op.registry import getCustomOp


def test_finnloop_cycle_estimate_models_pipeline_overlap():
    tensor_in = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1])
    tensor_mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1])
    tensor_out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    body_nodes = [
        helper.make_node(
            "StreamingFIFO_rtl",
            ["inp"],
            ["mid"],
            name="fifo0",
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            cycles_estimate=100,
        ),
        helper.make_node(
            "StreamingFIFO_rtl",
            ["mid"],
            ["out"],
            name="fifo1",
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            cycles_estimate=30,
        ),
    ]
    loop_body = helper.make_graph(
        body_nodes,
        "loop_body",
        [tensor_in],
        [tensor_out],
        value_info=[tensor_mid],
    )
    loop_node = helper.make_node(
        "FINNLoop",
        ["top_in"],
        ["top_out"],
        name="loop",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        body=loop_body,
        iteration=4,
        inputDataType="FLOAT32",
        outputDataType="FLOAT32",
    )

    # Fill latency is 100 + 30, steady-state interval is max(100, 30), and
    # loop-control overhead is 40 cycles per iteration.
    assert getCustomOp(loop_node).get_exp_cycles() == 130 + 3 * 100 + 4 * 40
