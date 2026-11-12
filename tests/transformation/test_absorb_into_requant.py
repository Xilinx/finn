# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.absorb_into_requant import (
    AbsorbElementwiseOpsIntoRequant,
)


def make_add_quant_model(add_shape, ishape=(1, 4, 8)):
    """Build an Add(const) -> Quant (8-bit, unit scale) model, which InferRequantLayer
    turns into ElementwiseAdd -> Requant."""
    inp = oh.make_tensor_value_info("in0", TensorProto.FLOAT, ishape)
    outp = oh.make_tensor_value_info("out0", TensorProto.FLOAT, ishape)
    add_node = oh.make_node("Add", ["in0", "add_param"], ["add_out"])
    quant_node = oh.make_node(
        "Quant",
        ["add_out", "scale", "zeropt", "bitwidth"],
        ["out0"],
        domain="qonnx.custom_op.general",
        signed=1,
        narrow=0,
        rounding_mode="ROUND",
    )
    model = ModelWrapper(
        qonnx_make_model(
            oh.make_graph([add_node, quant_node], "add-quant", [inp], [outp]),
        )
    )
    model.set_initializer("add_param", gen_finn_dt_tensor(DataType["INT6"], add_shape))
    model.set_initializer("scale", np.array(1.0, dtype=np.float32))
    model.set_initializer("zeropt", np.array(0.0, dtype=np.float32))
    model.set_initializer("bitwidth", np.array(8.0, dtype=np.float32))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


@pytest.mark.transform
@pytest.mark.parametrize(
    "add_shape,absorbed",
    [
        # per-channel (last axis) constants are absorbed into the Requant bias
        ((8,), True),
        ((1, 1, 8), True),
        # per-element constants (e.g. a positional encoding) are not supported by
        # Requant and must stay a separate ElementwiseAdd
        ((1, 4, 8), False),
    ],
)
def test_absorb_into_requant_channelwise_only(add_shape, absorbed):
    model = make_add_quant_model(add_shape)
    # INT9 inputs exceed the INT8 output range, so clipping is covered as well
    inp = {"in0": gen_finn_dt_tensor(DataType["INT9"], (1, 4, 8))}
    y_golden = oxe.execute_onnx(model, inp)["out0"]

    model = model.transform(to_hw.InferRequantLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    assert len(model.get_nodes_by_op_type("ElementwiseAdd")) == 1
    model = model.transform(AbsorbElementwiseOpsIntoRequant())

    assert len(model.get_nodes_by_op_type("Requant")) == 1
    assert len(model.get_nodes_by_op_type("ElementwiseAdd")) == (0 if absorbed else 1)
    y = oxe.execute_onnx(model, inp)[model.graph.output[0].name]
    assert np.allclose(y_golden, y)
