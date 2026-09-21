# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as ox
from finn.transformation.streamline.absorb import (
    AbsorbScalarBiasIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
)


def make_mt_add_model(num_ch=64, num_steps=15, out_dtype="UINT4", bias_value=-8.0):
    """Build a MultiThreshold -> Add(scalar) model.

    The MultiThreshold has out_bias=0/out_scale=1 and the given out_dtype; the
    downstream Add adds a scalar ``bias_value``, which is what
    AbsorbScalarBiasIntoMultiThreshold is meant to fold back into out_bias."""
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, num_ch])
    thres = oh.make_tensor_value_info("thres", TensorProto.FLOAT, [num_ch, num_steps])
    mt_out = oh.make_tensor_value_info("mt_out", TensorProto.FLOAT, [1, num_ch])
    bias = oh.make_tensor_value_info("bias", TensorProto.FLOAT, [1])
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, num_ch])

    mt_node = oh.make_node(
        "MultiThreshold",
        ["inp", "thres"],
        ["mt_out"],
        domain="qonnx.custom_op.general",
        out_dtype=out_dtype,
        out_scale=1.0,
        out_bias=0.0,
    )
    add_node = oh.make_node("Add", ["mt_out", "bias"], ["outp"])

    model = ModelWrapper(
        qonnx_make_model(
            oh.make_graph(
                name="mt-add",
                inputs=[inp],
                outputs=[outp],
                value_info=[thres, mt_out, bias],
                nodes=[mt_node, add_node],
            )
        )
    )
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # monotonic thresholds so the MultiThreshold is well-formed
    thres_values = np.sort(
        np.random.uniform(-4.0, 4.0, size=(num_ch, num_steps)).astype(np.float32), axis=1
    )
    model.set_initializer("thres", thres_values)
    model.set_initializer("bias", np.array([bias_value], dtype=np.float32))
    return model


@pytest.mark.streamline
# bias_value, max_bitwidth_increase, expect_absorbed, expected_out_dtype
@pytest.mark.parametrize(
    "bias_value, max_bitwidth_increase, expect_absorbed, expected_odt",
    [
        # centering/sign bias: UINT4 -> INT4, no width growth -> always absorbed
        (-8.0, 0, True, "INT4"),
        # large positive bias needs UINT5 (+1 bit): blocked at default tolerance
        (8.0, 0, False, "UINT4"),
        # same bias, but tolerance raised to allow +1 bit -> absorbed as UINT5
        (8.0, 1, True, "UINT5"),
        # straddling range [-5, 10] where the positive endpoint dominates:
        # needs signed INT5, exercises the signed-range datatype derivation
        (-5.0, 1, True, "INT5"),
    ],
)
def test_absorb_scalar_bias_into_multithreshold(
    bias_value, max_bitwidth_increase, expect_absorbed, expected_odt
):
    model = make_mt_add_model(bias_value=bias_value)
    new_model = model.transform(
        AbsorbScalarBiasIntoMultiThreshold(max_bitwidth_increase=max_bitwidth_increase)
    )

    mt_node = new_model.graph.node[0]
    assert mt_node.op_type == "MultiThreshold"
    mt_inst = getCustomOp(mt_node)

    if expect_absorbed:
        # Add node folded away, only MultiThreshold remains
        assert len(new_model.graph.node) == 1
        assert mt_inst.get_nodeattr("out_bias") == bias_value
        assert mt_inst.get_nodeattr("out_dtype") == expected_odt
        # folding the constant into out_bias must be numerically exact
        inp_dict = {"inp": np.random.uniform(-4.0, 4.0, size=(1, 64)).astype(np.float32)}
        assert ox.compare_execution(model, new_model, inp_dict)
    else:
        # absorption skipped: Add stays and MultiThreshold is untouched
        assert len(new_model.graph.node) == 2
        assert new_model.graph.node[1].op_type == "Add"
        assert mt_inst.get_nodeattr("out_bias") == 0.0
        assert mt_inst.get_nodeattr("out_dtype") == expected_odt


@pytest.mark.streamline
def test_absorb_sign_bias_alias_is_deprecated():
    """The old name must still work but emit a DeprecationWarning."""
    model = make_mt_add_model(bias_value=-8.0)
    with pytest.warns(DeprecationWarning):
        transform = AbsorbSignBiasIntoMultiThreshold()
    new_model = model.transform(transform)
    # behaves identically to the renamed transformation
    assert len(new_model.graph.node) == 1
    mt_inst = getCustomOp(new_model.graph.node[0])
    assert mt_inst.get_nodeattr("out_bias") == -8.0
    assert mt_inst.get_nodeattr("out_dtype") == "INT4"
