# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames, SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveTransposePastEltwise

# ResNet-50-like cases: after LowerConvsToMatMul the residual path is littered
# with NCHW<->NHWC transposes feeding the per-channel bias/scale ops between
# convolutions. Each case is (input feature-map shape, transpose perm, channel
# axis of the transpose *output*), covering both transpose directions.
NETWORK_CASES = [
    # NCHW -> NHWC (channel moves to last axis)
    ([1, 64, 56, 56], [0, 2, 3, 1], 3),
    ([1, 256, 14, 14], [0, 2, 3, 1], 3),
    # NHWC -> NCHW (channel moves back to axis 1)
    ([1, 14, 14, 256], [0, 3, 1, 2], 1),
]


def make_const_shape(out_shape, ch_axis, const_mode):
    if const_mode == "scalar":
        # a single broadcast scalar (e.g. a global requant scale)
        return [1]
    if const_mode == "per_channel":
        # a per-channel affine vector broadcasting on the channel axis
        shape = [1] * len(out_shape)
        shape[ch_axis] = out_shape[ch_axis]
        return shape
    # full per-element constant
    return list(out_shape)


@pytest.mark.streamline
@pytest.mark.parametrize("in_shape, perm, ch_axis", NETWORK_CASES)
@pytest.mark.parametrize("op_type", ["Add", "Mul"])
@pytest.mark.parametrize("const_mode", ["scalar", "per_channel", "full"])
def test_move_transpose_past_eltwise(in_shape, perm, ch_axis, op_type, const_mode):
    out_shape = [in_shape[p] for p in perm]
    const_shape = make_const_shape(out_shape, ch_axis, const_mode)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)
    a0 = helper.make_tensor_value_info("a0", TensorProto.FLOAT, const_shape)

    transp_node = helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm)
    eltwise_node = helper.make_node(op_type, ["transp_out", "a0"], ["outp"])

    graph = helper.make_graph(
        nodes=[transp_node, eltwise_node],
        name="mv-transpose-eltwise-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[a0],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise"))

    # the addend/scale must be a constant initializer for the transform to fire
    model.set_initializer("a0", gen_finn_dt_tensor(DataType["FLOAT32"], const_shape))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    idict = {model.get_first_global_in(): gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)}

    model_transformed = model.transform(MoveTransposePastEltwise())
    # the transform only rewires tensors; re-sort so the node list is topological
    model_transformed = model_transformed.transform(SortGraph())

    # functional equivalence: moving the transpose (and transposing the constant
    # by the inverse permutation) must not change the numerical result
    assert oxe.compare_execution(model, model_transformed, idict)

    # order swapped: the elementwise op now runs first, the transpose second
    assert model_transformed.graph.node[0].op_type == op_type
    assert model_transformed.graph.node[1].op_type == "Transpose"
    # the elementwise op now consumes the pre-transpose input directly
    assert model_transformed.graph.node[0].input[0] == model.get_first_global_in()
    # the transpose now consumes the elementwise output
    assert model_transformed.graph.node[1].input[0] == model_transformed.graph.node[0].output[0]


@pytest.mark.streamline
@pytest.mark.parametrize("op_type", ["Add", "Mul"])
def test_move_transpose_past_eltwise_dynamic_noop(op_type):
    # when the other elementwise input is dynamic (no constant initializer), the
    # transform must not fire -- this is the residual-join case that is handled by
    # the dedicated MoveTransposePastJoinAdd pass instead.
    in_shape = [1, 64, 56, 56]
    perm = [0, 2, 3, 1]
    out_shape = [in_shape[p] for p in perm]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    other = helper.make_tensor_value_info("other", TensorProto.FLOAT, out_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    transp_node = helper.make_node("Transpose", ["inp"], ["transp_out"], perm=perm)
    eltwise_node = helper.make_node(op_type, ["transp_out", "other"], ["outp"])

    graph = helper.make_graph(
        nodes=[transp_node, eltwise_node],
        name="mv-transpose-eltwise-noop",
        inputs=[inp, other],
        outputs=[outp],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mv_transpose_eltwise_noop"))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    model_transformed = model.transform(MoveTransposePastEltwise())

    # unchanged: transpose still first, elementwise second
    assert model_transformed.graph.node[0].op_type == "Transpose"
    assert model_transformed.graph.node[1].op_type == op_type
