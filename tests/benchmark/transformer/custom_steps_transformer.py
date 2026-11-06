# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import numpy as np
import qonnx.core.data_layout as DataLayout
import qonnx.transformation.infer_data_layouts as infer_data_layouts
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.remove import remove_node_and_rewire
from qonnx.util.basic import get_by_name
from qonnx.util.cleanup import cleanup_model

from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_streamline,
    step_tidy_up,
    verify_step,
)
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferGlobalAccPoolLayer,
)
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MoveMulPastJoinAdd,
    MoveMulPastJoinConcat,
    MoveOpPastFork,
    MoveScalarLinearPastInvariants,
    MoveScalarLinearPastSplit,
    MoveScalarMulPastMatMul,
)


@contextlib.contextmanager
def rank_checked_data_layouts():
    """Temporarily make qonnx's InferDataLayouts rank-aware.

    InferDataLayouts propagates the input layout of standard ONNX ops unchanged, also
    through rank-changing ops such as Reshape (e.g. a 4D NHWC annotation on a 3D
    token tensor). Subsequent Transposes permute this into meaningless layouts and
    the MultiThreshold channel axes end up wrong (RadioML/vision transformers).
    Within this context, a propagated layout is only kept if its rank matches the
    output tensor, otherwise the rank-based default is used.
    TODO: remove once fixed in qonnx (_dims_to_layout)."""
    original = infer_data_layouts._dims_to_layout

    def _dims_to_layout(model, node, ndims):
        layout = original(model, node, ndims)
        if layout and len(layout) != ndims:
            return {4: DataLayout.NHWC, 3: DataLayout.NWC, 2: DataLayout.NC}.get(
                ndims, DataLayout.UNKNOWN
            )
        return layout

    infer_data_layouts._dims_to_layout = _dims_to_layout
    try:
        yield
    finally:
        infer_data_layouts._dims_to_layout = original


class RemoveShapeIdentityOps(Transformation):
    """Remove shape/data-movement ops that leave their input unchanged: Reshape to
    the same shape (also after collapsing consecutive Reshapes), Concat of a single
    input, Slice over the full range with unit steps and 1x1 pooling. The transformer
    exports contain these around the (empty) token prepend/drop, the encoder blocks
    and the patch embedding, and they would otherwise block the HW conversion."""

    def apply(self, model):
        for node in model.graph.node:
            ishape = model.get_tensor_shape(node.input[0])
            oshape = model.get_tensor_shape(node.output[0])
            if node.op_type == "Reshape":
                producer = model.find_producer(node.input[0])
                if (
                    producer is not None
                    and producer.op_type == "Reshape"
                    and len(model.find_consumers(producer.output[0])) == 1
                ):
                    # Reshape -> Reshape: only the last target shape matters
                    node.input[0] = producer.input[0]
                    model.graph.node.remove(producer)
                    return model.transform(InferShapes()), True
                is_identity = ishape is not None and ishape == oshape
            elif node.op_type == "Concat":
                is_identity = len(node.input) == 1
            elif node.op_type == "Slice":
                steps = model.get_initializer(node.input[4]) if len(node.input) > 4 else None
                unit_steps = steps is None or bool(np.all(steps == 1))
                is_identity = unit_steps and ishape is not None and ishape == oshape
            elif node.op_type in ["AveragePool", "MaxPool"]:
                # 1x1 pooling with unit stride and no padding
                kernel = get_by_name(node.attribute, "kernel_shape")
                strides = get_by_name(node.attribute, "strides")
                pads = get_by_name(node.attribute, "pads")
                is_identity = (
                    kernel is not None
                    and all(k == 1 for k in kernel.ints)
                    and (strides is None or all(s == 1 for s in strides.ints))
                    and (pads is None or all(p == 0 for p in pads.ints))
                )
            else:
                continue
            if is_identity:
                remove_node_and_rewire(model, node)
                return model.transform(InferShapes()), True
        return model, False


class MoveChannelPreservingReshapeUp(Transformation):
    """Move a Reshape that keeps the last (channel) dimension upwards past
    MultiThresholds and Mul/Add with a constant operand broadcasting to the reshaped
    shape. The image->token reshape after the patch embedding (RadioML/vision) thus
    moves up to the embedding MatMul, so the elementwise stem ops (including the
    positional encoding) already operate on tokens and the remaining reshape sits
    between layers that tolerate it (see RemoveCNVtoFCFlatten)."""

    def apply(self, model):
        graph = model.graph
        for node in graph.node:
            if node.op_type != "Reshape":
                continue
            ishape = model.get_tensor_shape(node.input[0])
            oshape = model.get_tensor_shape(node.output[0])
            if ishape is None or oshape is None or ishape[-1] != oshape[-1]:
                continue
            producer = model.find_producer(node.input[0])
            if producer is None or producer.op_type not in ["MultiThreshold", "Mul", "Add"]:
                continue
            if len(model.find_consumers(producer.output[0])) != 1:
                continue
            if model.get_initializer(producer.input[0]) is not None:
                continue
            param = model.get_initializer(producer.input[1])
            if param is None:
                continue
            if producer.op_type in ["Mul", "Add"]:
                try:
                    broadcastable = np.broadcast_shapes(param.shape, tuple(oshape)) == tuple(oshape)
                except ValueError:
                    broadcastable = False
                if not broadcastable:
                    continue
            # rewire x -> producer -> y -> Reshape -> z  into  x -> Reshape -> y -> producer -> z
            x, y, z = producer.input[0], producer.output[0], node.output[0]
            node.input[0], node.output[0] = x, y
            producer.input[0], producer.output[0] = y, z
            model.set_tensor_shape(y, oshape)
            model.set_tensor_datatype(y, model.get_tensor_datatype(x))
            if producer.op_type == "MultiThreshold":
                # channels are on the last axis for any rank
                layout = {2: "NC", 3: "NWC", 4: "NHWC"}[len(oshape)]
                getCustomOp(producer).set_nodeattr("data_layout", layout)
            # keep the graph topologically sorted: Reshape now precedes its producer
            graph.node.remove(node)
            graph.node.insert(list(graph.node).index(producer), node)
            return model.transform(InferShapes()), True
        return model, False


def step_transformer_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Fold the export-time shape arithmetic and drop the shape no-ops before the
    # QONNX->FINN conversion.
    model = cleanup_model(model)
    model = model.transform(RemoveShapeIdentityOps())
    global_in = model.graph.input[0].name
    # A 4D input that is immediately transposed NHWC->NCHW (RadioML: 1x1x1024x2 I/Q
    # samples) is channels-last: annotate it, otherwise layout inference assumes
    # NCHW and all downstream layouts (and MultiThreshold channel axes) are wrong.
    consumers = model.find_consumers(global_in) or []
    if len(model.get_tensor_shape(global_in)) == 4 and any(
        n.op_type == "Transpose" and list(get_by_name(n.attribute, "perm").ints) == [0, 3, 1, 2]
        for n in consumers
    ):
        model.set_tensor_layout(global_in, DataLayout.NHWC)
    # Token inputs (language model) index an embedding table: annotate them as
    # unsigned integers covering the vocabulary, so the Gather becomes a Lookup layer
    # instead of being left on the host. The table is still produced by a Quant node
    # here, so take the vocabulary size from its shape.
    for node in model.graph.node:
        if node.op_type == "Gather" and node.input[1] == global_in:
            vocab_size = model.get_tensor_shape(node.input[0])[0]
            vocab_bits = max(1, int(np.ceil(np.log2(vocab_size))))
            model.set_tensor_datatype(global_in, DataType[f"UINT{vocab_bits}"])
    return model


def step_transformer_prepare(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Stock QONNX->FINN conversion and tidy-up (phase_prepare_model), with
    # rank-checked layout inference so the MultiThresholds get the right channel axis.
    with rank_checked_data_layouts():
        model = step_qonnx_to_finn(model, cfg)
        model = step_tidy_up(model, cfg)
    return model


def step_transformer_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Standard streamlining leaves scalar scales in front of the multi-head
    # attention: past the fork into Q/K/V, the head Split, the head-merge Concat and
    # the residual Add. Move them through so every MatMul/Split/Concat sees
    # quantized integer inputs. Only the scales feeding the Softmax remain (they
    # cannot be moved past it) and become ElementwiseMul layers.
    with rank_checked_data_layouts():
        model = step_streamline(model, cfg)
        for _ in range(3):
            for trn in [
                # Mean-pooling classifier head (RadioML/vision): once its input is
                # integer, turn the GlobalAveragePool into a GlobalAccPool and a
                # scalar 1/N Mul, which is then streamlined into the classifier
                # (fresh layouts: the conversion transposes NCHW inputs)
                InferDataLayouts(),
                InferGlobalAccPoolLayer(),
                MoveOpPastFork(["Mul"]),
                MoveScalarLinearPastSplit(),
                MoveMulPastJoinConcat(),
                MoveMulPastJoinAdd(),
                MoveScalarMulPastMatMul(),
                MoveScalarLinearPastInvariants(),
                Streamline(),
            ]:
                model = model.transform(trn)
        model = model.transform(MoveChannelPreservingReshapeUp())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model
