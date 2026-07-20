############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""
Custom build steps for the SigLIP (12-layer vision transformer, MLO) FINN build.

These follow the same self-contained ``step_xxx(model, cfg) -> model`` shape as
the BERT benchmark flow, but are trimmed to what the SigLIP-to-stitched-IP build
needs: light cleanup and a reference-IO generator for rtlsim verification.

Unlike the BERT flow, the SigLIP Conv patch-embedding head is *kept* and built
into hardware, so there is no ``step_remove_head`` here: the stock
``step_streamline`` lowers the Conv to a MatMul and the stock
``step_convert_to_hw`` infers the transformer kernels (Softmax, Gelu/PWPolyF,
LayerNorm, MVAU).
"""

import numpy as np
import os
import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
    SortCommutativeInputsInitializerLast,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps

from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.builder.build_dataflow_steps import verify_step
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
    CollapseRepeatedOp,
)
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveAddPastConv,
    MoveAddPastMul,
    MoveMulPastMaxPool,
    MoveOpPastFork,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres


def _add_missing_conv_kernel_shape(model: ModelWrapper) -> ModelWrapper:
    """Set ``kernel_shape`` on any Conv node that is missing it.

    ``kernel_shape`` is optional in the ONNX spec (inferable from the weight
    tensor), but qonnx's ``LowerConvsToMatMul`` assumes it is always present and
    crashes otherwise. The SigLIP patch-embedding Conv is exported without it, so
    we infer ``[k_h, k_w]`` from the weight's trailing spatial dims. The weight
    may be behind a Quant node, so we follow one producer hop if needed.
    """
    from onnx.helper import make_attribute
    from qonnx.util.basic import get_by_name

    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if get_by_name(node.attribute, "kernel_shape") is not None:
            continue
        weight_name = node.input[1]
        W = model.get_initializer(weight_name)
        if W is None:
            producer = model.find_producer(weight_name)
            if producer is not None and producer.op_type == "Quant":
                W = model.get_initializer(producer.input[0])
        if W is None:
            raise RuntimeError(
                f"Conv {node.name} is missing kernel_shape and its weight tensor "
                "could not be resolved to infer it."
            )
        # weight layout is [out_ch, in_ch, k_h, k_w]; take the spatial dims.
        kernel_shape = list(W.shape[2:])
        node.attribute.append(make_attribute("kernel_shape", kernel_shape))
    return model


def step_siglip_cleanup(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Basic graph cleanup / preparation.

    Sorts commutative inputs so initializers come last (so the streamlining
    absorb transforms find the parameter on input[1]), removes identity ops, and
    backfills the ``kernel_shape`` attribute the SigLIP Conv export omits.
    """
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(RemoveUnusedTensors())
    model = _add_missing_conv_kernel_shape(model)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


def step_extract_norm_scale_bias(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Extract LayerNorm learned scale/bias into trailing Mul/Add nodes.

    SigLIP's LayerNormalization nodes carry learned gamma/beta on inputs[1]/[2].
    ``InferLayerNorm`` only converts a LayerNorm whose scale is all-ones and bias
    is zero, so without this the LayerNorms are silently skipped in
    ``step_convert_to_hw``. This transform resets scale->1 / bias->0 on the
    LayerNorm and emits the learned params as a following Mul / Add, which
    streamlining then absorbs into neighbouring thresholds. In the BERT flow this
    ran inside ``step_remove_head``; we keep the Conv head, so it lives here.
    """
    model = model.transform(ExtractNormScaleBias())
    return model


class _StreamlinePreserveThreshShape(Transformation):
    """Copy of stock ``finn.transformation.streamline.Streamline`` with
    ``preserve_thresh_shape`` threaded into the two MultiThreshold absorb passes.

    The bundled ``Streamline`` hardcodes ``AbsorbAddIntoMultiThreshold()`` /
    ``AbsorbMulIntoMultiThreshold()`` with the default ``preserve_thresh_shape=
    False``, and exposes no hook to override it. With ``False``, a channelwise
    Add/Mul (e.g. the per-channel LayerNorm gamma/beta emitted by
    ``ExtractNormScaleBias``) gets folded into a *per-tensor* threshold, expanding
    it from ``(1, steps)`` to ``(C, steps)``. For SigLIP's ``[N, tokens, C]``
    activations that per-channel threshold then trips MultiThreshold's NCHW
    channel-axis assumption at execution (axis 1 = tokens, not channels), which is
    the ``streamlined_python`` verify crash. Passing ``True`` refuses that fold so
    thresholds stay per-tensor (global), matching the BERT flow.

    The transform list is otherwise identical to stock ``Streamline`` -- keep it in
    sync if the upstream list changes.
    """

    def __init__(self, preserve_thresh_shape=True):
        super().__init__()
        self.preserve_thresh_shape = preserve_thresh_shape

    def apply(self, model):
        p = self.preserve_thresh_shape
        streamline_transformations = [
            ConvertSubToAdd(),
            ConvertDivToMul(),
            BatchNormToAffine(),
            ConvertSignToThres(),
            MoveMulPastMaxPool(),
            AbsorbSignBiasIntoMultiThreshold(),
            MoveScalarLinearPastInvariants(),
            MoveAddPastMul(),
            MoveScalarAddPastMatMul(),
            MoveAddPastConv(),
            MoveScalarMulPastMatMul(),
            MoveScalarMulPastConv(),
            MoveAddPastMul(),
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),
            MoveMulPastMaxPool(),
            AbsorbAddIntoMultiThreshold(preserve_thresh_shape=p),
            FactorOutMulSignMagnitude(),
            AbsorbMulIntoMultiThreshold(preserve_thresh_shape=p),
            Absorb1BitMulIntoMatMul(),
            Absorb1BitMulIntoConv(),
        ]
        for trn in streamline_transformations:
            model = model.transform(trn)
            model = model.transform(RemoveIdentityOps())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(InferDataTypes())
        return (model, False)


def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """SigLIP replacement for stock ``step_streamline``.

    Identical to ``finn.builder.build_dataflow_steps.step_streamline`` (Conv-head
    lowering, bipolar->xnor, TopK absorb, layout inference, and the
    ``streamlined_python`` verify) except the streamlining pass is
    ``_StreamlinePreserveThreshShape(preserve_thresh_shape=True)`` instead of the
    stock ``Streamline()``. This keeps channelwise LayerNorm scale/bias out of the
    per-tensor thresholds so the transformer MultiThresholds stay executable (see
    ``_StreamlinePreserveThreshShape``).
    """
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(_StreamlinePreserveThreshShape(preserve_thresh_shape=True))
    need_lowering = len(model.get_nodes_by_op_type("Conv")) > 0
    if need_lowering:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(_StreamlinePreserveThreshShape(preserve_thresh_shape=True))
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model


def step_siglip_streamlining(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """SigLIP-specific streamlining, run after the stock ``step_streamline``.

    Stock streamlining leaves a scalar Mul stranded on the fork that feeds the
    per-layer q/k/v projection MatMuls (the SoftMax/attention scaling factor).
    Because that Mul is a *fork* node, neither ``MoveScalarMulPastMatMul`` nor
    ``AbsorbMulIntoMultiThreshold`` will touch it, so the MatMul inputs stay
    FLOAT32 and ``InferQuantizedMatrixVectorActivation`` skips them -- leaving
    unconverted MatMuls that split the graph into multiple dataflow partitions.

    Mirroring the BERT flow's ``step_bert_streamlining``: replicate the Mul down
    each fork branch (``MoveOpPastFork``), then move each per-branch Mul past its
    MatMul and absorb the resulting scale/shift into the neighbouring thresholds
    so the MatMul inputs become integer again.
    """
    model = model.transform(AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(AbsorbAddIntoMultiThreshold())
    model = model.transform(AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())

    # Push the scalar Mul across the q/k/v fork so each branch owns a copy.
    model = model.transform(MoveOpPastFork(["Mul"]))

    model = model.transform(MoveScalarMulPastMatMul())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(AbsorbAddIntoMultiThreshold())
    model = model.transform(AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())

    model = model.transform(CollapseRepeatedOp("Mul", lambda x, y: y * x))

    model = model.transform(GiveUniqueNodeNames())
    return model


def _absorb_signed_ln_scale(model: ModelWrapper):
    """Absorb a signed per-channel LayerNorm scale (gamma in {-1,+1}) sitting in
    a ``Mul -> MultiThreshold -> MatMul -> [Mul(scale)] -> Add(bias)`` chain.

    Background (see LOOP_ROLLING_SPEEDBUMP.md): two SigLIP LayerNorms have a
    single negative scale channel, which streamlining cannot absorb (only
    positive muls fold into thresholds). The stranded signed Mul makes those two
    encoder layers structurally differ from the other ten, so loop rolling's
    "all layers identical" assertion fails.

    Since |gamma| == 1 on the affected channels (pure sign), for each negative
    channel ``c`` the Mul is removed exactly by:
      1. threshold row flip:  T'[c] = -T[c][::-1]   (reverse-count for negated input)
      2. weight row negate:   W'[c, :] = -W[c, :]
      3. bias compensation:   bias'[j] += scale[j] * W[c, j] * (n_steps + 2*out_bias)

    The reverse threshold turns the per-channel count ``k`` into ``n_steps - k``,
    and the MultiThreshold carries an ``out_bias``; combining both, the flipped
    channel's contribution to the MatMul input shifts by ``W[c,j]*(n_steps +
    2*out_bias)``, which the downstream Add bias absorbs (scaled by the Mul
    scale). ``n_steps`` is the number of thresholds per channel (``T.shape[1]``).

    Returns the number of Mul nodes absorbed.
    """
    absorbed = 0
    for mul in list(model.graph.node):
        if mul.op_type != "Mul" or model.is_fork_node(mul) or model.is_join_node(mul):
            continue
        gamma = model.get_initializer(mul.input[1])
        if gamma is None:
            continue
        gamma = np.asarray(gamma).reshape(-1)
        # only handle the pure-sign per-channel case (magnitude already absorbed)
        if not (gamma.ndim == 1 and gamma.size > 1 and (gamma < 0).any()):
            continue
        if not np.all(np.isin(gamma, (-1.0, 1.0))):
            continue
        neg = np.where(gamma < 0)[0]

        mt = model.find_consumer(mul.output[0])
        if mt is None or mt.op_type != "MultiThreshold":
            continue
        mm = model.find_consumer(mt.output[0])
        if mm is None or mm.op_type != "MatMul" or mt.output[0] != mm.input[0]:
            continue

        T = model.get_initializer(mt.input[1])  # [C, steps]
        W = model.get_initializer(mm.input[1])  # [C_in, C_out]
        if T is None or W is None:
            continue
        if T.shape[0] != gamma.size or W.shape[0] != gamma.size:
            continue

        # MultiThreshold output bias; the reverse-count constant depends on it.
        out_bias = 0.0
        for a in mt.attribute:
            if a.name == "out_bias":
                out_bias = a.f
        n_steps = T.shape[1]
        factor = n_steps + 2.0 * out_bias

        # downstream scale (Mul) and bias (Add), used for constant compensation
        scale_node = model.find_consumer(mm.output[0])
        scale = None
        bias_node = None
        if scale_node is not None and scale_node.op_type == "Mul":
            scale = model.get_initializer(scale_node.input[1])
            bias_node = model.find_consumer(scale_node.output[0])
        else:
            bias_node = scale_node
        if bias_node is None or bias_node.op_type != "Add":
            continue
        bias = model.get_initializer(bias_node.input[1])
        if bias is None:
            continue

        s = np.ones(W.shape[1], dtype=np.float64) if scale is None else np.asarray(scale).reshape(-1)

        T_new = T.copy().astype(np.float64)
        W_new = W.copy().astype(np.float64)
        bias_new = np.asarray(bias).reshape(-1).astype(np.float64).copy()
        for c in neg:
            T_new[c] = -T[c][::-1]
            bias_new += s * W[c, :] * factor  # absorb reverse-count + out_bias constant
            W_new[c, :] = -W[c, :]

        # Equivalence check on the local MultiThreshold->MatMul->[Mul]->Add chain.
        # Full-graph execute_onnx can't validate this: MultiThreshold assumes the
        # channel axis is 1 (NCHW) but the transformer activations are [N, tokens,
        # channels], so we reproduce the chain in numpy over random channel inputs.
        _verify_local_chain(
            gamma, T, W, s, np.asarray(bias).reshape(-1).astype(np.float64),
            out_bias, T_new, W_new, bias_new, neg, mt.name,
        )

        model.set_initializer(mt.input[1], T_new.astype(T.dtype))
        model.set_initializer(mm.input[1], W_new.astype(W.dtype))
        model.set_initializer(bias_node.input[1], bias_new.reshape(bias.shape).astype(bias.dtype))

        # remove the signed Mul: wire its input straight into the MultiThreshold
        mt.input[0] = mul.input[0]
        model.graph.node.remove(mul)
        absorbed += 1
    return model, absorbed


def _multithreshold_count(v, T, out_bias):
    """Per-channel MultiThreshold: count(v[c] >= T[c]) + out_bias. v,T over channels."""
    out = np.empty(v.shape[0], dtype=np.float64)
    for c in range(v.shape[0]):
        out[c] = np.count_nonzero(v[c] >= T[c]) + out_bias
    return out


def _verify_local_chain(gamma, T, W, s, bias, out_bias, T_new, W_new, bias_new, neg, name):
    """Assert original vs rewritten chain agree over random channel-vector inputs.

    original:  (MT(gamma * x) )        @ W  * s + bias
    rewritten: (MT_new(x) )            @ W_new * s + bias_new
    """
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(8):
        x = rng.uniform(-4.0, 4.0, size=gamma.size)
        y_orig = s * (_multithreshold_count(gamma * x, T, out_bias) @ W) + bias
        y_new = s * (_multithreshold_count(x, T_new, out_bias) @ W_new) + bias_new
        max_err = max(max_err, float(np.abs(y_orig - y_new).max()))
    assert max_err < 1e-6, (
        f"Signed LN-scale absorption at {name} is not equivalent "
        f"(max abs err {max_err:.3e}); threshold/weight/bias compensation is wrong."
    )


def step_absorb_signed_ln_scale(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Make all encoder layers structurally identical for loop rolling by
    absorbing the stranded signed LayerNorm-scale Muls (see
    ``_absorb_signed_ln_scale`` and LOOP_ROLLING_SPEEDBUMP.md).

    Self-verifies equivalence per rewritten chain: ``_absorb_signed_ln_scale``
    reproduces each ``MultiThreshold -> MatMul -> [Mul] -> Add`` chain in numpy
    before and after the edit and asserts they match, so a mis-derived
    compensation fails loudly at build time. (A full-graph ``execute_onnx`` can't
    be used here: MultiThreshold assumes an NCHW channel axis, but the
    transformer activations are ``[N, tokens, channels]``.)
    """
    model, absorbed = _absorb_signed_ln_scale(model)
    print(
        f"step_absorb_signed_ln_scale: absorbed {absorbed} signed LayerNorm-scale "
        "Mul(s) (per-chain equivalence verified)"
    )
    model = model.transform(GiveUniqueNodeNames())
    return model


def step_generate_reference_io(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Generate the golden reference IO by executing the current graph.

    Placed BEFORE ``step_qonnx_to_finn`` (BERT-style) so the reference is computed
    on the raw QONNX graph, independent of ConvertQONNXtoFINN. This lets
    ``QONNX_TO_FINN_PYTHON`` verification measure the Quant->MultiThreshold
    conversion fidelity, and every later step (qonnx_to_finn, streamline,
    convert_to_hw, specialize, loop_rolling, ...) is checked against the true
    QONNX output. All those steps are equivalence-preserving, so the reference
    stays valid down to the hardware.

    Saves ``input.npy`` and ``expected_output.npy`` (feeding ``verify_input_npy``
    / ``verify_expected_output_npy``). NOTE: must run before ``step_streamline``
    -- the post-streamline graph has per-channel MultiThresholds that trip the
    executor's NCHW channel-axis assumption and can't be executed.
    """
    input_m = model.graph.input[0]
    in_shape = [d.dim_value for d in input_m.type.tensor_type.shape.dim]
    out_name = model.graph.output[0].name

    # pixel_values are normalised image tensors; a small uniform range keeps the
    # reference execution numerically sane for verification.
    in_tensor = np.random.uniform(-1.0, 1.0, size=in_shape).astype(np.float32)

    # The persisted input MUST be exactly the array we execute on. execute_onnx
    # can mutate its input dict / arrays in place (especially with
    # return_full_exec_context=True), so we snapshot the input to disk and hand
    # the executor an independent copy. Otherwise input.npy and
    # expected_output.npy desync and every downstream verify_step compares
    # against an unreproducible reference and always FAILs.
    np.save(os.path.join(cfg.output_dir, "input.npy"), in_tensor)
    exec_ctx = oxe.execute_onnx(
        model, {input_m.name: in_tensor.copy()}, return_full_exec_context=True
    )
    np.save(os.path.join(cfg.output_dir, "expected_output.npy"), exec_ctx[out_name])
    np.savez(os.path.join(cfg.output_dir, "expected_context.npz"), **exec_ctx)

    # Self-check: re-execute on the persisted input and confirm it reproduces the
    # saved expected output. This makes a reference desync fail loudly here rather
    # than silently downstream in verify_step.
    check = oxe.execute_onnx(model, {input_m.name: in_tensor.copy()})[out_name]
    max_err = float(np.abs(check - exec_ctx[out_name]).max())
    assert max_err < 1e-4, (
        f"Reference IO is not self-consistent (max abs err {max_err:.3e}); "
        "input.npy does not reproduce expected_output.npy."
    )
    return model
