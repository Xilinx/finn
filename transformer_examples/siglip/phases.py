# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""SigLIP-specific overrides for the phase-based FINN pipeline."""

from __future__ import annotations

import numpy as np
import warnings
from copy import deepcopy
from json import load
from onnx import AttributeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.remove import RemoveIdentityOps, RemoveUnusedNodes
from qonnx.util.cleanup import cleanup_model

from finn.builder.build_dataflow_config import VerificationStepType
from finn.builder.build_dataflow_phases import phase_optimize_model
from finn.builder.build_dataflow_steps import step_tidy_up, verify_step
from finn.transformation.general import ApplyConfig
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias


class _DuplicateSafeModelWrapper(ModelWrapper):
    """Rename every occurrence when a tensor is used twice by one node.

    QONNX ``ModelWrapper.rename_tensor`` currently updates only the first
    matching input and output on each node. SigLIP's exact GELU contains
    ``Mul(x, x)`` nodes, so ordinary readable-name cleanup otherwise leaves a
    dangling copy of the old tensor name.
    """

    def rename_tensor(self, old_name: str, new_name: str) -> None:
        super().rename_tensor(old_name, new_name)
        for node in self.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[index] = new_name
            for index, output_name in enumerate(node.output):
                if output_name == old_name:
                    node.output[index] = new_name

    def transform(self, transformation, *args, **kwargs):
        model = super().transform(transformation, *args, **kwargs)
        if not isinstance(model, _DuplicateSafeModelWrapper):
            model = _DuplicateSafeModelWrapper(model.model, fix_missing_initializer_valueinfo=False)
        return model


def _select_graph_output(model: ModelWrapper, output_name: str) -> ModelWrapper:
    matches = [value_info for value_info in model.graph.output if value_info.name == output_name]
    if len(matches) != 1:
        available = [value_info.name for value_info in model.graph.output]
        raise ValueError(f"Expected output {output_name!r}; available outputs are {available}")
    selected = deepcopy(matches[0])
    model.graph.ClearField("output")
    model.graph.output.append(selected)
    return model.transform(RemoveUnusedNodes())


def _siglip_quant_filter(max_bit_width: int):
    """Convert static QV activation quantizers within the configured width."""

    def filter_function(model, quant_node):
        if quant_node.op_type == "BipolarQuant":
            return True
        bit_width = model.get_initializer(quant_node.input[3])
        if bit_width is None:
            raise ValueError("Quant nodes must have a static bit width")
        if float(np.asarray(bit_width).reshape(-1)[0]) > max_bit_width:
            return False
        return model.get_initializer(quant_node.input[2]) is not None

    return filter_function


def phase_prepare_siglip(model, cfg):
    """Prepare QV-LSQ QONNX and convert eligible activation quantizers."""

    if not isinstance(model, _DuplicateSafeModelWrapper):
        model = _DuplicateSafeModelWrapper(model.model, fix_missing_initializer_valueinfo=False)
    model = _select_graph_output(model, "image_embeds")

    qonnx_count = sum(
        len(model.get_nodes_by_op_type(op_type)) for op_type in ("BinaryQuant", "Quant", "Trunc")
    )
    if qonnx_count:
        model = cleanup_model(model)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The Quant node with name: .* was not converted to a MultiThreshold.*",
            )
            model = model.transform(
                ConvertQONNXtoFINN(
                    filter_function=_siglip_quant_filter(cfg.max_multithreshold_bit_width)
                )
            )
        model = _fold_quantized_matmul_into_multithreshold(model)
        if VerificationStepType.QONNX_TO_FINN_PYTHON in cfg._resolve_verification_steps():
            verify_step(model, cfg, "finn_onnx_python", need_parent=False)
    return step_tidy_up(model, cfg)


def _extract_layernorm_affine(model: ModelWrapper) -> ModelWrapper:
    model = model.transform(ExtractNormScaleBias(), cleanup=False)
    model = model.transform(SortGraph())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    return model.transform(InferDataTypes())


def _tensor_datatype(model: ModelWrapper, tensor_name: str):
    try:
        return model.get_tensor_datatype(tensor_name)
    except Exception:
        return None


def _scalar_initializer(model: ModelWrapper, tensor_name: str):
    value = model.get_initializer(tensor_name)
    if value is None or value.size != 1:
        return None
    return value.reshape(-1)[0]


def _constant_binary_operand(model: ModelWrapper, node, dynamic_input: str):
    """Return the constant operand of a binary node with one expected input."""

    if len(node.input) != 2 or dynamic_input not in node.input:
        return None
    other_index = 1 - list(node.input).index(dynamic_input)
    return model.get_initializer(node.input[other_index])


def _integer_code_datatype(model: ModelWrapper, tensor_name: str, producers):
    dtype = _tensor_datatype(model, tensor_name)
    if dtype is not None and dtype.is_integer():
        return dtype

    # Immediately after QONNX-to-FINN conversion, the Add which changes a
    # MultiThreshold count into a signed code is still annotated FLOAT32.
    # Prove its integer range from the operator parameters instead.
    add = producers.get(tensor_name)
    if add is None or add.op_type != "Add":
        return None
    dynamic_inputs = [name for name in add.input if model.get_initializer(name) is None]
    constant_inputs = [name for name in add.input if model.get_initializer(name) is not None]
    if len(dynamic_inputs) != 1 or len(constant_inputs) != 1:
        return None
    add_value = model.get_initializer(constant_inputs[0])
    if add_value.size != 1 or not np.equal(add_value, np.round(add_value)).all():
        return None
    upstream = producers.get(dynamic_inputs[0])
    if upstream is None or upstream.op_type != "MultiThreshold":
        return None
    upstream_thresholds = model.get_initializer(upstream.input[1])
    upstream_inst = getCustomOp(upstream)
    upstream_scale = upstream_inst.get_nodeattr("out_scale")
    upstream_bias = upstream_inst.get_nodeattr("out_bias")
    if (
        upstream_thresholds is None
        or int(upstream_scale) != upstream_scale
        or upstream_scale <= 0
        or int(upstream_bias) != upstream_bias
    ):
        return None
    low = int(upstream_bias + add_value.reshape(-1)[0])
    high = int(low + upstream_scale * upstream_thresholds.shape[1])
    for candidate in DataType.get_accumulator_dt_cands():
        candidate_dtype = DataType[candidate]
        if candidate_dtype.allowed(low) and candidate_dtype.allowed(high):
            return candidate_dtype
    return None


def _fold_quantized_matmul_into_multithreshold(model: ModelWrapper) -> ModelWrapper:
    """Fold a quantized SigLIP projection into integer accumulator thresholds.

    Match quantized projection paths produced by QONNX-to-FINN conversion::

        Mul(act, act_scale) -> MatMul(weights) -> Mul(weight_scale) -> Add(bias)
          -> [Reshape/Transpose] -> [Mul(head_scale)] -> MultiThreshold

    The rewritten path performs the MatMul on the integer activation and weight
    codes, then thresholds that accumulator before the shape-only operations.
    Only full-range signed ROUND/HALF_EVEN quantizers are accepted.
    """

    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    for node in model.graph.node:
        for tensor_name in node.input:
            consumers.setdefault(tensor_name, []).append(node)

    def only_consumer(tensor_name, op_type):
        matches = consumers.get(tensor_name, [])
        if len(matches) == 1 and matches[0].op_type == op_type:
            return matches[0]
        return None

    modified = False
    for matmul in list(model.get_nodes_by_op_type("MatMul")):
        weights = model.get_initializer(matmul.input[1])
        weight_dtype = _tensor_datatype(model, matmul.input[1])
        if (
            weights is None
            or weights.ndim < 2
            or weight_dtype is None
            or not weight_dtype.is_integer()
            or not np.array_equal(weights, np.round(weights))
        ):
            continue

        activation_mul = producers.get(matmul.input[0])
        if activation_mul is None or activation_mul.op_type != "Mul":
            continue
        activation_input = None
        activation_dtype = None
        activation_scale = None
        for input_name in activation_mul.input:
            input_dtype = _integer_code_datatype(model, input_name, producers)
            scale = _constant_binary_operand(model, activation_mul, input_name)
            if input_dtype is not None and input_dtype.is_integer() and scale is not None:
                activation_input = input_name
                activation_dtype = input_dtype
                activation_scale = np.asarray(scale)
                break
        if activation_input is None or activation_scale.size != 1:
            continue

        weight_mul = only_consumer(matmul.output[0], "Mul")
        if weight_mul is None:
            continue
        weight_scale = _constant_binary_operand(model, weight_mul, matmul.output[0])
        if weight_scale is None:
            continue

        bias_add = only_consumer(weight_mul.output[0], "Add")
        if bias_add is None:
            continue
        bias = _constant_binary_operand(model, bias_add, weight_mul.output[0])
        if bias is None:
            continue

        shape_nodes = []
        chain_output = bias_add.output[0]
        chain_consumers = consumers.get(chain_output, [])
        while len(chain_consumers) == 1 and chain_consumers[0].op_type in {
            "Reshape",
            "Transpose",
        }:
            shape_node = chain_consumers[0]
            if shape_node.op_type == "Reshape" and (
                len(shape_node.input) != 2 or model.get_initializer(shape_node.input[1]) is None
            ):
                break
            shape_nodes.append(shape_node)
            chain_output = shape_node.output[0]
            chain_consumers = consumers.get(chain_output, [])

        head_mul = None
        head_scale = np.asarray(1.0, dtype=np.float32)
        if len(chain_consumers) == 1 and chain_consumers[0].op_type == "Mul":
            head_mul = chain_consumers[0]
            head_scale = _constant_binary_operand(model, head_mul, chain_output)
            if head_scale is None or np.asarray(head_scale).size != 1:
                continue
            chain_output = head_mul.output[0]

        multithreshold = only_consumer(chain_output, "MultiThreshold")
        if multithreshold is None:
            continue
        thresholds = model.get_initializer(multithreshold.input[1])
        if thresholds is None or thresholds.ndim != 2 or thresholds.shape[0] != 1:
            continue
        mt_inst = getCustomOp(multithreshold)
        if mt_inst.get_nodeattr("out_scale") != 1.0 or mt_inst.get_nodeattr("out_bias") != 0.0:
            continue

        output_add = only_consumer(multithreshold.output[0], "Add")
        if output_add is None:
            continue
        output_bias = _constant_binary_operand(model, output_add, multithreshold.output[0])
        if output_bias is None or np.asarray(output_bias).size != 1:
            continue
        output_mul = only_consumer(output_add.output[0], "Mul")
        if output_mul is None:
            continue
        output_scale = _constant_binary_operand(model, output_mul, output_add.output[0])
        if output_scale is None or np.asarray(output_scale).size != 1:
            continue

        output_channels = weights.shape[-1]
        weight_scale = np.asarray(weight_scale, dtype=np.float64).reshape(-1)
        bias = np.asarray(bias, dtype=np.float64).reshape(-1)
        activation_scale = float(np.asarray(activation_scale, dtype=np.float64).reshape(-1)[0])
        head_scale = float(np.asarray(head_scale, dtype=np.float64).reshape(-1)[0])
        output_scale = float(np.asarray(output_scale, dtype=np.float64).reshape(-1)[0])
        output_bias = float(np.asarray(output_bias, dtype=np.float64).reshape(-1)[0])
        if weight_scale.size not in (1, output_channels) or bias.size not in (
            1,
            output_channels,
        ):
            continue
        scales = np.concatenate(
            [
                np.asarray([activation_scale, head_scale, output_scale]),
                weight_scale,
            ]
        )
        if not np.isfinite(scales).all() or not np.isfinite(bias).all() or np.any(scales <= 0):
            continue

        num_thresholds = thresholds.shape[1]
        num_levels = num_thresholds + 1
        if num_levels & (num_levels - 1) or output_bias != -(num_levels // 2):
            continue
        output_dtype = DataType[mt_inst.get_nodeattr("out_dtype")]
        if output_dtype.signed() or output_dtype.bitwidth() != num_levels.bit_length() - 1:
            continue
        lower_levels = output_bias + np.arange(num_thresholds, dtype=np.float64)
        expected_thresholds = ((lower_levels + 0.5) * output_scale).astype(np.float32)
        lower_even = (lower_levels.astype(np.int64) % 2) == 0
        expected_thresholds[lower_even] = np.nextafter(
            expected_thresholds[lower_even], np.float32(np.inf)
        )
        if not np.array_equal(thresholds, expected_thresholds.reshape(1, -1)):
            continue

        accumulator_boundaries = (
            (lower_levels.reshape(1, -1) + 0.5) * output_scale - bias.reshape(-1, 1) * head_scale
        ) / (activation_scale * weight_scale.reshape(-1, 1) * head_scale)
        # MultiThreshold uses >=. At an exact half-way boundary, HALF_EVEN
        # transitions immediately when the lower level is odd, but only above
        # the boundary when the lower level is even.
        integer_thresholds = np.ceil(accumulator_boundaries)
        integer_thresholds[:, lower_even] = np.floor(accumulator_boundaries[:, lower_even]) + 1
        float_thresholds = integer_thresholds.astype(np.float32)
        if (
            not np.isfinite(integer_thresholds).all()
            or integer_thresholds.min() < DataType["INT32"].min()
            or integer_thresholds.max() > DataType["INT32"].max()
            or not np.array_equal(float_thresholds.astype(np.float64), integer_thresholds)
        ):
            continue

        max_activation = max(abs(activation_dtype.min()), abs(activation_dtype.max()))
        max_accumulator = max_activation * np.abs(weights.astype(np.int64)).sum(axis=0).max()
        if max_accumulator > 2**24:
            continue

        matmul.input[0] = activation_input
        model.set_tensor_datatype(activation_input, activation_dtype)
        model.set_tensor_datatype(matmul.output[0], DataType["INT32"])
        model.set_initializer(multithreshold.input[1], float_thresholds)
        multithreshold.input[0] = matmul.output[0]
        mt_inst.set_nodeattr("data_layout", "NHWC")
        if shape_nodes:
            old_mt_output = multithreshold.output[0]
            accumulator_shape = model.get_tensor_shape(matmul.output[0])
            mt_output_dtype = _tensor_datatype(model, old_mt_output)
            new_mt_output = model.make_new_valueinfo_name()
            model.set_tensor_shape(new_mt_output, accumulator_shape)
            if mt_output_dtype is not None:
                model.set_tensor_datatype(new_mt_output, mt_output_dtype)
            multithreshold.output[0] = new_mt_output
            shape_nodes[0].input[0] = new_mt_output
            shape_nodes[-1].output[0] = old_mt_output
        modified = True

    if modified:
        model = model.transform(RemoveUnusedNodes())
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(SortGraph())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    return model


def _absorb_pre_matmul_dequant(model: ModelWrapper) -> ModelWrapper:
    """Move scalar activation dequantization after static-weight MatMul."""

    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    for node in model.graph.node:
        for tensor_name in node.input:
            consumers.setdefault(tensor_name, []).append(node)

    modified = False
    for matmul in model.get_nodes_by_op_type("MatMul"):
        weight_name = matmul.input[1]
        weight_dtype = _tensor_datatype(model, weight_name)
        if model.get_initializer(weight_name) is None:
            continue
        if weight_dtype is None or not weight_dtype.is_integer():
            continue

        pre_mul = producers.get(matmul.input[0])
        if pre_mul is None or pre_mul.op_type != "Mul":
            continue
        int_input = None
        pre_scale = None
        for index, input_name in enumerate(pre_mul.input):
            input_dtype = _tensor_datatype(model, input_name)
            candidate_scale = _scalar_initializer(model, pre_mul.input[1 - index])
            if input_dtype is not None and input_dtype.is_integer() and candidate_scale is not None:
                int_input = input_name
                pre_scale = candidate_scale
                break
        if int_input is None:
            continue

        post_consumers = consumers.get(matmul.output[0], [])
        if len(post_consumers) != 1 or post_consumers[0].op_type != "Mul":
            continue
        post_mul = post_consumers[0]
        post_scale_name = next(
            (
                input_name
                for input_name in post_mul.input
                if input_name != matmul.output[0] and model.get_initializer(input_name) is not None
            ),
            None,
        )
        if post_scale_name is None:
            continue

        post_scale = model.get_initializer(post_scale_name)
        model.set_initializer(post_scale_name, post_scale * pre_scale)
        matmul.input[0] = int_input
        model.set_tensor_datatype(matmul.output[0], DataType["INT32"])
        modified = True

    if modified:
        model = model.transform(RemoveUnusedNodes())
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    return model


def _integerize_static_lhs_matmul(model: ModelWrapper) -> ModelWrapper:
    """Integerize exactly quantized static attention-pool queries."""

    consumers = {}
    for node in model.graph.node:
        for tensor_name in node.input:
            consumers.setdefault(tensor_name, []).append(node)

    modified = False
    for matmul in model.get_nodes_by_op_type("MatMul"):
        lhs_name, rhs_name = matmul.input
        lhs_value = model.get_initializer(lhs_name)
        lhs_dtype = _tensor_datatype(model, lhs_name)
        rhs_dtype = _tensor_datatype(model, rhs_name)
        if lhs_value is None or model.get_initializer(rhs_name) is not None:
            continue
        if rhs_dtype is None or not rhs_dtype.is_integer():
            continue
        if lhs_dtype is not None and lhs_dtype.is_integer():
            continue

        post_consumers = consumers.get(matmul.output[0], [])
        if len(post_consumers) != 1 or post_consumers[0].op_type != "Mul":
            continue
        post_mul = post_consumers[0]
        scale_name = next(
            (
                input_name
                for input_name in post_mul.input
                if input_name != matmul.output[0]
                and _scalar_initializer(model, input_name) is not None
            ),
            None,
        )
        if scale_name is None:
            continue

        unique_values = np.sort(np.unique(lhs_value))
        max_abs = np.float32(np.max(np.abs(unique_values)))
        level_candidates = max_abs / np.arange(1, 128, dtype=np.float32)
        nonzero_candidates = np.concatenate(
            [np.abs(unique_values), np.diff(unique_values), level_candidates]
        )
        nonzero_candidates = nonzero_candidates[nonzero_candidates > 1e-12]
        if nonzero_candidates.size == 0:
            quantum = np.float32(1.0)
            integer_value = np.zeros_like(lhs_value, dtype=np.float32)
        else:
            quantum = None
            integer_value = None
            for candidate in np.sort(np.unique(nonzero_candidates)):
                candidate = np.float32(candidate)
                candidate_integer = np.round(lhs_value / candidate).astype(np.float32)
                if np.min(candidate_integer) < -128 or np.max(candidate_integer) > 127:
                    continue
                if np.array_equal(candidate_integer * candidate, lhs_value):
                    quantum = candidate
                    integer_value = candidate_integer
                    break
        if quantum is None:
            continue

        model.set_initializer(lhs_name, integer_value)
        model.set_tensor_datatype(lhs_name, DataType["INT8"])
        model.set_tensor_datatype(matmul.output[0], DataType["INT32"])
        scale = model.get_initializer(scale_name)
        model.set_initializer(scale_name, scale * quantum)
        modified = True

    if modified:
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
    return model


def _folding_config_nodes(graph, hierarchy=None):
    """Yield the config key and node pairs used by ``ApplyConfig``."""

    for node in graph.node:
        config_key = node.name if hierarchy is None else f"{hierarchy}_{node.name}"
        if node.domain.startswith("finn.custom_op.fpgadataflow"):
            yield config_key, node
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                nested_hierarchy = f"{config_key}_{attribute.name}"
                yield from _folding_config_nodes(attribute.g, nested_hierarchy)


def make_siglip_folding_step(config_path, require_complete: bool):
    """Apply the entries which exist at the current shuffle-decomposition stage."""

    def step_apply_siglip_folding(model, cfg):
        with open(config_path, encoding="utf-8") as config_file:
            full_config = load(config_file)

        configurable_nodes = dict(_folding_config_nodes(model.graph))
        configured_keys = set(full_config) - {"Defaults"}
        current_keys = set(configurable_nodes)
        if require_complete:
            missing = {
                key
                for key in current_keys - configured_keys
                if configurable_nodes[key].op_type != "FINNLoop"
            }
            unused = configured_keys - current_keys
            if missing or unused:
                raise RuntimeError(
                    "SigLIP folding profile does not match the decomposed graph: "
                    f"missing={sorted(missing)}, unused={sorted(unused)}"
                )

        stage_config = {"Defaults": full_config.get("Defaults", {})}
        for key, node in configurable_nodes.items():
            # Shuffles do not have their final names before decomposition, and
            # FINNLoop itself has no folding parameters. A no-op backend entry
            # lets ApplyConfig verify every current custom node without warnings.
            stage_config[key] = full_config.get(key, {"backend": "fpgadataflow"})

        return model.transform(
            ApplyConfig(
                stage_config,
                node_filter=lambda node: node.domain.startswith("finn.custom_op.fpgadataflow"),
            )
        )

    suffix = "final" if require_complete else "pre_decomposition"
    step_apply_siglip_folding.__name__ = f"step_apply_siglip_folding_{suffix}"
    return step_apply_siglip_folding


def phase_optimize_siglip(model, cfg):
    """Streamline SigLIP and expose its quantized MatMuls and LayerNorms."""

    model = phase_optimize_model(model, cfg)
    model = _extract_layernorm_affine(model)
    model = _absorb_pre_matmul_dequant(model)
    model = _integerize_static_lhs_matmul(model)
    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "siglip_optimized_python", need_parent=False)
    return model
