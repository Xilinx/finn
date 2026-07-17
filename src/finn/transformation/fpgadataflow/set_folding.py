# Copyright (C) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import functools
import inspect
import itertools
import json
import math
import numpy as np
import random
import warnings
from dataclasses import dataclass
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames

import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as elementwise_binary_hls
from finn.analysis.fpgadataflow.dataflow_performance import (
    FINNLOOP_ITERATION_OVERHEAD_CYCLES,
    accumulate_frame_boundary_cycles,
    dataflow_performance,
    node_frame_boundary_cycles,
    node_throughput_cycles,
)
from finn.analysis.fpgadataflow.res_estimation import resource_attr_variants
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.basic import part_map
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.platforms import platforms


def divisors(num):
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


def first_divisor_at_least(num, lower_bound):
    for divisor in divisors(num):
        if divisor >= lower_bound:
            return divisor
    return num


def common_divisors(numbers):
    separate_divisors = []
    for num in numbers:
        individual_divisors = list(divisors(num))
        separate_divisors.append(individual_divisors)

    return functools.reduce(np.intersect1d, separate_divisors)


# HLS elementwise operations use PE as their folding attribute.
ELEMENTWISE_BINARY_OPS = [
    op_type
    for op_type, cls in inspect.getmembers(elementwise_binary_hls, inspect.isclass)
    if issubclass(cls, elementwise_binary_hls.ElementwiseBinaryOperation_hls)
]


RESOURCE_KEYS = ("LUT", "BRAM_18K", "URAM", "DSP")
# Fallback normalization when a device capacity is unavailable.
DEFAULT_RESOURCE_SCALE = {
    "LUT": 100000.0,
    "BRAM_18K": 1000.0,
    "URAM": 100.0,
    "DSP": 1000.0,
}
BOARD_ALIASES = {
    "Ultra96-V2": "Ultra96",
}
BOARD_RESOURCE_CAPACITY = {
    # Vivado 2024.2 xcvc1902-vsva2197 device resources. FINN counts each
    # 36-Kb block RAM as two BRAM_18K resources.
    "VCK190": {
        "LUT": 899840.0,
        "FF": 1799680.0,
        "BRAM_18K": 1934.0,
        "URAM": 463.0,
        "DSP": 1968.0,
    },
}
PART_RESOURCE_CAPACITY = {
    "xcvc1902-vsva2197-2MP-e-S": BOARD_RESOURCE_CAPACITY["VCK190"],
}


@dataclass(frozen=True)
class FoldingKnob:
    scope_path: tuple
    node_name: str
    attr_name: str
    values: tuple
    kind: str
    initial_index: int = 0


def aggregate_resources(resource_dict):
    total = {key: 0.0 for key in RESOURCE_KEYS}
    for node_resources in resource_dict.values():
        for key in RESOURCE_KEYS:
            total[key] += float(node_resources.get(key, 0.0))
    return total


def capacity_from_board(board):
    if board is None:
        return None
    board = BOARD_ALIASES.get(board, board)
    if board in BOARD_RESOURCE_CAPACITY:
        return dict(BOARD_RESOURCE_CAPACITY[board])
    if board not in platforms:
        return None
    platform = platforms[board]()
    resources = np.asarray(platform.compute_resources, dtype=np.float64)
    resources = np.sum(resources, axis=0)
    return {
        "LUT": float(resources[0]),
        "FF": float(resources[1]),
        "BRAM_18K": float(resources[2]),
        "URAM": float(resources[3]),
        "DSP": float(resources[4]),
    }


def capacity_from_part(fpgapart):
    if fpgapart is None:
        return None
    if fpgapart in PART_RESOURCE_CAPACITY:
        return dict(PART_RESOURCE_CAPACITY[fpgapart])
    for board, part in part_map.items():
        if part == fpgapart:
            capacity = capacity_from_board(board)
            if capacity is not None:
                return capacity
    return None


def resource_capacity(board, fpgapart):
    capacity = capacity_from_board(board)
    if capacity is not None:
        return capacity
    return capacity_from_part(fpgapart)


def attr_values_with_current(node_inst, attr_name, values, kind, include_current=True):
    values = list(values)
    try:
        current = node_inst.get_nodeattr(attr_name)
    except AttributeError:
        current = values[0] if values else None

    if kind == "parallelism":
        values = sorted(set(int(value) for value in values if int(value) > 0))
        if include_current and current is not None and int(current) not in values:
            values.append(int(current))
            values = sorted(set(values))
        initial_index = 0
    else:
        values = list(dict.fromkeys(values))
        if include_current and current is not None and current not in values:
            values.insert(0, current)
        initial_index = values.index(current) if current in values else 0

    return tuple(values), initial_index


class ResourceAwareFoldingPPO:
    """PPO search over FINN folding and resource attributes.

    Deterministic refinement keeps compiler results stable. Designs that meet
    the cycle target minimize estimated resources; otherwise the search chooses
    the fastest configuration that fits the device.
    """

    pe_ops = [
        "DuplicateStreams_hls",
        "GlobalAccPool_hls",
        "PWPolyF_rtl",
        "ElementwiseAdd_rtl",
        "ElementwiseMul_rtl",
        "Thresholding_hls",
        "Thresholding_rtl",
        *ELEMENTWISE_BINARY_OPS,
    ]
    simd_ops = [
        "AddCLSToken_rtl",
        "FMPadding_rtl",
        "FMPadding_Pixel_hls",
        "ConvolutionInputGenerator_rtl",
        "StreamingSplit_hls",
        "StreamingConcat_hls",
        "HWSoftmax_hls",
        "HWSoftmax_rtl",
        "InnerShuffle_rtl",
        "LayerNorm_rtl",
        "OuterShuffle_hls",
        "SelectToken_rtl",
        "Shuffle",
    ]
    depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]

    def __init__(
        self,
        model,
        target_cycles_per_frame,
        fpgapart,
        board=None,
        mvau_wwidth_max=36,
        resource_limit=1.0,
        episodes=6,
        rollout_steps=32,
        update_epochs=4,
        gamma=0.92,
        gae_lambda=0.85,
        clip_ratio=0.2,
        policy_lr=0.05,
        value_lr=0.05,
        entropy_coef=0.01,
        seed=0,
    ):
        # Knobs address nodes by name, so the root and loop bodies need unique names.
        self.model = model.transform(GiveUniqueNodeNames(), make_deepcopy=False, cleanup=False)
        self.target_cycles_per_frame = max(1, int(target_cycles_per_frame))
        self.fpgapart = fpgapart
        self.board = board
        self.mvau_wwidth_max = mvau_wwidth_max
        self.resource_limit = float(resource_limit)
        self.episodes = int(episodes)
        self.rollout_steps = int(rollout_steps)
        self.update_epochs = int(update_epochs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.policy_lr = float(policy_lr)
        self.value_lr = float(value_lr)
        self.entropy_coef = float(entropy_coef)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.py_rng = random.Random(self.seed)
        self.capacity = resource_capacity(board, fpgapart)
        self.scopes = {(): self.model}
        self._collect_scopes((), self.model)
        self.knobs = self._collect_knobs()
        self.initial_indices = tuple(knob.initial_index for knob in self.knobs)
        self.eval_cache = {}
        self.node_cycle_cache = {}
        self.node_resource_cache = {}
        self.target_feasibility = None

    def _collect_scopes(self, scope_path, model):
        for node in model.graph.node:
            if node.op_type != "FINNLoop":
                continue
            body = getCustomOp(node).get_nodeattr("body")
            body = body.transform(GiveUniqueNodeNames(), make_deepcopy=False, cleanup=False)
            body_path = scope_path + (node.name,)
            self.scopes[body_path] = body
            self._collect_scopes(body_path, body)

    def _node_by_name(self, scope_path, node_name):
        for node in self.scopes[scope_path].graph.node:
            if node.name == node_name:
                return node
        raise KeyError((scope_path, node_name))

    def _add_knob(
        self,
        knobs,
        scope_path,
        node,
        attr_name,
        values,
        kind="parallelism",
        include_current=True,
    ):
        if not values:
            return
        node_inst = getCustomOp(node)
        values, initial_index = attr_values_with_current(
            node_inst, attr_name, values, kind, include_current=include_current
        )
        if len(values) <= 1:
            if len(values) == 1:
                node_inst.set_nodeattr(attr_name, values[0])
            return
        knobs.append(FoldingKnob(scope_path, node.name, attr_name, values, kind, initial_index))

    def _mvau_simd_values(self, node_inst, max_simd):
        is_hls = node_inst.onnx_node.op_type == "MVAU_hls"
        ret = []
        min_simd = 1
        if is_hls:
            # HLS MVAU codegen requires the matrix width per SIMD group to
            # stay within the hlslib static array bounds.
            min_simd = first_divisor_at_least(max_simd, int(math.ceil(max_simd / 1024)))
        legal_values = []
        for simd_val in divisors(max_simd):
            if simd_val < min_simd:
                continue
            legal_values.append(simd_val)
            if not is_hls:
                ret.append(simd_val)
                continue
            try:
                weight_width = node_inst.get_input_datatype(1).bitwidth() * int(simd_val)
            except Exception:
                weight_width = 0
            if weight_width <= self.mvau_wwidth_max:
                ret.append(simd_val)
        return ret or legal_values or [max_simd]

    def _collect_knobs(self):
        knobs = []
        for scope_path, model in self.scopes.items():
            for node in model.graph.node:
                if not (is_hls_node(node) or is_rtl_node(node)) or node.op_type == "FINNLoop":
                    continue
                op_type = node.op_type
                node_inst = getCustomOp(node)
                if op_type in ["MVAU_hls", "MVAU_rtl"]:
                    max_simd = node_inst.get_nodeattr("MW")
                    max_pe = node_inst.get_nodeattr("MH")
                    self._add_knob(
                        knobs,
                        scope_path,
                        node,
                        "SIMD",
                        self._mvau_simd_values(node_inst, max_simd),
                        include_current=False,
                    )
                    self._add_knob(knobs, scope_path, node, "PE", divisors(max_pe))
                    if op_type == "MVAU_rtl":
                        self._add_knob(
                            knobs,
                            scope_path,
                            node,
                            "pumpedCompute",
                            [0, 1],
                            kind="resource",
                        )
                elif op_type in self.pe_ops:
                    try:
                        max_pe = node_inst.get_nodeattr("NumChannels")
                    except AttributeError:
                        max_pe = node_inst.get_normal_input_shape()[-1]
                    self._add_knob(knobs, scope_path, node, "PE", divisors(max_pe))
                elif op_type == "LabelSelect_hls":
                    self._add_knob(
                        knobs,
                        scope_path,
                        node,
                        "PE",
                        divisors(node_inst.get_nodeattr("Labels")),
                    )
                elif op_type in self.depthwise_op_exceptions:
                    max_pe = node_inst.get_nodeattr("Channels")
                    self._add_knob(knobs, scope_path, node, "PE", divisors(max_pe))
                    if op_type in ["VVAU_hls", "VVAU_rtl"]:
                        max_simd = int(np.prod(node_inst.get_nodeattr("Kernel")))
                        self._add_knob(knobs, scope_path, node, "SIMD", divisors(max_simd))
                elif op_type in self.simd_ops:
                    if op_type.startswith("ConvolutionInputGenerator"):
                        depthwise = node_inst.get_nodeattr("depthwise")
                        if depthwise == 0:
                            max_simd = node_inst.get_nodeattr("IFMChannels")
                            self._add_knob(knobs, scope_path, node, "SIMD", divisors(max_simd))
                            if op_type == "ConvolutionInputGenerator_rtl":
                                self._add_knob(knobs, scope_path, node, "parallel_window", [0, 1])
                    elif op_type == "StreamingConcat_hls" or op_type == "StreamingSplit_hls":
                        channels_per_stream = node_inst.get_nodeattr("ChannelsPerStream")
                        self._add_knob(
                            knobs,
                            scope_path,
                            node,
                            "SIMD",
                            common_divisors(channels_per_stream),
                        )
                    elif op_type == "LayerNorm_rtl":
                        dim = int(node_inst.get_normal_input_shape()[-1])
                        values = []
                        for simd_val in divisors(dim):
                            if dim // simd_val > 12:
                                values.append(simd_val)
                            else:
                                break
                        self._add_knob(knobs, scope_path, node, "SIMD", values)
                    elif op_type == "InnerShuffle_rtl":
                        # InnerShuffle swaps the final two dimensions; its RTL
                        # constructor requires SIMD to divide the penultimate
                        # input dimension (the final output dimension).
                        max_simd = int(node_inst.get_nodeattr("in_shape")[-2])
                        self._add_knob(knobs, scope_path, node, "SIMD", divisors(max_simd))
                    elif op_type.startswith("HWSoftmax"):
                        max_simd = int(node_inst.get_normal_input_shape()[-1])
                        self._add_knob(knobs, scope_path, node, "SIMD", divisors(max_simd))
                    elif op_type == "OuterShuffle_hls":
                        # OuterShuffle vectorizes the internal transpose dimension.
                        # Normal input divisors can include SIMD values rejected by
                        # its constructor, and code generation divides loop bounds by SIMD.
                        simd_dimensions = [
                            int(node_inst.get_normal_input_shape()[-1]),
                            int(node_inst.get_normal_output_shape()[-1]),
                            int(node_inst.get_nodeattr("transpose_in_shape")[-1]),
                            int(node_inst.get_nodeattr("transpose_out_shape")[-1]),
                        ]
                        simd_dimensions.extend(
                            int(value)
                            for value in node_inst.get_nodeattr("loop_coeffs")
                            if int(value) != 1
                        )
                        max_simd = functools.reduce(math.gcd, simd_dimensions)
                        self._add_knob(knobs, scope_path, node, "SIMD", divisors(max_simd))
                    elif op_type == "Shuffle":
                        input_width = int(node_inst.get_normal_input_shape()[-1])
                        output_width = int(node_inst.get_normal_output_shape()[-1])
                        max_simd = math.gcd(input_width, output_width)
                        self._add_knob(knobs, scope_path, node, "SIMD", divisors(max_simd))
                    else:
                        max_simd = node_inst.get_nodeattr("NumChannels")
                        self._add_knob(knobs, scope_path, node, "SIMD", divisors(max_simd))

                for attr_name, values in resource_attr_variants(node_inst):
                    # RTL MVAU code generation currently rejects LUT compute,
                    # even though the inherited resType schema advertises it.
                    if op_type == "MVAU_rtl" and attr_name == "resType":
                        continue
                    self._add_knob(knobs, scope_path, node, attr_name, values, kind="resource")

        return knobs

    def _indices_from_model(self):
        indices = []
        for knob in self.knobs:
            node_inst = getCustomOp(self._node_by_name(knob.scope_path, knob.node_name))
            value = node_inst.get_nodeattr(knob.attr_name)
            if value in knob.values:
                indices.append(knob.values.index(value))
            elif knob.kind == "parallelism":
                value = int(value)
                index = min(range(len(knob.values)), key=lambda ind: abs(knob.values[ind] - value))
                indices.append(index)
            else:
                indices.append(knob.initial_index)
        return tuple(indices)

    def _set_indices(self, indices):
        for knob, index in zip(self.knobs, indices):
            node_inst = getCustomOp(self._node_by_name(knob.scope_path, knob.node_name))
            node_inst.set_nodeattr(knob.attr_name, knob.values[int(index)])
        for model in self.scopes.values():
            self._repair_dependencies(model)
        return self._indices_from_model()

    def _repair_dependencies(self, model):
        for node in model.graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)
            if op_type == "MVAU_rtl":
                # Pumping preserves stream rate with half as many physical SIMD lanes.
                # Enabling it as SIMD grows avoids inefficient high-PE folds.
                pumped_compute = 1 if node_inst.get_nodeattr("SIMD") > 1 else 0
                node_inst.set_nodeattr("pumpedCompute", pumped_compute)
            if op_type in self.depthwise_op_exceptions:
                pe = node_inst.get_nodeattr("PE")
                simd = 1
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    max_pe = node_inst.get_nodeattr("Channels")
                    if pe < max_pe:
                        node_inst.set_nodeattr("SIMD", 1)
                    simd = node_inst.get_nodeattr("SIMD")
                swu_node = model.find_producer(node.input[0])
                is_swu_node = swu_node is not None and swu_node.op_type.startswith(
                    "ConvolutionInputGenerator"
                )
                if is_swu_node:
                    swu_node_inst = getCustomOp(swu_node)
                    swu_node_inst.set_nodeattr("SIMD", pe)
                    if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                        parallel_window = 1 if op_type.startswith("VVAU") and simd > 1 else 0
                        swu_node_inst.set_nodeattr("parallel_window", parallel_window)
            elif op_type == "ConvolutionInputGenerator_rtl":
                try:
                    depthwise = node_inst.get_nodeattr("depthwise")
                    simd = node_inst.get_nodeattr("SIMD")
                    max_simd = node_inst.get_nodeattr("IFMChannels")
                    if depthwise == 0 and simd < max_simd:
                        node_inst.set_nodeattr("parallel_window", 0)
                except AttributeError:
                    pass

    def _resource_norms(self, resources):
        norms = {}
        for key in RESOURCE_KEYS:
            usage = float(resources.get(key, 0.0))
            if self.capacity is None:
                denom = DEFAULT_RESOURCE_SCALE[key]
            else:
                denom = float(self.capacity.get(key, 0.0))
            if denom <= 0.0:
                norms[key] = 0.0 if usage <= 0.0 else float("inf")
            else:
                norms[key] = usage / denom
        return norms

    def _scope_throughput_performance(self, scope_path=()):
        max_cycles = 0
        max_node_name = ""
        cycle_score = 0
        model = self.scopes[scope_path]
        boundary_state_at_output = {}
        for node in model.graph.node:
            if node.op_type == "FINNLoop":
                node_inst = getCustomOp(node)
                body_perf = self._scope_throughput_performance(scope_path + (node.name,))
                iteration = max(1, int(node_inst.get_nodeattr("iteration")))
                node_cycles = iteration * (
                    body_perf["max_cycles"] + FINNLOOP_ITERATION_OVERHEAD_CYCLES
                )
                cycle_score += iteration * body_perf["cycle_score"]
                path_boundary_state = (0, False)
            elif is_hls_node(node) or is_rtl_node(node) or node.op_type == "Shuffle":
                node_inst = getCustomOp(node)
                attr_values = []
                for attr_name in ["PE", "SIMD", "parallel_window", "pumpedCompute"]:
                    try:
                        value = node_inst.get_nodeattr(attr_name)
                    except AttributeError:
                        continue
                    attr_values.append((attr_name, value))
                cache_key = (scope_path, node.name, tuple(attr_values))
                if cache_key not in self.node_cycle_cache:
                    self.node_cycle_cache[cache_key] = (
                        node_throughput_cycles(node_inst),
                        node_frame_boundary_cycles(node_inst),
                    )
                throughput_cycles, boundary_cycles = self.node_cycle_cache[cache_key]
                predecessor_states = [
                    boundary_state_at_output.get(input_name, (0, False))
                    for input_name in node.input
                ]
                candidate_boundary_states = [
                    accumulate_frame_boundary_cycles(state, boundary_cycles)
                    for state in predecessor_states
                ] or [accumulate_frame_boundary_cycles((0, False), boundary_cycles)]
                path_boundary_state = max(candidate_boundary_states, key=lambda state: state[0])
                path_boundary_cycles = path_boundary_state[0]
                node_cycles = throughput_cycles + path_boundary_cycles
            else:
                continue
            for output_name in node.output:
                boundary_state_at_output[output_name] = path_boundary_state
            cycle_score += node_cycles
            if node_cycles > max_cycles:
                max_cycles = node_cycles
                max_node_name = "/".join(scope_path + (node.name,))
        return {
            "max_cycles": max_cycles,
            "max_cycles_node_name": max_node_name,
            "cycle_score": cycle_score,
        }

    def _all_resources(self):
        report = {}
        for scope_path, model in self.scopes.items():
            for node in model.graph.node:
                if not (is_hls_node(node) or is_rtl_node(node)):
                    continue
                name = "/".join(scope_path + (node.name,))
                node_inst = getCustomOp(node)
                attr_values = []
                for attr_name in [
                    "PE",
                    "SIMD",
                    "parallel_window",
                    "pumpedCompute",
                    "ram_style",
                    "resType",
                ]:
                    try:
                        value = node_inst.get_nodeattr(attr_name)
                    except AttributeError:
                        continue
                    if isinstance(value, list):
                        value = tuple(value)
                    attr_values.append((attr_name, value))
                cache_key = (scope_path, node.name, tuple(attr_values))
                if cache_key not in self.node_resource_cache:
                    self.node_resource_cache[cache_key] = node_inst.node_res_estimation(
                        self.fpgapart
                    )
                report[name] = self.node_resource_cache[cache_key]
        return aggregate_resources(report)

    def _annotate_and_commit_scopes(self):
        for scope_path in sorted(self.scopes, key=len, reverse=True):
            model = self.scopes[scope_path]
            model = model.transform(AnnotateCycles(), make_deepcopy=False, cleanup=False)
            self.scopes[scope_path] = model
            if scope_path:
                parent_path = scope_path[:-1]
                loop_node = self._node_by_name(parent_path, scope_path[-1])
                getCustomOp(loop_node).set_nodeattr("body", model.graph)
        self.model = self.scopes[()]

    def _evaluate_indices(self, indices):
        indices = self._set_indices(indices)
        cached = self.eval_cache.get(indices)
        if cached is not None:
            return cached

        try:
            perf = self._scope_throughput_performance()
            resources = self._all_resources()
        except Exception as error:
            evaluation = {
                "indices": indices,
                "valid": False,
                "meets_target": False,
                "max_cycles": float("inf"),
                "cycle_score": float("inf"),
                "resources": {key: float("inf") for key in RESOURCE_KEYS},
                "resource_norms": {key: float("inf") for key in RESOURCE_KEYS},
                "resource_score": float("inf"),
                "resource_pressure": float("inf"),
                "quality": -float("inf"),
                "error": str(error),
            }
            self.eval_cache[indices] = evaluation
            return evaluation

        max_cycles = max(1, int(perf["max_cycles"]))
        norms = self._resource_norms(resources)
        resource_pressure = max(norms.values()) if norms else 0.0
        resource_score = sum(value for value in norms.values() if np.isfinite(value))
        valid = resource_pressure <= self.resource_limit + 1.0e-9
        meets_target = max_cycles <= self.target_cycles_per_frame
        evaluation = {
            "indices": indices,
            "valid": valid,
            "meets_target": meets_target,
            "max_cycles": max_cycles,
            "cycle_score": perf["cycle_score"],
            "resources": resources,
            "resource_norms": norms,
            "resource_score": resource_score,
            "resource_pressure": resource_pressure,
        }
        evaluation["quality"] = self._quality(evaluation)
        self.eval_cache[indices] = evaluation
        return evaluation

    def _quality(self, evaluation):
        max_cycles = evaluation["max_cycles"]
        if not np.isfinite(max_cycles):
            max_cycles = self.target_cycles_per_frame * 100.0
        if not evaluation["valid"]:
            pressure = evaluation["resource_pressure"]
            overuse = max(0.0, pressure - self.resource_limit) if np.isfinite(pressure) else 10.0
            return -1000.0 - (1000.0 * overuse) - math.log1p(max_cycles)
        if evaluation["meets_target"]:
            slack = max(0.0, self.target_cycles_per_frame - max_cycles)
            slack /= max(self.target_cycles_per_frame, 1)
            return 1000.0 - (10.0 * evaluation["resource_score"]) - (0.01 * slack)
        target_ratio = self.target_cycles_per_frame / max(max_cycles, 1)
        return (100.0 * target_ratio) - (0.1 * evaluation["resource_score"])

    def _is_better(self, candidate, reference):
        if reference is None:
            return True
        if candidate["valid"] != reference["valid"]:
            return candidate["valid"]
        if candidate["valid"]:
            if candidate["meets_target"] != reference["meets_target"]:
                return candidate["meets_target"]
            if candidate["meets_target"]:
                if candidate["resource_score"] < reference["resource_score"] - 1.0e-9:
                    return True
                if candidate["resource_score"] > reference["resource_score"] + 1.0e-9:
                    return False
                candidate_slack = abs(self.target_cycles_per_frame - candidate["max_cycles"])
                reference_slack = abs(self.target_cycles_per_frame - reference["max_cycles"])
                return candidate_slack < reference_slack
            if candidate["max_cycles"] != reference["max_cycles"]:
                return candidate["max_cycles"] < reference["max_cycles"]
            if candidate["cycle_score"] != reference["cycle_score"]:
                return candidate["cycle_score"] < reference["cycle_score"]
            return candidate["resource_score"] < reference["resource_score"] - 1.0e-9
        if candidate["resource_pressure"] != reference["resource_pressure"]:
            return candidate["resource_pressure"] < reference["resource_pressure"]
        return candidate["max_cycles"] < reference["max_cycles"]

    def _state_vector(self, evaluation):
        max_cycles = evaluation["max_cycles"]
        if not np.isfinite(max_cycles):
            max_cycles = self.target_cycles_per_frame * 100.0
        state = [
            1.0,
            self.target_cycles_per_frame / max(max_cycles, 1.0),
            max_cycles / max(self.target_cycles_per_frame, 1.0),
            1.0 if evaluation["meets_target"] else 0.0,
            1.0 if evaluation["valid"] else 0.0,
            min(evaluation["resource_score"], 10.0) / 10.0,
            min(evaluation["resource_pressure"], 10.0) / 10.0,
            min(evaluation["cycle_score"] / max(self.target_cycles_per_frame, 1), 100.0) / 100.0,
        ]
        for knob, index in zip(self.knobs, evaluation["indices"]):
            denom = max(len(knob.values) - 1, 1)
            state.append(float(index) / denom)
        return np.asarray(state, dtype=np.float64)

    def _valid_action_mask(self, indices):
        mask = np.zeros(2 * len(self.knobs), dtype=bool)
        for knob_index, (knob, index) in enumerate(zip(self.knobs, indices)):
            if index < len(knob.values) - 1:
                mask[2 * knob_index] = True
            if index > 0:
                mask[2 * knob_index + 1] = True
        if len(mask) and not np.any(mask):
            mask[:] = True
        return mask

    def _apply_action_to_indices(self, indices, action):
        if len(indices) == 0:
            return tuple(indices), False
        knob_index = int(action) // 2
        direction = 1 if int(action) % 2 == 0 else -1
        indices = list(indices)
        old_index = indices[knob_index]
        new_index = old_index + direction
        new_index = max(0, min(new_index, len(self.knobs[knob_index].values) - 1))
        indices[knob_index] = new_index
        if new_index == old_index:
            return tuple(indices), False
        repaired_indices = self._set_indices(tuple(indices))
        return repaired_indices, repaired_indices != tuple(indices) or new_index != old_index

    def _softmax(self, logits, mask):
        logits = np.asarray(logits, dtype=np.float64)
        logits = np.nan_to_num(logits, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        masked_logits = np.full_like(logits, -1.0e9)
        masked_logits[mask] = logits[mask]
        masked_logits -= np.max(masked_logits)
        probs = np.exp(masked_logits)
        probs *= mask
        total = np.sum(probs)
        if total <= 0.0:
            return mask.astype(np.float64) / max(np.sum(mask), 1)
        return probs / total

    def _sample_action(self, state, mask, policy_w, policy_b, value_w, value_b):
        probs = self._softmax(state @ policy_w + policy_b, mask)
        action = int(self.rng.choice(np.arange(len(probs)), p=probs))
        logp = math.log(max(probs[action], 1.0e-12))
        value = float(state @ value_w + value_b)
        return action, logp, value, probs

    def _compute_advantages(self, rewards, values):
        advantages = np.zeros(len(rewards), dtype=np.float64)
        gae = 0.0
        next_value = 0.0
        for index in reversed(range(len(rewards))):
            delta = rewards[index] + self.gamma * next_value - values[index]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[index] = gae
            next_value = values[index]
        returns = advantages + values
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1.0e-8)
        return advantages, returns

    def _ppo_update(self, rollout, policy_w, policy_b, value_w, value_b):
        if not rollout:
            return policy_w, policy_b, value_w, value_b
        states = np.asarray([item["state"] for item in rollout], dtype=np.float64)
        actions = np.asarray([item["action"] for item in rollout], dtype=np.int64)
        old_logps = np.asarray([item["logp"] for item in rollout], dtype=np.float64)
        values = np.asarray([item["value"] for item in rollout], dtype=np.float64)
        rewards = np.asarray([item["reward"] for item in rollout], dtype=np.float64)
        masks = np.asarray([item["mask"] for item in rollout], dtype=bool)

        advantages, returns = self._compute_advantages(rewards, values)
        for _ in range(self.update_epochs):
            policy_grad_w = np.zeros_like(policy_w)
            policy_grad_b = np.zeros_like(policy_b)
            value_grad_w = np.zeros_like(value_w)
            value_grad_b = 0.0

            for state, action, old_logp, advantage, ret, mask in zip(
                states, actions, old_logps, advantages, returns, masks
            ):
                probs = self._softmax(state @ policy_w + policy_b, mask)
                logp = math.log(max(probs[action], 1.0e-12))
                ratio = math.exp(logp - old_logp)
                use_grad = True
                if advantage >= 0.0 and ratio > 1.0 + self.clip_ratio:
                    use_grad = False
                if advantage < 0.0 and ratio < 1.0 - self.clip_ratio:
                    use_grad = False

                if use_grad:
                    grad_logits = -probs
                    grad_logits[action] += 1.0
                    grad_logits *= ratio * advantage
                    if self.entropy_coef:
                        uniform = mask.astype(np.float64) / max(np.sum(mask), 1)
                        grad_logits += self.entropy_coef * (uniform - probs)
                    policy_grad_w += np.outer(state, grad_logits)
                    policy_grad_b += grad_logits

                prediction = float(state @ value_w + value_b)
                value_error = ret - prediction
                value_grad_w += value_error * state
                value_grad_b += value_error

            scale = 1.0 / len(rollout)
            policy_w += self.policy_lr * scale * policy_grad_w
            policy_b += self.policy_lr * scale * policy_grad_b
            value_w += self.value_lr * scale * value_grad_w
            value_b += self.value_lr * scale * value_grad_b
        return policy_w, policy_b, value_w, value_b

    def _best_action_from_current(self, current_eval, positive_only=True, require_meet=False):
        current_indices = current_eval["indices"]
        best_eval = None
        best_indices = None
        if positive_only:
            action_range = range(0, 2 * len(self.knobs), 2)
        else:
            action_range = range(2 * len(self.knobs))
        for action in action_range:
            self._set_indices(current_indices)
            candidate_indices, changed = self._apply_action_to_indices(current_indices, action)
            if not changed:
                continue
            candidate_eval = self._evaluate_indices(candidate_indices)
            if require_meet and not candidate_eval["meets_target"]:
                continue
            if not candidate_eval["valid"]:
                continue
            if best_eval is None or self._is_better(candidate_eval, best_eval):
                best_eval = candidate_eval
                best_indices = candidate_eval["indices"]
        self._set_indices(current_indices)
        return best_indices, best_eval

    def _node_target_cycles(self, scope_path):
        target = self.target_cycles_per_frame
        parent_path = ()
        for loop_name in scope_path:
            loop_node = self._node_by_name(parent_path, loop_name)
            loop_inst = getCustomOp(loop_node)
            iteration = max(1, int(loop_inst.get_nodeattr("iteration")))
            target = int(math.floor(target / iteration)) - FINNLOOP_ITERATION_OVERHEAD_CYCLES
            parent_path += (loop_name,)
        return max(1, target)

    def _target_dsp_lower_bound(self):
        """Return an independent-node DSP lower bound for meeting the target.

        For every hardware node, enumerate only controls that can affect its
        cycles or DSP usage and retain the least-DSP target-meeting choice.
        Summing those independent minima is optimistic, so exceeding the board
        DSP capacity is a proof that no design in this knob space can fit.
        """

        base_indices = self.initial_indices
        knob_groups = {}
        for knob_index, knob in enumerate(self.knobs):
            key = (knob.scope_path, knob.node_name)
            if knob.kind == "parallelism" or knob.attr_name in ["pumpedCompute", "resType"]:
                knob_groups.setdefault(key, []).append(knob_index)

        total_dsp = 0.0
        local_target_reachable = True
        try:
            self._set_indices(base_indices)
            for scope_path, model in self.scopes.items():
                node_target = self._node_target_cycles(scope_path)
                for node in model.graph.node:
                    if node.op_type == "FINNLoop" or not (is_hls_node(node) or is_rtl_node(node)):
                        continue
                    key = (scope_path, node.name)
                    knob_indices = knob_groups.get(key, [])
                    best_dsp = float("inf")
                    value_ranges = [range(len(self.knobs[index].values)) for index in knob_indices]
                    for value_indices in itertools.product(*value_ranges):
                        candidate = list(base_indices)
                        for knob_index, value_index in zip(knob_indices, value_indices):
                            candidate[knob_index] = value_index
                        self._set_indices(tuple(candidate))
                        node_inst = getCustomOp(self._node_by_name(scope_path, node.name))
                        if node_throughput_cycles(node_inst) > node_target:
                            continue
                        dsp = float(node_inst.node_res_estimation(self.fpgapart).get("DSP", 0))
                        best_dsp = min(best_dsp, dsp)
                    if not np.isfinite(best_dsp):
                        local_target_reachable = False
                    else:
                        total_dsp += best_dsp
        finally:
            self._set_indices(base_indices)
        return total_dsp, local_target_reachable

    def _balanced_target_start(self):
        """Build a resource-efficient per-node seed for the global search.

        MVAU has two coupled parallelism dimensions. Moving only one step at a
        time can strand a search in a high-PE/low-SIMD state at the DSP limit,
        even when a faster low-PE/high-SIMD state uses fewer DSPs. Enumerating
        the legal PE x SIMD pairs per MVAU is small and supplies PPO with a
        balanced, globally evaluated seed without encoding model-specific
        folding values.
        """

        current_indices = self._set_indices(self.initial_indices)
        knob_groups = {}
        for knob_index, knob in enumerate(self.knobs):
            if knob.kind != "parallelism":
                continue
            key = (knob.scope_path, knob.node_name)
            knob_groups.setdefault(key, []).append(knob_index)

        for (scope_path, node_name), knob_indices in knob_groups.items():
            node = self._node_by_name(scope_path, node_name)
            target = self._node_target_cycles(scope_path)
            attr_to_knob = {self.knobs[index].attr_name: index for index in knob_indices}

            if node.op_type in ["MVAU_hls", "MVAU_rtl"] and {
                "PE",
                "SIMD",
            }.issubset(attr_to_knob):
                pe_index = attr_to_knob["PE"]
                simd_index = attr_to_knob["SIMD"]
                best_indices = None
                best_key = None
                fastest_indices = None
                fastest_key = None
                for pe_value_index in range(len(self.knobs[pe_index].values)):
                    for simd_value_index in range(len(self.knobs[simd_index].values)):
                        candidate = list(current_indices)
                        candidate[pe_index] = pe_value_index
                        candidate[simd_index] = simd_value_index
                        repaired = self._set_indices(tuple(candidate))
                        cycles = node_throughput_cycles(
                            getCustomOp(self._node_by_name(scope_path, node_name))
                        )
                        resources = getCustomOp(
                            self._node_by_name(scope_path, node_name)
                        ).node_res_estimation(self.fpgapart)
                        norms = self._resource_norms(resources)
                        resource_score = sum(norms.values())
                        fastest_candidate_key = (cycles, resource_score)
                        if fastest_key is None or fastest_candidate_key < fastest_key:
                            fastest_key = fastest_candidate_key
                            fastest_indices = repaired
                        if cycles > target:
                            continue
                        candidate_key = (resource_score, target - cycles)
                        if best_key is None or candidate_key < best_key:
                            best_key = candidate_key
                            best_indices = repaired
                selected_indices = best_indices if best_indices is not None else fastest_indices
                current_indices = self._set_indices(selected_indices)
                continue

            for knob_index in knob_indices:
                knob = self.knobs[knob_index]
                for value_index in range(len(knob.values)):
                    candidate = list(current_indices)
                    candidate[knob_index] = value_index
                    repaired = self._set_indices(tuple(candidate))
                    cycles = node_throughput_cycles(
                        getCustomOp(self._node_by_name(scope_path, node_name))
                    )
                    current_indices = repaired
                    if cycles <= target:
                        break

        return self._evaluate_indices(current_indices)

    def _deterministic_target_search(self, start_eval):
        best_eval = start_eval
        current_eval = start_eval
        while not current_eval["meets_target"]:
            candidate_indices, candidate_eval = self._best_action_from_current(current_eval)
            if candidate_eval is None or not self._is_better(candidate_eval, current_eval):
                break
            current_eval = self._evaluate_indices(candidate_indices)
            if self._is_better(current_eval, best_eval):
                best_eval = current_eval
        return best_eval

    def _repair_meeting_capacity(self, start_eval):
        """Reduce capacity violations without giving up a met cycle target.

        The balanced seed chooses cheap node-local parallelism, but the sum can
        still exceed a board resource. Reaching a valid design may require a
        non-adjacent resource choice (for example ``auto`` directly to
        ``distributed`` RAM) or enabling pumped compute on several MVAUs.
        Sweep resource-only controls first because they preserve the balanced
        cycle point and have a much smaller Cartesian space. Only reopen the
        full per-node PE/SIMD space if resource controls alone cannot fit.
        """

        if not start_eval["meets_target"]:
            return start_eval

        def repair_key(evaluation):
            return (
                0 if evaluation["valid"] else 1,
                evaluation["resource_pressure"],
                evaluation["resource_score"],
                abs(self.target_cycles_per_frame - evaluation["max_cycles"]),
            )

        all_knob_groups = {}
        resource_knob_groups = {}
        for knob_index, knob in enumerate(self.knobs):
            key = (knob.scope_path, knob.node_name)
            all_knob_groups.setdefault(key, []).append(knob_index)
            if knob.kind == "resource":
                resource_knob_groups.setdefault(key, []).append(knob_index)

        def sweep(current_eval, knob_groups):
            while True:
                improved = False
                for knob_indices in knob_groups.values():
                    current_indices = current_eval["indices"]
                    best_candidate = current_eval
                    value_ranges = [range(len(self.knobs[index].values)) for index in knob_indices]
                    for value_indices in itertools.product(*value_ranges):
                        candidate = list(current_indices)
                        for knob_index, value_index in zip(knob_indices, value_indices):
                            candidate[knob_index] = value_index
                        if tuple(candidate) == current_indices:
                            continue
                        candidate_eval = self._evaluate_indices(tuple(candidate))
                        if not candidate_eval["meets_target"]:
                            continue
                        if repair_key(candidate_eval) < repair_key(best_candidate):
                            best_candidate = candidate_eval
                    self._set_indices(current_indices)
                    if best_candidate["indices"] != current_indices:
                        current_eval = self._evaluate_indices(best_candidate["indices"])
                        improved = True
                if not improved:
                    return current_eval

        current_eval = sweep(start_eval, resource_knob_groups)
        if current_eval["valid"]:
            return current_eval
        return sweep(current_eval, all_knob_groups)

    def _minimize_meeting_resources(self, start_eval):
        if not start_eval["meets_target"] or not start_eval["valid"]:
            return start_eval
        knob_groups = {}
        for knob_index, knob in enumerate(self.knobs):
            key = (knob.scope_path, knob.node_name)
            knob_groups.setdefault(key, []).append(knob_index)
        current_eval = start_eval
        while True:
            improved = False
            for knob_indices in knob_groups.values():
                best_candidate = current_eval
                current_indices = current_eval["indices"]
                value_ranges = [range(len(self.knobs[index].values)) for index in knob_indices]
                for value_indices in itertools.product(*value_ranges):
                    candidate = list(current_indices)
                    for knob_index, value_index in zip(knob_indices, value_indices):
                        candidate[knob_index] = value_index
                    if tuple(candidate) == current_indices:
                        continue
                    candidate_eval = self._evaluate_indices(tuple(candidate))
                    if not candidate_eval["valid"] or not candidate_eval["meets_target"]:
                        continue
                    if self._is_better(candidate_eval, best_candidate):
                        best_candidate = candidate_eval
                self._set_indices(current_indices)
                if best_candidate["indices"] != current_indices:
                    current_eval = self._evaluate_indices(best_candidate["indices"])
                    improved = True
            if not improved:
                break
        return current_eval

    def _maximize_valid_utilization(self, start_eval):
        if not start_eval["valid"]:
            return start_eval
        mvau_knobs = {}
        for knob_index, knob in enumerate(self.knobs):
            if knob.kind != "parallelism" or knob.attr_name not in ["PE", "SIMD"]:
                continue
            key = (knob.scope_path, knob.node_name)
            mvau_knobs.setdefault(key, {})[knob.attr_name] = knob_index
        current_eval = start_eval
        while True:
            current_indices = current_eval["indices"]
            best_candidate = None

            def consider(candidate_eval):
                nonlocal best_candidate
                if not candidate_eval["valid"]:
                    return
                improves_cycles = candidate_eval["max_cycles"] < current_eval["max_cycles"]
                same_cycles = candidate_eval["max_cycles"] == current_eval["max_cycles"]
                improves_util = (
                    candidate_eval["resource_score"] > current_eval["resource_score"] + 1.0e-9
                )
                if not improves_cycles and not (same_cycles and improves_util):
                    return
                if best_candidate is None:
                    best_candidate = candidate_eval
                elif candidate_eval["max_cycles"] < best_candidate["max_cycles"]:
                    best_candidate = candidate_eval
                elif (
                    candidate_eval["max_cycles"] == best_candidate["max_cycles"]
                    and candidate_eval["resource_score"] > best_candidate["resource_score"]
                ):
                    best_candidate = candidate_eval

            for action in range(0, 2 * len(self.knobs), 2):
                self._set_indices(current_indices)
                candidate_indices, changed = self._apply_action_to_indices(current_indices, action)
                if not changed:
                    continue
                consider(self._evaluate_indices(candidate_indices))

            # A PE/SIMD exchange can preserve throughput while changing resource
            # cost. Enumerate the legal pairs because either single step may be invalid.
            for (scope_path, node_name), attr_indices in mvau_knobs.items():
                if not {"PE", "SIMD"}.issubset(attr_indices):
                    continue
                node = self._node_by_name(scope_path, node_name)
                if node.op_type not in ["MVAU_hls", "MVAU_rtl"]:
                    continue
                pe_index = attr_indices["PE"]
                simd_index = attr_indices["SIMD"]
                for pe_value_index in range(len(self.knobs[pe_index].values)):
                    for simd_value_index in range(len(self.knobs[simd_index].values)):
                        candidate = list(current_indices)
                        candidate[pe_index] = pe_value_index
                        candidate[simd_index] = simd_value_index
                        self._set_indices(current_indices)
                        candidate_indices = self._set_indices(tuple(candidate))
                        if candidate_indices == current_indices:
                            continue
                        consider(self._evaluate_indices(candidate_indices))
            self._set_indices(current_indices)
            if best_candidate is None:
                break
            current_eval = self._evaluate_indices(best_candidate["indices"])
        return current_eval

    def _random_valid_start(self, best_indices):
        if len(self.knobs) == 0:
            return tuple()
        choices = [self.initial_indices, best_indices]
        base = list(self.py_rng.choice(choices))
        for _ in range(max(1, len(self.knobs) // 2)):
            action = self.py_rng.randrange(2 * len(self.knobs))
            base, _ = self._apply_action_to_indices(tuple(base), action)
            base = list(base)
        evaluation = self._evaluate_indices(tuple(base))
        return evaluation["indices"] if evaluation["valid"] else best_indices

    def _run_ppo(self, start_eval):
        if len(self.knobs) == 0 or self.episodes <= 0 or self.rollout_steps <= 0:
            return start_eval

        state_dim = len(self._state_vector(start_eval))
        action_dim = 2 * len(self.knobs)
        policy_w = self.rng.normal(0.0, 0.02, (state_dim, action_dim))
        policy_b = np.zeros(action_dim, dtype=np.float64)
        value_w = self.rng.normal(0.0, 0.02, state_dim)
        value_b = 0.0
        best_eval = start_eval

        for episode in range(self.episodes):
            if episode % 3 == 0:
                current_indices = self.initial_indices
            elif episode % 3 == 1:
                current_indices = best_eval["indices"]
            else:
                current_indices = self._random_valid_start(best_eval["indices"])
            current_eval = self._evaluate_indices(current_indices)
            rollout = []
            for _ in range(self.rollout_steps):
                state = self._state_vector(current_eval)
                mask = self._valid_action_mask(current_eval["indices"])
                if not np.any(mask):
                    break
                action, logp, value, _ = self._sample_action(
                    state, mask, policy_w, policy_b, value_w, value_b
                )
                old_eval = current_eval
                candidate_indices, changed = self._apply_action_to_indices(
                    old_eval["indices"], action
                )
                if not changed:
                    reward = -0.01
                    candidate_eval = old_eval
                else:
                    candidate_eval = self._evaluate_indices(candidate_indices)
                    reward = candidate_eval["quality"] - old_eval["quality"]
                    if not candidate_eval["valid"] and old_eval["valid"]:
                        reward = -1.0
                        candidate_eval = old_eval
                        self._set_indices(old_eval["indices"])
                    if not np.isfinite(reward):
                        reward = -1.0 if not candidate_eval["valid"] else 0.0
                rollout.append(
                    {
                        "state": state,
                        "action": action,
                        "logp": logp,
                        "value": value,
                        "reward": reward,
                        "mask": mask,
                    }
                )
                current_eval = candidate_eval
                if self._is_better(current_eval, best_eval):
                    best_eval = current_eval
            policy_w, policy_b, value_w, value_b = self._ppo_update(
                rollout, policy_w, policy_b, value_w, value_b
            )
        return best_eval

    def optimize(self):
        if len(self.knobs) == 0:
            self._annotate_and_commit_scopes()
            return self.model

        start_eval = self._evaluate_indices(self.initial_indices)
        target_proven_infeasible = False
        if len(self.scopes) > 1 and self.capacity is not None and self.capacity.get("DSP", 0) > 0:
            dsp_lower_bound, local_target_reachable = self._target_dsp_lower_bound()
            dsp_capacity = float(self.capacity["DSP"])
            target_proven_infeasible = (
                not local_target_reachable or dsp_lower_bound > dsp_capacity + 1.0e-9
            )
            self.target_feasibility = {
                "local_target_reachable": local_target_reachable,
                "DSP_lower_bound": dsp_lower_bound,
                "DSP_capacity": dsp_capacity,
                "target_proven_infeasible": target_proven_infeasible,
            }

        if target_proven_infeasible:
            best_eval = self._deterministic_target_search(start_eval)
        else:
            balanced_eval = self._balanced_target_start()
            if balanced_eval["meets_target"]:
                balanced_eval = self._repair_meeting_capacity(balanced_eval)
            if balanced_eval["valid"]:
                best_eval = balanced_eval
            else:
                best_eval = self._deterministic_target_search(start_eval)
        if not best_eval["meets_target"]:
            best_eval = self._maximize_valid_utilization(best_eval)
        if best_eval["meets_target"]:
            best_eval = self._minimize_meeting_resources(best_eval)

        ppo_eval = self._run_ppo(best_eval)
        ppo_improved = False
        if self._is_better(ppo_eval, best_eval):
            best_eval = ppo_eval
            ppo_improved = True

        if ppo_improved and best_eval["meets_target"]:
            best_eval = self._minimize_meeting_resources(best_eval)

        self._set_indices(best_eval["indices"])
        self._annotate_and_commit_scopes()
        if self.target_feasibility is not None:
            self.model.set_metadata_prop(
                "resource_aware_folding_target_feasibility",
                json.dumps(self.target_feasibility),
            )
        return self.model


class SetFolding(Transformation):
    """Attempt to set parallelism attributes in all nodes to meet a specific
    target expressed as cycles per frame target_cycles_per_frame. For each
    HLSCustomOp node type, the attribute may vary but is typically one of {PE, SIMD},
    and has a certain allowed-maximum value and divisibility constraints,
    which SetFolding will take into account.

    If fpgapart is provided, SetFolding uses a resource-aware PPO search. This
    search minimizes the sum of estimated LUT, BRAM18, URAM, and DSP fractions
    of board capacity for a design that meets the target cycles. Deterministic
    refinement exhaustively checks every knob combination within each node, so
    the returned target-meeting design is a per-node coordinate minimum under
    those analytical cycle/resource models. It is not a claim of globally
    minimal post-synthesis resources. If the target is too high to meet, it
    chooses the fastest valid design it can find and uses remaining valid board
    capacity as a tie-breaker. Without fpgapart, SetFolding keeps the legacy
    greedy behavior.

    In the returned model, each node's
    cycles_estimate attribute will be set to its estimated number of cycles.

    In the legacy greedy path, if two_pass_relaxation is enabled, SetFolding
    will internally run a second time if the target cycles from the first pass
    could not be achieved, instead using the achievable target (which may be
    constrained by a single node) to obtain a balanced pipeline.

    Notable exceptions and special behavior:

    The legacy greedy path has special handling for dense convolution/FC compute
    engines ("MVAU"/MatrixVectorActivation), which have two attributes (PE and
    SIMD):

    * first increases SIMD while weight stream width per PE is <= mvau_wwidth_max
      (configurable in the SetFolding initializer, defaults to 36)
    * then increases PE until the target is met or max PE reached

    When folding depthwise convolutions ("VVAU"/VectorVectorActivation)
    or spatial reduction ops (Pool_Batch):

    * the producer of the node is expected to be a ConvolutionInputGenerator
      with depthwise=1, whose SIMD value will be set equal to the PE value of
      its consumer node
    * the VVAU also supports SIMD ("input window") parallelism next to
      PE ("channels"), but current ConvInpGen limitations require PE to be fully
      unfolded before SIMD is increased
    """

    def __init__(
        self,
        target_cycles_per_frame=1000,
        mvau_wwidth_max=36,
        two_pass_relaxation=True,
        fpgapart=None,
        board=None,
        resource_aware=None,
        resource_limit=1.0,
        ppo_episodes=6,
        ppo_rollout_steps=32,
        ppo_update_epochs=4,
        ppo_seed=0,
    ):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation
        self.fpgapart = fpgapart
        self.board = board
        self.resource_aware = (fpgapart is not None) if resource_aware is None else resource_aware
        self.resource_limit = resource_limit
        self.ppo_episodes = ppo_episodes
        self.ppo_rollout_steps = ppo_rollout_steps
        self.ppo_update_epochs = ppo_update_epochs
        self.ppo_seed = ppo_seed

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc < self.target_cycles_per_frame:
                # finish if target met
                break

    def apply(self, model):
        if self.resource_aware:
            model = model.transform(GiveUniqueNodeNames())
            optimizer = ResourceAwareFoldingPPO(
                model,
                self.target_cycles_per_frame,
                self.fpgapart,
                board=self.board,
                mvau_wwidth_max=self.mvau_wwidth_max,
                resource_limit=self.resource_limit,
                episodes=self.ppo_episodes,
                rollout_steps=self.ppo_rollout_steps,
                update_epochs=self.ppo_update_epochs,
                seed=self.ppo_seed,
            )
            model = optimizer.optimize()
            return (model, False)
        return self._apply_legacy(model)

    def _apply_legacy(self, model):
        graph = model.graph
        # these ops use PE parallelism, up to a max value of NumChannels
        pe_ops = [
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
            *ELEMENTWISE_BINARY_OPS,
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        # ConvolutionInputGenerator has a special case when depthwise=1
        # ConvolutionInputGenerator_rtl supports additional parallelism by
        # setting parallel_window=1 mode after maxing out SIMD
        simd_ops = [
            "FMPadding_rtl",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_rtl",
            "StreamingSplit_hls",
            "StreamingConcat_hls",
            "LayerNorm_rtl",
            "Shuffle",
        ]
        # these ops are preceded by depthwise SWG and have special behavior,
        # as explained in the SetFolding docstring
        depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]
        for node in graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)
            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                max_simd = node_inst.get_nodeattr("MW")
                max_pe = node_inst.get_nodeattr("MH")
                node_inst.set_nodeattr("PE", 1)
                min_simd = 1
                if op_type == "MVAU_hls":
                    min_simd = first_divisor_at_least(max_simd, int(math.ceil(max_simd / 1024)))
                node_inst.set_nodeattr("SIMD", min_simd)
                # increase SIMD until either we meet
                # the target or weight stream becomes
                # too wide
                for simd_val in divisors(max_simd):
                    if simd_val < min_simd:
                        continue
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.target_cycles_per_frame:
                        # finish if target met
                        break
                    if (
                        node_inst.get_input_datatype(1).bitwidth() * node_inst.get_nodeattr("SIMD")
                        > self.mvau_wwidth_max
                    ):
                        # revert if we've gone above width threshold
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        break
                # increase PE until target met or reached max_pe
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in pe_ops:
                # Note: Keep original behavior for all custom-ops defining the
                # NumChannels attribute as it is
                try:
                    max_pe = node_inst.get_nodeattr("NumChannels")
                # Note: Some of the recent additions do not define the
                # NumChannels attribute
                except AttributeError:
                    # We can extract the channels from the normal, i.e., not
                    # folded, shape of the input in these cases
                    max_pe = node_inst.get_normal_input_shape()[-1]
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "LabelSelect_hls":
                max_pe = node_inst.get_nodeattr("Labels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in depthwise_op_exceptions:
                # init/reset SIMD of VVAU
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    node_inst.set_nodeattr("SIMD", 1)
                max_pe = node_inst.get_nodeattr("Channels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                # increase SIMD for VVAU once PE is exhausted
                pe = node_inst.get_nodeattr("PE")
                cyc = node_inst.get_exp_cycles()
                if (
                    op_type in ["VVAU_hls", "VVAU_rtl"]
                    and pe == max_pe
                    and cyc > self.target_cycles_per_frame
                ):
                    max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                # also set the folding of the upsteam DW SWU
                # which must be identical to this node
                swu_node = model.find_producer(node.input[0])
                if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                    swu_node_inst = getCustomOp(swu_node)
                    swu_node_inst.set_nodeattr("SIMD", pe)
                    # enable parallel_window mode of RTL SWG if needed
                    if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                        if op_type.startswith("VVAU") and node_inst.get_nodeattr("SIMD") > 1:
                            swu_node_inst.set_nodeattr("parallel_window", 1)
                        else:
                            swu_node_inst.set_nodeattr("parallel_window", 0)
                else:
                    if op_type in ["VVAU_hls", "VVAU_rtl"]:
                        ksize = np.prod(node_inst.get_nodeattr("Kernel"))
                    elif op_type == "Pool_hls":
                        ksize = node_inst.get_nodeattr("KernelSize")
                    else:
                        raise Exception("Undefined edge case for %s" % op_type)
                    if ksize != 1:  # pointwise vvau/pool lack a SWU
                        raise Exception("Expected SWU on DW op input, found " + swu_node.op_type)
            elif op_type in simd_ops:
                if op_type.startswith("ConvolutionInputGenerator"):
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = node_inst.get_nodeattr("IFMChannels")
                        # init/reset parallel_window mode of RTL SWG
                        if op_type == "ConvolutionInputGenerator_rtl":
                            node_inst.set_nodeattr("parallel_window", 0)
                        self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                        # enable parallel_window mode of RTL SWG if needed
                        simd = node_inst.get_nodeattr("SIMD")
                        cyc = node_inst.get_exp_cycles()
                        if (
                            op_type == "ConvolutionInputGenerator_rtl"
                            and simd == max_simd
                            and cyc > self.target_cycles_per_frame
                        ):
                            node_inst.set_nodeattr("parallel_window", 1)
                    else:
                        # depthwise SWGs are handled separately
                        continue
                elif op_type == "StreamingConcat_hls" or op_type == "StreamingSplit_hls":
                    node_inst.set_nodeattr("SIMD", 1)
                    channels_per_stream = node_inst.get_nodeattr("ChannelsPerStream")
                    for simd_val in common_divisors(channels_per_stream):
                        node_inst.set_nodeattr("SIMD", simd_val)
                        cyc = node_inst.get_exp_cycles()
                        if cyc < self.target_cycles_per_frame:
                            break
                elif op_type == "LayerNorm_rtl":
                    node_inst.set_nodeattr("SIMD", 1)
                    dim = int(node_inst.get_normal_input_shape()[-1])
                    for simd_val in divisors(dim):
                        if dim // simd_val > 12:
                            node_inst.set_nodeattr("SIMD", simd_val)
                            cyc = node_inst.get_exp_cycles()
                            if cyc < self.target_cycles_per_frame:
                                break
                        else:
                            break
                else:
                    max_simd = node_inst.get_nodeattr("NumChannels")
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
            else:
                warnings.warn("SetFolding doesn't know how to handle op_type " + op_type)

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        if self.two_pass_relaxation:
            perf_dict = model.analysis(dataflow_performance)
            if perf_dict["max_cycles"] > self.target_cycles_per_frame:
                # run again, but with lower target (that we managed) -- this
                # may be coming from a single node's constraints, but we want
                # to balance the entire dataflow pipeline instead
                # no two_pass_relaxation this time -- no guarantee we'll
                # converge otherwise
                warnings.warn(
                    "Node %s is bottleneck with %d cycles, running second pass"
                    % (perf_dict["max_cycles_node_name"], perf_dict["max_cycles"])
                )
                model = model.transform(
                    SetFolding(
                        target_cycles_per_frame=perf_dict["max_cycles"],
                        mvau_wwidth_max=self.mvau_wwidth_max,
                        two_pass_relaxation=False,
                    )
                )

        return (model, False)
