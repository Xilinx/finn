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

import copy
import functools

# Inspect information on Python objects like modules
import inspect
import numpy as np
import scipy
import time
import warnings
from onnx import TensorProto, helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor

# Import the elementwise binary operation module to extract names of all
# specializations (which require PE parallelism to be configured)
import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as elementwise_binary_hls
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import part_map
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.platforms import DEFAULT_RES_LIMITS, platforms


def divisors(num):
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


def allowed_divisors(cap, bounding_value_exponent=1, max_padding_count=0, skip_folding=False):
    """
    compute all possible folding factors for a given
    upper bound variable

    max_padding_count allows generating values with the assumption
    that the bounding variable could be padded by up to that many
    elements, which dramatically increases the possible folding
    parameters with even a small amount of extra values

    bounding_value_exponent, if set to two, forces the folding factors into
    square roots of the bounding variable (applicable in some cases)
    """

    all_divs = []
    all_bounding_values = []
    factors = []
    if skip_folding:
        all_divs = [1]
        all_bounding_values = [cap]
    else:
        for i in range(cap, cap + max_padding_count + 1):
            for x in range(1, i + 1):
                if (i**bounding_value_exponent % x) == 0:
                    if (x not in all_divs) and (x <= cap) and (i // x not in factors):
                        all_divs.append(x)
                        all_bounding_values.append(i)
                        factors.append(i // x)

    return zip(*sorted(zip(all_divs, all_bounding_values)))


class GridEncoder:
    """Encode ordered discrete folding values into a continuous index space so a
    continuous global optimizer (scipy.dual_annealing) can search over them, and
    decode a searched point back to real values. A parameter with a single
    possible value is fixed and carries no encoded dimension. This replaces the
    equivalent wrapdisc GridVar/Objective functionality."""

    _EPS = 1e-9

    def __init__(self, choices):
        # choices: one ordered value tuple per parameter
        self.choices = [tuple(c) for c in choices]

    @property
    def bounds(self):
        # half-step bounds keep boundary indices reachable and rounding-safe
        return [
            (-0.5 + self._EPS, (len(c) - 1) + 0.5 - self._EPS) for c in self.choices if len(c) > 1
        ]

    def encode(self, decoded):
        return [float(c.index(v)) for c, v in zip(self.choices, decoded) if len(c) > 1]

    def decode(self, encoded):
        out, it = [], iter(encoded)
        for c in self.choices:
            if len(c) > 1:
                idx = min(max(int(round(next(it))), 0), len(c) - 1)
                out.append(c[idx])
            else:
                out.append(c[0])
        return out


class Parameter:
    def __init__(
        self,
        name=None,  # SWU_SIMD, MVAU_SIMD, MVAU_PE etc
        target_value_name=None,
        target_value=None,
        bound_name=None,
        bound_value=None,
        bound_value_last=None,
        update_threshold_input=False,
        update_weights_input=False,
        # every construction in generate_parameter_set updates both io shapes
        update_input_tensor_shape=True,
        update_output_tensor_shape=True,
        node=None,  # node instance!
        node_index=None,
        op_type=None,
        model=None,
    ):
        self.name = name
        self.target_value_name = target_value_name
        self.target_value = target_value
        self.bound_name = bound_name
        self.bound_value = bound_value
        self.bound_value_last = bound_value_last
        self.update_threshold_input = update_threshold_input
        self.update_weights_input = update_weights_input
        self.update_input_tensor_shape = update_input_tensor_shape
        self.update_output_tensor_shape = update_output_tensor_shape
        self.node = node
        self.node_index = node_index
        self.op_type = op_type
        self.model = model

    def update_threshold_tensor(self):
        if self.op_type in ["Thresholding_hls", "Thresholding_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("NumChannels")

        elif self.op_type in ["VVAU_hls", "VVAU_rtl"]:
            input_index = 2
            dim0 = self.node.get_nodeattr("Channels")
            if len(self.model.graph.node[self.node_index].input) < 3:
                # if the MVAU doesnt have a threshold input, just skip
                return

        elif self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            input_index = 2
            dim0 = self.node.get_nodeattr("MH")
            if len(self.model.graph.node[self.node_index].input) < 3:
                # if the MVAU doesnt have a threshold input, just skip
                return

        # thresholding nodes have a weight matrix which needs to be
        # adjusted if padding or cropping were introduced
        # MVAU and VVAU nodes can also have it so we stay flexible

        T = self.model.get_initializer(self.model.graph.node[self.node_index].input[input_index])

        adt = self.model.get_tensor_datatype(
            self.model.graph.node[self.node_index].input[input_index]
        )
        T_new = gen_finn_dt_tensor(adt, (dim0, T.shape[1]))
        T_new[...] = 0

        T_new[: min(dim0, T.shape[0]), :] = T[: min(dim0, T.shape[0]), :]

        self.model.set_initializer(self.model.graph.node[self.node_index].input[input_index], T_new)

        self.model.set_tensor_shape(
            self.model.graph.node[self.node_index].input[input_index], T_new.shape
        )

    def update_weight_tensor(self):
        if self.op_type in ["VVAU_hls", "VVAU_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("Channels")
            dim1 = self.node.get_nodeattr("Kernel")

        elif self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            input_index = 1
            dim0 = self.node.get_nodeattr("MW")
            dim1 = self.node.get_nodeattr("MH")

        W = self.model.get_initializer(self.model.graph.node[self.node_index].input[input_index])

        if self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            if (dim0, dim1) == W.shape:
                return False

        if self.op_type in ["VVAU_hls", "VVAU_rtl"]:
            if W.shape[0] == dim0 and W.shape[-2:] == tuple(dim1):
                return False

        wdt = self.model.get_tensor_datatype(
            self.model.graph.node[self.node_index].input[input_index]
        )

        if self.op_type in ["MVAU_hls", "MVAU_rtl"]:
            W_new = gen_finn_dt_tensor(wdt, (dim0, dim1))
            W_new[...] = 0

            W_new[: min(dim0, W.shape[0]), : min(dim1, W.shape[1])] = W[
                : min(dim0, W.shape[0]), : min(dim1, W.shape[1])
            ]
            self.model.set_initializer(self.model.graph.node[self.node_index].input[1], W_new)

        if self.op_type in ["VVAU_hls", "VVAU_rtl"]:
            W_new = gen_finn_dt_tensor(wdt, (dim0, W.shape[1], dim1[0], dim1[1]))
            W_new[...] = 0

            W_new[
                : min(dim0, W.shape[0]), :, : min(dim1[0], W.shape[2]), : min(dim1[1], W.shape[3])
            ] = W[
                : min(dim0, W.shape[0]), :, : min(dim1[0], W.shape[2]), : min(dim1[1], W.shape[3])
            ]

            self.model.set_initializer(
                self.model.graph.node[self.node_index].input[input_index], W_new
            )

        self.model.set_tensor_shape(self.model.graph.node[self.node_index].input[1], W_new.shape)

        return True

    def apply_value(self, final=True):
        # update the target value being optimized
        self.node.set_nodeattr(self.target_value_name, self.target_value)

        # if the bounding value has changed (ie,. MW of an MVAU) as
        # a result of padding the node, update it as well
        # if self.bound_value != self.bound_value_last:
        if self.bound_name is not None:
            self.node.set_nodeattr(self.bound_name, self.bound_value)

        # make certain parallel window is set right
        if self.bound_name == "IFMChannels":
            if self.target_value < self.bound_value:
                self.node.set_nodeattr("parallel_window", 0)

        # if this is the end of the minimizer routine, we update the tensor
        # shapes as well to retain functional correctness
        if final:
            # first the io tensors only
            if self.update_input_tensor_shape:
                new_shape = self.node.get_normal_input_shape()
                self.model.set_tensor_shape(
                    self.model.graph.node[self.node_index].input[0], new_shape
                )

            if self.update_output_tensor_shape:
                new_shape = self.node.get_normal_output_shape()
                self.model.set_tensor_shape(
                    self.model.graph.node[self.node_index].output[0], new_shape
                )

            if self.update_threshold_input:
                self.update_threshold_tensor()

            if self.update_weights_input:
                self.update_weight_tensor()


class MetaParameter:
    """
    A meta parameter defines a single optimizable integer value (meta_value)
    which translates into a set of finn-onnx graph node attributes
    which are tighly linked together (called values)

    Examples:
    -SIMD and PE values of a VVAU + SIMD of the SWU if necessary
    -SIMD value of an SWU and the PE and SIMD values of an MVAU (convolution)
    -SWU and Pool layer SIMD values (max pooling using SWU)

    - NOTE that MVAU PE and SIMD values are optimized independently, since
    - both 1-2 and 2-1 SIMD-PE combinations would have the same meta value
    - while having different resource characteristics

    All possible (legal) combinations of real values are stored in a list and an
    address translation is performed to map each meta_value to a set
    of real values when applying them
    """

    def __init__(
        self,
        name=None,
        meta_value=None,  # current value
        possible_values=[],  # all possible values
        real_values=[],  # list of real values for each possible value
        model=None,
        node_index=None,
    ):
        self.name = name
        self.meta_value = None
        assert len(real_values) == len(possible_values)
        self.possible_values = possible_values
        self.real_values = real_values
        self.model = model
        self.updated = False
        self.index = 0
        self.node_index = node_index

        """
        we build up a list of unique nodes related to this meta parameter
        for future cycle calculations
        """

        # sort the values first
        pairs = [
            (x, y)
            for (x, y) in sorted(
                zip(self.possible_values, self.real_values), key=lambda pair: pair[0]
            )
        ]
        self.possible_values = [x[0] for x in pairs]
        self.real_values = [x[1] for x in pairs]

        self.unique_nodes = []
        for val in real_values[0]:
            if val.node not in self.unique_nodes:
                self.unique_nodes.append(val.node)

    def update_value(self, value):
        if self.meta_value == value:
            self.updated = False
        else:
            self.meta_value = value
            self.updated = True

    def apply_value(self, final=False, filter=["PE", "SIMD", "parallel_window"]):
        # make sure to run this once before minimizing
        self.index = self.possible_values.index(self.meta_value)
        for val in self.real_values[self.index]:
            if val.target_value_name in filter:
                val.apply_value(final)

    def get_cycles(self):
        """
        This function assumes all parameters in the unique nodes are
        updated.
        """
        return max([n.get_exp_cycles() for n in self.unique_nodes])


class ParameterSet:
    def __init__(self):
        self.parameters = []
        self.index_list = []
        self.nodes = []

    def filter(self, params_to_filter):
        # filter parameters we want to use in the set
        # useful for multi-pass optimization
        self.parameters = [x for x in self.parameters if x.name in params_to_filter]

    def get_max_cycles(self):
        return max([n.get_exp_cycles() for n in self.nodes])

    def get_vals(self):
        return [p.value for p in self.parameters]

    def get_min_vals(self):
        # get minimum possible folding values in the set
        return [p.possible_values[0] for p in self.parameters]

    def get_max_vals(self):
        # get maximum possible folding values in the set
        return [p.possible_values[-1] for p in self.parameters]

    def add_all_params_to_index_list(self):
        self.index_list = [x for x in range(len(self.parameters))]

    def set_values(self, values):
        for i in range(len(self.index_list)):
            self.parameters[self.index_list[i]].update_value(values[i])

    def apply_updates(self, final=False, filter=[]):
        # a
        for i in self.index_list:
            self.parameters[i].apply_value(final, filter)

    def assign_involved_nodes(self):
        nodes = []
        for i in range(len(self.index_list)):
            p = self.parameters[self.index_list[i]]
            for node in p.unique_nodes:
                nodes.append(node)
        self.nodes = list(set(nodes))  # make this unique


class Optimizer:
    """
    Class responsible for the 'inner loop' of the folding optimization.
    We set all minimizer-specific Hyper-parameters here, model
    node & parameter partitioning, minimizer instantation,
    cost model function and the overarching loop of minimizing the
    partitions are performed in this class.

    How the optimizer decides what to fold
    --------------------------------------
    EVERY hardware node is folded generically: for each node it reads whichever
    of PE / SIMD the node declares and sweeps the legal folding factors of that
    parameter. The maximum useful value comes straight off the tensor shapes::

        max PE   = output channels = get_normal_output_shape()[-1]
        max SIMD = input channels  = get_normal_input_shape()[-1]

    Only a handful of things are genuinely op-specific, and each is one of the
    short tables below -- easy to see, and easy to add to or remove. The fifth
    exception is the SWG pairing rules (``_pair_*``): a
    ConvolutionInputGenerator must be folded in lockstep with the layer it feeds.
    """

    # A VVAU folds SIMD over its kernel window, not its input channels, so the
    # input-shape rule above does not apply.
    SIMD_MAX_OVERRIDE = {
        "VVAU_hls": lambda inst: int(np.prod(inst.get_nodeattr("Kernel"))),
        "VVAU_rtl": lambda inst: int(np.prod(inst.get_nodeattr("Kernel"))),
    }

    # Parameters we deliberately keep unfolded (fixed at 1).
    DO_NOT_FOLD = {
        # Folding LabelSelect can ruin fmax; revisit once an RTL LabelSelect / a
        # safe topk-to-label heuristic exists.
        ("LabelSelect_hls", "PE"),
        # A VVAU_hls only supports SIMD (kernel-window) folding in its RTL sibling.
        ("VVAU_hls", "SIMD"),
        # Multi-stream concat/split fold their SIMD over per-stream channel
        # counts, which needs the dedicated handling in the naive folder; the
        # optimizer does not model that yet, so leave them unfolded rather than
        # fold them wrongly.
        ("StreamingConcat_hls", "SIMD"),
        ("StreamingSplit_hls", "SIMD"),
    }

    # Ops that may be channel-padded to unlock finer folding factors, mapping
    # (op_type, param) -> the node attribute carrying the padded channel count.
    # Nodes absent here are still folded, just never padded. VVAU/SWG SIMD are
    # omitted on purpose: their SIMD tracks a kernel size that must not be padded.
    PADDING_BOUND_ATTR = {
        ("MVAU_hls", "SIMD"): "MW",
        ("MVAU_rtl", "SIMD"): "MW",
        ("MVAU_hls", "PE"): "MH",
        ("MVAU_rtl", "PE"): "MH",
        ("VVAU_hls", "PE"): "Channels",
        ("VVAU_rtl", "PE"): "Channels",
        ("Thresholding_hls", "PE"): "NumChannels",
        ("Thresholding_rtl", "PE"): "NumChannels",
        ("AddStreams_hls", "PE"): "NumChannels",
        ("ChannelwiseOp_hls", "PE"): "NumChannels",
        ("DuplicateStreams_hls", "PE"): "NumChannels",
        ("StreamingMaxPool_hls", "PE"): "NumChannels",
        ("StreamingMaxPool_rtl", "PE"): "NumChannels",
        ("FMPadding_hls", "SIMD"): "NumChannels",
        ("FMPadding_rtl", "SIMD"): "NumChannels",
        ("FMPadding_Pixel_hls", "SIMD"): "NumChannels",
        ("DownSampler_hls", "SIMD"): "NumChannels",
    }

    # Ops whose weight / threshold initializers must be resized when their
    # folding (and therefore channel count, under padding) changes.
    OPS_WITH_WEIGHTS = {"MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl"}
    OPS_WITH_THRESHOLDS = {
        "MVAU_hls",
        "MVAU_rtl",
        "VVAU_hls",
        "VVAU_rtl",
        "Thresholding_hls",
        "Thresholding_rtl",
    }

    # ConvolutionInputGenerator ("SWG") op types. An SWG is always fused to the
    # layer it feeds and is folded by a pairing rule, never on its own.
    SWG_OPS = {"ConvolutionInputGenerator_hls", "ConvolutionInputGenerator_rtl"}

    # Categorical resource-selection attributes the optimizer can tune to trade
    # one resource for another (BRAM<->URAM via ram_style, LUT<->DSP via resType).
    RESOURCE_TYPE_ATTRS = ["ram_style", "resType", "ram_style_thresholds"]

    # Resource-type values the optimizer must NOT choose, because
    # node_res_estimation does not account for their real cost and would treat
    # them as "free": "distributed" maps memory to LUTRAM but is estimated as
    # 0 LUT/BRAM/URAM, so an unconstrained search would hide all memory there
    # instead of making a real BRAM<->URAM trade.
    RESOURCE_TYPE_EXCLUDED_VALUES = {
        "ram_style": {"distributed"},
        "ram_style_thresholds": {"distributed"},
    }

    # rough LUT cost of an HLS DWC relative to an RTL one (DWC-avoidance heuristic)
    HLS_DWC_COST_PENALTY = 8

    @staticmethod
    def max_folding_factor(node_inst, op_type, param):
        """Largest useful value for a folding parameter (``PE`` or ``SIMD``) on a
        node, read from the tensor shapes with the documented per-op exceptions."""
        if param == "PE":
            return int(node_inst.get_normal_output_shape()[-1])
        # param == "SIMD"
        if op_type in Optimizer.SIMD_MAX_OVERRIDE:
            return Optimizer.SIMD_MAX_OVERRIDE[op_type](node_inst)
        return int(node_inst.get_normal_input_shape()[-1])

    def __init__(
        self,
        model,
        name,
        targets,
        hard_constraint_target="max_cycles",
        target_cycles_per_frame=1,
        padding=0,
        maxfun_per_parameter=100,
        fpgapart="xc7z020clg400-1",
        parameters_to_apply=["SIMD", "PE", "parallel_window", *RESOURCE_TYPE_ATTRS],
        enable_folding_dwc_heuristic=True,
        verbose=False,
        mvau_wwidth_max=1024,
        # slack on the hard (max_cycles) constraint: violated only if
        # metric*this > target
        value_to_minimize_relaxation=0.98,
        init_run=False,
        # --- scipy dual_annealing driver ---
        # global iteration cap (maxfun = effort*N is usually binding first)
        maxiter=200,
        # acceptance-distribution parameter (scipy range (-1e4, -5]);
        # less negative = more accepting
        accept=-0.5,
        # visiting-distribution shape parameter (scipy range (1, 3])
        visit=2.0,
        seed=None,
        # large FINITE cost for a hard-constraint violation, so dual_annealing
        # can climb out of an infeasible region instead of aborting on inf/nan
        infeasible_penalty=1e6,
        pad_io_nodes=False,
        # resource-type attributes (ram_style/resType) the optimizer may tune to
        # trade BRAM<->URAM and LUT<->DSP; empty means "fold only, don't retune".
        resource_type_params=[],
        # per-resource multipliers in the cost function (see cost_model)
        resource_weights=None,
        # may weight memories be placed in URAM? (requires making them
        # runtime-writeable, so it is opt-in at the SetFolding level)
        allow_uram_weights=False,
    ):
        self.params = None
        self.targets = targets
        self.target_cycles_per_frame = target_cycles_per_frame
        self.padding = padding
        self.mvau_wwidth_max = mvau_wwidth_max
        self.model = model
        self.pad_io_nodes = pad_io_nodes
        self.name = name
        self.fpgapart = fpgapart
        self.init_run = init_run
        self.maxiter = maxiter
        self.accept = accept
        self.visit = visit
        self.seed = seed
        self.infeasible_penalty = infeasible_penalty

        # 0-100, relax whether we MUST hit the required bounding value,
        # for example max_cycles
        self.value_to_minimize_relaxation = value_to_minimize_relaxation
        self.maxfun_per_parameter = maxfun_per_parameter

        self.hard_constraint_target = hard_constraint_target
        self.parameters_to_apply = parameters_to_apply
        self.enable_folding_dwc_heuristic = enable_folding_dwc_heuristic
        self.verbose = verbose
        self.resource_type_params = resource_type_params
        self.resource_weights = resource_weights or {}
        self.allow_uram_weights = allow_uram_weights

    def compute_hls_dwc_cost(
        self, model, nodes, lut_capacity, hls_dwc_cost_penalty=HLS_DWC_COST_PENALTY
    ):
        # Given a set of nodes and a model,
        # consider the stream widths between all adjacent nodes
        # and apply a cost penalty if the shapes mismatch relative
        # to the cost of introducing a DataWidthConverter

        # this heuristic is critical for preventing overuse of
        # DWCs with enormous resource costs

        # hls_dwc_cost_penalty is a rough heuristic for how much
        # an HLS variant consumes in LUTs

        cost = 0
        for node in nodes:
            prod = model.find_producer(node.onnx_node.input[0])

            # check if this is not the first node of a model
            if prod is not None:
                output_name = prod.output[0]
                prod_inst = getCustomOp(prod)
                inWidth = prod_inst.get_outstream_width()
                outWidth = prod_inst.get_instream_width()

                n0_out_shape = prod_inst.get_folded_output_shape()

                # mvau has a special case with external memory
                # where we have to consider a different input
                if (
                    node.onnx_node.op_type.startswith("MVAU")
                    and node.get_nodeattr("mem_mode") == "external"
                ) or (node.onnx_node.op_type.startswith("StreamingConcat")):
                    # get input idx
                    in_idx = None
                    for idx, n_input in enumerate(node.onnx_node.input):
                        if output_name == n_input:
                            in_idx = idx
                    assert in_idx is not None, "Malformed model"
                    n1_in_shape = node.get_folded_input_shape(in_idx)
                else:
                    # use default folded input shape
                    n1_in_shape = node.get_folded_input_shape()

                # dwcs cannot be inserted between mvau/vvau and pool/swg
                # so we only run it for other combinations
                if not (
                    (
                        prod.name.startswith("ConvolutionInputGenerator")
                        or prod.name.startswith("Pool")
                    )
                    and (
                        node.onnx_node.name.startswith("Pool")
                        or node.onnx_node.name.startswith("MVAU")
                        or node.onnx_node.name.startswith("VVAU")
                    )
                ):
                    n1_in_shape = node.get_folded_input_shape()

                    # check if we need a DWC
                    if (
                        np.prod(n0_out_shape) != np.prod(n1_in_shape)
                        or n0_out_shape[-1] != n1_in_shape[-1]
                    ):
                        # HLS DWC needed, expensive
                        if (max(inWidth, outWidth) % min(inWidth, outWidth) != 0) or (
                            np.prod(n0_out_shape) != np.prod(n1_in_shape)
                        ):
                            cost += ((inWidth + outWidth) * hls_dwc_cost_penalty) / lut_capacity

                        # RTL DWC can be used cheaply
                        else:
                            cost += (inWidth + outWidth) / lut_capacity

        return cost

    def cost_model(self, param_guess, opt):
        """
        Score a folding configuration for the optimizer (lower is better).

        The score sums each soft target (resource) normalized by its budget and
        scaled by an optional per-resource weight, so a higher weight steers the
        optimizer away from that resource. This is how ``prefer_memory`` /
        ``prefer_compute`` bias BRAM<->URAM and LUT<->DSP. The hard constraint
        (usually ``max_cycles``) is enforced last with a large finite penalty.
        Extra heuristics (e.g. the DWC-avoidance term) are added on top.
        """
        cost = 0

        # 1. apply the folding parameters
        opt.params.set_values(param_guess)
        opt.params.apply_updates(final=False, filter=self.parameters_to_apply)

        # 2. compute results
        cycles = opt.params.get_max_cycles()
        resources = self.get_resources(opt.params.nodes)
        metrics = {**{"max_cycles": cycles}, **resources}

        # 3. update cost based on all minimizable targets
        # the hard constraint (usually max_cycles) enforces
        # which target MUST be met.
        constraint_penalty = 0
        for value_to_minimize in opt.targets:
            if value_to_minimize != opt.hard_constraint_target:
                weight = self.resource_weights.get(value_to_minimize, 1.0)
                cost += weight * metrics[value_to_minimize] / opt.targets[value_to_minimize]
            else:
                # large FINITE penalty (not inf) so the minimizer can still
                # climb out of an infeasible region: if even the initial guess
                # violates the constraint (e.g. target unreachable), an inf cost
                # would abort dual_annealing outright. The penalty grows with the
                # violation ratio and always dominates the resource cost, so any
                # feasible point is preferred over any infeasible one.
                target = opt.targets[value_to_minimize]
                violation = metrics[value_to_minimize] * self.value_to_minimize_relaxation / target
                if violation > 1:
                    constraint_penalty = self.infeasible_penalty * violation

        # 4. Add additional heuristic costs

        # 4.1 DWC heuristic to decrease the use of HLS DWCs
        # which can have massive LUT resource consumption
        # increases. All pairs are considered because
        # we optimize partitions left to right and consider
        # the DWC between a node and its left neighbor
        if self.enable_folding_dwc_heuristic:
            cost += self.compute_hls_dwc_cost(opt.model, opt.params.nodes, opt.targets["LUT"])

        # apply the hard-constraint penalty last so it dominates the cost
        cost += constraint_penalty

        return cost

    def execute_minimizer(self, discrete_args, init_guess):
        """
        the specific minimizer for performing the parameter optimization
        for a single parameter set is called with this function.
        discrete_args are the ordered possible values per parameter; they are
        encoded to a continuous index space for scipy.dual_annealing and the
        objective is cached over decoded solutions to avoid redundant evaluation.
        """
        encoder = GridEncoder(discrete_args)
        bounds = encoder.bounds

        if len(bounds) == 0:
            return np.array(init_guess)

        encoded_init_guess = encoder.encode(init_guess)

        cache = {}

        def objective(encoded, opt):
            if any(np.isnan(v) for v in encoded):
                return float("nan")
            decoded = tuple(encoder.decode(encoded))
            if decoded not in cache:
                cache[decoded] = self.cost_model(decoded, opt)
            return cache[decoded]

        optimal_args = scipy.optimize.dual_annealing(
            func=objective,
            x0=encoded_init_guess,
            maxiter=self.maxiter,
            accept=self.accept,
            visit=self.visit,
            maxfun=self.maxfun_per_parameter * len(init_guess),
            seed=self.seed,
            args=(self,),
            bounds=bounds,
        )

        optimized_params = np.array(encoder.decode(optimal_args.x))

        return optimized_params

    def optimize(
        self,
        initial_guess="max",
        max_nodes_in_partition=2,
        target_parameters=["SIMD", "PE"],
    ):
        """
        A single optimization pass across an entire model
        initial guess can be "min" or "max" for what folding values to use
        at the start of optimization
        min = least folding (makes sense when the hard constraint is resource use)
        max = maximum folding (makes sense when the hard constraint is max_cycles)
        It is critical to select these values in a way that lets the optimizer know
        a legal solution exists for the problem, otherwise it will give up after a set
        number of iterations

        we peform partition splitting in this function
        """

        # 1. Split parameters into partitions to optimize locally.
        # partitioning is node-based via max_nodes_in_partition below; the
        # partitions/param-count split here is vestigial (partitions is reset to
        # 0 before use), so we just build the flat index list.
        indexes = self.params.index_list = [x for x in range(len(self.params.parameters))]

        if initial_guess == "min":
            init_guess = self.params.get_min_vals()
        elif initial_guess == "max":
            init_guess = self.params.get_max_vals()
        self.params.set_values(init_guess)

        self.params.apply_updates(filter=target_parameters)
        self.params.assign_involved_nodes()
        params = self.params.parameters

        # node-based partitioning
        partitions = 0
        old_node_index = 0
        index_partitions = []
        init_guess_partitions = []
        params_partitions = []

        tmp_index_partitions = []
        tmp_init_guess_partitions = []
        tmp_params_partitions = []

        i = 0
        nodes_in_partition = 1
        for param in params:
            if param.name in target_parameters:
                new_node_index = param.node_index

                if new_node_index != old_node_index:
                    nodes_in_partition += 1

                if nodes_in_partition > max_nodes_in_partition:
                    # store set and start a new one
                    if len(tmp_index_partitions) > 0:
                        index_partitions.append(tmp_index_partitions)
                        init_guess_partitions.append(tmp_init_guess_partitions)
                        params_partitions.append(tmp_params_partitions)
                        tmp_index_partitions = []
                        tmp_init_guess_partitions = []
                        tmp_params_partitions = []
                        partitions += 1
                        nodes_in_partition = 1
                if nodes_in_partition <= max_nodes_in_partition:
                    tmp_index_partitions.append(indexes[i])
                    tmp_init_guess_partitions.append(init_guess[i])
                    tmp_params_partitions.append(params[i])

                old_node_index = new_node_index
            i += 1

        # add remaining lefover tail partition
        if len(tmp_index_partitions) > 0:
            if len(tmp_index_partitions) > 0:
                index_partitions.append(tmp_index_partitions)
                init_guess_partitions.append(tmp_init_guess_partitions)
                params_partitions.append(tmp_params_partitions)
                partitions += 1

        # 2. Perform local optimization of partitions
        for p in range(partitions):
            # generate discrete argument list based on possible values
            # this is the input for the scipy minimizer
            discrete_args = []
            for arg in params_partitions[p]:
                discrete_args.append(tuple(arg.possible_values))

            # filter out parameters to the ones of the requested partition
            self.params.index_list = index_partitions[p]
            self.params.assign_involved_nodes()

            # fetch the respective initial list of parameters
            # it is very important that the initial guess is feasible
            # for the minimizer so that the cost_model call returns a non-infinity cost
            # otherwise the optimizer might give up believing there is no solution
            init_guess = init_guess_partitions[p]

            # an initial run to get resource consumption bounds
            if self.init_run:
                optimized_params = init_guess
            else:
                optimized_params = self.execute_minimizer(discrete_args, init_guess)

            # apply final values, adjusting the model accordingly
            self.params.set_values(optimized_params)
            self.params.apply_updates(final=True, filter=target_parameters)

    def get_resources(self, nodes):
        resources = {}
        for n in nodes:
            resources[n] = n.node_res_estimation(self.fpgapart)
        return aggregate_dict_keys(resources)

    def generate_parameter_set(self):
        """Extract every optimizable folding parameter from the model.

        Each hardware node contributes one MetaParameter per foldable attribute
        it declares -- PE / SIMD generically, plus the resource-type attributes
        (ram_style / resType) when resource-type tuning is enabled. The one
        exception is the ConvolutionInputGenerator (SWG): it is fused to the
        layer it feeds, so it is handled by the pairing rules in
        ``_extract_swg_pair`` which also fold that consumer and ask us to skip
        it. See the rule tables near the top of this module.
        """
        graph = self.model.graph
        parameters = []
        skip_next = 0

        for node_index in range(len(graph.node)):
            if skip_next > 0:
                skip_next -= 1
                continue
            node = graph.node[node_index]
            if node is None or node.op_type == "StreamingDataWidthConverter":
                continue
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue

            max_padding = self._max_padding_for_node(node_index)

            if node.op_type in self.SWG_OPS:
                metas, consumed = self._extract_swg_pair(node, node_index, max_padding)
                skip_next = consumed
            else:
                metas = self._extract_node(node, node_index, max_padding)
            parameters.extend(metas)

        self.params = ParameterSet()
        self.params.parameters = parameters

    def _max_padding_for_node(self, node_index):
        """Padding budget for a node, forced to 0 for the model's IO nodes
        unless the user explicitly allows padding them."""
        last_index = len(self.model.graph.node) - 1
        if not self.pad_io_nodes and node_index in (0, last_index):
            return 0
        return self.padding

    # ------------------------------------------------------------------
    # Generic node: fold whichever of PE / SIMD the node declares
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Parameter construction. Every rule below builds the same two objects, and
    # spelling out a ten-keyword constructor each time buried what actually
    # differs between the rules in boilerplate. These two helpers hold the
    # invariants -- the model, and the fact that a folding change on a
    # weight-bearing layer must reshape its weight/threshold tensors -- so each
    # rule reads as the rule and nothing else.
    # ------------------------------------------------------------------

    def _param(
        self,
        name,
        attr,
        value,
        node,
        node_index,
        op_type,
        bound_name=None,
        bound_value=None,
        update_weights=False,
        update_thresholds=False,
        reshapes_io=True,
    ):
        return Parameter(
            name=name,
            target_value_name=attr,
            target_value=value,
            bound_name=bound_name,
            bound_value=bound_value,
            update_threshold_input=update_thresholds,
            update_weights_input=update_weights,
            update_input_tensor_shape=reshapes_io,
            update_output_tensor_shape=reshapes_io,
            node=node,
            node_index=node_index,
            op_type=op_type,
            model=self.model,
        )

    def _meta(self, name, values, real_values, node_index):
        return MetaParameter(
            name=name,
            meta_value=values[0],
            possible_values=list(values),
            real_values=real_values,
            model=self.model,
            node_index=node_index,
        )

    def _extract_node(self, node, node_index, max_padding):
        inst = getCustomOp(node)
        op_type = node.op_type
        declared = inst.get_nodeattr_types()
        metas = []

        for param in ("SIMD", "PE"):
            if param not in declared or (op_type, param) in self.DO_NOT_FOLD:
                continue
            try:
                metas.append(self._make_folding_meta(inst, node_index, op_type, param, max_padding))
            except Exception as e:
                # be robust to op types the generic extraction cannot size: leave
                # them at their current folding rather than aborting the pass
                warnings.warn(
                    f"SetFolding: could not extract {param} for {op_type} "
                    f"(node {node_index}), leaving it unfolded: {e}"
                )

        # resource-type selection the optimizer tunes to trade BRAM<->URAM and
        # LUT<->DSP while fitting the folding within the device budget
        for param in self.resource_type_params:
            if param in declared:
                meta = self._make_resource_type_meta(inst, node_index, op_type, param)
                if meta is not None:
                    metas.append(meta)
        return metas

    def _make_folding_meta(self, inst, node_index, op_type, param, max_padding):
        """Build the MetaParameter that sweeps ``param`` (PE or SIMD) of a node
        over its legal folding factors, padding the channel count when allowed."""
        max_value = self.max_folding_factor(inst, op_type, param)
        bound_attr = self.PADDING_BOUND_ATTR.get((op_type, param))
        padding = max_padding if bound_attr is not None else 0

        possible_values, bounding_values = allowed_divisors(max_value, 1, padding)
        update_weights = op_type in self.OPS_WITH_WEIGHTS
        update_thresholds = op_type in self.OPS_WITH_THRESHOLDS

        seen_factors = []
        kept_values = []
        real_values = []
        for value, bounding_value in zip(possible_values, bounding_values):
            # a VVAU folds SIMD over its (unpaddable) kernel window
            if param == "SIMD" and op_type in self.SIMD_MAX_OVERRIDE:
                bounding_value = inst.get_nodeattr("Kernel")
            factor = int(np.prod(bounding_value)) // value
            if factor in seen_factors:
                continue
            seen_factors.append(factor)
            kept_values.append(value)
            real_values.append(
                [
                    self._param(
                        f"{op_type}_{param}",
                        param,
                        value,
                        inst,
                        node_index,
                        op_type,
                        bound_name=bound_attr,
                        bound_value=bounding_value,
                        update_weights=update_weights,
                        update_thresholds=update_thresholds,
                    )
                ]
            )

        return self._meta(param, kept_values, real_values, node_index)

    def _make_resource_type_meta(self, inst, node_index, op_type, param):
        """Build the MetaParameter that lets the optimizer pick a node's
        ram_style / resType (used to trade BRAM<->URAM and LUT<->DSP). Values the
        estimator cannot cost (see self.RESOURCE_TYPE_EXCLUDED_VALUES) are dropped, and
        the attribute is skipped entirely if the resource estimator is blind to it
        (all choices give the same estimate) -- optimizing such a lever would only
        waste search time and pick an estimator-unjustified value."""
        excluded = set(self.RESOURCE_TYPE_EXCLUDED_VALUES.get(param, set()))
        # URAM weight memories require runtime_writeable_weights=1 on Ultrascale
        # (asserted in MVAU/VVAU generate_infra_hdl); don't let the optimizer
        # pick "ultra" for a weight ram_style unless the user opted in. Nodes
        # without the attr (e.g. SWG line buffers) are unaffected.
        if param == "ram_style" and "runtime_writeable_weights" in inst.get_nodeattr_types():
            if inst.get_nodeattr("runtime_writeable_weights") != 1 and not self.allow_uram_weights:
                excluded = excluded | {"ultra"}
        allowed = sorted(set(inst.get_nodeattr_types()[param][3]) - excluded)
        if len(allowed) < 2:
            return None
        if not self._estimator_distinguishes(inst, param, allowed):
            return None
        real_values = [
            [self._param(param, param, choice, inst, node_index, op_type, reshapes_io=False)]
            for choice in allowed
        ]
        return self._meta(param, allowed, real_values, node_index)

    def _estimator_distinguishes(self, inst, param, allowed):
        """Whether node_res_estimation reacts to ``param`` for this node, i.e. at
        least two allowed values give different resource estimates. Used to skip
        resource-type levers the estimator is blind to."""
        original = inst.get_nodeattr(param)
        estimates = set()
        try:
            for choice in allowed:
                inst.set_nodeattr(param, choice)
                estimates.add(tuple(sorted(inst.node_res_estimation(self.fpgapart).items())))
        except Exception:
            return False
        finally:
            inst.set_nodeattr(param, original)
        return len(estimates) > 1

    # ------------------------------------------------------------------
    # Special pairing rules: a ConvolutionInputGenerator (SWG) must be folded
    # in lockstep with the layer it feeds. Each rule below is self-contained so
    # rules can be added or removed without touching the others.
    # ------------------------------------------------------------------
    def _extract_swg_pair(self, swg_node, node_index, max_padding):
        """Route an SWG to the pairing rule for its (fused) consumer and return
        ``(meta_parameters, nodes_to_skip)`` -- the consumer is folded jointly,
        so the caller skips it."""
        consumer = self.model.graph.node[node_index + 1]
        swg = getCustomOp(swg_node)
        consumer_inst = getCustomOp(consumer)

        if consumer.op_type in ("Pool_rtl", "Pool_hls"):
            metas = self._pair_swg_with_pool(
                swg, swg_node, consumer_inst, consumer, node_index, max_padding
            )
        elif consumer.op_type in ("MVAU_hls", "MVAU_rtl"):
            metas = self._pair_swg_with_mvau(
                swg, swg_node, consumer_inst, consumer, node_index, max_padding
            )
        elif consumer.op_type in ("VVAU_hls", "VVAU_rtl"):
            metas = self._pair_swg_with_vvau(
                swg, swg_node, consumer_inst, consumer, node_index, max_padding
            )
        else:
            raise AssertionError(
                "ConvolutionInputGenerator feeds unsupported op "
                f"{consumer.op_type}; expected Pool/MVAU/VVAU"
            )
        return metas, 1

    def _swg_simd_choices(self, swg, max_padding):
        """Legal SWG SIMD values and the (possibly padded) IFMChannels each
        implies, shared by all three pairing rules."""
        ifm_channels = swg.get_nodeattr("IFMChannels")
        kernel_size = int(np.prod(swg.get_nodeattr("ConvKernelDim")))
        simd_values, ifm_values = allowed_divisors(ifm_channels, 1, max_padding)
        return list(simd_values), list(ifm_values), kernel_size

    def _swg_params(self, swg, swg_node, node_index, simd, ifm_channels, parallel_window=None):
        """The SWG side of every pairing rule: its SIMD, bounded by the (possibly
        padded) channel count, and -- for the MVAU/VVAU rules -- whether it emits
        the whole kernel window at once. Order matters: the consumer's parameters
        are appended after these."""
        params = [
            self._param(
                "SWU_SIMD",
                "SIMD",
                simd,
                swg,
                node_index,
                swg_node.op_type,
                bound_name="IFMChannels",
                bound_value=ifm_channels,
            )
        ]
        if parallel_window is not None:
            params.append(
                self._param(
                    "SWU_parallel_window",
                    "parallel_window",
                    parallel_window,
                    swg,
                    node_index,
                    swg_node.op_type,
                )
            )
        return params

    def _pair_swg_with_pool(self, swg, swg_node, pool, pool_node, node_index, max_padding):
        """SWG -> Pool (max pooling): SWG SIMD and Pool PE move together and share
        the same (padded) channel count."""
        assert swg.get_nodeattr("depthwise") == 1
        simd_values, ifm_values, _ = self._swg_simd_choices(swg, max_padding)

        real_values = [
            self._swg_params(swg, swg_node, node_index, simd, ifm)
            + [
                self._param(
                    "Pool_PE",
                    "PE",
                    simd,
                    pool,
                    node_index + 1,
                    pool_node.op_type,
                    bound_name="Channels",
                    bound_value=ifm,
                )
            ]
            for simd, ifm in zip(simd_values, ifm_values)
        ]
        return [self._meta("SIMD", simd_values, real_values, node_index)]

    def _pair_swg_with_mvau(self, swg, swg_node, mvau, mvau_node, node_index, max_padding):
        """SWG -> MVAU (dense convolution): optimize SWG SIMD together with the
        MVAU SIMD (folding MW = kernel * IFMChannels), plus the MVAU PE (folding
        MH) as an independent parameter. parallel_window on the SWG is enabled
        once the MVAU SIMD reaches the full input channel count."""
        simd_values, ifm_values, kernel_size = self._swg_simd_choices(swg, max_padding)
        weight_bits = mvau.get_input_datatype(1).bitwidth()

        # --- joint SWG SIMD + MVAU SIMD (over MW) ---
        simd_real_values = []
        simd_meta_values = []
        seen_factors = []
        for ifm_channels in set(ifm_values):
            mw = kernel_size * ifm_channels
            mvau_simd_values, mw_values = allowed_divisors(mw, 1, 0)
            for mvau_simd, mw_value in zip(mvau_simd_values, mw_values):
                # mvau_simd <= ifm_channels folds within one kernel position
                # (parallel_window=0); mvau_simd a multiple of ifm_channels needs
                # the SWG to emit the full window (parallel_window=1), which lets
                # the MVAU fold over the whole MW (up to kernel*IFMChannels).
                channels_ok = (ifm_channels % mvau_simd == 0) or (mvau_simd % ifm_channels == 0)
                if not (
                    channels_ok
                    and mvau_simd not in simd_meta_values
                    and mw_value // mvau_simd not in seen_factors
                    and (weight_bits * mvau_simd) < self.mvau_wwidth_max
                    and mvau_simd > (mw_value / 1024)
                ):
                    continue
                simd_meta_values.append(mvau_simd)
                seen_factors.append(mw_value // mvau_simd)

                parallel_window = int(mvau_simd >= ifm_channels)
                swg_simd_value = ifm_channels if parallel_window else mvau_simd
                simd_real_values.append(
                    self._swg_params(
                        swg, swg_node, node_index, swg_simd_value, ifm_channels, parallel_window
                    )
                    + [
                        self._param(
                            "MVAU_SIMD",
                            "SIMD",
                            mvau_simd,
                            mvau,
                            node_index + 1,
                            mvau_node.op_type,
                            bound_name="MW",
                            bound_value=mw_value,
                            update_weights=True,
                            update_thresholds=True,
                        )
                    ]
                )

        simd_meta = self._meta("SIMD", simd_meta_values, simd_real_values, node_index)

        # --- independent MVAU PE (over MH); a conv MVAU's MH is never padded ---
        # allowed_divisors already returns one value per distinct folding
        # factor, so no extra de-duplication is needed here.
        mh = mvau.get_nodeattr("MH")
        pe_values, mh_values = allowed_divisors(mh, 1, 0)
        pe_real_values = [
            [
                self._param(
                    "MVAU_PE",
                    "PE",
                    pe,
                    mvau,
                    node_index + 1,
                    mvau_node.op_type,
                    bound_name="MH",
                    bound_value=mh_value,
                    update_weights=True,
                    update_thresholds=True,
                )
            ]
            for pe, mh_value in zip(pe_values, mh_values)
        ]
        return [simd_meta, self._meta("PE", pe_values, pe_real_values, node_index)]

    def _pair_swg_with_vvau(self, swg, swg_node, vvau, vvau_node, node_index, max_padding):
        """SWG -> VVAU (depthwise convolution): a single meta parameter drives
        VVAU PE (= SWG SIMD, over the channels) and VVAU SIMD (over the kernel
        window), enabling SWG parallel_window once PE saturates the channels."""
        assert swg.get_nodeattr("depthwise") == 1
        _, ifm_values, _ = self._swg_simd_choices(swg, max_padding)
        kernel_dim = swg.get_nodeattr("ConvKernelDim")

        real_values = []
        meta_values = []
        seen_pe_factors = []
        for ifm_channels in set(ifm_values):
            # PE folds the channels; its SIMD cannot be padded (it is a kernel size)
            pe_values, pe_bounds = allowed_divisors(ifm_channels, 1, 0)
            for pe, pe_bound in zip(pe_values, pe_bounds):
                pe_factor = pe_bound // pe
                if pe_factor in seen_pe_factors:
                    continue
                seen_pe_factors.append(pe_factor)

                if pe < ifm_channels:
                    simd_limit = 1
                    parallel_window = 0
                else:
                    simd_limit = int(np.prod(kernel_dim))
                    parallel_window = 1

                vvau_simd_values, _ = allowed_divisors(simd_limit, 1, 0)
                seen_simd_factors = []
                for vvau_simd in vvau_simd_values:
                    if (
                        vvau_simd * pe in meta_values
                        or int(np.prod(kernel_dim)) // vvau_simd in seen_simd_factors
                    ):
                        continue
                    meta_values.append(vvau_simd * pe)
                    seen_simd_factors.append(int(np.prod(kernel_dim)) // vvau_simd)

                    real_values.append(
                        self._swg_params(
                            swg, swg_node, node_index, pe, ifm_channels, parallel_window
                        )
                        + [
                            self._param(
                                "VVAU_SIMD",
                                "SIMD",
                                vvau_simd,
                                vvau,
                                node_index + 1,
                                vvau_node.op_type,
                                bound_name="Kernel",
                                bound_value=[kernel_dim[0], kernel_dim[1]],
                                update_weights=True,
                                update_thresholds=True,
                            ),
                            self._param(
                                "VVAU_PE",
                                "PE",
                                pe,
                                vvau,
                                node_index + 1,
                                vvau_node.op_type,
                                bound_name="Channels",
                                bound_value=pe_bound,
                                update_weights=True,
                                update_thresholds=True,
                            ),
                        ]
                    )

        return [self._meta("SIMD", meta_values, real_values, node_index)]


def insert_and_size_fifos(
    model_dir, model, board, fpga_part, consider_dwc_costs, auto_fifo_strategy
):
    """
    force a fifo sizing step after folding to test the resource consumption
    and throughput changes introduced by fifo sizing. This pass must be
    performed using tree-based TAV generation. Otherwise,
    it will take an extremely long amount of time.
    """
    # deferred: build_dataflow_steps imports SetFolding, so a top-level import
    # here would be circular
    from finn.builder.build_dataflow_steps import step_set_fifo_depths  # noqa: PLC0415

    if not consider_dwc_costs:
        model = model.transform(InsertDWC())

    # this scoring copy is sized before step_minimize_bit_width has run, but
    # RTL Thresholding codegen (needed by the sizer's characterization pass)
    # requires integer thresholds -- round them here like the real flow will
    model = model.transform(RoundAndClipThresholds())

    # per this function's contract, the in-loop sizing must run tree-model TAV
    # generation and stay synthesis-free: the default (rtlsim characterization,
    # trailing re-synth) invokes vitis_hls per node per scoring candidate. The
    # TAV knobs only exist on trees that carry the analytic FIFO sizer.
    tav_kwargs = {}
    try:
        # deferred on purpose: these enums only exist on trees that carry the
        # analytic FIFO sizer, so this is a feature probe rather than a dependency
        from finn.builder.build_dataflow_config import (  # noqa: PLC0415
            TAVGenerationMethod,
            TAVUtilizationMethod,
        )

        tav_kwargs = dict(
            tav_generation_strategy=TAVGenerationMethod.TREE_MODEL,
            tav_utilization_strategy=TAVUtilizationMethod.CONSERVATIVE_RELAXATION,
            skip_resynth_during_fifo_sizing=True,
        )
    except ImportError:
        pass

    cfg = DataflowBuildConfig(
        output_dir="",
        auto_fifo_depths=True,
        split_large_fifos=True,
        auto_fifo_strategy=auto_fifo_strategy,
        folding_config_file=None,
        synth_clk_period_ns=5.0,
        fpga_part=fpga_part,
        steps=["step_set_fifo_depths"],
        generate_outputs=[],
        board=board,
        extract_hw_config=False,
        **tav_kwargs,
    )

    model = step_set_fifo_depths(model, cfg)

    return model


def common_divisors(numbers):
    separate_divisors = []
    for num in numbers:
        individual_divisors = list(divisors(num))
        separate_divisors.append(individual_divisors)

    return functools.reduce(np.intersect1d, separate_divisors)


# Find the op-type names for all HLS specializations of elementwise binary
# operations
ELEMENTWISE_BINARY_OPS = [
    op_type
    for op_type, cls in inspect.getmembers(elementwise_binary_hls, inspect.isclass)
    if issubclass(cls, elementwise_binary_hls.ElementwiseBinaryOperation_hls)
]


class SetFolding(Transformation):

    """
    Set the parallelism (folding) attributes of every hardware node in a FINN
    dataflow graph. Each node's parallelism is one of {PE, SIMD}; SetFolding
    reads whichever the node declares and respects its divisibility constraints.

    Two styles are available (``style``):

    * ``"optimizer"`` (default) -- a resource-aware search (simulated annealing)
      that, depending on ``target_cycles_per_frame``, either minimizes resources
      while meeting a throughput target, or maximizes throughput within the
      device resource budget (``target_cycles_per_frame=None``). It can also bias
      BRAM<->URAM / LUT<->DSP usage via ``prefer_memory`` / ``prefer_compute``.
    * ``"naive"`` -- the legacy greedy per-node folder.

    The optimizer folds every node generically; genuinely op-specific behavior is
    kept as a few small, explicit rules near the top of this module (SWG pairing,
    SIMD_MAX_OVERRIDE, DO_NOT_FOLD, PADDING_BOUND_ATTR) that are easy to extend.

    If ``folding_maximum_padding`` is greater than 0, folding-factor restrictions
    are relaxed by padding channel counts where that helps; padding & cropping
    DWCs are inserted downstream as needed.

    In the returned model, each node's cycles_estimate attribute is set to its
    estimated number of cycles.
    """

    # Invariants, not knobs. They were instance attributes assigned in __init__,
    # which read as configuration and invited callers to flip them; nothing in
    # the flow ever did, and two of them cannot be flipped safely.

    # the folders apply() can dispatch to; also the accepted values of the build
    # config's `folding_style`, so keep the two in step
    STYLES = ("optimizer", "naive")

    # the throughput target is always the cost model's hard constraint
    hard_constraint_target = "max_cycles"
    target_resources = ["LUT", "BRAM_18K", "DSP", "URAM"]

    # binary-search steps when maximizing throughput within a resource budget
    # (target_cycles_per_frame=None)
    MAXIMIZE_SEARCH_STEPS = 15
    # cost-model weight applied to the resource the caller asked to spare via
    # prefer_memory / prefer_compute (higher = avoided more strongly)
    PREFERENCE_PENALTY = 8.0
    # nodes grouped per local optimization partition in the folding pass
    NODES_PER_PARTITION = 3
    # nodes grouped per partition in the (larger) resource-type tuning pass
    RESOURCE_PARTITION_SIZE = 8
    # Always tune each node's ram_style/resType so the search can trade
    # BRAM<->URAM and LUT<->DSP to fit a higher-throughput folding in budget.
    # These attributes do not enter any cycle formula, so this can only change
    # how resources are realized, never the achieved throughput.
    optimize_resource_types = True
    resource_type_params = list(Optimizer.RESOURCE_TYPE_ATTRS)
    optimize_folding = True
    # DWCs are inserted by a later build step; folding only accounts for them
    insert_dwcs = False
    consider_dwc_costs = True

    def __init__(
        self,
        target_cycles_per_frame=None,
        platform="Pynq-Z1",
        devices=1,
        style="optimizer",
        # --- resource preferences -------------------------------------------
        prefer_memory=None,
        prefer_compute=None,
        resource_weights=None,
        # --- folding search knobs -------------------------------------------
        folding_effort=250,
        folding_maximum_padding=0,
        folding_pad_io_nodes=False,
        folding_max_attempts=1,
        strict_budget=False,
        allow_uram_weights=False,
        folding_search_timeout_s=None,
        mvau_wwidth_max=1024,
        enable_folding_dwc_heuristic=True,
        enable_folding_fifo_heuristic=False,
        auto_fifo_strategy="analytic",
        # --- naive-style knobs (style="naive" only) -------------------------
        two_pass_relaxation=True,
        # --- simulated-annealing internals (optimizer style only) -----------
        # None means "use Optimizer's own default"; the values live there, in the
        # class that consumes them, rather than being restated here
        maxiter=None,
        accept=None,
        visit=None,
        seed=None,
        verbose=False,
    ):
        """
        target_cycles_per_frame
            A throughput ceiling to meet at minimum resource cost. ``None``
            switches the objective: maximize throughput within the budget.
        platform / devices
            Board (and device count) whose resource budget bounds the search.
        style
            ``"optimizer"`` (resource-aware search) or ``"naive"`` (the legacy
            greedy per-node folder in ``set_folding_naive``).
        prefer_memory / prefer_compute / resource_weights
            Bias, not permission: ram_style and resType are always tuned so the
            search can trade BRAM<->URAM and LUT<->DSP. ``"bram"``/``"uram"``
            and ``"lut"``/``"dsp"`` penalize the resource NOT named;
            ``resource_weights`` overrides both with explicit multipliers.
        folding_effort
            Annealing evaluations per parameter; 50-100 is the useful range.
        folding_maximum_padding / folding_pad_io_nodes
            Opt-in channel padding for finer folding factors, and whether the
            IO layers may be padded too (if so, the host must pad its input and
            crop its output to match). Both off by default.

        Three that are easy to get wrong:

        allow_uram_weights
            Lets *weight* memories go to URAM, which requires
            ``runtime_writeable_weights=1``. Off by default because that changes
            the deployment contract -- the driver must load those weights rather
            than the bitstream carrying them. The cost of leaving it off is real:
            on a part where BRAM is the binding constraint there may be no
            feasible folding without it (MobileNet on ZCU104 fits only by moving
            three MVAUs to URAM this way).
        strict_budget
            Raise rather than warn when the returned folding exceeds the budget.
            Either way the verdict is recorded in ``last_search_report``; an
            over-budget folding is never returned silently.
        folding_search_timeout_s
            Wall-clock budget for the maximize-mode search. Fast targets cost far
            more to probe than slow ones (every evaluation reshapes the weight
            tensors), so the search goes cheapest-first and keeps the best
            feasible point found when the budget runs out. Truncation is reported
            as ``search_truncated``, never hidden. ``None`` means no limit.

        In the returned model each node's cycles_estimate attribute is set to
        its estimated number of cycles.
        """
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.platform = platform
        self.devices = devices
        self.fpgapart = part_map[self.platform]
        self.style = style

        self.mvau_wwidth_max = mvau_wwidth_max
        self.padding = folding_maximum_padding
        self.pad_io_nodes = folding_pad_io_nodes
        self.max_attempts = folding_max_attempts
        self.strict_budget = strict_budget
        self.allow_uram_weights = allow_uram_weights
        self.search_timeout_s = folding_search_timeout_s
        self.effort = folding_effort
        # populated by the throughput search; see _record_search_result
        self.last_search_report = None
        self.enable_folding_dwc_heuristic = enable_folding_dwc_heuristic
        self.enable_folding_fifo_heuristic = enable_folding_fifo_heuristic
        self.auto_fifo_strategy = auto_fifo_strategy

        # simulated-annealing internals
        self.maxiter = maxiter
        self.accept = accept
        self.visit = visit
        self.seed = seed
        self.verbose = verbose

        # naive-style only
        self.two_pass_relaxation = two_pass_relaxation

        # prefer_memory / prefer_compute only bias the resource-type choice via
        # cost-model weights; they are neutral when left as None.
        self.resource_weights = self._build_resource_weights(
            prefer_memory, prefer_compute, resource_weights
        )

    @staticmethod
    def _build_resource_weights(prefer_memory, prefer_compute, overrides):
        """Turn the prefer_memory / prefer_compute shorthands (and any explicit
        overrides) into per-resource cost multipliers. A higher weight steers
        the optimizer away from that resource, so we penalize the resource the
        user did NOT prefer."""
        weights = {"LUT": 1.0, "BRAM_18K": 1.0, "DSP": 1.0, "URAM": 1.0}
        if prefer_memory == "bram":
            weights["URAM"] = SetFolding.PREFERENCE_PENALTY
        elif prefer_memory == "uram":
            weights["BRAM_18K"] = SetFolding.PREFERENCE_PENALTY
        elif prefer_memory is not None:
            raise ValueError("prefer_memory must be 'bram', 'uram' or None")
        if prefer_compute == "lut":
            weights["DSP"] = SetFolding.PREFERENCE_PENALTY
        elif prefer_compute == "dsp":
            weights["LUT"] = SetFolding.PREFERENCE_PENALTY
        elif prefer_compute is not None:
            raise ValueError("prefer_compute must be 'lut', 'dsp' or None")
        if overrides:
            weights.update(overrides)
        return weights

    def _make_optimizer(self, model, name, targets, init_run):
        """Construct an Optimizer sharing this transformation's configuration."""
        return Optimizer(
            model,
            name,
            targets,
            self.hard_constraint_target,
            padding=0 if init_run else self.padding,
            fpgapart=self.fpgapart,
            maxfun_per_parameter=self.effort,
            enable_folding_dwc_heuristic=self.enable_folding_dwc_heuristic,
            verbose=self.verbose,
            mvau_wwidth_max=self.mvau_wwidth_max,
            init_run=init_run,
            pad_io_nodes=self.pad_io_nodes,
            resource_type_params=self.resource_type_params,
            resource_weights=self.resource_weights,
            allow_uram_weights=self.allow_uram_weights,
            seed=self.seed,
            # only override what the caller actually set
            **{
                k: v
                for k, v in (
                    ("maxiter", self.maxiter),
                    ("accept", self.accept),
                    ("visit", self.visit),
                )
                if v is not None
            },
        )

    def _probe_fastest_throughput(self, model):
        """Fold every node maximally to measure the fastest (fewest cycles per
        frame) the graph can run -- the floor the throughput search cannot beat."""
        probe = self._make_optimizer(
            copy.deepcopy(model), "throughput_probe", {"max_cycles": 1}, True
        )
        probe.generate_parameter_set()
        probe.optimize(
            max_nodes_in_partition=1, target_parameters=["SIMD", "PE", "parallel_window"]
        )
        probed = probe.model.transform(AnnotateCycles())
        return probed.analysis(dataflow_performance)["max_cycles"]

    def _resource_budgets(self):
        """Per-resource budget for the target platform (raw counts scaled by the
        default utilization limits)."""
        limits = DEFAULT_RES_LIMITS
        totals = {"LUT": 0, "BRAM_18K": 0, "URAM": 0, "DSP": 0}
        for r in platforms[self.platform](self.devices).resource_count_dict.values():
            for key in totals:
                totals[key] += r[key]
        return {
            "LUT": max(limits[0] * totals["LUT"], 0.001),
            "BRAM_18K": max(limits[2] * totals["BRAM_18K"], 0.001),
            "URAM": max(limits[3] * totals["URAM"], 0.001),
            "DSP": max(limits[4] * totals["DSP"], 0.001),
        }

    def _aggregate_resources(self, model):
        estimates = {}
        for node in model.graph.node:
            inst = getCustomOp(node)
            estimates[inst] = inst.node_res_estimation(self.fpgapart)
        return aggregate_dict_keys(estimates)

    def _as_built_if_useful(self, model, metrics, targets):
        """Re-score on the as-built graph only when it can change the verdict.

        If the search's own scoring already exceeds the budget, the as-built
        graph cannot rescue it -- the verdict is "does not fit" either way -- so
        the extra minimisation pass is pure cost. It is precisely the infeasible
        fallback (minimal folding, deepest/narrowest weight memories) on which
        MinimizeAccumulatorWidth is slowest, so skipping it there matters.
        """
        if any(metrics.get(r, 0.0) > targets[r] for r in self.target_resources):
            return None
        try:
            return self._resources_as_built(model)
        except Exception as e:  # noqa: BLE001 - never lose the result over a verdict
            warnings.warn(
                f"SetFolding: could not re-score the folding on the as-built graph "
                f"({type(e).__name__}: {e}); the budget verdict is the search's own "
                f"scoring, which does not include step_minimize_bit_width.",
                stacklevel=2,
            )
            return None

    def _resources_as_built(self, model):
        """Resources of the graph the *build* will actually carry.

        The search scores candidates on the freshly folded graph, but every real
        flow runs step_minimize_bit_width immediately after folding, and that
        step does not only shrink things: MinimizeAccumulatorWidth and
        RoundAndClipThresholds can widen accumulators and thresholds. Measured on
        cnv-w1a2 (Pynq-Z1), a folding SetFolding scored at 33428 LUT / 142 BRAM
        -- comfortably inside a 37240 / 224 budget -- becomes 40025 LUT / 278
        BRAM once those run, i.e. over budget on both. Reporting the pre-step
        numbers would therefore certify designs that do not fit.

        Used for the *verdict* only, not inside the search loop, so it costs one
        extra pass per SetFolding call rather than one per candidate.
        """
        m = copy.deepcopy(model).transform(GiveUniqueNodeNames())
        m = m.transform(MinimizeWeightBitWidth())
        m = m.transform(MinimizeAccumulatorWidth())
        m = m.transform(InferDataTypes())
        return self._aggregate_resources(m)

    def apply_optimized_folding(self, model):
        """
        Resource-aware folding using simulated annealing.

        Depending on ``target_cycles_per_frame`` this either minimizes resources
        while meeting a throughput target, or maximizes throughput within the
        platform resource budget. Either way the search runs the cost model over
        candidate PE/SIMD (and, when preferences are set, ram_style/resType)
        settings, optionally accounting for DWC and FIFO costs.
        """
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        maximize = self.target_cycles_per_frame is None

        # fastest folding the graph allows + the platform resource budget
        fastest_cycles = self._probe_fastest_throughput(model)
        budgets = self._resource_budgets()

        attempts = (
            max(self.max_attempts, self.MAXIMIZE_SEARCH_STEPS) if maximize else self.max_attempts
        )

        # max_cycles is a placeholder here; _fold_at sets it per candidate target
        targets = {"max_cycles": fastest_cycles, **budgets}
        opt_template = self._make_optimizer(model, "folding", targets, False)
        opt_template.generate_parameter_set()

        model = self._search_folding(opt_template, targets, fastest_cycles, attempts, maximize)

        if self.insert_dwcs:
            # only needed if downstream steps will not insert DWCs themselves
            model = model.transform(InsertDWC())
            model = model.transform(SpecializeLayers(self.fpgapart))

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        if self.allow_uram_weights:
            self._enable_runtime_writeable_for_uram(model)

        if self.pad_io_nodes:
            model = self._retype_padded_io(model)

        return (model, False)

    @staticmethod
    def _sweep_targets(lo, hi, n):
        """``n`` cycles-per-frame probes spread geometrically over [lo, hi].

        Geometric, not linear: the two ends routinely differ by four or five
        orders of magnitude (a maximally folded MobileNet is ~1e5 cycles/frame,
        the minimally folded one ~5e7), so linear spacing would put every probe
        at the slow end and never look at the interesting region.
        """
        lo, hi = max(int(lo), 1), max(int(hi), 1)
        if hi <= lo or n <= 1:
            return [hi]
        ratio = (hi / lo) ** (1.0 / (n - 1))
        out = []
        for i in range(n):
            t = int(round(lo * (ratio**i)))
            t = min(max(t, lo), hi)
            if t not in out:
                out.append(t)
        if hi not in out:
            out.append(hi)
        return sorted(out)

    def _search_folding(self, opt_template, targets, fastest_cycles, attempts, maximize):
        """Fold the model, choosing the throughput target.

        * meet-target mode (a target was given, single attempt): fold once to meet
          the target with minimal resources.
        * otherwise: find the fastest throughput (fewest cycles per frame) whose
          folding still fits the device resource budget.

        The search does NOT assume that resource use is monotonic in the
        throughput target, because it is not. BRAM in particular is
        non-monotonic: at minimal folding each weight memory is deep and narrow
        and therefore packs into BRAM badly, so *more* parallelism can cost
        *less* BRAM. MobileNet on a ZCU104 is the worked example -- it needs
        990-1040 BRAM_18K at minimum folding against a 488 budget, but only ~530
        at the far more parallel hand-tuned folding.

        A plain bisection on "fits -> go faster, else go slower" silently breaks
        on that shape: the slow end is infeasible, so the invariant it relies on
        never holds, no feasible point is ever recorded, and it falls through to
        the slowest folding -- an answer that is both slow *and* over budget.

        So instead: sample feasibility across the whole range first (phase A),
        then refine around the best feasible point found (phase B). Only a
        candidate that was *verified* feasible can ever be returned as the
        answer, so a wrong monotonicity guess can cost search efficiency but
        cannot produce a wrong result.
        """
        if not maximize and attempts <= 1:
            model, fits, metrics, achieved = self._fold_at(
                opt_template, targets, self.target_cycles_per_frame
            )
            self._record_search_result(
                fits,
                metrics,
                targets,
                achieved,
                feasible_found=fits,
                searched=False,
                as_built=self._as_built_if_useful(model, metrics, targets),
            )
            return model

        slowest = (
            self.target_cycles_per_frame if not maximize else self._slowest_cycles(opt_template)
        )
        lo, hi = max(int(fastest_cycles), 1), max(int(slowest), int(fastest_cycles), 1)

        # split the budget between sampling and refinement
        n_sweep = max(4, (attempts + 1) // 2)
        n_refine = max(0, attempts - n_sweep)

        best = None  # (achieved_cycles, model, metrics) -- verified feasible
        fallback = None  # (achieved_cycles, model, metrics) at the slow end
        infeasible_below = None  # fastest probed target known NOT to fit
        deadline = None if self.search_timeout_s is None else time.time() + self.search_timeout_s
        self._search_truncated = False

        def out_of_time():
            if deadline is not None and time.time() > deadline:
                self._search_truncated = True
                return True
            return False

        # ---- phase A: sample feasibility across the range ------------------
        # Slowest (cheapest to evaluate) first: a fast target drives every node
        # to maximum parallelism, and each cost-model evaluation then reshapes
        # the full weight tensors. Cheapest-first means that if the budget runs
        # out we still hold the feasible points, rather than having spent it all
        # on the expensive end.
        for t in reversed(self._sweep_targets(lo, hi, n_sweep)):
            if out_of_time():
                break
            model, fits, metrics, achieved = self._fold_at(opt_template, targets, t)
            if fits:
                if best is None or achieved < best[0]:
                    best = (achieved, model, metrics)
            else:
                if infeasible_below is None or t > infeasible_below:
                    # the slowest target that still failed: nothing at or below
                    # it is worth refining into
                    infeasible_below = t
            if fallback is None or achieved > fallback[0]:
                fallback = (achieved, model, metrics)

        # ---- phase B: refine between the best feasible point and the ------
        # fastest infeasible one below it
        if best is not None and n_refine:
            fast_bound = 1 if infeasible_below is None else infeasible_below + 1
            slow_bound = best[0]
            for _ in range(n_refine):
                if slow_bound - fast_bound <= 1 or out_of_time():
                    break
                mid = (fast_bound + slow_bound) // 2
                model, fits, metrics, achieved = self._fold_at(opt_template, targets, mid)
                if fits:
                    if achieved < best[0]:
                        best = (achieved, model, metrics)
                    slow_bound = min(mid, achieved)
                else:
                    fast_bound = mid + 1

        if best is not None:
            self._record_search_result(
                True,
                best[2],
                targets,
                best[0],
                feasible_found=True,
                as_built=self._as_built_if_useful(best[1], best[2], targets),
            )
            return best[1]

        # Nothing in the sampled range fits. Return the least-resource folding
        # we saw, but say so loudly -- this is not a solution to the problem
        # that was asked.
        self._record_search_result(
            False,
            fallback[2],
            targets,
            fallback[0],
            feasible_found=False,
            as_built=self._as_built_if_useful(fallback[1], fallback[2], targets),
        )
        return fallback[1]

    def _record_search_result(
        self,
        fits,
        metrics,
        targets,
        achieved_cycles,
        feasible_found,
        searched=True,
        as_built=None,
    ):
        """Publish the budget verdict instead of leaving the caller to guess.

        Before this, maximize mode could return a folding that exceeded the
        device budget with no signal whatsoever: the transformation looked like
        it had succeeded. Anyone using SetFolding outside a harness that
        re-derives the resource estimate itself had no way to tell.
        """
        as_built = metrics if as_built is None else as_built
        # judge on the as-built graph: what the search scored is not what the
        # build ships (see _resources_as_built)
        over = {
            r: (as_built.get(r, 0.0), targets[r])
            for r in self.target_resources
            if as_built.get(r, 0.0) > targets[r]
        }
        scored_over = {
            r: metrics.get(r, 0.0)
            for r in self.target_resources
            if metrics.get(r, 0.0) > targets[r]
        }
        fits = fits and not over
        self.last_search_report = {
            "fits_budget": bool(fits and not over),
            "resources_as_built": {r: as_built.get(r, 0.0) for r in self.target_resources},
            "feasible_solution_found": bool(feasible_found),
            "achieved_cycles_per_frame": achieved_cycles,
            "resources": {r: metrics.get(r, 0.0) for r in self.target_resources},
            "budget": {r: targets[r] for r in self.target_resources},
            "over_budget": {r: {"used": u, "budget": b} for r, (u, b) in over.items()},
            # True when the search's own scoring thought it fitted but the
            # as-built graph does not -- the blind spot, made visible
            "missed_by_search_scoring": bool(over and not scored_over),
            "search_truncated": bool(getattr(self, "_search_truncated", False)),
        }
        if over:
            detail = ", ".join(
                f"{r}: {u:.0f} > {b:.0f} ({100 * u / b:.0f}% of budget)"
                for r, (u, b) in sorted(over.items())
            )
            msg = (
                f"SetFolding: the returned folding does NOT fit the {self.platform} "
                f"resource budget ({detail}). "
                + (
                    "This solution was expected to fit and does not."
                    if feasible_found
                    else (
                        "No folding in the searched throughput range fitted the budget, so the "
                        "least-resource folding was returned; it is not a solution."
                        if searched
                        else "Meeting the requested target_cycles_per_frame costs more than "
                        "the device has; relax the target or the folding will not fit."
                    )
                )
            )
            if over and not scored_over:
                msg += (
                    " NOTE: the folding search scored this as fitting; it only exceeds the "
                    "budget once step_minimize_bit_width (MinimizeAccumulatorWidth / "
                    "RoundAndClipThresholds) has run, which the search does not model."
                )
            if self.strict_budget:
                raise RuntimeError(msg + " (strict_budget=True)")
            warnings.warn(msg, stacklevel=2)

    def _fold_at(self, opt_template, targets, target_cycles):
        """Fold a fresh copy of the model to meet ``target_cycles`` (minimizing
        resources and honoring resource-type preferences). Returns
        ``(model, fits_budget, metrics, achieved_cycles)``.

        ``achieved_cycles`` is what the folded graph actually reaches, which is
        not the same as ``target_cycles``: discrete folding factors mean the
        solver usually lands somewhere near the target rather than on it. The
        search compares candidates on what they achieved, never on what they
        were asked for."""
        targets["max_cycles"] = target_cycles
        opt = copy.deepcopy(opt_template)
        opt.targets = targets
        opt.generate_parameter_set()
        opt.target_cycles_per_frame = target_cycles

        # first pass: fold parallelism (PE/SIMD) to meet the target
        if self.optimize_folding:
            opt.optimize(
                max_nodes_in_partition=self.NODES_PER_PARTITION,
                target_parameters=["SIMD", "PE"],
            )
        # second pass: trade BRAM<->URAM / LUT<->DSP via ram_style/resType
        if self.optimize_resource_types:
            opt.optimize(
                max_nodes_in_partition=min(len(opt.model.graph.node), self.RESOURCE_PARTITION_SIZE),
                target_parameters=list(Optimizer.RESOURCE_TYPE_ATTRS),
            )

        # score the fit on the folded model, optionally including the resources of
        # inserted DWCs and sized FIFOs (experimental; off by default)
        scored_model = opt.model
        if self.enable_folding_fifo_heuristic:
            scored_model = opt.model.transform(InsertDWC())
            scored_model = scored_model.transform(SpecializeLayers(self.fpgapart))
            scored_model = insert_and_size_fifos(
                "folded_model.onnx",
                scored_model,
                self.platform,
                self.fpgapart,
                self.consider_dwc_costs,
                self.auto_fifo_strategy,
            )
            scored_model = scored_model.transform(SpecializeLayers(self.fpgapart))

        metrics = self._aggregate_resources(scored_model)
        fits = all(metrics[r] <= targets[r] for r in self.target_resources)
        # in place, no deepcopy and no cleanup pass: this runs once per search
        # probe (~15 per call) and opt.model is already a private copy, so the
        # default make_deepcopy=True/cleanup=True would copy and re-clean a
        # whole graph per probe -- on MobileNet that dominated the search.
        opt.model = opt.model.transform(AnnotateCycles(), make_deepcopy=False, cleanup=False)
        achieved = opt.model.analysis(dataflow_performance)["max_cycles"]
        return opt.model, fits, metrics, achieved

    def _slowest_cycles(self, opt_template):
        """Cycles per frame at minimal folding -- the least-resource
        configuration, used as the slow bound of the throughput search."""
        opt = copy.deepcopy(opt_template)
        opt.generate_parameter_set()
        opt.params.add_all_params_to_index_list()
        opt.params.set_values(opt.params.get_min_vals())
        opt.params.apply_updates(final=True, filter=["SIMD", "PE", "parallel_window"])
        model = opt.model.transform(AnnotateCycles())
        return model.analysis(dataflow_performance)["max_cycles"]

    def _enable_runtime_writeable_for_uram(self, model):
        """MVAU/VVAU assert that URAM weight memories are runtime-writeable, so a
        node the optimizer put in URAM must carry the flag or HDL generation
        fails later. Only touches nodes the optimizer actually moved to URAM."""
        for node in model.graph.node:
            inst = getCustomOp(node)
            types = inst.get_nodeattr_types()
            if "ram_style" not in types or "runtime_writeable_weights" not in types:
                continue
            if inst.get_nodeattr("ram_style") == "ultra":
                inst.set_nodeattr("runtime_writeable_weights", 1)

    def _retype_padded_io(self, model):
        """Rewrite the graph input/output tensor shapes to the padded shapes.
        Only has an effect when padding changed the IO layer channel counts; the
        host must then pad its input / crop its output to match."""
        input_shape = getCustomOp(model.graph.node[0]).get_normal_input_shape()
        output_shape = getCustomOp(model.graph.node[-1]).get_normal_output_shape()
        output_name = model.graph.output[0].name

        if len(model.graph.input) != 0:
            model.graph.input.remove(model.graph.input[0])
        model.graph.input.append(
            helper.make_tensor_value_info(
                model.graph.node[0].input[0], TensorProto.FLOAT, [*input_shape]
            )
        )
        if len(model.graph.output) != 0:
            model.graph.output.remove(model.graph.output[0])
        model.graph.output.append(
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [*output_shape])
        )
        return model

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc < self.target_cycles_per_frame:
                # finish if target met
                break

    def apply_naive_folding(self, model):
        """
        A naive folding optimizer implementation

        If two_pass_relaxation is enabled,
        SetFolding will internally run a second time if the target cycles from the
        first pass could not be achieved, instead using the achievable target (which
        may be constrained by a single node) to obtain a balanced pipeline.

        Notable exceptions and special behavior:

        When folding dense convolution/FC compute engines ("MVAU"/MatrixVectorActivation),
        which have two attributes (PE and SIMD):

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
                node_inst.set_nodeattr("SIMD", 1)
                # increase SIMD until either we meet
                # the target or weight stream becomes
                # too wide
                for simd_val in divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.target_cycles_per_frame and simd_val > (max_simd / 1024):
                        # finish if target met and simd value is not too low
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
                        style="naive",
                        folding_maximum_padding=0,
                    )
                )

        # necessary final transforms
        if self.insert_dwcs:
            model.transform(InsertDWC())

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        return (model, False)

    def apply(self, model):
        # Checked rather than treated as "anything that is not naive": this used
        # to fall through to the optimizer for any unrecognized string, so a
        # misspelled style silently selected a folder the caller did not ask for
        # and the build looked fine.
        if self.style not in self.STYLES:
            raise ValueError(
                f"unknown SetFolding style {self.style!r}; expected one of "
                + ", ".join(repr(s) for s in self.STYLES)
            )
        if self.style == "naive":
            # the naive folder has no maximize-throughput mode: it needs a
            # concrete cycles-per-frame target to fold against
            if self.target_cycles_per_frame is None:
                raise ValueError(
                    "style='naive' requires a target_cycles_per_frame; "
                    "leaving it None (maximize throughput) is only supported by "
                    "style='optimizer'"
                )
            return self.apply_naive_folding(model)
        return self.apply_optimized_folding(model)
