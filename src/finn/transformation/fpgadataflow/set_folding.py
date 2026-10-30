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

# Inspect information on Python objects like modules
import inspect
import numpy as np
import warnings
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames

# Import the elementwise binary operation module to extract names of all
# specializations (which require PE parallelism to be configured)
import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as elementwise_binary_hls
import finn.custom_op.fpgadataflow.rtl.elementwise_binary_rtl as elementwise_binary_rtl
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.fpgadataflow import is_hls_node, is_rtl_node


def divisors(num):
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


def common_divisors(numbers):
    separate_divisors = []
    for num in numbers:
        individual_divisors = list(divisors(num))
        separate_divisors.append(individual_divisors)

    # cast back to Python ints: np.intersect1d yields numpy ints, which
    # set_nodeattr rejects for int attributes
    return [int(x) for x in functools.reduce(np.intersect1d, separate_divisors)]


# Find the op-type names for all HLS and RTL specializations of elementwise
# binary operations
ELEMENTWISE_BINARY_OPS = [
    op_type
    for op_type, cls in inspect.getmembers(elementwise_binary_hls, inspect.isclass)
    if issubclass(cls, elementwise_binary_hls.ElementwiseBinaryOperation_hls)
] + [
    op_type
    for op_type, cls in inspect.getmembers(elementwise_binary_rtl, inspect.isclass)
    if issubclass(cls, elementwise_binary_rtl.ElementwiseBinary_rtl)
]


class SetFolding(Transformation):
    """Attempt to set parallelism attributes in all nodes to meet a specific
    target expressed as cycles per frame target_cycles_per_frame. For each
    HLSCustomOp node type, the attribute may vary but is typically one of {PE, SIMD},
    and has a certain allowed-maximum value and divisibility constraints,
    which SetFolding will take into account. Note that the algorithm implemented
    by SetFolding is very simple and it is often possible to hand-tune the returned
    parallelism configuration for better results.

    In the returned model, each node's
    cycles_estimate attribute will be set to its estimated number of cycles.

    If two_pass_relaxation is enabled,
    SetFolding will internally run a second time if the target cycles from the
    first pass could not be achieved, instead using the achievable target (which
    may be constrained by a single node) to obtain a balanced pipeline.

    Notable exceptions and special behavior:

    When folding DotProd (standard conv/FC), which has two attributes (PE and SIMD):

    * first increases SIMD while weight stream width per PE is <= mvau_wwidth_max
      (configurable in the SetFolding initializer, defaults to 36)
    * then increases PE until the target is met or max PE reached

    When folding DotProd (depthwise conv):

    * increases PE (channels) first, then SIMD (kernel elements) once PE is
      fully unfolded
    """

    def __init__(self, target_cycles_per_frame=1000, mvau_wwidth_max=36, two_pass_relaxation=True):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc < self.target_cycles_per_frame:
                # finish if target met
                break

    def apply(self, model):
        graph = model.graph
        # these ops use PE parallelism, up to a max value of NumChannels
        pe_ops = [
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Pool_hls",
            "PWPolyF_rtl",
            "Requant_hls",
            "Requant_rtl",
            "Thresholding_hls",
            "Thresholding_rtl",
            *ELEMENTWISE_BINARY_OPS,
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        simd_ops = [
            "FMPadding_Pixel_hls",
            "StreamingSplit_hls",
            "StreamingConcat_hls",
            "LayerNorm_rtl",
            "HWSoftmax_hls",
            "HWSoftmax_rtl",
            "Shuffle",
        ]
        for node in graph.node:
            # Shuffle is still a generic layer here (it is decomposed into
            # Inner/OuterShuffle stages by step_transpose_decomposition, which
            # inherit its SIMD), so it has to be folded before specialization
            if not (is_hls_node(node) or is_rtl_node(node) or node.op_type == "Shuffle"):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)
            if op_type in ["DotProd_hls", "DotProd_rtl"]:
                depthwise = node_inst.get_nodeattr("depthwise")
                if depthwise:
                    # Depthwise: PE = channels, then SIMD = kernel elements
                    max_pe = node_inst.get_nodeattr("OFMChannels")
                    max_simd = int(np.prod(node_inst.get_nodeattr("KernelSize")))
                    node_inst.set_nodeattr("PE", 1)
                    node_inst.set_nodeattr("SIMD", 1)
                    self.optimize_attribute_val(node_inst, max_pe, "PE")
                    pe = node_inst.get_nodeattr("PE")
                    cyc = node_inst.get_exp_cycles()
                    if pe == max_pe and cyc > self.target_cycles_per_frame:
                        self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                else:
                    # Standard: SIMD-then-PE, same as MVAU
                    max_simd = node_inst.MW
                    max_pe = node_inst.MH
                    node_inst.set_nodeattr("PE", 1)
                    node_inst.set_nodeattr("SIMD", 1)
                    for simd_val in divisors(max_simd):
                        prev_simd_val = node_inst.get_nodeattr("SIMD")
                        node_inst.set_nodeattr("SIMD", simd_val)
                        cyc = node_inst.get_exp_cycles()
                        if cyc < self.target_cycles_per_frame:
                            break
                        if (
                            node_inst.get_input_datatype(1).bitwidth()
                            * node_inst.get_nodeattr("SIMD")
                            > self.mvau_wwidth_max
                        ):
                            node_inst.set_nodeattr("SIMD", prev_simd_val)
                            break
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
            elif op_type in simd_ops:
                if op_type == "StreamingConcat_hls" or op_type == "StreamingSplit_hls":
                    node_inst.set_nodeattr("SIMD", 1)
                    channels_per_stream = node_inst.get_nodeattr("ChannelsPerStream")
                    for simd_val in common_divisors(channels_per_stream):
                        node_inst.set_nodeattr("SIMD", simd_val)
                        cyc = node_inst.get_exp_cycles()
                        if cyc < self.target_cycles_per_frame:
                            break
                elif op_type == "Shuffle":
                    # SIMD must divide the innermost dimension of the input (for the
                    # decomposition) and of the output (InnerShuffle folding);
                    # values without a valid decomposition estimate 0 cycles
                    node_inst.set_nodeattr("SIMD", 1)
                    in_dim = node_inst.get_normal_input_shape()[-1]
                    out_dim = node_inst.get_normal_output_shape()[-1]
                    for simd_val in common_divisors([in_dim, out_dim]):
                        prev_simd = node_inst.get_nodeattr("SIMD")
                        node_inst.set_nodeattr("SIMD", simd_val)
                        cyc = node_inst.get_exp_cycles()
                        if cyc == 0 and simd_val > 1:
                            node_inst.set_nodeattr("SIMD", prev_simd)
                            continue
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
