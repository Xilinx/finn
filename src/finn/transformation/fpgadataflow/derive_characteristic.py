# Copyright (C) 2022, Xilinx, Inc.
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


"""FIFO sizing from token access vectors (TAVs): generating the TAVs.

The flow, as ``step_set_fifo_depths`` runs it:

1. ``JustInTimeSynthesize`` builds IP for the nodes that have to be simulated.
2. ``DeriveTokenAccessVectors`` gives every characterizable node its token
   access vectors (TAVs) ``io_chrc_in``/``io_chrc_out``: cumulative
   per-cycle transaction counts over two periods, one row per stream. This is
   the only place a TAV is written. Nothing below modifies one. A TAV comes
   from the node's tree model (``fifo_sizing.tree_models``) or from rtlsim.
3. ``DeriveFIFOSizes`` (``fifo_sizing.derive_fifo_sizes``) reads the TAVs,
   derives a wall-clock schedule for every edge with the chained-TAV pass and
   writes the resulting depths to ``inFIFODepths``/``outFIFODepths``.

This module holds steps 1 and 2.
"""

import os
import qonnx.custom_op.registry as registry
import warnings
from qonnx.transformation.base import NodeLocalTransformation, Transformation

from finn.transformation.fpgadataflow.fifo_sizing.tree_models.base import get_tree_model
from finn.transformation.fpgadataflow.prepare_ip import _codegen_single_node
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

#: Ops that are never characterized: fork, join and the FIFOs themselves.
_UNCHARACTERIZED_OPS = [
    "AddStreams_hls",
    "DuplicateStreams_hls",
    "StreamingFIFO_hls",
    "StreamingFIFO_rtl",
]


def is_stream_join(node):
    """True if ``node`` merges two streamed inputs into one output.

    Matches ``AddStreams`` (deprecated) and every elementwise binary operation
    (``ElementwiseAdd``, ``ElementwiseMul``, ...) whose inputs are *both*
    streams. Joins are not characterized: rtlsim characterization drives
    ``in0`` only and would stall waiting on the second stream. The sizer treats
    them as transparent (output token *k* leaves once token *k* is in on both
    inputs) and gives their inputs the join window floor.
    """
    if node is None or not node.op_type.startswith(("AddStreams", "Elementwise")):
        return False
    if len(node.input) < 2:
        return False
    inst = registry.getCustomOp(node)
    attr_types = inst.get_nodeattr_types()
    for style in ("lhs_style", "rhs_style"):
        if style in attr_types and inst.get_nodeattr(style) != "input":
            return False
    return True


class JustInTimeSynthesize(Transformation):
    def __init__(self, part, clk_period, only_without_tree_model=False):
        super().__init__()
        self.part = part
        self.clk_period = clk_period
        self.only_without_tree_model = only_without_tree_model

    def apply(self, model):
        for node in model.graph.node:
            inst = registry.getCustomOp(node)
            if (is_hls_node(node) or is_rtl_node(node)) and (
                (
                    (get_tree_model(inst) is None and self.only_without_tree_model)
                    or not self.only_without_tree_model
                )
                and (inst.get_nodeattr("io_chrc_in") == "")
            ):
                _codegen_single_node(
                    node,
                    model,
                    self.part,
                    self.clk_period,
                )

                op_type = node.op_type
                if is_hls_node(node):
                    try:
                        # ensure that code is generated
                        assert (
                            inst.get_nodeattr("code_gen_dir_ipgen") != ""
                        ), """Node
                        attribute "code_gen_dir_ipgen" is empty. Please run
                        transformation PrepareIP first."""
                        if os.path.isdir(inst.get_nodeattr("ipgen_path")) or inst.get_nodeattr(
                            "code_gen_dir_ipgen"
                        ) not in inst.get_nodeattr("ipgen_path"):
                            # call the compilation function for this node
                            inst.ipgen_singlenode_code()
                        else:
                            warnings.warn("Using pre-existing IP for %s" % node.name)
                        # ensure that executable path is now set
                        assert (
                            inst.get_nodeattr("ipgen_path") != ""
                        ), """Transformation
                        HLSSynthIP was not successful. Node attribute "ipgen_path"
                        is empty."""
                    except KeyError:
                        raise Exception("Custom op_type %s is currently not supported." % op_type)

        model = model.transform(ReplaceVerilogRelPaths())
        for node in model.graph.node:
            inst = registry.getCustomOp(node)
            if (
                (is_hls_node(node) or is_rtl_node(node))
                and (
                    (get_tree_model(inst) is None and self.only_without_tree_model)
                    or not self.only_without_tree_model
                )
                and node.op_type not in _UNCHARACTERIZED_OPS
                and not is_stream_join(node)
                and (inst.get_nodeattr("rtlsim_so") == "")
            ):
                try:
                    inst.prepare_rtlsim()
                    # ensure that executable path is now set
                    assert (
                        inst.get_nodeattr("rtlsim_so") != ""
                    ), "Failed to prepare RTLSim, no rtlsim_so attribute found."
                except KeyError:
                    raise Exception("Custom op_type %s is currently not supported." % op_type)

        model = model.transform(SetExecMode("rtlsim"))

        return (model, False)


class DeriveTokenAccessVectors(NodeLocalTransformation):
    """For each node in the graph, derive the token access vectors, either from
    the node's tree model or by rtlsim of the node in isolation, and store them
    in ``io_chrc_in``/``io_chrc_out``.

    * period (int) desired period over which the characteristic function
      will be derived.

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(
        self,
        model,
        period,
        strategy,
        fpga_part,
        clk_period,
        num_workers=None,
        nodes_to_ignore=[],
    ):
        super().__init__(num_workers=num_workers)
        self.model = model
        self.period = period
        self.strategy = strategy
        self.fpga_part = fpga_part
        self.clk_period = clk_period
        self.nodes_to_ignore = set(nodes_to_ignore)

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                inst = registry.getCustomOp(node)
                if node.name in self.nodes_to_ignore:
                    return (node, False)
                # Forks, joins and FIFOs get no TAV; the sizing pass treats
                # them as transparent.
                if op_type not in _UNCHARACTERIZED_OPS and not is_stream_join(node):
                    inst.derive_token_access_vectors(
                        model=self.model,
                        period=self.period,
                        strategy=self.strategy,
                        fpga_part=self.fpga_part,
                        clk_period=self.clk_period,
                        op_type=op_type,
                    )
            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)
