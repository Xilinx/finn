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


import numpy as np
import os
import qonnx.custom_op.registry as registry
import sys
import warnings
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import NodeLocalTransformation, Transformation

from finn.transformation.fpgadataflow.prepare_ip import _codegen_single_node
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import (
    compress_numpy_to_string,
    decompress_string_to_numpy,
    save_tav_npy,
    stretch,
)
from finn.util.fpgadataflow import is_hls_node, is_rtl_node


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
                    (inst.get_tree_model() is None and self.only_without_tree_model)
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
                    (inst.get_tree_model() is None and self.only_without_tree_model)
                    or not self.only_without_tree_model
                )
                and (
                    node.op_type
                    not in [
                        "AddStreams_hls",
                        "DuplicateStreams_hls",
                        "StreamingFIFO_hls",
                        "StreamingFIFO_rtl",
                    ]
                )
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
    """For each node in the graph, run rtlsim to obtain the i/o
    characteristic function for FIFO sizing and set the attribute.
    It is assumed that the PrepareRTLSim transformation was already
    called on the graph.

    This transformation performs rtlsim for each node, so it will run for
    some time (minutes to hours depending on configuration).

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
                # lookup op_type in registry of CustomOps
                inst = registry.getCustomOp(node)
                if node.name in self.nodes_to_ignore:
                    return (node, False)

                # Fork and join nodes are not characterized in their own right:
                # HandleBranches propagates their neighbours' vectors onto them.
                # A join must be skipped whatever backend is selected, because
                # derive_token_access_vectors_using_rtlsim drives "in0" only and
                # would stall waiting on the second stream.
                if op_type not in [
                    "AddStreams_hls",
                    "DuplicateStreams_hls",
                    "StreamingFIFO_hls",
                    "StreamingFIFO_rtl",
                ] and not is_stream_join(node):
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


def get_top_producer_period(node, model):
    highest_period = 0
    for indx, input_name in enumerate(node.input):
        # prod_node = model.find_producer(input_name)
        prod_node = find_non_dwc_producer(model, node)

        if prod_node is not None and registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out"):
            prod_chrc = decompress_string_to_numpy(
                registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out")
            )[0]
            cons_chrc = decompress_string_to_numpy(
                registry.getCustomOp(prod_node).get_nodeattr("io_chrc_in")
            )[0]
            period = max(len(prod_chrc) // 2, len(cons_chrc) // 2)
            highest_period = max(period, highest_period)
    return highest_period, prod_node


def get_top_consumer_period(node, model):
    highest_period = 0
    for indx, output_name in enumerate(node.output):
        # prod_node = model.find_consumer(output_name)
        prod_node = find_non_dwc_consumer(model, node)

        if prod_node is not None and registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out"):
            prod_chrc = decompress_string_to_numpy(
                registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out")
            )[0]
            cons_chrc = decompress_string_to_numpy(
                registry.getCustomOp(prod_node).get_nodeattr("io_chrc_in")
            )[0]
            period = max(len(prod_chrc) // 2, len(cons_chrc) // 2)
            highest_period = max(period, highest_period)
    return highest_period, prod_node


def max_throughput(trace, max_depth=10, min_size=10):
    """
    Recursively find the maximum throughput (delta / time) from a cumulative trace.

    Parameters:
        trace (np.ndarray): 1D cumulative access trace.
        max_depth (int): maximum depth of recursive splitting.
        min_size (int): minimum size of segment allowed for consideration.

    Returns:
        float: maximum throughput found in any segment.
    """
    segments = [(0, len(trace) - 1)]
    best_throughput = 0.0

    for _ in range(max_depth):
        max_local_throughput = 0
        max_segment = None

        # Evaluate current segments
        for start, end in segments:
            duration = end - start
            if duration < min_size:
                continue
            delta = trace[end] - trace[start]
            throughput = delta / duration
            if throughput > max_local_throughput:
                max_local_throughput = throughput
                max_segment = (start, end)

        if max_segment is None:
            break

        best_throughput = max(best_throughput, max_local_throughput)

        # Subdivide the fastest segment if large enough
        start, end = max_segment
        mid = (start + end) // 2
        if (mid - start) < min_size or (end - mid) < min_size:
            break

        segments = [s for s in segments if s != max_segment]
        segments += [(start, mid), (mid, end)]

    return best_throughput


def get_throughput(node, dir="in"):
    # calculate all budgets for nodes faster than the global period

    trace = None
    throughput = 0
    inst = registry.getCustomOp(node)
    if inst.get_nodeattr(f"io_chrc_{dir}_stretch") != "":
        trace = decompress_string_to_numpy(inst.get_nodeattr(f"io_chrc_{dir}_stretch"))[0]
        period = len(trace) // 2
    else:
        if inst.get_nodeattr(f"io_chrc_{dir}") != "":
            trace = decompress_string_to_numpy(inst.get_nodeattr(f"io_chrc_{dir}"))[0]
            period = len(trace) // 2
        else:
            period = 0
    if period != 0:
        # throughput = max_throughput(trace,min_size=int(np.sqrt(period)))
        throughput = trace[-1] / inst.get_nodeattr("io_chrc_period")
    # throughput = max_throughput(trace,min_size=1000)
    return throughput


def get_consumer(node, model):
    for indx, output_name in enumerate(node.output):
        cons = model.find_consumer(output_name)
        return cons


def get_true_period(node):
    in_chrc = decompress_string_to_numpy(node.get_nodeattr("io_chrc_in"))[0]
    out_chrc = decompress_string_to_numpy(node.get_nodeattr("io_chrc_out"))[0]

    return max(len(in_chrc) // 2, len(out_chrc) // 2)


def get_branch_nodes(last_node, model):
    branch_nodes = []
    while last_node.op_type != "DuplicateStreams_hls":
        branch_nodes.append(last_node)
        last_node = model.find_producer(last_node.input[0])
    return branch_nodes, last_node


def get_branch_volume(as_node, indx, model):
    last_node = model.find_producer(as_node.input[indx])
    branch_nodes, ds_node = get_branch_nodes(last_node, model)
    branch = [as_node, *branch_nodes, ds_node]

    # now perform volume calculation based on characteristic functions
    # note that the nodes are reversed, we start at addstreams node
    volume = 0
    max_i = 0
    max_period = 0
    latency = 0
    for i, node in enumerate(branch[1:]):
        volume += 1  # placeholder
        period = registry.getCustomOp(node).get_nodeattr("io_chrc_period")
        if period > max_period:
            max_period = period
            max_i = i

        # actual calculation has to consider the exp cycles and total nr of elements.
        # maybe maximum amount of values per period?
        # we can do this sort of calc by comparing the first consumed token to the
        # last produced token in some form.

    return volume, branch, max_i + 1, latency, max_period


def find_non_dwc_producer(model, node):
    producer = model.find_producer(node.input[0])
    if producer is None:
        return None
    if "StreamingDataWidthConverter" in producer.name:
        producer = model.find_producer(producer.input[0])
    return producer


def find_non_dwc_consumer(model, node):
    consumer = model.find_consumer(node.output[0])
    if consumer is None:
        return None
    if "StreamingDataWidthConverter" in consumer.name:
        consumer = model.find_consumer(consumer.output[0])
    return consumer


def calculate_peak_volume_delta(b0_lat, node_0, b1_lat, node_1, period_0, period_1, global_period):
    n0 = registry.getCustomOp(node_0)
    n1 = registry.getCustomOp(node_1)
    p0_v = decompress_string_to_numpy(n0.get_nodeattr("io_chrc_out"))[0]
    p1_v = decompress_string_to_numpy(n1.get_nodeattr("io_chrc_out"))[0]

    p0_v = stretch(p0_v, global_period)
    p1_v = stretch(p1_v, global_period)

    # pad vectors with latency
    p0_v = np.concatenate((np.zeros(b0_lat, dtype=p0_v.dtype), p0_v))
    p1_v = np.concatenate((np.zeros(b1_lat, dtype=p1_v.dtype), p1_v))

    if len(p0_v) > len(p1_v):
        # pad p1_v end
        last = p1_v[-1]
        p1_v = np.concatenate((p1_v, np.array([last] * (len(p0_v) - len(p1_v)), dtype=p1_v.dtype)))
    else:
        # pad p0_v end
        last = p0_v[-1]
        p0_v = np.concatenate((p0_v, np.array([last] * (len(p1_v) - len(p0_v)), dtype=p0_v.dtype)))

    p = max(len(p0_v), len(p1_v))

    max_positive_delta = 0
    max_negative_delta = 0
    peak_b0 = 0
    peak_b1 = 0
    peak_deltas = [0, 0]

    for i in range(p):
        delta = p0_v[i] - p1_v[i]
        if delta > max_positive_delta:
            max_positive_delta = delta
            peak_deltas[0] = delta
        if delta < max_negative_delta:
            max_negative_delta = delta
            peak_deltas[1] = delta * -1

        peak_b0 = max(p0_v[i], peak_b0)
        peak_b1 = max(p1_v[i], peak_b1)

    final_fifos = [int(max(0, (b1_lat)) + peak_deltas[1]), int(max(0, (b0_lat)) + peak_deltas[0])]
    return final_fifos


def compute_node_latency_init_periods(node, branch_max):
    cons_chrc = decompress_string_to_numpy(node.get_nodeattr("io_chrc_in"))[0]
    prod_chrc = decompress_string_to_numpy(node.get_nodeattr("io_chrc_out"))[0]

    cons_chrc = stretch(cons_chrc, branch_max)
    prod_chrc = stretch(prod_chrc, branch_max)

    def max_dist(a, b):
        a_last = a[-1]
        b_last = b[-1]

        idx_a = np.argmax(a == a_last)
        idx_b = np.argmax(b == b_last)

        return abs(idx_a - idx_b)

    max_distance = max_dist(cons_chrc, prod_chrc)
    return max_distance


def get_full_branch_latency(nodes, branch_max):
    total_latency = 0
    for node in nodes:
        total_latency += compute_node_latency_init_periods(registry.getCustomOp(node), branch_max)
    return total_latency


def assign_extra_fifo_volume(as_node, model, global_period):
    assert len(as_node.input) > 1

    b0 = get_branch_volume(as_node, 0, model)
    b1 = get_branch_volume(as_node, 1, model)
    if b0 is None or b1 is None or b0[1][-1] is not b1[1][-1]:
        # Not a fork/join pair this pass can describe: either input does not
        # trace back to a fork, or the two inputs come from *different* forks.
        # Leave the edges to the chained-TAV pass, which derives depths from
        # arrival times on the DAG and needs no pairing at all.
        return 0
    _, branch_0, _, _, period_0 = b0
    _, branch_1, _, _, period_1 = b1

    # propagate a characteristic onto the duplicatestreams node. Normally this is
    # inherited from its producer's output TAV; when the DuplicateStreams forks the
    # global input it has no producer, so fall back to the input TAV of the first
    # layer it feeds on the model branch -- the rate at which tokens leave the fork
    # equals the rate that layer accepts them.
    ds_node = registry.getCustomOp(branch_0[-1])
    prod_node = model.find_producer(branch_0[-1].input[0])

    if prod_node is not None:
        src_inst = registry.getCustomOp(prod_node)
        tav_ds = src_inst.get_nodeattr("io_chrc_out")
        tav_stretched_ds = src_inst.get_nodeattr("io_chrc_out_stretch")
        tav_pad_ds = src_inst.get_nodeattr("io_chrc_out_original")
    else:
        src_inst = registry.getCustomOp(model.find_consumer(branch_0[-1].output[0]))
        tav_ds = src_inst.get_nodeattr("io_chrc_in")
        tav_stretched_ds = src_inst.get_nodeattr("io_chrc_in_stretch")
        tav_pad_ds = src_inst.get_nodeattr("io_chrc_in_original")

    period_ds = get_true_period(src_inst)
    ds_node.set_nodeattr("io_chrc_in", tav_ds)
    ds_node.set_nodeattr("io_chrc_out", tav_ds)

    ds_node.set_nodeattr("io_chrc_in_original", tav_pad_ds)
    ds_node.set_nodeattr("io_chrc_out_original", tav_pad_ds)

    ds_node.set_nodeattr("io_chrc_in_stretch", tav_stretched_ds)
    ds_node.set_nodeattr("io_chrc_out_stretch", tav_stretched_ds)

    ds_node.set_nodeattr("io_chrc_period", period_ds)

    # last node with latencies version. Stretch both branches to a common,
    # non-zero period: a bypass branch can contain only the DuplicateStreams,
    # whose period is still unset here (0), which would collapse the stretch to
    # an empty vector.
    branch_period = max(period_0, period_1, global_period)
    latency_to_first_output_0 = get_full_branch_latency(branch_0[1:], branch_period)
    latency_to_first_output_1 = get_full_branch_latency(branch_1[1:], branch_period)
    peak_deltas = calculate_peak_volume_delta(
        latency_to_first_output_0,
        branch_0[1],
        latency_to_first_output_1,
        branch_1[1],
        period_0,
        period_1,
        global_period,
    )

    # latency_delta = max(latency_0, latency_1) - min(latency_0, latency_1)
    # peak delta should also contain additional fifos
    # for any latency differences between nodes
    # here we take the sum input to output latency
    # of each node in a branch and take the
    # last node's volume at that clock
    # This is a severe over-estimation to improve in the future

    addstrm_node_inst = registry.getCustomOp(as_node)

    add_strm_child = get_consumer(as_node, model)
    volumes = [0, 0]

    volumes[0] = peak_deltas[1]
    volumes[1] = peak_deltas[0]

    ds_node.set_nodeattr("extra_branch_fifos", volumes)

    old_sizes = ds_node.get_nodeattr("outFIFODepths")
    old_sizes[0] += volumes[0]
    old_sizes[1] += volumes[1]
    ds_node.set_nodeattr("outFIFODepths", old_sizes)

    # Propagate the join node's characteristic from its consumer so its own
    # output FIFO can be sized downstream. A terminal join (whose outputs are
    # global outputs) has no consumer -- nothing to size there.
    if add_strm_child is not None:
        tav = registry.getCustomOp(add_strm_child).get_nodeattr("io_chrc_in")
        tav_pad = registry.getCustomOp(add_strm_child).get_nodeattr("io_chrc_in_original")

        period_add = get_true_period(registry.getCustomOp(add_strm_child))

        addstrm_node_inst.set_nodeattr("io_chrc_in", tav)
        addstrm_node_inst.set_nodeattr("io_chrc_out", tav)

        addstrm_node_inst.set_nodeattr("io_chrc_out_original", tav_pad)
        addstrm_node_inst.set_nodeattr("io_chrc_in_original", tav_pad)

        addstrm_node_inst.set_nodeattr("io_chrc_period", period_add)
    return sum(volumes)


def is_stream_join(node):
    """True if ``node`` merges two streamed inputs into one output.

    ``AddStreams`` was FINN's dedicated two-input adder. It is deprecated --
    ``InferAddStreamsLayer`` now redirects to ``InferElementwiseBinaryOperation``
    and no ``AddStreams_hls`` custom op is registered any more -- so a residual
    branch reconverges on an ``ElementwiseAdd`` today. Both names are matched so
    that a graph built either way is sized.

    An ``ElementwiseAdd`` is only a join when *both* of its inputs are streams.
    With a constant right-hand side it is a unary passthrough that happens to
    have two ONNX inputs, and handing it to the branch machinery would have it
    hunt for a DuplicateStreams fork that is not there.
    """
    if node is None or not node.op_type.startswith(("AddStreams", "ElementwiseAdd")):
        return False
    if len(node.input) < 2:
        return False
    inst = registry.getCustomOp(node)
    attr_types = inst.get_nodeattr_types()
    for style in ("lhs_style", "rhs_style"):
        if style in attr_types and inst.get_nodeattr(style) != "input":
            return False
    return True


class HandleBranches(Transformation):
    """Given a characterized model, additionally generate the token
    access vectors for DuplicateStreams and stream joins such that no
    deadlocks occur. These nodes were not characterized in the
    DeriveTokenAccessVectors step and must inherit the edge node
    token access vectors of the faster of the two branches'.
    The inherited token access vector is also further padded in this
    case to simulate additional stalling on the faster branch.
    We expect the stretching operation afterwards to stretch the
    faster branch 'less' due to this padding, thus introducing FIFO
      depth during the DeriveFIFOSizes transform
    """

    def __init__(self, model, period):
        super().__init__()
        self.model = model
        self.period = period

    def apply(self, model: ModelWrapper):
        depth_added = 0
        # A join node is fed by a DuplicateStreams fork; it needs the faster
        # branch's FIFO stretched so the slower branch does not deadlock.
        join_nodes = [node for node in model.graph.node if is_stream_join(node)]
        if len(join_nodes) == 0:
            warnings.warn("No stream-join nodes found, skipping")
            return (model, False)
        # assign_extra_fifo_volume returns 0 for a join it cannot pair with a
        # fork, so a graph mixing describable and non-describable joins gets the
        # bypass sizing on the ones that are and nothing on the ones that are
        # not, instead of an exception for the whole graph.

        for join_node in join_nodes:
            depth_added += assign_extra_fifo_volume(join_node, model, self.period)

        return (model, False)


class ProducerDelayCharacteristicFunctions(NodeLocalTransformation):
    """Prerequisite: DeriveTokenAccessVectors already called on graph.
    For each node in the graph, use the accumulated I/O characteristic function
    and delay it if there is a difference in periods between the producer and consumer.
    This step adjusts for a delayed consumer and a fast producer so that additional
    depth is not introduced by stretching the consumer too much in the next step
    The consumer is 'faster' than what an immediate stretch might produce if
    we dont adjust for the latency of the producer's output starting to arrive

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
      period (int or None) the period to stretch the individual node chr function dumps to.
    """

    def __init__(self, num_workers=None, period=None, nodes_to_ignore=[]):
        super().__init__(num_workers=num_workers)
        self.period = period
        self.nodes_to_ignore = set(nodes_to_ignore)

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                # lookup op_type in registry of CustomOps
                prod = registry.getCustomOp(node)

                if node.op_type in [
                    "DuplicateStreams_hls",
                    "StreamingFIFO_hls",
                    "StreamingFIFO_rtl",
                ]:
                    return (node, False)

                if node.name in self.nodes_to_ignore:
                    return (node, False)

                prod_chrc_out = decompress_string_to_numpy(prod.get_nodeattr("io_chrc_out"))[0]
                period = len(prod_chrc_out) // 2
                prod.set_nodeattr("io_chrc_period", period)

                model = self.ref_input_model
                for output_name in node.output:
                    # cons = model.find_consumer(output_name)
                    cons = find_non_dwc_consumer(model, node)
                    if cons is None:
                        continue

                    cons = registry.getCustomOp(cons)
                    if cons.get_nodeattr("io_chrc_in") == "":
                        # consumer is an uncharacterized join -- no input
                        # pattern to match against
                        continue
                    cons_chrc_in = decompress_string_to_numpy(cons.get_nodeattr("io_chrc_in"))[0]

                    diff = len(cons_chrc_in) - len(prod_chrc_out)

                    if diff > 0:
                        # stretching
                        prod_chrc_out_stretch = stretch(prod_chrc_out, len(cons_chrc_in))

                        # padding
                        # prod_chrc_out_stretch = np.concatenate(
                        #     [prod_chrc_out, np.array([prod_chrc_out[-1]] * diff)]
                        # )

                        prod.set_nodeattr(
                            "io_chrc_out_stretch",
                            save_tav_npy(
                                prod, "io_chrc_out_stretch", np.array([prod_chrc_out_stretch])
                            ),
                        )

            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)


class DelayCharacteristicFunctions(NodeLocalTransformation):
    """Prerequisite: DeriveTokenAccessVectors already called on graph.
    For each node in the graph, use the accumulated I/O characteristic function
    and delay it if there is a difference in periods between the producer and consumer.
    This step adjusts for a delayed consumer and a fast producer so that additional
    depth is not introduced by stretching the consumer too much in the next step
    The consumer is 'faster' than what an immediate stretch might produce if
    we dont adjust for the latency of the producer's output starting to arrive

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
      period (int or None) the period to stretch the individual node chr function dumps to.
    """

    def __init__(self, num_workers=None, period=None, nodes_to_ignore=[]):
        super().__init__(num_workers=num_workers)
        self.period = period
        self.nodes_to_ignore = set(nodes_to_ignore)

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                # lookup op_type in registry of CustomOps
                # prod = registry.getCustomOp(node)

                if node.op_type in [
                    "DuplicateStreams_hls",
                    "StreamingFIFO_hls",
                    "StreamingFIFO_rtl",
                ]:
                    return (node, False)
                # assert not (op_type.startswith("StreamingFIFO")), "Found existing FIFOs"
                # we allow a FIFO, it will get removed in the next transform and is used to
                # fill in a bypass branch
                if node.name in self.nodes_to_ignore:
                    return (node, False)

                    # perform stretching if necessary
                # prod_period = prod.get_nodeattr("io_chrc_period")

                model = self.ref_input_model
                for input_name in node.input:
                    # prod = model.find_producer(input_name)
                    prod = find_non_dwc_producer(model, node)
                    if prod is None:
                        continue

                    prod = registry.getCustomOp(prod)

                    prod_chrc_out = decompress_string_to_numpy(prod.get_nodeattr("io_chrc_out"))[0]
                    # period = len(prod_chrc_out) // 2

                    cons = registry.getCustomOp(node)
                    cons_chrc_in = decompress_string_to_numpy(cons.get_nodeattr("io_chrc_in"))[0]

                    cons_period = len(cons_chrc_in) // 2

                    cons.set_nodeattr("io_chrc_period", cons_period)

                    np.set_printoptions(threshold=sys.maxsize)

                    diff = len(prod_chrc_out) - len(cons_chrc_in)

                    if diff > 0:
                        # stretch
                        cons_chrc_in_stretch = stretch(cons_chrc_in, len(prod_chrc_out))

                        # padding
                        # cons_chrc_in_stretch = np.concatenate(
                        #     [np.array([cons_chrc_in[-1]] * diff), cons_chrc_in]
                        # )
                        #
                        cons.set_nodeattr(
                            "io_chrc_in_stretch",
                            save_tav_npy(
                                cons, "io_chrc_in_stretch", np.array([cons_chrc_in_stretch])
                            ),
                        )

                    # setting these parameters here will make final
                    # characterization func comparisons impossible!
                    cons.set_nodeattr(
                        "io_chrc_in", save_tav_npy(cons, "io_chrc_in", np.array([cons_chrc_in]))
                    )

            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)


def inter_token_gaps(tav):
    if tav is None or tav.size == 0:
        return np.array([1]), np.array([0])  # reasonable defaults

    # Find indices where tokens are added (nonzero diff indicates a new token)
    token_times = np.flatnonzero(np.diff(tav) > 0) + 1  # +1 to align with time index

    if token_times.size < 2:
        # Not enough token events to compute gaps
        # Default gap of 1 between tokens (or 0 if no tokens)
        return np.array([1]), token_times

    # Compute gaps between token emissions
    # median = np.median
    gaps = np.diff(token_times)
    #  median_gap = np.array([int(np.median(gaps))])
    return gaps, token_times  # ,gaps_min


def _curve_to_times(curve):
    """Cumulative token curve -> per-token event times (cycle when token i,
    1-indexed, becomes available/consumed)."""
    total = int(curve[-1])
    return np.searchsorted(curve, np.arange(1, total + 1), side="left")


def _times_to_attr(times):
    return compress_numpy_to_string(np.asarray([times], dtype=np.int64))


class ChainComposeTAVs(Transformation):
    """Compose the isolated per-node token access vectors along the dataflow
    chain: each node's schedule is shifted by the lateness of its input
    arrivals (blocking-read semantics, prefix-max of per-token delays), and
    the resulting effective output schedule feeds the next node.

    Isolated curves assume input always available (over-stating run-ahead:
    open-loop) while the stretch pass assumes the producer slows to the
    consumer's rate (under-stating bursts: closed-loop). The composed curves
    are the middle ground -- free-running but input-constrained. They are
    stored as io_chrc_in/out_composed event-time arrays for DeriveFIFOSizes'
    chain_composed strategy.

    Joins compose with the elementwise-max of both arrivals; nodes without
    characterization break the chain (edges touching them fall back to the
    default sizing strategy).
    """

    def apply(self, model):
        eff_out = {}  # tensor name -> np.array of token arrival times

        for node in model.graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            inst = registry.getCustomOp(node)
            chrc_in = inst.get_nodeattr("io_chrc_in")
            chrc_out = inst.get_nodeattr("io_chrc_out")
            if chrc_in == "" or chrc_out == "":
                continue
            in_times = curve_to_times(decompress_string_to_numpy(chrc_in)[0])
            out_times = curve_to_times(decompress_string_to_numpy(chrc_out)[0])
            if len(in_times) == 0 or len(out_times) == 0:
                continue

            # arrival schedule of this node's input tokens; graph inputs and
            # chain breaks count as always-available (zero delay)
            arrivals = []
            for inp in node.input:
                if inp in eff_out:
                    arrivals.append(eff_out[inp])
            arrival = None
            if arrivals:
                n = min(min(len(a) for a in arrivals), len(in_times))
                arrival = arrivals[0][:n]
                for a in arrivals[1:]:
                    arrival = np.maximum(arrival, a[:n])

            if arrival is None:
                shift_in = np.zeros(len(in_times), dtype=np.int64)
            else:
                n = len(arrival)
                delay = np.maximum(0, arrival - in_times[:n])
                shift_in = np.zeros(len(in_times), dtype=np.int64)
                shift_in[:n] = np.maximum.accumulate(delay)
                if n < len(in_times):
                    shift_in[n:] = shift_in[n - 1]

            actual_in = in_times + shift_in

            # dependency: inputs consumed by the native emission time of each
            # output token; its shift propagates to the output
            in_curve = decompress_string_to_numpy(chrc_in)[0]
            dep = in_curve[np.minimum(out_times, len(in_curve) - 1)].astype(np.int64)
            shift_out = np.where(dep >= 1, shift_in[np.maximum(dep - 1, 0)], 0)
            actual_out = out_times + shift_out

            inst.set_nodeattr("io_chrc_in_composed", _times_to_attr(actual_in))
            inst.set_nodeattr("io_chrc_out_composed", _times_to_attr(actual_out))
            for out in node.output:
                eff_out[out] = actual_out

        return (model, False)


def swg_edge_burst_floor(prod_node, cons_node):
    """Analytic FIFO floor for edges touching a sliding-window generator.

    The TAV comparison stretches the SWG's curve to the consumer's period,
    which flattens its window bursts: after the line buffer fills, the SWG
    emits a full ROW of windows back-to-back (verified in stitched-IP rtlsim:
    cnv-w1a1 needs exactly OFMDim_w x k^2 x C/SIMD = 270 slots on SWG->MVAU
    where the stretched TAV delta is ~1). Floor the depth at one output-row
    burst on SWG outputs, and at the line-buffer fill on SWG inputs.
    """

    def swg_attrs(node):
        inst = registry.getCustomOp(node)
        k = inst.get_nodeattr("ConvKernelDim")
        ifm = inst.get_nodeattr("IFMDim")
        ofm = inst.get_nodeattr("OFMDim")
        ch = inst.get_nodeattr("IFMChannels")
        simd = inst.get_nodeattr("SIMD")
        return k, ifm, ofm, ch, simd

    floor = 0
    try:
        if prod_node is not None and prod_node.op_type.startswith("ConvolutionInputGenerator"):
            k, ifm, ofm, ch, simd = swg_attrs(prod_node)
            window_beats = int(np.prod(k)) * ch // simd
            if ofm[0] > 1:  # 2D: one output row of windows
                floor = max(floor, ofm[1] * window_beats)
            else:  # 1D: a couple of windows
                floor = max(floor, 2 * window_beats)
        if cons_node is not None and cons_node.op_type.startswith("ConvolutionInputGenerator"):
            k, ifm, ofm, ch, simd = swg_attrs(cons_node)
            if ifm[0] > 1:  # 2D: line-buffer fill of (k_h - 1) input rows
                floor = max(floor, (k[0] - 1) * ifm[1] * ch // simd)
            else:
                floor = max(floor, k[1] * ch // simd)
    except Exception:
        return 0
    return int(floor)


#: Ops that fan out or join without a token access vector of their own. They are
#: treated as transparent in the chained-TAV pass: a token handed to them is
#: handed on in the same cycle, which is what the hardware does.
_TRANSPARENT_OPS = (
    "DuplicateStreams_hls",
    "AddStreams_hls",
    "ElementwiseAdd_hls",
    "ElementwiseAdd_rtl",
)


#: Diagnostics opt-in: when true the per-edge trace also carries the raw
#: (write, read, native-read) schedules the depth was derived from. Off by
#: default -- these are per-token arrays.
_TRACE_SCHEDULES = False


def arrival_of(schedule, counts):
    """Time at which ``counts[i]`` tokens have arrived, given per-token times.

    An empty schedule means no token ever arrives, which constrains nothing, so
    every arrival time is 0. Indexing would otherwise run off the empty array.
    """
    if len(schedule) == 0:
        return np.zeros(len(counts), dtype=np.int64)
    idx = np.clip(counts, 0, len(schedule)) - 1
    return np.where(counts > 0, schedule[np.maximum(idx, 0)], 0)


def peak_occupancy(write_times, read_times):
    """Largest number of tokens simultaneously in flight on one edge."""
    n = min(len(write_times), len(read_times))
    if n == 0:
        return 0
    consumed = np.searchsorted(read_times, write_times[:n], side="right")
    return int(np.max(np.arange(1, n + 1) - consumed))


def _peak_occupancy_periodic(write_times, read_times, period, per_frame=0, both_frames=True):
    """Peak occupancy in steady state, when both schedules repeat every frame.

    ``_peak_occupancy`` measures one frame in isolation, so it can never report
    more than a frame's tokens however far behind the consumer is. That is the
    wrong answer wherever the consumer lags the producer by more than one frame
    -- radioml's residual edge (``Thresholding_rtl_5`` around the whole
    attention block) needs 1976 tokens on a 1024-token-per-frame edge, because
    attention takes about two frame periods to deliver the other input of the
    join.

    In steady state the design runs one frame per ``period``, so frame *f*
    writes token *i* at ``w_i + f*period`` and reads it at ``r_i + f*period``,
    and occupancy is periodic::

        occ(t) = sum over f of g(t - f*period),   g(u) = |w <= u| - |r <= u|

    ``g`` has finite support (both counts reach n), so the sum is finite and
    the peak follows from a merge of the two schedules -- no tiling of the
    per-cycle arrays, which is what made the previous multi-frame experiment
    cost ``frames`` times the memory of every node's schedule and put resnet50
    out of reach.
    """
    n = min(len(write_times), len(read_times))
    if n == 0:
        return 0
    w = np.sort(np.asarray(write_times[:n], dtype=np.int64))
    r = np.sort(np.asarray(read_times[:n], dtype=np.int64))
    if period <= 0:
        return _peak_occupancy(w, r)
    # A token access vector stores two periods, so these schedules already hold
    # two frames of tokens. Extending *those* with period ``period`` would count
    # every token twice.
    #
    # Both stored periods are steady state as the node measured them, but the
    # wall-clock propagation makes the first the frame that fills the pipeline
    # and the second the first fully-pipelined one, and neither is reliably the
    # worse -- on cnv-w2a2 the first frame alone loses 16% of throughput
    # (133681 cy against its 115206 optimum), because the run-ahead that needs
    # the buffer has not built up by frame 0. So evaluate both and take the
    # larger.
    #
    # In frame 1 a producer faster than its consumer has had a frame to get
    # ahead, and on an ordinary chain edge that run-ahead is exactly what the
    # buffer is for. Re-anchoring the two frames onto a common period to remove
    # it was tried and rejected: it takes cnv-w2a2 straight back to the frame-0
    # answer and its 16% loss.
    #
    # ``both_frames=False`` is for edges the caller will not relax -- a join's
    # input and anything inside a reconvergent branch. There the run-ahead is
    # charged in full because there is no relaxation left to trim it, and it
    # compounds: resnet50 took 835-1762 on 20-odd such edges whose ground truth
    # is 25-400 (+22 kB) purely from frame 1. Frame 0 already carries the
    # latency difference that a join actually has to buffer -- which is the
    # quantity that matters there -- so measuring it alone loses nothing on
    # those edges and is verified on the board not to.
    if per_frame and 2 * per_frame <= n:
        fill = _peak_occupancy_periodic(w[:per_frame], r[:per_frame], period)
        if not both_frames:
            return fill
        steady = _peak_occupancy_periodic(w[n - per_frame : n], r[n - per_frame : n], period)
        if both_frames == "steady":
            return steady
        return max(fill, steady)
    lo = int(min(w[0], r[0]))
    hi = int(max(w[-1], r[-1]))
    # how many periods of history can still be in flight
    reps = int((hi - lo) // period) + 2
    # occupancy only changes at an event, so those are the only candidates
    cand = np.unique(np.concatenate([w, r]) % period) + lo - (lo % period)
    cand = np.concatenate([cand, cand + period])
    occ = np.zeros(len(cand), dtype=np.int64)
    for k in range(reps):
        u = cand + k * period
        occ += np.searchsorted(w, u, side="right") - np.searchsorted(r, u, side="right")
    return int(max(0, occ.max()))


def _causal_writes(write_times, in_scheds):
    """Hold each output token until the inputs that carry it have arrived.

    A token access vector is a *steady-state* trace with an arbitrary phase: it
    is accumulated from the middle of a multi-period rtlsim, so the first output
    it records belongs to a frame whose inputs were consumed before the window
    opened. Its within-window ``first_write - first_read`` is therefore not the
    node's latency, and reading one out of it under-states any node that has to
    gather several inputs per output.

    tr-language shows how far that goes. A 64:1 width converter measures
    ``first_read = 0, first_write = 1`` -- one cycle, where physically it cannot
    emit until 64 input words have arrived, and at that point in the graph they
    arrive one every four cycles. The pass put the MLP branch's first token 198
    cycles behind its sibling where the board needs about 404, and sized the
    residual FIFO at 52 against a requirement of 106; on hardware that costs
    66.9% of the design's throughput.

    So impose the dependency the trace cannot express: with ``N_in`` inputs and
    ``N_out`` outputs a frame, output *j* cannot precede input
    ``ceil((j+1) * N_in / N_out)``. That is exact for a rate converter, a valid
    lower bound for anything that consumes a prefix to produce a prefix, and
    vacuous for a node that expands (the bound lands on input 1).

    **Only over the first frame.** There the pipeline starts empty, so output
    *j* provably comes from this frame's inputs. In steady state it does not: a
    compacting node emits its first outputs of frame *f* from inputs it took
    during frame *f-1*, and index-aligning the two schedules then pushes its
    whole trace a frame late. Applied to every frame this deadlocked resnet50 --
    three ``ConvolutionInputGenerator`` output edges collapsed from 48-50 to
    4-16 against a ground truth of 45-47, because their producers were being
    held back into the next frame and the occupancy against them vanished.
    """
    if not in_scheds:
        return write_times
    n_out = len(write_times)
    if n_out == 0:
        return write_times
    out = np.array(write_times, dtype=np.int64, copy=True)
    fill = max(1, n_out // 2)  # a token access vector stores two frames
    idx = np.arange(1, fill + 1, dtype=np.int64)
    for sched, _ in in_scheds.values():
        n_in = len(sched)
        if n_in == 0 or n_in <= n_out:
            continue  # expanding or one-to-one: the clock already covers it
        need = np.minimum(-(-idx * (n_in // 2) // fill), n_in)
        out[:fill] = np.maximum(out[:fill], sched[need - 1])
    return np.maximum.accumulate(out)


def _burst_above_rate(read_times, rate):
    """Depth a consumer needs when its supply arrives at a constant ``rate``.

    The largest excursion of demand above a rate line,
    ``max over t1 < t2 of C(t2) - C(t1) - rate * (t2 - t1)``, which for a
    monotone read schedule is one running minimum.

    Beware what dominates this: a consumer with period ``T_c`` reading ``N``
    tokens a frame contributes ``N * (1 - T_c / global_period)`` from the frame
    as a whole, which for a consumer much faster than the pacer swamps any
    genuine short-timescale burst. That whole-frame term is only real if the
    consumer must keep its native rate; a consumer with slack can simply be
    stretched. Use ``_longest_read_run`` when the short-timescale part is what
    is wanted.
    """
    if len(read_times) == 0 or rate <= 0:
        return 0
    g = np.arange(1, len(read_times) + 1) - rate * read_times
    return int(np.ceil(np.max(g - np.minimum.accumulate(g))))


def _longest_read_run(read_times):
    """Most tokens a consumer takes back to back, one per cycle.

    The demand a buffer has to satisfy instantaneously, with no help from the
    producer: whatever rate the supply sustains on average, it cannot serve a
    run of one-per-cycle reads out of an empty FIFO. Unlike
    ``_burst_above_rate`` this carries no whole-frame term, so it does not grow
    just because the consumer happens to be much faster than the pacer.
    """
    if len(read_times) < 2:
        return int(len(read_times))
    gaps = np.diff(read_times)
    breaks = np.flatnonzero(gaps > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(read_times) - 1]))
    return int(np.max(ends - starts) + 1)


def _stream_transactions(model, tensor):
    """Number of stream transactions one frame of ``tensor`` takes, or None."""
    shape = model.get_tensor_shape(tensor)
    if shape is None or len(shape) < 2:
        return None
    n = 1
    for d in shape[:-1]:
        n *= int(d)
    return n


def _warn_if_streams_differ(model, node):
    """Flag the shapes the single-schedule-per-node assumption cannot express.

    A node has one token access vector, so this pass gives every output the same
    write schedule and reads every input off the same clock. That is exact when
    the node's streams all carry the same number of transactions per frame --
    every op in cnv-w2a2, vgg10, mobilenetv1 and resnet50 -- and wrong when they
    do not, which is what ``StreamingSplit_hls`` and ``StreamingConcat_hls`` in
    the transformers do. Warn rather than guess: the token access vector does not
    carry the information needed to split the schedule, and inventing a division
    would produce a number that looks like a result.
    """
    for direction, tensors in (("output", node.output), ("input", node.input)):
        counts = set()
        for t in tensors:
            if direction == "output" and model.find_consumer(t) is None:
                continue
            if direction == "input" and model.find_producer(t) is None:
                continue
            n = _stream_transactions(model, t)
            if n is not None:
                counts.add(n)
        if len(counts) > 1:
            warnings.warn(
                "%s (%s) has %s streams of differing length %s; chained-TAV "
                "sizing assumes one schedule per node and will mis-size them"
                % (node.name, node.op_type, direction, sorted(counts))
            )


#: Diagnostics opt-in: when true the per-edge trace also carries the raw
#: (write, read, native-read) schedules the depth was derived from, so an
#: experiment can test a candidate rule against real schedules instead of
#: against summary statistics. Off by default -- these are per-token arrays.
_TRACE_SCHEDULES = False


def derive_chained_tav_depths(
    model,
    global_period=None,
    slack_relaxation=0.0,
    floor_mode="burst",
    throttled_cap=256,
    causal=True,
    trace=None,
    slack_side="chain",
    slack_scale=1.0,
    floor_rate="graph",
    small_peak=0.0,
    frames="both",
    cap_guard="down_chain",
):
    """Per-edge FIFO depths from token arrival times on the dataflow DAG.

    Returns ``{tensor_name: depth}``.

    Each node's token access vector is a cumulative per-cycle token count on a
    *local* clock: the schedule it would keep if nothing ever made it wait. In a
    real graph a node waits, so its local clock runs slower than the wall clock
    in a way that depends entirely on when its inputs turn up. Propagating that
    forward in topological order is a max-plus recurrence,

        R(c) = max(R(c - 1) + 1, earliest time the tokens read at c have arrived)

    which is a running maximum of ``req(c) - c``, so one vectorised pass per
    node. The depth an edge needs is then the peak of
    ``written_by(t) - read_by(t)`` over that schedule.

    Two things this gets right that comparing a stretched producer trace against
    a stretched consumer trace cannot:

    * **Run-ahead survives.** Stretching both traces to a common length asserts
      that producer and consumer move at the same rate, which is exactly the
      assumption a FIFO exists to break. A producer whose own upstream lets it
      finish a frame early *does* run ahead, and the buffer must hold the
      surplus. On cnv-w2a2 that is the difference between 208 and the ~1666 the
      board needs in front of ``MVAU_hls_3``.
    * **Starvation is charged upstream.** A node that looks fast in isolation may
      be starved by its own producer, in which case it never runs ahead and needs
      no buffer. Comparing local traces cannot see this and invents deep FIFOs
      after every width converter; the chained-TAV pass simply never gives those
      tokens an early arrival time.

    The graph's input is paced at the steady-state rate (one frame per
    ``global_period``). Without that the source is infinitely fast, every buffer
    upstream of the bottleneck grows without bound, and the numbers scale with
    how many frames you happen to simulate.

    ``slack_relaxation`` (0..1) trades a little of that back. The peak occupancy
    is the depth at which the producer *never* waits, which is more than
    throughput needs: a producer whose own upstream chain finishes a frame in
    ``T_up < global_period`` can afford to be blocked for the remaining
    ``global_period - T_up`` cycles per frame and still deliver on time. Draining
    ``k`` tokens takes ``k * global_period / N`` cycles, so up to
    ``N * (1 - T_up / global_period)`` tokens of the peak can be given up. At 1.0
    that whole allowance is spent; at 0.0 nothing is (the depth is what full
    decoupling costs). Nodes downstream of the bottleneck have ``T_up ==
    global_period`` and are never relaxed, which is the point -- blocking those
    blocks the bottleneck itself.
    """
    nodes = [n for n in model.graph.node if is_hls_node(n) or is_rtl_node(n)]

    tavs = {}
    curves = {}
    for node in nodes:
        try:
            inst = registry.getCustomOp(node)
            tavs[node.name] = _raw_tav(inst)
            curves[node.name] = _stream_curves(node, inst, *_raw_tav_rows(inst))
        except Exception:
            tavs[node.name] = (None, None)
            curves[node.name] = ({}, {}, False)

    periods = [len(t[0]) // 2 for t in tavs.values() if t[0] is not None]
    #: The caller passes the analytic estimate (``dataflow_performance``'s
    #: ``max_cycles``), which is what every op type's ``get_exp_cycles`` adds up
    #: to. For the ops in bnn-pynq and resnet50 that estimate is the measured
    #: period, but for the transformer ops it is not: radioml's
    #: ``StreamingSplit_hls``/``StreamingConcat_hls`` are freerunning at one word
    #: every five cycles, so they take 20480 cycles a frame where the estimate
    #: says 4096 -- and 20480 is exactly what the board measures. A steady-state
    #: rate taken from the estimate then has the whole graph running five times
    #: faster than it does, which the periodic occupancy reads as five frames of
    #: backlog on every edge below the bottleneck. The token access vectors are
    #: measured, so believe them: the pacer is the slowest of the two.
    measured = max(periods) if periods else 0
    global_period = max(int(global_period or 0), measured, 1)
    graph_inputs = {x.name for x in model.graph.input}
    #: The slowest node in the graph paces every stream in it. A node at that
    #: period is where a chain's slack accounting starts over: what happens
    #: further upstream cannot make the segment below it any tighter, because
    #: the pacer is already handing that segment one frame per global_period.
    #:
    #: Compared with a tolerance rather than for equality. A graph's slowest
    #: stage is usually several nodes wide and their periods differ by a cycle
    #: or two -- mobilenetv1's thresholds sit at 394274 and the width converters
    #: between them at 394272. An exact test resets at a threshold and is then
    #: re-poisoned by the converter two cycles below it, which switched the
    #: relaxation off for the entire network. The margin is wide enough to
    #: capture that and far narrower than any real second-slowest node
    #: (cnv-w2a2's runner-up is at 0.98 of its pacer).
    pacer_period = max([global_period] + periods) * 0.995

    arrival = {}  # tensor name -> per-token write times
    depths = {}
    #: tensor -> (tokens the producer writes on it per frame,
    #:            longest period in the chain feeding this edge,
    #:            the same thing to hand further downstream)
    #: The third differs from the second only at the pacer, which propagates "no
    #: constraint": the edge leaving the pacer has no slack, because blocking it
    #: blocks the pacer, but an edge two hops down is limited only by the segment
    #: since the pacer -- that segment is faster than the pacer by construction
    #: and can absorb being blocked.
    supply = {}

    #: Is this node, or anything it feeds, running at the pacer's period? If not,
    #: the node can be throttled and simply finish its frame later, so a buffer
    #: in front of it only has to smooth short-timescale mismatch -- it never has
    #: to hold a whole frame's worth of the producer running ahead.
    #:
    #: ``down_chain`` is the mirror of ``chain_period``'s upstream walk: the
    #: longest period between a node and the next pacer below it. A consumer
    #: whose downstream segment is faster than the pacer can be made to wait and
    #: still deliver its frame on time, exactly as a producer with upstream slack
    #: can be made to block -- so either side's slack shortens the buffer.
    drives_pacer = {}
    down_chain = {}
    down_prop = {}
    for node in reversed(nodes):
        tin = tavs[node.name][0]
        own = len(tin) // 2 if tin is not None else 0
        below = False
        downstream = []
        for t in node.output:
            cons = model.find_consumer(t)
            if cons is None:
                continue
            if drives_pacer.get(cons.name):
                below = True
            if cons.name in down_prop:
                downstream.append(down_prop[cons.name])
        drives_pacer[node.name] = own >= pacer_period or below
        chain = max([own] + downstream) if (downstream or own) else global_period
        down_chain[node.name] = chain
        down_prop[node.name] = 0 if own >= pacer_period else chain

    def chain_period(node, tin):
        own = len(tin) // 2 if tin is not None else 0
        upstream = [supply[t][2] for t in node.input if t in supply]
        chain = max([own] + upstream) if (upstream or own) else global_period
        propagated = 0 if own >= pacer_period else chain
        return chain, propagated

    #: Nodes that sit on a branch between a fork and the join it reconverges at.
    #: The slack relaxation says a producer with upstream slack can be blocked for
    #: ``global_period - T_up`` cycles a frame and still deliver on time. Its only
    #: deadline is the frame. Inside a reconvergent branch that is false: the
    #: deadline is the *join*, whose other input is arriving on its own schedule,
    #: so a cycle lost here is a cycle the join waits and a cycle the graph loses.
    #: Measured on resnet50: relaxing these gave the right total (215 kB against
    #: the ground truth's 206) at 1108271 cycles instead of 903174, +22.7%.
    def _reachable(tensor):
        seen, stack = set(), [model.find_consumer(tensor)]
        while stack:
            n = stack.pop()
            if n is None or n.name in seen:
                continue
            seen.add(n.name)
            for t in n.output:
                stack.append(model.find_consumer(t))
        return seen

    in_branch = set()
    for node in nodes:
        outs = [t for t in node.output if model.find_consumer(t) is not None]
        if len(outs) < 2:
            continue
        reach = [_reachable(t) for t in outs]
        shared = set.intersection(*reach)
        if not shared:
            continue  # the branches never meet again; ordinary chains
        for r in reach:
            in_branch |= r - shared

    def _record(**row):
        if trace is not None:
            trace.append(row)

    def idle_window_floor(tensor, curve, consumer):
        """Tokens that arrive while the consumer is not reading this stream.

        The peak occupancy is computed from the *stretched* read schedule, and
        stretching is what a large FIFO buys: the consumer reads each token the
        instant it turns up, so on any single stream the peak collapses towards
        one. That is right for a node that reads continuously and wrong for one
        that reads a stream inside a short window of its period and then does
        something else, because during the rest of the period the producer
        keeps delivering into a FIFO nobody is draining.

        The measure is the stream's *duty cycle* in the node's own schedule --
        the span from its first to its last read of that stream, over the
        node's period. What has to be buffered is the frame's tokens times the
        fraction of the period the node spends elsewhere.

        Radioml's four attention nodes are the case this exists for: each reads
        its K and V streams as 64 back-to-back words in cycles 5..68 of a
        4485-cycle period, a duty of 1.4%, so essentially a whole frame arrives
        while they are busy. At depth 2 those eight edges cost 10.9% of the
        design's throughput -- measured on the board, and restoring only them
        recovers the optimum exactly. A Thresholding or a width converter reads
        across its whole period, duty 1, and is charged nothing, which is why
        this leaves ordinary chains alone.
        """
        if curve is None or tensor not in supply or floor_mode == "none":
            return 0
        half = len(curve) // 2
        if half < 2:
            return 0
        per_frame = int(curve[half - 1])
        if per_frame <= 0:
            return 0
        reads = np.searchsorted(curve, np.arange(1, per_frame + 1), side="left")
        duty = min(1.0, float(reads[-1] - reads[0] + 1) / float(half))
        want = int(round(per_frame * (1.0 - duty)))
        if throttled_cap and consumer is not None and not drives_pacer.get(consumer, True):
            # a consumer that nothing at the pacer's period depends on can be
            # made to wait through its idle window instead of buffering it
            want = min(want, throttled_cap)
        return want

    def absorbed_frame_floor(model, node, tensor):
        """Depth for an edge whose consumer's own schedule says it reads nothing.

        A token access vector that ends at a cumulative count of zero says the
        node performed no read at all in the characterized window. A streaming
        layer cannot do that. A node that absorbs its whole input frame before
        it releases anything can, whenever the window that was stored happens to
        miss the absorb phase -- which is what a ``FINNLoop`` driven with more
        than one frame does, and why it is now characterized over exactly one.
        This stays as the guard for any operator that reaches the same state.

        There is therefore no schedule to compare against and no occupancy to
        measure -- but the requirement does not need a measurement. If the
        consumer holds its entire frame before releasing anything, and the
        producer upstream is running at the graph's steady-state rate, then
        every word of that frame is in flight at once and the edge has to hold
        all of them. One frame of the consumer's *folded* input words is that
        number, and it is a floor rather than an estimate: a shallower FIFO
        back-pressures the producer for the whole absorb phase, which on a loop
        body is most of the frame.

        Returning it here rather than in ``relaxed`` is deliberate. The
        relaxation trades depth for producer blocking, and blocking is precisely
        what this edge cannot tolerate: the consumer is not slow, it is
        *waiting*, and it will not start until the last word arrives.
        """
        cons = registry.getCustomOp(node)
        try:
            idx = list(node.input).index(tensor)
        except ValueError:
            return 0
        try:
            shape = cons.get_folded_input_shape(idx)
        except Exception:
            return 0
        if shape is None or len(shape) < 2:
            return 0
        n = 1
        for d in shape[:-1]:
            n *= int(d)
        return int(n)

    def _supply_deficit(write_times, read_times, local_reads, hold=0):
        """Tokens the consumer must find already buffered to never wait.

        ``read_times`` is the *stretched* schedule the arrival propagation
        produced: every read has already been pushed back to whenever its token
        turned up, so on that schedule the buffer is never short by
        construction. The consumer's own token access vector says when it *wants*
        each token -- ``local_reads``, its native offsets -- and anchoring that
        pattern at the earliest feasible frame start gives the demand curve the
        supply actually has to meet.

        The deficit is then the largest gap between that demand and the arrivals,

            max over j of  j - #{writes at or before r0 + local(j) - local(1) + hold}

        which is the same running-maximum the peak occupancy is, with the roles
        of the two schedules exchanged. Unlike ``_burst_above_rate`` it needs no
        rate line: it compares the consumer against the producer's real arrival
        times rather than against the graph's frame average, which on an
        ``FMPadding -> ConvolutionInputGenerator`` edge is two orders of
        magnitude apart.

        ``hold`` is how long the consumer may be started late -- its own slack.
        A consumer nothing at the pacer's period depends on can simply begin its
        frame later instead of buffering the head of it.
        """
        if write_times is None or local_reads is None or len(local_reads) == 0:
            return 0
        demand = read_times[0] + (local_reads - local_reads[0]) + int(hold)
        supplied = np.searchsorted(write_times, demand, side="right")
        return int(max(0, np.max(np.arange(1, len(local_reads) + 1) - supplied)))

    def relaxed(
        tensor,
        peak,
        read_times=None,
        consumer=None,
        join=False,
        burst_floor=0,
        write_times=None,
        local_reads=None,
    ):
        if slack_relaxation <= 0 or tensor not in supply or join:
            # ``join``: an edge feeding a node with more than one dynamic input is
            # never relaxed. Every term the relaxation trades away assumes the
            # consumer can simply be made to wait and finish its frame later --
            # true in a chain, false at a join. The early-arriving input of a join
            # cannot be throttled without throttling the fork it came from, which
            # starves the *other* branch, which is what the join is waiting for.
            # On hardware that is not a slowdown but a deadlock: resnet50 locks up
            # whenever a shortcut FIFO is shorter than the long branch's frame
            # occupancy (measured, sizing-log attempt 34). The peak occupancy is
            # exactly the storage that imbalance needs, so it is the floor here.
            #
            # The consumer's own burst demand is a second, independent floor:
            # not a relaxation to be traded away but a lower bound the peak
            # cannot see, because the peak is measured on a schedule that has
            # already been stretched to the supply's rate. Whichever is larger
            # wins.
            depth = max(int(peak), int(burst_floor))
            _record(
                tensor=tensor,
                consumer=consumer,
                peak=int(peak),
                burst_floor=int(burst_floor),
                depth=depth,
                join=join,
            )
            return depth
        per_frame, t_up, _ = supply[tensor]
        # Whichever side has more slack sets the allowance: the producer can be
        # blocked for global_period - t_up, the consumer can be made to wait for
        # global_period - down_chain, and either shortens the buffer by the same
        # arithmetic.
        #
        # Except when the producer's own chain is *at* the pacer. Then blocking
        # it is not a delay it can absorb, it is a cycle the whole graph loses,
        # and no amount of patience downstream buys that back -- so the
        # consumer's slack must not be allowed to excuse it. tr-vision's
        # `Reshape_rtl_2 -> ConvolutionInputGenerator_rtl_1` is exactly this
        # shape: producer at the pacer (65540), consumer with 50% slack
        # (32451). Taking the consumer's side allowed 9894 tokens away from a
        # peak of 3132, leaving the throttled floor of 256 where the board needs
        # 3201 -- and that one edge was the whole of tr-vision's +15.7%.
        t_down = down_chain.get(consumer, global_period)
        t_side = t_up
        if slack_side == "down":
            #: The relaxation depends on the *consumer's* segment only. Reducing
            #: an edge below its peak makes the producer block, and what that
            #: costs is paid by the consumer, which finds its tokens arriving
            #: late; the budget for that is the consumer's own slack. See
            #: ``CHAINED_TAV_SLACK_SIDE``.
            t_side = t_down
        elif slack_side == "max":
            t_side = max(t_up, t_down)
        elif t_up < pacer_period:
            t_side = min(t_up, t_down)
        t_side *= slack_scale
        allowance = per_frame * (1.0 - min(t_side, global_period) / float(global_period))
        after_slack = int(round(peak - slack_relaxation * allowance))
        #: The rate the burst floor charges demand against. ``graph`` is the
        #: edge's frame average, ``supply`` the producer's native rate (it
        #: delivers ``per_frame`` tokens in ``t_up`` cycles, then idles), and
        #: ``deficit`` abandons the rate line altogether for the producer's
        #: measured arrival times.
        rate = per_frame / float(global_period)
        if floor_rate == "supply":
            rate = per_frame / float(max(t_up, 1))
        floor = 0
        if read_times is not None and floor_mode != "none":
            if floor_mode == "burst":
                floor = _burst_above_rate(read_times, rate)
            elif floor_mode == "deficit":
                hold = max(0, global_period - t_down) if not drives_pacer.get(consumer, True) else 0
                floor = _supply_deficit(write_times, read_times, local_reads, hold)
            elif floor_mode.startswith("const:"):
                floor = int(floor_mode.split(":", 1)[1])
            else:
                floor = _longest_read_run(read_times)
            floor = min(peak, floor)
        capped = False
        #: What has to be true for the cap's trade -- less depth, more producer
        #: blocking -- to be free. ``down_chain`` is the shipped test: the
        #: producer's segment must be the slacker of the two. ``pacer`` is the
        #: narrower and, on every board-measured edge, the correct one: the
        #: producer must simply be blockable at all, i.e. its chain must not run
        #: at the pacer's period. See ``CHAINED_TAV_CAP_GUARD``.
        guard = (
            t_up < pacer_period
            if cap_guard == "pacer"
            else t_up <= down_chain.get(consumer, global_period)
        )
        if (
            throttled_cap
            and consumer is not None
            and not drives_pacer.get(consumer, True)
            and guard
        ):
            # The cap says a throttleable consumer needs only short-timescale
            # smoothing. It bounds the *floor*, not the slack result: if the
            # producer has no slack of its own -- it is at the pacer's period --
            # then blocking it is not free however patient its consumer is.
            #
            # The producer's slack has to be at least the consumer's for that to
            # hold. Capping trades depth for producer blocking, and the argument
            # that the blocking is absorbed assumes the segment above the edge
            # can hold a frame's slack somewhere -- which it can only if it is
            # the slacker of the two. tr-vision's
            # `Reshape_rtl_2 -> ConvolutionInputGenerator_rtl_1` is the
            # counterexample: producer chain at 0.80 of the pacer, consumer at
            # 0.40, capped from a peak of 3132 to 256 where the board needs
            # 3201, and that one edge is the whole of its +15.7%.
            floor = min(floor, throttled_cap)
            capped = True
        depth = max(after_slack, floor)
        if small_peak and peak < small_peak * per_frame and not capped:
            depth = 0
        _record(
            tensor=tensor,
            consumer=consumer,
            peak=int(peak),
            per_frame=int(per_frame),
            t_up=int(t_up),
            allowance=round(allowance, 1),
            after_slack=after_slack,
            floor=int(floor),
            capped=capped,
            run=int(_longest_read_run(read_times)) if read_times is not None else 0,
            burst=int(_burst_above_rate(read_times, per_frame / float(global_period)))
            if read_times is not None
            else 0,
            burst_supply=int(_burst_above_rate(read_times, per_frame / float(max(t_up, 1))))
            if read_times is not None
            else 0,
            deficit=_supply_deficit(write_times, read_times, local_reads, 0)
            if read_times is not None
            else 0,
            deficit_slack=_supply_deficit(
                write_times, read_times, local_reads, max(0, global_period - t_down)
            )
            if read_times is not None
            else 0,
            cons_period=int(len(tavs[consumer][0]) // 2)
            if consumer in tavs and tavs[consumer][0] is not None
            else 0,
            down_chain=int(down_chain.get(consumer, 0)),
            drives_pacer=bool(drives_pacer.get(consumer, True)),
            depth=int(depth),
            schedules=(write_times, read_times, local_reads) if _TRACE_SCHEDULES else None,
        )
        return depth

    for node in nodes:
        tin, tout = tavs[node.name]
        dyn_inputs = [t for t in node.input if model.find_producer(t) is not None]
        # a node the characterisation skipped (the fork/join ops) just forwards
        # its input timeline; it adds no schedule of its own
        transparent = tin is None or tout is None or node.op_type in _TRANSPARENT_OPS

        if transparent:
            if dyn_inputs:
                # the slowest input governs a join; a fork hands the same
                # timeline to every output
                srcs = [arrival[t] for t in dyn_inputs if t in arrival]
                if srcs:
                    n = min(len(s) for s in srcs)
                    out_sched = np.max(np.stack([s[:n] for s in srcs]), axis=0)
                else:
                    out_sched = None
                read_sched = out_sched
            else:
                out_sched = read_sched = None
            t_up, t_prop = chain_period(node, tin)
            is_join = len(dyn_inputs) > 1
            for t in dyn_inputs:
                if t in arrival and read_sched is not None:
                    peak = _peak_occupancy_periodic(
                        arrival[t],
                        read_sched,
                        global_period,
                        supply.get(t, (0,))[0],
                        both_frames=False if (is_join or node.name in in_branch) else frames,
                    )
                    depths[t] = relaxed(
                        t,
                        peak,
                        read_sched,
                        node.name,
                        join=is_join or node.name in in_branch,
                        write_times=arrival[t],
                        local_reads=None,  # a transparent node has no schedule
                    )
            for t in node.output:
                if out_sched is not None:
                    arrival[t] = out_sched
                    per_frame = max(1, len(out_sched) // 2)
                    supply[t] = (per_frame, t_up, t_prop)
            continue

        in_curves, out_curves, bound = curves[node.name]
        span = len(tin)
        cycles = np.arange(span, dtype=np.int64)

        req = np.zeros(span, dtype=np.int64)
        in_scheds = {}
        for t in node.input:
            curve = in_curves.get(t)
            if curve is None:
                continue
            curve_t = curve
            if t in arrival:
                sched = arrival[t]
            elif t in graph_inputs and model.find_producer(t) is None:
                # graph input: paced at the steady-state rate, so nothing
                # upstream of the bottleneck can run ahead without limit
                rate = global_period / max(int(curve[len(curve) // 2 - 1]), 1)
                sched = (np.arange(1, int(curve_t[-1]) + 1) * rate).astype(np.int64)
            else:
                continue  # weights / thresholds: no stream behind them
            if len(sched) == 0:
                # No token crosses this edge in the characterized window, so it
                # constrains nothing and there is no occupancy to measure --
                # the same state causal_writes() skips on ``n_in == 0``.
                #
                # It happens when a token access vector ends at a cumulative
                # count of zero, which a streaming layer never does but a node
                # that absorbs its whole input frame before releasing anything
                # can, if the stored window misses the absorb phase -- see
                # ``absorbed_frame_floor``.
                continue
            in_scheds[t] = (sched, curve_t)
            req = np.maximum(req, arrival_of(sched, curve_t))

        clock = cycles + np.maximum.accumulate(req - cycles)

        def _times(curve_t):
            n = int(curve_t[-1])
            return clock[
                np.minimum(np.searchsorted(curve_t, np.arange(1, n + 1), side="left"), span - 1)
            ]

        is_join = len(dyn_inputs) > 1
        for t, (sched, curve_t) in in_scheds.items():
            if model.find_producer(t) is None:
                continue
            if int(curve_t[-1]) == 0:
                # The consumer's own token access vector says it read nothing at
                # all in the characterized window, so there is no demand curve to
                # compare the arrivals against and the occupancy comes out zero.
                # Size the edge from the graph instead -- see
                # ``absorbed_frame_floor``, and note that leaving it at zero is
                # the one answer this shape cannot tolerate.
                depths[t] = max(depths.get(t, 0), absorbed_frame_floor(model, node, t))
                continue
            read_times = _times(curve_t)
            curve = in_curves[t]
            n_frame = int(curve[len(curve) // 2 - 1]) if len(curve) >= 2 else 0
            #: where the consumer *wants* each token, on its own unstretched
            #: clock -- the demand curve ``read_times`` is the stretched image of
            local_reads = np.minimum(
                np.searchsorted(curve_t, np.arange(1, int(curve_t[-1]) + 1), side="left"),
                span - 1,
            )
            peak = _peak_occupancy_periodic(
                sched,
                read_times,
                global_period,
                n_frame,
                both_frames=False if (is_join or node.name in in_branch) else frames,
            )
            depths[t] = relaxed(
                t,
                peak,
                read_times,
                node.name,
                join=is_join or node.name in in_branch,
                burst_floor=idle_window_floor(t, curve, node.name),
                write_times=sched,
                local_reads=local_reads,
            )

        t_up, t_prop = chain_period(node, tin)
        # the warning is about the single-schedule-per-node assumption, so it
        # only applies to nodes that still fall back to row 0
        if not bound:
            _warn_if_streams_differ(model, node)
        for t in node.output:
            curve = out_curves.get(t)
            if curve is None:
                continue
            wt = _times(curve)
            arrival[t] = _causal_writes(wt, in_scheds) if causal else wt
            per_frame_out = int(curve[len(curve) // 2 - 1]) if len(curve) >= 2 else int(curve[-1])
            supply[t] = (max(1, per_frame_out), t_up, t_prop)

    if trace is not None:
        trace.append({"global_period": int(global_period), "pacer_period": float(pacer_period)})

    return depths


class DeriveFIFOSizes(Transformation):
    """Prerequisite: DeriveTokenAccessVectors, ProducerDelayCharacteristic
    #  and DelayCharacteristic already called on graph.
    For each node in the graph, use the accumulated Token Access Vectors
    to perform FIFO sizing, setting the in/outFIFODepths attributes of HLSCustomOp
    nodes.
    """

    #: ``chained_tav`` only, and deliberately not build-config options: one
    #: value of each fits every model measured on the ZCU104 (cnv-w2a2, vgg10,
    #: cnv-w1a1, gtsrb, kws, tfc, cybsec, mobilenetv1, resnet50 and the three
    #: transformers), and a sizer that needs to be retuned per model is not a
    #: sizer. They are named here so the numbers are inspectable and so a sweep
    #: can subclass, not so a build config can drift.

    #: How much of a producer's upstream slack to spend rather than buffer. The
    #: chained-TAV peak is the depth at which the producer never blocks, which is
    #: more than throughput needs when its own chain finishes a frame early:
    #: draining k tokens costs k * period / N cycles, so up to
    #: N * (1 - T_up / period) tokens of the peak can be given up. Measured: 0.0
    #: leaves cnv-w2a2 at 24.5 kB against a ground truth of 10.1, 1.0 brings it
    #: to 13.0 with no loss of throughput.
    CHAINED_TAV_SLACK_RELAXATION = 1.0

    #: Lower bound on the relaxation: the consumer's largest excursion of demand
    #: above the edge's steady-state rate line. "none" costs mobilenetv1 41%.
    CHAINED_TAV_FLOOR = "burst"

    #: Depth cap for edges whose consumer neither runs at the pacer's period nor
    #: feeds anything that does -- blocking such a consumer only makes it finish
    #: its frame later. 256 is SplitLargeFIFOs' max_qsrl_depth, above which a
    #: FIFO stops fitting in SRLs, and is also the measured knee: 128 costs
    #: cnv-w2a2 2.8%, 512 costs mobilenetv1 6.8 kB.
    CHAINED_TAV_THROTTLED_CAP = 256

    #: Which side's slack the relaxation spends. ``chain`` is the shipped rule,
    #: ``min(t_up, down_chain)`` where the producer has slack of its own;
    #: ``down`` uses the consumer's segment alone and ``max`` the slower of the
    #: two. Back-solving the allowance from board-measured per-edge minimums
    #: says ``t_up`` should not appear -- see docs/fifo-sizing-workbench.md 6.2.
    CHAINED_TAV_SLACK_SIDE = "chain"

    #: Multiplier on whatever period the relaxation charges. An experiment knob:
    #: 1.0 is the shipped behaviour and anything else is a fitted constant, which
    #: is what this sizer is not allowed to ship.
    CHAINED_TAV_SLACK_SCALE = 1.0

    #: Rate reference for the burst floor: ``graph`` (the frame average, shipped)
    #: or ``supply`` (the producer's native rate).
    CHAINED_TAV_FLOOR_RATE = "graph"

    #: Zero any edge whose peak is below this fraction of a frame. Measured and
    #: rejected as a global rule -- costs mobilenetv1 30.5% (sizing-log 47).
    CHAINED_TAV_SMALL_PEAK = 0.0

    #: What the throttled cap requires of the producer before it will fire.
    #:
    #: ``down_chain`` (shipped) demands ``t_up <= down_chain[consumer]`` -- the
    #: producer's segment must have at least as much slack as the consumer's. It
    #: was introduced for tr-vision's ``Reshape_rtl_2 -> ConvolutionInputGenerator_rtl_1``
    #: on the reading that its producer sits "at 0.80 of the pacer". The trace
    #: says otherwise: that producer's chain period is 65544 against a pacer
    #: period of 65540, i.e. it *is* the pacer, and what actually protects the
    #: edge is that a pacer-rate producer cannot be blocked at all.
    #:
    #: ``pacer`` is that narrower condition, ``t_up < pacer_period``. Capping
    #: trades depth for producer blocking, so the thing that has to hold is that
    #: the producer can be blocked -- not that it is slacker than its consumer.
    #: The two differ on exactly the edges where both sides have slack, which is
    #: where mobilenetv1 keeps 8.65 kB it does not need.
    CHAINED_TAV_CAP_GUARD = "down_chain"

    #: Which of a token access vector's two stored frames the chain-edge peak is
    #: taken from. ``both`` (shipped) takes the larger, ``True`` being the same
    #: thing; ``False`` the fill frame alone, which is what join and
    #: reconvergent-branch edges always get; ``steady`` the second frame alone.
    CHAINED_TAV_FRAMES = "both"

    def __init__(
        self,
        num_workers=None,
        io_fifo_depth=5,
        period=None,
        nodes_to_ignore=[],
        global_offset_correction=False,
        heuristic_fifo_sizing_method="conservative_relaxation",
    ):
        super().__init__()
        self.io_fifo_depth = io_fifo_depth
        self.period = period
        self.minimum_size = 2
        self.nodes_to_ignore = set(nodes_to_ignore)
        self.global_budgets = []
        self.slowdown_so_far = [0, 0]
        self.fifos_removed = 0
        self.max_delay_so_far = 0
        self.nodes_parsed = 0
        self.global_offset_correction = global_offset_correction
        self.heuristic_fifo_sizing_method = heuristic_fifo_sizing_method
        self.delta_total_fifo_size = 0
        self.delta_adjusted_fifo_size = 0
        self.hybrid_fifo_size_rate = 0
        self.data_rate_total_fifo_size = 0
        self.data_rate_adjusted_fifo_size = 0
        self.hybrid_fifo_size = 0
        self.chained_tav_depths = None
        #: set to a list before ``apply`` to collect the per-edge derivation
        #: (peak, floor, slack, resulting depth). Diagnostics only; nothing in
        #: the pass reads it back.
        self.chained_tav_trace = None

    def apply(self, model):
        nodes = [node for node in model.graph.node]

        if self.heuristic_fifo_sizing_method == "chained_tav":
            # Two derivations of the same quantity, and the requirement is at
            # least both. They differ only in whether a node's writes are held
            # to the inputs that carry them (``causal_writes``):
            #
            #   * without it, a node's schedule keeps the phase its token access
            #     vector was measured with, so a producer's potential run-ahead
            #     survives, which residual branches need;
            #   * with it, a node that gathers many inputs per output shows the
            #     latency it really has, which the FIFO across a residual has to
            #     cover.
            #
            # Neither dominates: the bound raises latency downstream and lowers
            # occupancy on the bounded edge itself, so each pass is a lower
            # bound on the depth and the larger of the two is the safe answer.
            # Either one alone loses throughput or deadlocks on some graph shape.
            common = dict(
                global_period=self.period,
                slack_relaxation=self.CHAINED_TAV_SLACK_RELAXATION,
                floor_mode=self.CHAINED_TAV_FLOOR,
                throttled_cap=self.CHAINED_TAV_THROTTLED_CAP,
                slack_side=self.CHAINED_TAV_SLACK_SIDE,
                slack_scale=self.CHAINED_TAV_SLACK_SCALE,
                floor_rate=self.CHAINED_TAV_FLOOR_RATE,
                small_peak=self.CHAINED_TAV_SMALL_PEAK,
                frames=self.CHAINED_TAV_FRAMES,
                cap_guard=self.CHAINED_TAV_CAP_GUARD,
            )
            phased = derive_chained_tav_depths(
                model, causal=False, trace=self.chained_tav_trace, **common
            )
            self.chained_tav_depths = derive_chained_tav_depths(model, causal=True, **common)
            for tensor, depth in phased.items():
                if depth > self.chained_tav_depths.get(tensor, 0):
                    self.chained_tav_depths[tensor] = depth
            # InsertFIFO takes max(producer.outFIFODepths, consumer.inFIFODepths),
            # and a folding config may carry both -- resnet50's
            # U250_folding_config_live_fifo.json sets them on all 515 nodes, up to
            # 95924 deep. Those values then override the sizer wherever they are
            # larger, which was 238 of resnet50's 269 edges and 87% of what looked
            # like this pass oversizing. The arrival pass derives every edge, so it
            # owns every edge: clear both attributes here and write both below.
            # Scoped to chained_tav; the other strategies, which only ever write
            # outFIFODepths, keep the max() behaviour they were measured with.
            for node in model.graph.node:
                if is_hls_node(node) or is_rtl_node(node):
                    inst = registry.getCustomOp(node)
                    inst.set_nodeattr("inFIFODepths", [self.minimum_size] * len(node.input))
                    inst.set_nodeattr("outFIFODepths", [self.minimum_size] * len(node.output))

        for node in nodes:
            op_type = node.op_type
            if is_hls_node(node) or is_rtl_node(node):
                try:
                    # lookup op_type in registry of CustomOps
                    self.nodes_parsed += 1

                    if node.name in self.nodes_to_ignore:
                        continue

                    # DWC nodes ARE processed as producers: their output edge (e.g.
                    # DWC->ConvolutionInputGenerator, which rtlsim sizing finds needs
                    # hundreds of slots on cnv) would otherwise silently keep the
                    # default depth 2. Uncharacterized DWCs (non-multiple widths, no
                    # tree model) still fall through to depth 2 via the empty-chrc
                    # guards below.

                    assert not (op_type.startswith("StreamingFIFO")), "Found existing FIFOs"

                    prod = registry.getCustomOp(node)
                    out_fifo_depths = []
                    for indx, output_name in enumerate(node.output):
                        # Size the FIFO against the DIRECT consumer (the DWC itself,
                        # which is rate-matched to the producer's output width). Seeing
                        # *through* the DWC to the post-DWC consumer compares TAVs whose
                        # transaction counts differ by the width ratio, blowing the
                        # peak-delta up to ~a full frame (the producer->DWC over-size).
                        cons_node = model.find_consumer(output_name)
                        if cons_node is None:
                            # could be final node, will be overridden if so
                            # need an entry in the list anyway
                            out_fifo_depths.append(self.io_fifo_depth)
                            continue

                        cons = registry.getCustomOp(cons_node)

                        # chain-composed strategy: the occupancy between composed
                        # (input-arrival-constrained) schedules replaces the
                        # stretched-pair peak delta as the un-relaxed deficit;
                        # the budget-debited relaxation then trims it where
                        # backpressure absorbs the transient for free. Edges
                        # lacking composed data (joins, uncharacterized nodes)
                        # use the default machinery.
                        strategy_env = self.heuristic_fifo_sizing_method
                        composed_max = None
                        if (
                            strategy_env == "chain_composed"
                            and not is_stream_join(node)
                            and prod.get_nodeattr("io_chrc_out_composed") != ""
                            and cons.get_nodeattr("io_chrc_in_composed") != ""
                        ):
                            prod_ct = decompress_string_to_numpy(
                                prod.get_nodeattr("io_chrc_out_composed")
                            )[0]
                            cons_ct = decompress_string_to_numpy(
                                cons.get_nodeattr("io_chrc_in_composed")
                            )[0]
                            n = min(len(prod_ct), len(cons_ct))
                            if n > 0:
                                occ = np.searchsorted(
                                    prod_ct, cons_ct[:n], side="right"
                                ) - np.arange(n)
                                composed_max = int(max(0, occ.max()))

                        if self.chained_tav_depths is not None:
                            # depth already follows from the graph-wide arrival
                            # schedule; the per-edge trace comparison below is the
                            # thing it replaces
                            fifo_depth = self.chained_tav_depths.get(output_name, self.minimum_size)
                        elif node.op_type != "AddStreams_hls":
                            # determine which of prod and cons TAVs to compare
                            # based on which one was stretched
                            chr_pairs = []

                            if prod.get_nodeattr("io_chrc_out_stretch") != "":
                                chr_pairs.append(["io_chrc_out_stretch", "io_chrc_in"])

                            if cons.get_nodeattr("io_chrc_in_stretch") != "":
                                chr_pairs.append(["io_chrc_out", "io_chrc_in_stretch"])

                            if len(chr_pairs) == 0:
                                chr_pairs = [["io_chrc_out", "io_chrc_in"]]

                            depth_attempts = []
                            # currently only testing the first (main) pair

                            if (prod.get_nodeattr(chr_pairs[0][0])) == "":
                                out_fifo_depths.append(2)
                                continue

                            if (cons.get_nodeattr(chr_pairs[0][1])) == "":
                                # Consumer isn't characterized (e.g. a terminal
                                # join). For a DuplicateStreams side-channel
                                # output still apply the branch buffer volume
                                # from HandleBranches so the bypass FIFO that
                                # holds the model input for the whole model
                                # latency is sized rather than left at the
                                # default depth.
                                base = 2
                                if node.op_type == "DuplicateStreams_hls":
                                    base += prod.get_nodeattr("extra_branch_fifos")[indx]
                                out_fifo_depths.append(base)
                                continue

                            for pair in chr_pairs[:1]:
                                if (prod.get_nodeattr(pair[0])) != "":
                                    prod_chrc = decompress_string_to_numpy(
                                        prod.get_nodeattr(pair[0])
                                    )[0]
                                else:
                                    out_fifo_depths.append(2)
                                    continue

                                if (cons.get_nodeattr(pair[1])) != "":
                                    cons_chrc = decompress_string_to_numpy(
                                        cons.get_nodeattr(pair[1])
                                    )[0]
                                else:
                                    out_fifo_depths.append(2)
                                    continue

                                if len(cons_chrc) != len(prod_chrc):
                                    period_prod = max(len(prod_chrc) // 2, len(cons_chrc) // 2)
                                    cons_chrc = stretch(cons_chrc, period_prod * 2)
                                    prod_chrc = stretch(prod_chrc, period_prod * 2)
                                else:
                                    period_prod = len(prod_chrc) // 2

                                global_period = self.period

                                prod_original_chr = decompress_string_to_numpy(
                                    prod.get_nodeattr("io_chrc_out")
                                )[0]
                                cons_original_chr = decompress_string_to_numpy(
                                    cons.get_nodeattr("io_chrc_in")
                                )[0]

                                prod_chr_original = decompress_string_to_numpy(
                                    prod.get_nodeattr("io_chrc_out_original")
                                )[0]
                                cons_chr_original = decompress_string_to_numpy(
                                    cons.get_nodeattr("io_chrc_in_original")
                                )[0]

                                period_true = len(prod_original_chr) // 2

                                period_cons = len(cons_original_chr) // 2

                                # Step 1: Compute un-relaxed initial FIFO size guess
                                # a conservative estimate to further
                                # decrease in size using relaxation strategies

                                # find phase shift
                                pshift_min = 0

                                for pshift_cand in range(period_prod):
                                    prod_chrc_part = prod_chrc[pshift_cand:period_prod]
                                    cons_chrc_part = cons_chrc[: period_prod - pshift_cand]
                                    if (prod_chrc_part >= cons_chrc_part).all():
                                        pshift_min = pshift_cand
                                        break

                                # shift TAVs by that amount
                                pshift_min = max(0, pshift_min - max(0, period_true - period_cons))
                                prod_chrc_part = prod_chrc[pshift_min : (pshift_min + period_prod)]
                                cons_chrc_part = cons_chrc[:period_prod]
                                diff = prod_chrc_part - cons_chrc_part

                                # find peak delta between the two TAVs and use as initial FIFO guess
                                max_pos = np.argmax(diff)
                                fifo_depth_maximum = max(0, int(diff[max_pos]))
                                if composed_max is not None:
                                    # composed occupancy is the true un-relaxed
                                    # deficit (stretch flattens SWG bursts)
                                    fifo_depth_maximum = composed_max

                                # Step 2: Compute relaxation factors to refine
                                # the fifo size computed in Step 1
                                # using the original tav for determining data rates

                                parent_period, producer_node = get_top_producer_period(node, model)
                                consumer_period, consumer_node = get_top_consumer_period(
                                    node, model
                                )

                                gaps, token_times = inter_token_gaps(prod_chr_original)
                                gaps_cons, token_times_cons = inter_token_gaps(cons_chr_original)

                                local_max_delay_prod_list = sorted(gaps, reverse=True)
                                local_max_delay_cons_list = sorted(gaps_cons, reverse=True)

                                local_max_delay_prod = local_max_delay_prod_list[-1]
                                local_max_delay_cons = local_max_delay_cons_list[
                                    min(0, len(local_max_delay_cons_list) - 1)
                                ]

                                min_gap = min(
                                    len(local_max_delay_prod_list), len(local_max_delay_cons_list)
                                )

                                gap_ratios = np.array(
                                    local_max_delay_cons_list[:min_gap]
                                ) / np.array(local_max_delay_prod_list[:min_gap])

                                self.max_delay_so_far = max(
                                    self.max_delay_so_far, local_max_delay_prod
                                )

                                # Compute the slowdown numerator using the new logic
                                effective_depth = min(len(gap_ratios), fifo_depth_maximum)
                                remainder = fifo_depth_maximum - effective_depth

                                if len(gap_ratios) > 0:
                                    last_value = gap_ratios[-1]
                                else:
                                    last_value = 0
                                    # or raise an error if gap_ratios is
                                    # expected to have at least one element

                                slowdown_numerator = (
                                    sum(gap_ratios[:effective_depth]) + remainder * last_value
                                )

                                fifo_slowdown = slowdown_numerator / period_true
                                fifo_slowdown = sum(gap_ratios) / period_true

                                minimum_fifos_true = int(
                                    (local_max_delay_prod + local_max_delay_cons)
                                    / local_max_delay_prod
                                )
                                minimum_fifos = minimum_fifos_true

                                fifo_slowdown_rate = (
                                    minimum_fifos_true * local_max_delay_prod
                                ) / period_true

                                cycle_loss_of_fifo = max(
                                    1, local_max_delay_cons - local_max_delay_prod
                                )
                                parent_period = min(parent_period, global_period)

                                # ======= TOLERABLE SLOWDOWN CALCULATION =========================
                                tolerable_slowdown_parent = max(
                                    0,
                                    1
                                    - (
                                        parent_period / (global_period - self.slowdown_so_far[indx])
                                    ),
                                )
                                tolerable_slowdown_prod = max(
                                    0,
                                    1
                                    - (period_prod / (global_period - self.slowdown_so_far[indx])),
                                )
                                tolerable_slowdown = min(
                                    [tolerable_slowdown_parent, tolerable_slowdown_prod]
                                )

                                # The slack a removed FIFO slot "spends" is shared by the
                                # whole chain: debit it from a running budget instead of
                                # letting every edge claim the full global slack
                                # independently. Without the debit each SWG->MVAU edge of a
                                # conv pipeline relaxes to depth 2 and the compounded
                                # stalls surface at the bottleneck (cnv-w1a1: interval
                                # 32849 -> 62548 in stitched-IP rtlsim, -47% throughput).
                                avail_slack = max(
                                    0, global_period - self.slowdown_so_far[indx] - period_true
                                )
                                prod_loss = avail_slack // cycle_loss_of_fifo
                                # the bubbles from removed slots are eaten by the CONSUMER,
                                # so its (debited) slack bounds the relaxation as well: a
                                # fast SWG feeding a near-bottleneck MVAU must keep its
                                # buffer even though the SWG itself has plenty of slack
                                cons_loss = (
                                    max(0, global_period - self.slowdown_so_far[indx] - period_cons)
                                    // cycle_loss_of_fifo
                                )
                                ignorable_fifos = int(max(0, min([prod_loss, cons_loss])))

                                if producer_node is not None:
                                    if producer_node.op_type.startswith("DuplicateStreams"):
                                        ignorable_fifos = 0
                                if consumer_node is not None:
                                    if consumer_node.op_type.startswith("AddStreams"):
                                        ignorable_fifos = 0

                                minimized_depth = max(2, fifo_depth_maximum - ignorable_fifos)
                                minimum_fifos = max(1, minimum_fifos - ignorable_fifos)

                                # debit the slack actually consumed by the removed slots
                                removed_slots = min(ignorable_fifos, fifo_depth_maximum)
                                self.slowdown_so_far[indx] += removed_slots * cycle_loss_of_fifo

                                if fifo_slowdown > tolerable_slowdown:
                                    fifos_to_remove = int(
                                        fifo_depth_maximum * tolerable_slowdown / fifo_slowdown
                                    )
                                else:
                                    fifos_to_remove = fifo_depth_maximum

                                if fifo_slowdown_rate > tolerable_slowdown:
                                    fifos_to_remove_rate = int(
                                        minimum_fifos_true * tolerable_slowdown / fifo_slowdown_rate
                                    )
                                else:
                                    fifos_to_remove_rate = minimum_fifos_true

                                delta_fifo_size_post_adjustment = max(
                                    0, fifo_depth_maximum - max(fifos_to_remove, ignorable_fifos)
                                )
                                # print("fifos to remove: ", fifos_to_remove)
                                delta_fifo_size_post_adjustment_rate = max(
                                    0, minimum_fifos_true - fifos_to_remove_rate
                                )

                                hybrid_size = max(minimum_fifos, delta_fifo_size_post_adjustment)
                                hybrid_size_rate = max(
                                    delta_fifo_size_post_adjustment,
                                    delta_fifo_size_post_adjustment_rate,
                                )

                                self.delta_total_fifo_size += fifo_depth_maximum
                                self.delta_adjusted_fifo_size += delta_fifo_size_post_adjustment

                                self.data_rate_total_fifo_size += minimum_fifos_true
                                self.data_rate_adjusted_fifo_size += minimum_fifos
                                self.hybrid_fifo_size += hybrid_size
                                self.hybrid_fifo_size_rate += hybrid_size_rate

                                strategy = self.heuristic_fifo_sizing_method
                                if strategy in ("conservative_relaxation", "chain_composed"):
                                    # minimized TAV different
                                    fifo_depth = minimized_depth
                                elif strategy == "aggressive_relaxation":
                                    # minimized delta based, uses slowdown tracking
                                    fifo_depth = delta_fifo_size_post_adjustment
                                elif strategy == "no_relaxation":
                                    # maximum from TAV comparisons
                                    fifo_depth = fifo_depth_maximum

                                # print(
                                #     f"initial size, new sizes: "
                                #     f"{fifo_depth_maximum}, "
                                #     f"{minimized_depth}, "
                                #     f"{self.delta_adjusted_fifo_size}, "
                                #     f"{self.hybrid_fifo_size}, "
                                #     f"{self.hybrid_fifo_size_rate}, "
                                #     f"{self.data_rate_adjusted_fifo_size}"
                                # )

                                # override for testing:
                                # fifo_depth = delta_fifo_size_post_adjustment

                                # print(f"sized {node.name} with {fifo_depth} ")
                                depth_attempts.append(fifo_depth)
                            fifo_depth = min(depth_attempts)
                            if composed_max is None:
                                # SWG bursts survive relaxation; composed curves
                                # already carry them (floors over-provision deep
                                # nets: mobilenet 92 -> 582 KiB)
                                fifo_depth = max(fifo_depth, swg_edge_burst_floor(node, cons_node))
                        else:
                            fifo_depth = 0

                        if self.chained_tav_depths is not None:
                            # HandleBranches' fork/join imbalance is already part of
                            # the arrival schedule -- adding it again double-counts
                            pass
                        elif node.op_type == "DuplicateStreams_hls":
                            # propagate slowdown
                            if indx == 0:
                                self.slowdown_so_far[1] = self.slowdown_so_far[0]

                            extra_volume = prod.get_nodeattr("extra_branch_fifos")[indx]
                            fifo_depth += extra_volume
                        else:
                            extra_volume = prod.get_nodeattr("extra_branch_fifos")[0]
                            fifo_depth += extra_volume

                        out_fifo_depths.append(max(fifo_depth, self.minimum_size))

                        if self.chained_tav_depths is not None:
                            # ... and the matching half on the consumer, so the
                            # max() in InsertFIFO is a no-op rather than a way for
                            # a stale attribute to win
                            in_depths = cons.get_nodeattr("inFIFODepths")
                            for i, inp in enumerate(cons_node.input):
                                if inp == output_name:
                                    in_depths[i] = max(fifo_depth, self.minimum_size)
                            cons.set_nodeattr("inFIFODepths", in_depths)

                        if is_stream_join(node):
                            self.slowdown_so_far[0] = max(self.slowdown_so_far)

                    # Outside the per-output loop: several branches above end in
                    # `continue` after appending their depth, so persisting
                    # inside the loop would drop those entries -- a terminal
                    # node's io_fifo_depth among them.
                    prod.set_nodeattr("outFIFODepths", out_fifo_depths)

                    # finally, check node inputs to ensure FIFOs are added to
                    # any top-level inputs (at least self.io_fifo_depth deep)
                    in_fifo_depths = prod.get_nodeattr("inFIFODepths")
                    for i, input_name in enumerate(node.input):
                        if input_name in [x.name for x in model.graph.input]:
                            in_fifo_depths[i] = max(self.io_fifo_depth, in_fifo_depths[i])
                    prod.set_nodeattr("inFIFODepths", in_fifo_depths)

                except KeyError:
                    raise Exception("Custom op_type %s is currently not supported." % op_type)

        return (model, False)
