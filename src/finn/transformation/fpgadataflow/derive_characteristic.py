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


import qonnx.custom_op.registry as registry
import warnings
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance, max_remaining_period
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import NodeLocalTransformation
import numpy as np
from finn.util.basic import decompress_string_to_numpy, compress_numpy_to_string, stretch
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from qonnx.transformation.base import Transformation
import copy

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
        self, model, period, strategy, fpga_part, clk_period, num_workers=None, manual_bypass=False,nodes_to_ignore=[]
    ):
        super().__init__(num_workers=num_workers)
        self.model = model
        self.period = period
        self.strategy = strategy
        self.fpga_part = fpga_part
        self.clk_period = clk_period
        self.manual_bypass = manual_bypass
        self.nodes_to_ignore = set(nodes_to_ignore)

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                # lookup op_type in registry of CustomOps
                print("deriving: ", node.name)
                inst = registry.getCustomOp(node)
                if node.name in self.nodes_to_ignore:
                    print(f"ignoring derivation of node {node.name}")
                    return (node, False)

                if op_type not in ["AddStreams_hls","DuplicateStreams_hls", "StreamingFIFO_hls","StreamingFIFO_rtl"]:
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

    def apply(self, model: ModelWrapper):
        (model, run_again) = super().apply(model)
        if not self.manual_bypass:
            return (model, run_again)

        return (model, run_again)





class LocalStretchCharacteristicFunctions(NodeLocalTransformation):
    """Prerequisite: DeriveTokenAccessVectors already called on graph.
    For each node in the graph, use the accumulated I/O characteristic function
    and stretch it if there is a difference in periods between the producer and consumer.

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
                if node.name in self.nodes_to_ignore or node.op_type in [
                    "AddStreams_hls",
                    "DuplicateStreams_hls",
                    "StreamingFIFO_hls",
                    "StreamingFIFO_rtl",
                ]:
                    return (node, False)

                # model = self.ref_input_model

                # lookup op_type in registry of CustomOps
                prod = registry.getCustomOp(node)

                prod_chrc_out_original = decompress_string_to_numpy(
                    prod.get_nodeattr("io_chrc_out")
                )[0]
                prod_chrc_in_original = decompress_string_to_numpy(prod.get_nodeattr("io_chrc_in"))[
                    0
                ]

                prod_chrc_out = prod_chrc_out_original
                prod_chrc_in = prod_chrc_in_original

                compressed_prod_chrc_out = compress_numpy_to_string(np.array([prod_chrc_out]))
                compressed_prod_chrc_in = compress_numpy_to_string(np.array([prod_chrc_in]))

                period = max(len(prod_chrc_in), len(prod_chrc_out))

                # def remove_trailing_duplicates_keep_one(arr):
                #     arr = np.asarray(arr)
                #     if arr.size == 0:
                #         return arr

                #     last_val = arr[-1]
                #     # Find index where values stop being the same as the last value (from the end)
                #     i = len(arr) - 1
                #     while i > 0 and arr[i - 1] == last_val:
                #         i -= 1

                #     # Keep everything before the trailing duplicates + one final instance
                #     return np.concatenate((arr[:i], [last_val]))

                # def remove_leading_duplicates_keep_one(arr):
                #     arr = np.asarray(arr)
                #     if arr.size == 0:
                #         return arr

                #     first_val = arr[0]
                #     # Find index where values stop being the same as
                #     # the first value (from the start)
                #     i = 0
                #     while i < len(arr) - 1 and arr[i + 1] == first_val:
                #         i += 1

                #     # Keep one leading instance, then the rest
                #     return np.concatenate(([first_val], arr[i + 1 :]))

                #  prod_chrc_in_local = remove_trailing_duplicates_keep_one(prod_chrc_in)
                #     prod_chrc_out_local = remove_trailing_duplicates_keep_one(prod_chrc_out)

                # prod_chrc_in_local = remove_leading_duplicates_keep_one(prod_chrc_in_local)
                # prod_chrc_out_local = remove_leading_duplicates_keep_one(prod_chrc_out_local)

                # perform stretching if necessary
                prod_chrc_in = stretch(prod_chrc_in, period)
                prod_chrc_out = stretch(prod_chrc_out, period)

                compressed_prod_chrc_in = compress_numpy_to_string(np.array([prod_chrc_in]))
                compressed_prod_chrc_out = compress_numpy_to_string(np.array([prod_chrc_out]))

                prod.set_nodeattr("io_chrc_in", compressed_prod_chrc_in)
                prod.set_nodeattr("io_chrc_out", compressed_prod_chrc_out)

                # prod_chrc_in = stretch(prod_chrc_in, self.period)
                # prod_chrc_out = stretch(prod_chrc_out, self.period)

                # compressed_prod_chrc_in = compress_numpy_to_string(np.array([prod_chrc_in]))
                # compressed_prod_chrc_out = compress_numpy_to_string(np.array([prod_chrc_out]))

            #   prod.set_nodeattr("io_chrc_in_global_stretch", compressed_prod_chrc_in)
            #   prod.set_nodeattr("io_chrc_out_global_stretch", compressed_prod_chrc_out)

            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)





def get_top_producer_period(node, model):

    highest_period = 0
    for indx, input_name in enumerate(node.input):
        prod_node = model.find_producer(input_name)    
        if prod_node is not None:
            if prod_node.op_type.startswith("StreamingDataWidthConverter"):
                return get_top_producer_period(prod_node, model)
            prod_chrc = decompress_string_to_numpy(registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out"))[0]
            cons_chrc = decompress_string_to_numpy(registry.getCustomOp(prod_node).get_nodeattr("io_chrc_in"))[0]
            period = max(len(prod_chrc) // 2, len(cons_chrc) // 2)
            highest_period = max(period, highest_period)
    return highest_period, prod_node


def get_top_consumer_period(node, model):

    highest_period = 0
    for indx, output_name in enumerate(node.output):
        prod_node = model.find_consumer(output_name)    
        if prod_node is not None:
            if prod_node.op_type.startswith("StreamingDataWidthConverter"):
                return get_top_consumer_period(prod_node, model)

            prod_chrc = decompress_string_to_numpy(registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out"))[0]
            cons_chrc = decompress_string_to_numpy(registry.getCustomOp(prod_node).get_nodeattr("io_chrc_in"))[0]
            period = max(len(prod_chrc) // 2, len(cons_chrc) // 2)
            highest_period = max(period, highest_period)
    return highest_period, prod_node




import numpy as np

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
        new_segments = []
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



def get_nodes_until_converging(node, model):
    
    init_node = node
    count = 0
    while node is not None:
        if node.name.startswith("DuplicateStreams"):
            return count
        node = model.find_producer(node.input[0])
        count+=1
    return count

def get_throughput(node,dir="in"):

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
       #throughput = max_throughput(trace,min_size=1000)
    return throughput

def get_parent_throughput(node, model):

    throughputs = []
    for indx, input_name in enumerate(node.input):
        prod_node = model.find_producer(input_name)
        if prod_node is not None:
            throughputs.append(get_throughput(prod_node,"out"))
        else:
            throughputs.append(0)
    return max(throughputs)


def get_parent(node, model):

    for indx, input_name in enumerate(node.input):
        prod_node = model.find_producer(input_name)
        if prod_node is not None:
            return prod_node
        else:
            return None
    return None



def get_consumer(node, model):

    for indx, output_name in enumerate(node.output):
        cons = model.find_consumer(output_name)
        return cons


def get_consumer_throughput(node, model):

    throughputs = []
    for indx, output_name in enumerate(node.output):
        prod_node = model.find_consumer(output_name)
        if prod_node is not None:
            throughputs.append(get_throughput(prod_node,"in"))
        else:
            throughputs.append(0)
    return max(throughputs)

def get_true_period(node):

    in_chrc = decompress_string_to_numpy(node.get_nodeattr("io_chrc_in"))[0]
    out_chrc = decompress_string_to_numpy(node.get_nodeattr("io_chrc_out"))[0]

    return max(len(in_chrc)//2,len(out_chrc)//2)


def get_branch_nodes(last_node,model):
    branch_nodes = []
    while last_node.op_type != "DuplicateStreams_hls":
        branch_nodes.append(last_node)
        last_node = model.find_producer(last_node.input[0])
    return branch_nodes,last_node                

def get_branch_volume(as_node, indx, model):

    last_node = model.find_producer(as_node.input[indx])
    branch_nodes,ds_node = get_branch_nodes(last_node,model)
    branch = [as_node, *branch_nodes, ds_node]


    # now perform volume calculation based on characteristic functions
    # note that the nodes are reversed, we start at addstreams node
    volume = 0
    max_i = 0
    max_period = 0
    latency = 0
    for i, node in enumerate(branch[1:]):
        ##print("traversing node in branch ", indx)
        #print("i = ", i)
        volume +=1 # placeholder
        period = registry.getCustomOp(node).get_nodeattr("io_chrc_period")
        if period > max_period:
            max_period = period
            max_i = i
        
        # actual calculation has to consider the exp cycles and total nr of elements.
        # maybe maximum amount of values per period?
        # we can do this sort of calc by comparing the first consumed token to the 
        # last produced token in some form.
    print("returning vol,max_i,lat: ", volume, max_i,latency)

    return volume,branch, max_i+1, latency, max_period

def assign_max_period(as_node, indx, model, max_period):
    last_node = model.find_producer(as_node.input[indx])
    branch_nodes,ds_node = get_branch_nodes(last_node,model)
    branch = [as_node, *branch_nodes, ds_node]

    for i, node in enumerate(branch[1:]):
        inst = registry.getCustomOp(node)
    #    print(f"assigning {max_period} to {node.name}")

    
    head_node = branch[-2]
    inst = registry.getCustomOp(head_node)
   # print(f"assigning {1} to {head_node.name}")


def calculate_peak_volume_delta(b0_lat, node_0, b1_lat, node_1, period_0, period_1, global_period):


    peak_delta = 0

    n0 = registry.getCustomOp(node_0)
    n1 = registry.getCustomOp(node_1)

    p0 = get_true_period(n0) + b0_lat
    p1 = get_true_period(n1) + b1_lat



    # if (n0.get_nodeattr("io_chrc_out_global_stretch")) != "":
    #     p0_v = decompress_string_to_numpy(n0.get_nodeattr("io_chrc_out_global_stretch"))[0]
    # else:
    #     p0_v = decompress_string_to_numpy(n0.get_nodeattr("io_chrc_out"))[0]

    # if (n1.get_nodeattr("io_chrc_out_global_stretch")) != "":
    #     p1_v = decompress_string_to_numpy(n1.get_nodeattr("io_chrc_out_global_stretch"))[0]
    # else:
    #     p1_v = decompress_string_to_numpy(n1.get_nodeattr("io_chrc_out"))[0]

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
        p1_v = np.concatenate((p1_v, np.array([last]*(len(p0_v)-len(p1_v)), dtype=p1_v.dtype)))
    else:
        # pad p0_v end
        last = p0_v[-1]
        p0_v = np.concatenate((p0_v, np.array([last]*(len(p1_v)-len(p0_v)), dtype=p0_v.dtype)))

    
    p = max(len(p0_v), len(p1_v))

    max_positive_delta = 0
    max_negative_delta = 0
    max_i = 0
    peak_b0 = 0
    peak_b1 = 0
    peak_deltas = [0,0]


    for i in range(p):
        delta = p0_v[i]-p1_v[i]
        if delta > max_positive_delta:
            max_positive_delta = delta
            peak_deltas[0] = delta
        if delta < max_negative_delta:
            max_negative_delta = delta
            peak_deltas[1] = delta * -1

        peak_b0 = max(p0_v[i], peak_b0)
        peak_b1 = max(p1_v[i], peak_b1)

    final_fifos = [int(max(0,(b1_lat))+peak_deltas[1]), int(max(0,(b0_lat))+peak_deltas[0])]
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

        last_output = len(cons_chrc)
        first_input = cons_chrc[0]
        first_input_cycle = 0
        #first read
        for cycle, el in enumerate(cons_chrc[1:]):
            if first_input != el:
                first_input_cycle = cycle + 1
                first_input = el
                break

        first_output = prod_chrc[0]
        first_output_cycle = 0
        #first write
        for cycle, el in enumerate(prod_chrc[1:]):
            if first_output != el:
                first_output_cycle = cycle + 1
                first_output = el
                break

        return max(first_output_cycle - first_input_cycle, first_input_cycle-first_output_cycle)

def compute_node_latency_reversed(node):

        cons_chrc = decompress_string_to_numpy(node.get_nodeattr("io_chrc_in"))[0]
        prod_chrc = decompress_string_to_numpy(node.get_nodeattr("io_chrc_out"))[0]

        for cycle, el in enumerate(reversed(prod_chrc[:-1])):
            if first_input != el:
                first_input_cycle = cycle
                first_input = el
                break

        first_input = cons_chrc[-1]
        first_input_cycle = None

        for cycle, el in enumerate(reversed(cons_chrc[:-1])):
            if first_input != el:
                first_input_cycle = cycle
                first_input = el
                break

        return first_input_cycle


def get_full_branch_latency(nodes, branch_max):
    total_latency = 0
    for node in nodes:
        total_latency += compute_node_latency_init_periods(registry.getCustomOp(node), branch_max)
    return total_latency

def assign_extra_fifo_volume(as_node,model, global_period):
    assert len(as_node.input) > 1

    volume_0, branch_0, max_i_0, latency_0, period_0 = get_branch_volume(as_node,0, model)
    volume_1, branch_1, max_i_1, latency_1, period_1 = get_branch_volume(as_node,1, model)
    faster_indx = 0 if volume_0 < volume_1 else 1
    volume_dif = max(volume_0, volume_1) - min(volume_0, volume_1)

    assign_max_period(as_node, 0, model, period_0)
    assign_max_period(as_node, 1, model, period_1)


    # propagate the producer to duplicatestreams node
    ds_node = registry.getCustomOp(branch_0[-1])   
    prod_node = model.find_producer(branch_0[-1].input[0])

    period_ds = get_true_period(registry.getCustomOp(prod_node))

    tav_ds =  registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out")
    tav_stretched_ds =  registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out_stretch")
    tav_pad_ds =  registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out_original")
    #tav_local_ds =  registry.getCustomOp(prod_node).get_nodeattr("io_chrc_out_global_stretch")

    ds_node.set_nodeattr("io_chrc_in",tav_ds)
    ds_node.set_nodeattr("io_chrc_out", tav_ds)

    ds_node.set_nodeattr("io_chrc_in_original",tav_pad_ds)
    ds_node.set_nodeattr("io_chrc_out_original", tav_pad_ds)

    ds_node.set_nodeattr("io_chrc_in_stretch",tav_stretched_ds)
    ds_node.set_nodeattr("io_chrc_out_stretch", tav_stretched_ds)


    # ds_node.set_nodeattr("io_chrc_in_global_stretch",tav_local_ds)
    # ds_node.set_nodeattr("io_chrc_out_global_stretch", tav_local_ds)

    ds_node.set_nodeattr("io_chrc_period",period_ds)

    # last node with latencies version
    latency_to_first_output_0 = get_full_branch_latency(branch_0[1:], period_0)
    latency_to_first_output_1 = get_full_branch_latency(branch_1[1:], period_1)
    peak_deltas =  calculate_peak_volume_delta(latency_to_first_output_0, branch_0[1], latency_to_first_output_1, branch_1[1], period_0, period_1, global_period)


    latency_delta = max(latency_0, latency_1) - min(latency_0, latency_1)
    # peak delta should also contain additional fifos for any latency differences between nodes
    # here we take the sum input to output latency of each node in a branch and take the 
    # last node's volume at that clock

    addstrm_node_inst = registry.getCustomOp(as_node)

    add_strm_child = get_consumer(as_node, model)
    volumes = [0,0]

    if peak_deltas[0] > peak_deltas[1]:
        faster_indx = 0   
    else:
        faster_indx = 1

    volumes[0] = peak_deltas[1]
    volumes[1] = peak_deltas[0]

    print([volumes[0], volumes[1]])
    ds_node.set_nodeattr("extra_branch_fifos", volumes)

    old_sizes = ds_node.get_nodeattr("outFIFODepths")
    old_sizes[0] += volumes[0]
    old_sizes[1] += volumes[1]
    ds_node.set_nodeattr("outFIFODepths", old_sizes)

    # propagate the slower branch to addstreams node

    b_to_propagate = branch_1 if faster_indx == 0 else branch_0


    tav = registry.getCustomOp(add_strm_child).get_nodeattr("io_chrc_in")
   # tav_local = registry.getCustomOp(add_strm_child).get_nodeattr("io_chrc_in_global_stretch")
    tav_pad = registry.getCustomOp(add_strm_child).get_nodeattr("io_chrc_in_original")


    # attempt to introduce more branching
    b0_last = registry.getCustomOp(b_to_propagate[0])
    b1_last = registry.getCustomOp(b_to_propagate[1])

    period_add = get_true_period(registry.getCustomOp(add_strm_child))

    addstrm_node_inst.set_nodeattr("io_chrc_in", tav)
    addstrm_node_inst.set_nodeattr("io_chrc_out", tav)

    # addstrm_node_inst.set_nodeattr("io_chrc_in_global_stretch", tav_local)
    # addstrm_node_inst.set_nodeattr("io_chrc_out_global_stretch", tav_local)

    addstrm_node_inst.set_nodeattr("io_chrc_out_original", tav_pad)
    addstrm_node_inst.set_nodeattr("io_chrc_in_original", tav_pad)

    addstrm_node_inst.set_nodeattr("io_chrc_period",period_add)
    return sum(volumes)


class HandleBranches(Transformation):
    """ Given a characterized model, additionally generate the token access vectors for DuplicateStreams
     and AddStreams such that no deadlocks occur. These nodes were not characterized
     in the DeriveTokenAccessVectors step and must inherit the edge node token access vectors
     of the faster of the two branches'. The inherited token access vector is also further padded in this case to
     simulate additional stalling on the faster branch. We expect the stretching operation afterwards to stretch the faster 
     branch 'less' due to this padding, thus introducing FIFO depth during the DeriveFIFOSizes transform

    """

    def __init__(self,model, period):
        super().__init__()
        self.model = model
        self.period = period

    def apply(self, model: ModelWrapper):

        depth_added = 0
        addstrm_nodes = model.get_nodes_by_op_type("AddStreams_hls")
        if len(addstrm_nodes) == 0:
            warnings.warn("No AddStreams nodes found, skipping")
            return (model, False)
        
        for addstrm_node in addstrm_nodes:
            depth_added += assign_extra_fifo_volume(addstrm_node, model, self.period)

    

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
            print(f"PRODUCER delaying {node.name}")
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
                    cons = model.find_consumer(output_name)
                    if cons is None:
                        print("first node, skip")
                        continue

                    cons = registry.getCustomOp(cons)
                    cons_chrc_in = decompress_string_to_numpy(cons.get_nodeattr("io_chrc_in"))[0]

                    # cons_period = len(cons_chrc_in) // 2

                    diff = len(cons_chrc_in) - len(prod_chrc_out)

                    if diff > 0:
                        prod_chrc_out_stretch = stretch(prod_chrc_out, len(cons_chrc_in))
                        # prod_chrc_out_pad_end = np.concatenate(
                        #     [prod_chrc_out, np.array([prod_chrc_out[-1]] * diff)]
                        # )
                        # prod_chrc_out_pad_start = np.concatenate(
                        #     [np.array([prod_chrc_out[-1]] * diff), prod_chrc_out]
                        # )

                        prod.set_nodeattr(
                            "io_chrc_out_stretch",
                            compress_numpy_to_string(np.array([prod_chrc_out_stretch])),
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
            print(f"delaying {node.name}'s consumer")
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
                    print(f"ignoring delaying of node {node.name} consumers")
                    return (node, False)

                    # perform stretching if necessary
                # prod_period = prod.get_nodeattr("io_chrc_period")

                model = self.ref_input_model
                for input_name in node.input:
                    prod = model.find_producer(input_name)
                    if prod is None:
                        print("last node, skip")
                        continue

                    prod = registry.getCustomOp(prod)

                    prod_chrc_out = decompress_string_to_numpy(prod.get_nodeattr("io_chrc_out"))[0]
                    # period = len(prod_chrc_out) // 2

                    cons = registry.getCustomOp(node)
                    cons_chrc_in = decompress_string_to_numpy(cons.get_nodeattr("io_chrc_in"))[0]

                    cons_period = len(cons_chrc_in) // 2

                    cons.set_nodeattr("io_chrc_period", cons_period)

                    # c0_in = cons_chrc_in[:cons_period]
                    # c1_in = cons_chrc_in[cons_period:]

                    import sys

                    np.set_printoptions(threshold=sys.maxsize)

                    diff = len(prod_chrc_out) - len(cons_chrc_in)

                    if diff > 0:
                        print("padding cons input")

                        cons_chrc_in_stretch = stretch(cons_chrc_in, len(prod_chrc_out))
                        # cons_chrc_in_pad_end = np.concatenate(
                        #     [cons_chrc_in, np.array([cons_chrc_in[-1]] * diff)]
                        # )
                        # cons_chrc_in_pad_start = np.concatenate(
                        #     [np.array([cons_chrc_in[-1]] * diff), cons_chrc_in]
                        # )

                        cons.set_nodeattr(
                            "io_chrc_in_stretch",
                            compress_numpy_to_string(np.array([cons_chrc_in_stretch])),
                        )

                    compressed_cons_chrc_in = compress_numpy_to_string(np.array([cons_chrc_in]))
                    # compressed_cons_chrc_out = compress_numpy_to_string(np.array([cons_chrc_out]))

                    # setting these parameters here will make final
                    # characterization func comparisons impossible!
                    cons.set_nodeattr("io_chrc_in", compressed_cons_chrc_in)
                    print(f"updated {cons.onnx_node.name} period to {len(cons_chrc_in)}")

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
        return np.array([1]), token_times  # Default gap of 1 between tokens (or 0 if no tokens)

    # Compute gaps between token emissions
    #median = np.median
    gaps = np.diff(token_times)
    #  median_gap = np.array([int(np.median(gaps))])
    return gaps, token_times#,gaps_min




def remove_trailing_duplicates_keep_one(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr

    last_val = arr[-1]
    # Find index where values stop being the same as the last value (from the end)
    i = len(arr) - 1
    while i > 0 and arr[i - 1] == last_val:
        i -= 1

    # Keep everything before the trailing duplicates + one final instance
    return np.concatenate((arr[:i], [last_val]))


def remove_leading_duplicates_keep_one(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr

    first_val = arr[0]
    # Find index where values stop being the same as the first value (from the start)
    i = 0
    while i < len(arr) - 1 and arr[i + 1] == first_val:
        i += 1

    # Keep one leading instance, then the rest
    return np.concatenate(([first_val], arr[i+1:]))


def compute_max_buffer_size(producer_tav, consumer_tav, period, pshift):
    producer_tav_part = producer_tav[pshift : (pshift + period)]
    consumer_tav_part = consumer_tav[:period]
    diff = producer_tav_part - consumer_tav_part
    max_pos = np.argmax(diff)
    fifo_depth_maximum = max(0, int(diff[max_pos]))
    return fifo_depth_maximum



class DeriveFIFOSizes(Transformation):
    """Prerequisite: DeriveTokenAccessVectors, ProducerDelayCharacteristic
    #  and DelayCharacteristic already called on graph.
    For each node in the graph, use the accumulated Token Access Vectors
    to perform FIFO sizing, setting the in/outFIFODepths attributes of HLSCustomOp
    nodes.
    """

    def __init__(
        self,
        num_workers=None,
        io_fifo_depth=8,
        period=None,
        nodes_to_ignore=[],
        global_offset_correction=False,
        tav_utilization_strategy="conservative_relaxation",
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
        self.tav_utilization_strategy = tav_utilization_strategy
        self.delta_total_fifo_size = 0
        self.delta_adjusted_fifo_size = 0
        self.hybrid_fifo_size_rate = 0
        self.data_rate_total_fifo_size = 0
        self.data_rate_adjusted_fifo_size = 0
        self.hybrid_fifo_size = 0

    def apply(self, model):
        nodes = [node for node in model.graph.node]

        for node in nodes:
            op_type = node.op_type
            if is_hls_node(node) or is_rtl_node(node):
                try:
                    # lookup op_type in registry of CustomOps
                    self.nodes_parsed += 1

                    if node.name in self.nodes_to_ignore:
                        continue

                    assert not (op_type.startswith("StreamingFIFO")), "Found existing FIFOs"

                    prod = registry.getCustomOp(node)
                    out_fifo_depths = []
                    for indx, output_name in enumerate(node.output):
                        cons_node = model.find_consumer(output_name)
                        if cons_node is None:
                            # could be final node, will be overridden if so
                            # need an entry in the list anyway
                            out_fifo_depths.append(self.io_fifo_depth)
                            continue

                        cons = registry.getCustomOp(cons_node)

                        if node.op_type != "AddStreams_hls":
                            # determine which of prod and cons we vary
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
                                out_fifo_depths.append(2)
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
                                # find phase shift
                                pshift_min = 0

                                for pshift_cand in range(period_prod):
                                    prod_chrc_part = prod_chrc[pshift_cand:period_prod]
                                    cons_chrc_part = cons_chrc[: period_prod - pshift_cand]
                                    if (prod_chrc_part >= cons_chrc_part).all():
                                        pshift_min = pshift_cand
                                        break

                                parent_period, producer_node = get_top_producer_period(node, model)
                                consumer_period, consumer_node = get_top_consumer_period(
                                    node, model
                                )

                                if global_period < period_prod:
                                    global_period = period_prod


                                pshift_min = max(0, pshift_min - max(0, period_true - period_cons))

                                prod_chrc_part = prod_chrc[pshift_min : (pshift_min + period_prod)]
                                cons_chrc_part = cons_chrc[:period_prod]

                                # using the original tav for determining data rates
                                gaps, token_times = inter_token_gaps(prod_chr_original)
                                gaps_cons, token_times_cons = inter_token_gaps(cons_chr_original)

                                local_max_delay_prod_list = sorted(gaps, reverse=True)
                                local_max_delay_cons_list = sorted(gaps_cons, reverse=True)

                                local_max_delay_prod = local_max_delay_prod_list[-1]
                                local_max_delay_cons = local_max_delay_cons_list[
                                    min(1, len(local_max_delay_cons_list) - 1)
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

                                diff = prod_chrc_part - cons_chrc_part

                                # Step 2: Get the index of the maximum
                                max_pos = np.argmax(diff)
                                fifo_depth_maximum = max(0, int(diff[max_pos]))

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
                                # tolerable_slowdown_cons = max(
                                #     0,
                                #     1
                                #     - (
                                #         consumer_period
                                #         / (global_period - self.slowdown_so_far[indx])
                                #     ),
                                # )

                                tolerable_slowdown = min(
                                    [tolerable_slowdown_parent, tolerable_slowdown_prod]
                                )

                                prod_loss = (global_period - period_true) // cycle_loss_of_fifo
                                cons_loss = (global_period - period_cons) // cycle_loss_of_fifo
                                pred_loss = (global_period - parent_period) // cycle_loss_of_fifo

                                ignorable_fifos = int(max(0,min(prod_loss, cons_loss, pred_loss)))

                                if producer_node is not None:
                                    if producer_node.op_type.startswith("DuplicateStreams"):
                                        ignorable_fifos = 0
                                if consumer_node is not None:
                                    if consumer_node.op_type.startswith("AddStreams"):
                                        ignorable_fifos = 0

                                minimized_depth = max(2, fifo_depth_maximum - ignorable_fifos)
                                minimum_fifos = max(1, minimum_fifos - ignorable_fifos)

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
                                    0, fifo_depth_maximum - max(fifos_to_remove, ignorable_fifos )
                                )
                                #print("fifos to remove: ", fifos_to_remove)
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

                                if self.tav_utilization_strategy == "conservative_relaxation":
                                    # minimized TAV different
                                    fifo_depth = minimized_depth
                                elif self.tav_utilization_strategy == "aggressive_relaxation":
                                    # minimized delta based, uses slowdown tracking
                                    fifo_depth = delta_fifo_size_post_adjustment
                                elif self.tav_utilization_strategy == "no_relaxation":
                                    # maximum from TAV comparisons
                                    fifo_depth = fifo_depth_maximum


                                # override for testing:
                                #fifo_depth = delta_fifo_size_post_adjustment

                                #print(f"sized {node.name} with {fifo_depth} ")
                                depth_attempts.append(fifo_depth)
                            fifo_depth = min(depth_attempts)
                        else:
                            fifo_depth = 0

                        if node.op_type == "DuplicateStreams_hls":
                            # propagate slowdown
                            if indx == 0:
                                self.slowdown_so_far[1] = self.slowdown_so_far[0]

                            extra_volume = prod.get_nodeattr("extra_branch_fifos")[indx]
                            fifo_depth += extra_volume
                        else:
                            extra_volume = prod.get_nodeattr("extra_branch_fifos")[0]
                            fifo_depth += extra_volume

                        out_fifo_depths.append(max(fifo_depth, self.minimum_size))

                        prod.set_nodeattr("outFIFODepths", out_fifo_depths)

                        in_fifo_depths = prod.get_nodeattr("inFIFODepths")
                        for i, input_name in enumerate(node.input):
                            if input_name in [x.name for x in model.graph.input]:
                                in_fifo_depths[i] = max(self.io_fifo_depth, in_fifo_depths[i])
                        prod.set_nodeattr("inFIFODepths", in_fifo_depths)

                        if node.op_type == "AddStreams_hls":
                            self.slowdown_so_far[0] = max(self.slowdown_so_far)

                except KeyError:
                    raise Exception("Custom op_type %s is currently not supported." % op_type)

        #print("final sizes for each strategy: ",self.delta_total_fifo_size, self.delta_adjusted_fifo_size, self.data_rate_total_fifo_size,self.data_rate_adjusted_fifo_size,self.hybrid_fifo_size, self.hybrid_fifo_size_rate)
        return (model, False)
