# Copyright (c) 2020, Xilinx
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
import math
import numpy as np
import warnings
from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.general import GiveUniqueNodeNames
from finn.core.rtlsim_exec import (
    _reset_rtlsim,
    _toggle_clk,
)
from finn.util.fpgadataflow import (
    pyverilate_stitched_ip,
)


def set_signal(sim, keyw, value):
    for i in range(len(sim.inputs)):
        input_name = sim.inputs[i][0]
        if keyw in input_name:
            sim.io[input_name] = value


def optimize_depth(depth):
    if depth <= 2:
        return 2
    if depth <= 32:
        return 32
    if depth <= 1024:
        return int(2 ** math.ceil(math.log2(depth)))
    return int(math.ceil(depth / 1024))


class SetFIFODepths(Transformation):
    """Determines minimum depths of StreamingFIFOs through RTLSim.
    We assume we get a dataflow partition (all nodes are dataflow, no FIFOs)
    We set initial depths very high (16k), run sim with multiple
    images on input (random/constant data) and keep track of maximum
    occupancy counts in each FIFO."""

    def __init__(self, fpgapart, clk_ns=10.0):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns

    def apply(self, model):

        orig_model = model

        # work on a copy of the model
        model = copy.deepcopy(model)

        # change external to decoupled and warn user;
        # this way we are sure we have exactly one input/output
        for node in model.graph.node:
            node = getCustomOp(node)
            node.set_nodeattr("inFIFODepth", 2 ** 14)
            node.set_nodeattr("outFIFODepth", 2 ** 14)
            if node.onnx_node.op_type == "StreamingFCLayer_Batch":
                mmode = node.get_nodeattr("mem_mode")
                if mmode == "external":
                    node.set_nodeattr("mem_mode", "decoupled")
                    warnings.warn(
                        "Changed mem_mode from external to decoupled for "
                        + node.onnx_node.name
                    )

        # insert stream infrastructure (DWC/FIFO)
        model = model.transform(InsertDWC())
        model = model.transform(InsertFIFO())
        model = model.transform(GiveUniqueNodeNames())

        # gather FIFO names, check they are of expected depth
        fifos = {}
        for node in model.graph.node:
            if node.op_type == "StreamingFIFO":
                consumer = model.find_consumers(node.output[0])
                if consumer is not None:
                    consumer = consumer[0].name
                producer = model.find_producer(node.input[0])
                if producer is not None:
                    producer = producer.name
                fifos[node.name] = {
                    "depth": 0,
                    "consumer": consumer,
                    "producer": producer,
                }
                node = getCustomOp(node)
                # check depths
                # if model came in with FIFOs, the depths will not have been updated
                if node.get_nodeattr("depth") != 2 ** 14:
                    node.set_nodeattr("depth", 2 ** 14)

        # insert FIFOs and do all transformations for RTLsim
        model = model.transform(AnnotateCycles())
        perf = model.analysis(dataflow_performance)
        latency = perf["critical_path_cycles"]
        max_cycles = perf["max_cycles"]
        model = model.transform(PrepareIP(self.fpgapart, self.clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(self.fpgapart, self.clk_ns))
        model.set_metadata_prop("exec_mode", "rtlsim")

        # calculate input frequency (number of cycles for each input word)
        first_node = getCustomOp(model.graph.node[0])
        ncycles_per_input = math.ceil(
            perf["max_cycles"]
            / (
                np.prod(first_node.get_folded_input_shape())
                / first_node.get_folded_input_shape()[-1]
            )
        )

        # set sufficiently large threshold for 1 image to  fully execute and exit
        ncycles = int(latency + max_cycles)

        # prepare pyverilator model
        sim = pyverilate_stitched_ip(model)

        _reset_rtlsim(sim)
        _toggle_clk(sim)

        # set all input valids to 0 and output readies to 1
        # set input data to some constant
        set_signal(sim, "tvalid", 0)
        set_signal(sim, "tready", 1)
        set_signal(sim, "tdata", 0)

        while ncycles > 0:
            _toggle_clk(sim)
            # set/unset valids
            if ncycles % ncycles_per_input == 0:
                set_signal(sim, "tvalid", 1)
            else:
                set_signal(sim, "tvalid", 0)

            # check/update all fifo counts
            for key in fifos:
                current_state = sim.internals["finn_design_i"][key]["inst"][
                    key + "_" + key
                ]["state"]
                current_addr = sim.internals["finn_design_i"][key]["inst"][
                    key + "_" + key
                ]["addr"]
                if current_state == 2:
                    current_count = current_addr + 2
                else:
                    current_count = current_state
                if current_count > fifos[key]["depth"]:
                    fifos[key]["depth"] = current_count
            ncycles = ncycles - 1

        # for each node in the original graph, determine in/outFIFODepth
        ret = {}
        for key in fifos:
            predecessor_node = fifos[key]["producer"]
            if predecessor_node is not None:
                if predecessor_node not in ret:
                    ret[predecessor_node] = {"inFIFODepth": 0, "outFIFODepth": 0}
                out_depth = ret[predecessor_node]["outFIFODepth"]
                ret[predecessor_node]["outFIFODepth"] = max(
                    out_depth, fifos[key]["depth"]
                )

            succcessor_node = fifos[key]["consumer"]
            if succcessor_node is not None:
                if succcessor_node not in ret:
                    ret[succcessor_node] = {"inFIFODepth": 0, "outFIFODepth": 0}
                in_depth = ret[succcessor_node]["inFIFODepth"]
                ret[succcessor_node]["inFIFODepth"] = max(in_depth, fifos[key]["depth"])

        # tweak and apply depths to original model
        for node in orig_model.graph.node:
            if node.name in ret:
                depths = ret[node.name]
                node = getCustomOp(node)
                node.set_nodeattr("inFIFODepth", optimize_depth(depths["inFIFODepth"]))
                node.set_nodeattr(
                    "outFIFODepth", optimize_depth(depths["outFIFODepth"])
                )

        return (orig_model, False)
