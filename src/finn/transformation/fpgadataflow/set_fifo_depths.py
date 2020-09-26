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
from finn.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from finn.core.rtlsim_exec import (
    _reset_rtlsim,
    _toggle_clk,
)
from finn.util.fpgadataflow import (
    pyverilate_stitched_ip,
)


def reset_implementation(node):
    node.set_nodeattr("code_gen_dir_ipgen", "")
    node.set_nodeattr("ipgen_path", "")
    node.set_nodeattr("ip_path", "")


def set_signal(sim, keyw, value):
    for i in range(len(sim.inputs)):
        input_name = sim.inputs[i][0]
        if keyw in input_name:
            sim.io[input_name] = value


def get_signal(sim, keyw):
    for i in range(len(sim.outputs)):
        output_name = sim.outputs[i][0]
        if keyw in output_name:
            return sim.io[output_name]


def optimize_depth(depth):
    if depth <= 2:
        return 2
    if depth <= 32:
        return 32
    if depth <= 1024:
        return int(2 ** math.ceil(math.log2(depth)))
    return int(math.ceil(depth / 1024) * 1024)


class SetFIFODepths(Transformation):
    """Determines minimum depths of StreamingFIFOs through RTLSim.
    We assume we get a dataflow partition (all nodes are dataflow, no FIFOs)
    We set initial depths very high (16k), run sim with multiple
    images on input (random/constant data) and keep track of maximum
    occupancy counts in each FIFO."""

    def __init__(self, fpgapart, clk_ns=10.0, max_qsrl_depth=256, max_depth=2 ** 14):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.max_qsrl_depth = max_qsrl_depth
        self.max_depth = max_depth

    def apply(self, model):

        # change external to decoupled and warn user
        # this way we are sure we have exactly one input/output
        modified_fc_nodes = []
        for node in model.graph.node:
            node = getCustomOp(node)
            node.set_nodeattr("inFIFODepth", self.max_depth)
            node.set_nodeattr("outFIFODepth", self.max_depth)
            if node.onnx_node.op_type == "StreamingFCLayer_Batch":
                mmode = node.get_nodeattr("mem_mode")
                if mmode == "external":
                    modified_fc_nodes.append(node.onnx_node.name)
                    node.set_nodeattr("mem_mode", "decoupled")
                    reset_implementation(node)
                    warnings.warn(
                        "Changed mem_mode from external to decoupled for "
                        + node.onnx_node.name
                    )

        # insert stream infrastructure (DWC/FIFO)
        model = model.transform(InsertDWC())
        model = model.transform(InsertFIFO())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        # gather FIFO names, check they are of expected depth
        fifos = {}
        for node in model.graph.node:
            if node.op_type == "StreamingFIFO":
                fifos[node.name] = 0
                node = getCustomOp(node)
                # check depths and fix as necessary
                if node.get_nodeattr("depth") != self.max_depth:
                    node.set_nodeattr("depth", self.max_depth)

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
        ncycles_per_input = max(
            1,
            int(
                math.ceil(
                    perf["max_cycles"]
                    / (
                        np.prod(first_node.get_folded_input_shape())
                        / first_node.get_folded_input_shape()[-1]
                    )
                )
            ),
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

        output_detected = False
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
                if current_count > fifos[key]:
                    fifos[key] = current_count

            # since latency estimation is very pessimistic, detect first output
            # and fast-forward the sim
            if get_signal(sim, "tvalid") != 0 and not output_detected:
                ncycles = max_cycles
                output_detected = True
            else:
                ncycles = ncycles - 1

        if not output_detected:
            warnings.warn(
                "No output detected, calculated FIFO depths may not be correct"
            )

        # Apply depths back into the model;
        # also set in/outFIFODepth to zero for non-FIFO
        # nodes, preventing further FIFO insertion
        for node in model.graph.node:
            # set FIFO depth, reset FIFO implementation,
            # and set implementation/ram styles
            if node.op_type == "StreamingFIFO":
                assert node.name in fifos, "FIFO node not found in size dictionary"
                # set depth of FIFO
                depth = optimize_depth(fifos[node.name])
                node_inst = getCustomOp(node)
                node_inst.set_nodeattr("depth", depth)
                # Set FIFO implementation/ram styles
                if depth > self.max_qsrl_depth:
                    node_inst.set_nodeattr("impl_style", "vivado")
                    node_inst.set_nodeattr("ram_style", "auto")
                else:
                    node_inst.set_nodeattr("impl_style", "rtl")
                # reset implementation
                reset_implementation(node_inst)
                del fifos[node.name]
            else:
                getCustomOp(node).set_nodeattr("inFIFODepth", 0)
                getCustomOp(node).set_nodeattr("outFIFODepth", 0)
                # for every FC node we changed from external to decoupled,
                # change back and reset implementation
                if node.op_type == "StreamingFCLayer_Batch":
                    if node.name in modified_fc_nodes:
                        node_inst = getCustomOp(node)
                        node_inst.set_nodeattr("mem_mode", "external")
                        reset_implementation(node_inst)
                        modified_fc_nodes.remove(node.name)

        assert (
            len(modified_fc_nodes) == 0 and len(fifos.keys()) == 0
        ), "FIFO/FC nodes left untouched after model reconfiguration"

        # Remove FIFOs which have depth <= 2
        shallow_fifos = []
        # First, bypass them
        for node in model.graph.node:
            if (
                node.op_type == "StreamingFIFO"
                and getCustomOp(node).get_nodeattr("depth") <= 2
            ):
                shallow_fifos.append(node)
                consumers = model.find_consumers(node.output[0])
                if consumers is None:
                    producer = model.find_producer(node.input[0])
                    for idx, inp in enumerate(producer.output):
                        if inp == node.input[0]:
                            producer.output[idx] = node.output[0]
                else:
                    assert len(consumers) == 1, "Fanout detected from FIFO output"
                    consumer = consumers[0]
                    # set fifo input tensor as new input tensor of second node
                    for idx, inp in enumerate(consumer.input):
                        if inp == node.output[0]:
                            consumer.input[idx] = node.input[0]
        # now filter out
        for node_to_remove in shallow_fifos:
            model.graph.node.remove(node_to_remove)

        return (model, False)
