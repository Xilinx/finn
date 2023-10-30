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
from onnx import TensorProto, helper
from pyverilator.util.axi_utils import reset_rtlsim, toggle_clk
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    SortGraph,
)

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.util.pyverilator import pyverilate_stitched_ip, verilator_fifosim


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
        # Q_srl FIFOs do not benefit from size < 32
        # add some slack
        return 32
    # otherwise leave as is
    # will be rounded to nearest power of two for Vivado-style FIFO
    return int(depth)


class RemoveShallowFIFOs(Transformation):
    """Remove small FIFOs as the streaming components have depth-2 FIFOs on the
    input/outputs by default."""

    # TODO add unit test

    def __init__(self, shallow_threshold=2):
        self.shallow_threshold = shallow_threshold

    def apply(self, model):
        shallow_fifos = []
        for node in model.graph.node:
            if len(node.input) > 0:
                is_first_node = model.find_producer(node.input[0]) is None
            else:
                is_first_node = True
            if (
                node.op_type == "StreamingFIFO"
                and getCustomOp(node).get_nodeattr("depth") <= self.shallow_threshold
                and (not is_first_node)
            ):
                # bypass shallow fifos
                shallow_fifos.append(node)
                consumers = model.find_consumers(node.output[0])
                if consumers == []:
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


class CapConvolutionFIFODepths(Transformation):
    """Make the size of FIFOs for convolution layers smaller where possible.
    Will be automatically called from InsertAndSetFIFODepths if the appropriate
    constructor flag is set.

    Constructor arguments:

    :parameter max_qsrl_depth: FIFOs deeper than this will use Vivado IP
        instead of Verilog FIFOs (Q_srl.v)

    Assumed input graph properties:

    - all nodes are fpgadataflow nodes
    - FIFOs inserted with InsertAndSetFIFODepths

    Output:

    - graph with smaller-depth FIFOs for convolutions

    Background:
    The simulation-based rtlsim_exec tends to overestimate the required depth
    of FIFOs between the ConvolutionInputGenerator (here called SWG) and the
    MatrixVectorActivation (here called MVAU). As the SWG has an internal buffer of 1
    image row, we use this as a rule of thumb to set FIFO depth to be no larger
    than 1 row.
    """

    # TODO add unit test

    def __init__(self, max_qsrl_depth=256):
        super().__init__()
        self.max_qsrl_depth = max_qsrl_depth

    def apply(self, model):
        # TODO move this to own transformation
        for node in model.graph.node:
            # look for following pattern:
            # ConvolutionInputGenerator -> StreamingFIFO -> MatrixVectorActivation
            if node.op_type == "StreamingFIFO":
                fifo_prod = model.find_producer(node.input[0])
                fifo_cons = model.find_consumer(node.output[0])
                if fifo_prod is None:
                    continue
                if fifo_prod.op_type != "ConvolutionInputGenerator":
                    continue
                if fifo_cons is None:
                    continue
                if fifo_cons.op_type != "MatrixVectorActivation":
                    continue
                op_inst = getCustomOp(node)
                depth = op_inst.get_nodeattr("depth")
                # SWG has an internal buffer of 1 row, so we use this as a
                # rule of thumb to set FIFO depth to be no larger than 1 row
                (bs, h, w, ifold, simd) = op_inst.get_folded_input_shape()
                new_depth = optimize_depth(w * ifold)
                new_depth = min(new_depth, depth)
                op_inst.set_nodeattr("depth", new_depth)
                # Set FIFO implementation/ram styles
                if new_depth > self.max_qsrl_depth:
                    op_inst.set_nodeattr("impl_style", "vivado")
                    op_inst.set_nodeattr("ram_style", "auto")
                else:
                    op_inst.set_nodeattr("impl_style", "rtl")

        return (model, False)


class InsertAndSetFIFODepths(Transformation):
    """Insert appropriate-depth StreamingFIFOs through RTLSim that preserve
    throughput in the created accelerator.

    Constructor arguments:

    :parameter clk_ns: clock period (used for IP preparation)
    :parameter max_qsrl_depth: FIFOs deeper than this will use Vivado IP
        instead of Verilog FIFOs (Q_srl.v)
    :parameter max_depth: how deep the "max"-sized FIFOs initially inserted
        will be. If set to None, use the tensor size as the depth
    :parameter swg_exception: call CapConvolutionFIFODepths to make convolution FIFOs
        smaller where appropriate
    :parameter vivado_ram_style: the StreamingFIFO.ram_style attribute to be used
        for large FIFOs implemented by Vivado afterwards

    Assumed input graph properties:

    - all nodes are fpgadataflow nodes
    - no FIFOs inserted,
    - (inFIFODepths/outFIFODepths attrs will be ignored)

    Output:

    - graph with appropriate-depth FIFOs inserted

    Background:
    Even with all FINN HLS fpgadatflow layers appropriately parallelized, it is
    necessary to insert FIFOs between them to prevent stalls due to bursty
    behavior. The sizes of those FIFOs are hard to predict analytically, so
    we do the following:

    - insert deep (=tensor size) FIFOs between all fpgadataflow nodes
    - create stitched design
    - run through rtlsim with stream of multiple random input images (to fill pipeline)
    - keep track of observed maximum occupancy for each FIFO during rtlsim
    - when sim finished, update each FIFO depth to maximum observed occupancy
      and set inFIFODepths/outFIFODepths attrs to that depth as well

    """

    def __init__(
        self,
        fpgapart,
        clk_ns=10.0,
        max_qsrl_depth=256,
        max_depth=None,
        swg_exception=True,
        vivado_ram_style="auto",
        force_python_sim=False,
    ):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.max_qsrl_depth = max_qsrl_depth
        self.max_depth = max_depth
        self.swg_exception = swg_exception
        self.vivado_ram_style = vivado_ram_style
        self.force_python_sim = force_python_sim

    def apply(self, model):
        # these optypes may potentially use external weights
        # we'll temporarily change them to use decoupled mode for FIFO sizing
        extw_optypes = ["MatrixVectorActivation", "VectorVectorActivation"]
        # change external to decoupled and warn user
        # this way we are sure we have exactly one input/output
        modified_fc_nodes = []
        for node in model.graph.node:
            # verify assumptions
            assert is_fpgadataflow_node(node), "Found non-fpgadataflow node: " + str(node)
            assert node.op_type != "StreamingFIFO", "Found existing StreamingFIFO node"
            node = getCustomOp(node)
            ifd = node.get_nodeattr("inFIFODepths")
            ofd = node.get_nodeattr("outFIFODepths")
            if self.max_depth is not None:
                ifd = [self.max_depth] * len(ifd)
                ofd = [self.max_depth] * len(ofd)
            else:
                # set each FIFO to its tensor size
                # (except stream width hence the :-1)
                for i in range(len(ifd)):
                    ifd[i] = np.prod(node.get_folded_input_shape(i)[:-1])
                for o in range(len(ofd)):
                    ofd[o] = np.prod(node.get_folded_output_shape(o)[:-1])
            node.set_nodeattr("inFIFODepths", ifd)
            node.set_nodeattr("outFIFODepths", ofd)

            if node.onnx_node.op_type in extw_optypes:
                mmode = node.get_nodeattr("mem_mode")
                if mmode == "external":
                    modified_fc_nodes.append(node.onnx_node.name)
                    node.set_nodeattr("mem_mode", "decoupled")
                    reset_implementation(node)
                    warnings.warn(
                        "Changed mem_mode from external to decoupled for " + node.onnx_node.name
                    )

        # insert stream infrastructure (DWC/FIFO)
        model = model.transform(InsertDWC())
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        # gather FIFO names, check they are of expected depth
        fifos = {}
        fifo_nodes = model.get_nodes_by_op_type("StreamingFIFO")
        for node in fifo_nodes:
            fifos[node.name] = 0
            node = getCustomOp(node)
            node.set_nodeattr("depth_monitor", 1)
            node.set_nodeattr("impl_style", "rtl")
            # check depths and fix as necessary
            if (self.max_depth is not None) and (node.get_nodeattr("depth") != self.max_depth):
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

        if self.force_python_sim:
            # do rtlsim in Python for FIFO sizing
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

            reset_rtlsim(sim)
            toggle_clk(sim)

            # set all input valids to 0 and output readies to 1
            # set input data to some constant
            set_signal(sim, "tvalid", 0)
            set_signal(sim, "tready", 1)
            set_signal(sim, "tdata", 0)

            output_detected = False
            while ncycles > 0:
                toggle_clk(sim)
                # set/unset valids
                if ncycles % ncycles_per_input == 0:
                    set_signal(sim, "tvalid", 1)
                else:
                    set_signal(sim, "tvalid", 0)

                # since latency estimation is very pessimistic, detect first output
                # and fast-forward the sim
                if get_signal(sim, "tvalid") != 0 and not output_detected:
                    ncycles = max_cycles
                    output_detected = True
                else:
                    ncycles = ncycles - 1

            if not output_detected:
                warnings.warn("No output detected, calculated FIFO depths may not be correct")
        else:
            # do rtlsim in C++ for FIFO sizing
            # determine # inputs for FIFO sizing according to topology type
            swg_nodes = [x for x in model.graph.node if "ConvolutionInputGenerator" in x.op_type]
            if len(swg_nodes) == 0:
                # MLP, no layer overlap
                # assuming half the nodes are now FIFOs, use half the # of
                # nodes as # inputs to drive the imulation
                n_inputs = int(len(model.graph.node) / 2)
            else:
                # convnet, two inputs are typically enough to fill entire
                # layer pipeline due to overlaps
                n_inputs = 2
            sim = verilator_fifosim(model, n_inputs)

        for ind, node in enumerate(fifo_nodes):
            maxcount_name = "maxcount_%d" % ind
            if ind == 0:
                maxcount_name = "maxcount"
            fifos[node.name] = sim[maxcount_name]

        # Apply depths back into the model;
        # also set in/outFIFODepths to zero for non-FIFO
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
                node_inst.set_nodeattr("depth_monitor", 0)
                # exception for top-level IO FIFOs which cause a bug in simulation
                # (top-level IOs should not have impl_style=vivado)
                toplevel_in = node.input[0] in [x.name for x in model.graph.input]
                toplevel_out = node.output[0] in [x.name for x in model.graph.output]
                toplevel_style_exception = toplevel_in or toplevel_out
                # Set FIFO implementation/ram styles
                if (depth > self.max_qsrl_depth) and (not toplevel_style_exception):
                    node_inst.set_nodeattr("impl_style", "vivado")
                    node_inst.set_nodeattr("ram_style", self.vivado_ram_style)
                else:
                    node_inst.set_nodeattr("impl_style", "rtl")
                # reset implementation
                reset_implementation(node_inst)
                del fifos[node.name]
            else:
                # (removed setting of node FIFO size attributes to 0 here)
                # for every extw node we changed from external to decoupled,
                # change back and reset implementation
                if node.op_type in extw_optypes:
                    if node.name in modified_fc_nodes:
                        node_inst = getCustomOp(node)
                        node_inst.set_nodeattr("mem_mode", "external")
                        reset_implementation(node_inst)
                        modified_fc_nodes.remove(node.name)

        assert (
            len(modified_fc_nodes) == 0 and len(fifos.keys()) == 0
        ), "FIFO/FC nodes left untouched after model reconfiguration"

        # handle custom sizing for SWG FIFOs if desired
        if self.swg_exception:
            model = model.transform(CapConvolutionFIFODepths(max_qsrl_depth=self.max_qsrl_depth))
        # remove shallow FIFOs
        model = model.transform(RemoveShallowFIFOs())

        # reflect final values in attributes
        for node in model.graph.node:
            if node.op_type != "StreamingFIFO":
                node_inst = getCustomOp(node)
                fifodepth_in = []
                for node_inp in node.input:
                    prod = model.find_producer(node_inp)
                    if prod is None:
                        # no producer for this input
                        if node_inp in [x.name for x in model.graph.input]:
                            # top-level input with no FIFO
                            fifodepth_in.append(0)
                        else:
                            # FIFO depth attr applies only to dynamic attributes
                            pass
                    else:
                        # there is a producer for this input
                        if prod.op_type == "StreamingFIFO":
                            prod_inst = getCustomOp(prod)
                            fifodepth_in.append(prod_inst.get_nodeattr("depth"))
                        else:
                            # explicitly no FIFO on this dynamic input
                            fifodepth_in.append(0)
                fifodepth_out = []
                for node_out in node.output:
                    cons = model.find_consumer(node_out)
                    if cons is None:
                        # no consumer for this output
                        if node_out in [x.name for x in model.graph.output]:
                            # top-level output with no FIFO
                            fifodepth_out.append(0)
                        else:
                            # FIFO depth attr applies only to dynamic attributes
                            pass
                    else:
                        # there is a consumer for this input
                        if cons.op_type == "StreamingFIFO":
                            cons_inst = getCustomOp(cons)
                            fifodepth_out.append(cons_inst.get_nodeattr("depth"))
                        else:
                            # explicitly no FIFO on this dynamic output
                            fifodepth_out.append(0)
                node_inst.set_nodeattr("inFIFODepths", fifodepth_in)
                node_inst.set_nodeattr("outFIFODepths", fifodepth_out)

        return (model, False)


def get_fifo_split_configs(depth, max_qsrl_depth=256, max_vivado_depth=32768):
    """Break non-power-of-2 sized FIFO depths into several ones"""

    def floor_pow2(x):
        if (x & (x - 1) == 0) and x != 0:
            return x
        else:
            return 1 << ((x - 1).bit_length() - 1)

    def decompose_pow2(x):
        if x <= max_qsrl_depth:
            return [x]
        else:
            r = floor_pow2(x)
            if x == r:
                return [x]
            else:
                return [r, *decompose_pow2(x - r)]

    ret = []
    # trivial case: for small FIFOs, return as-is with rtl style
    if depth <= max_qsrl_depth:
        return [(depth, "rtl")]
    # first pass: ensure max depth is respected
    # (restricted by Vivado AXIS infra IP)
    remainder = depth
    while remainder != 0:
        if remainder > max_vivado_depth:
            ret.append(max_vivado_depth)
            remainder -= max_vivado_depth
        else:
            ret.append(remainder)
            remainder = 0
    # second pass: break non-power-of-2 sized FIFOs
    # into several ones

    ret_pass2 = list(map(decompose_pow2, ret))
    # unpack list of lists
    ret_pass2 = [x for dec_list in ret_pass2 for x in dec_list]

    # finally, add impl_style to each split FIFO
    ret_final = []
    for cand_depth in ret_pass2:
        if cand_depth <= max_qsrl_depth:
            ret_final.append((cand_depth, "rtl"))
        else:
            ret_final.append((cand_depth, "vivado"))

    return ret_final


class SplitLargeFIFOs(Transformation):
    """Split large FIFOs before implementation, for two reasons:

    - impl_style="vivado" supports a max depth of 32k. Any larger
      FIFOs must be implemented as a sequence of smaller FIFOs.
    - impl_style="vivado" requires power-of-two depths, which is
      normally handled by rounding up to the nearest power-of-two.
      So a FIFO of size 8196 normally gets rounded-up to a depth of
      16384 and wastes a lot of resources. Here, instead, we split
      this up into two FIFOs of depth 8192 + 4.

    """

    def __init__(self, max_qsrl_depth=256, max_vivado_depth=32768):
        super().__init__()
        self.max_qsrl_depth = max_qsrl_depth
        self.max_vivado_depth = max_vivado_depth

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "StreamingFIFO":
                n_inst = getCustomOp(node)
                depth = n_inst.get_nodeattr("depth")
                cfgs = get_fifo_split_configs(depth, self.max_qsrl_depth, self.max_vivado_depth)
                if len(cfgs) > 1:
                    fld_shape = n_inst.get_folded_output_shape()
                    dtype = n_inst.get_nodeattr("dataType")
                    ram_style = n_inst.get_nodeattr("ram_style")
                    shape = model.get_tensor_shape(node.input[0])
                    for i, (fifo_depth, impl_style) in enumerate(cfgs):
                        if i == 0:
                            inp = node.input[0]
                        else:
                            inp = node.name + "_" + str(i - 1) + "_out"
                        if i == len(cfgs) - 1:
                            outp = node.output[0]
                        else:
                            outp = node.name + "_" + str(i) + "_out"
                            out_tensor = helper.make_tensor_value_info(
                                outp, TensorProto.FLOAT, shape
                            )
                            graph.value_info.append(out_tensor)
                            model.set_tensor_datatype(out_tensor.name, DataType[dtype])
                        fifo_node = helper.make_node(
                            "StreamingFIFO",
                            [inp],
                            [outp],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            depth=fifo_depth,
                            folded_shape=fld_shape,
                            dataType=dtype,
                            impl_style=impl_style,
                            ram_style=ram_style,
                            name=node.name + "_" + str(i),
                        )
                        graph.node.insert(node_ind + i, fifo_node)

                    graph.node.remove(node)
                    graph_modified = True
        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(GiveReadableTensorNames())
        return (model, False)
