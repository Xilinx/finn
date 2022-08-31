# Copyright (c) 2022, Xilinx
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
import qonnx.custom_op.registry as registry
import warnings
from pyverilator.util.axi_utils import _read_signal, reset_rtlsim, rtlsim_multi_io
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import NodeLocalTransformation

from finn.util.fpgadataflow import is_fpgadataflow_node


class DeriveCharacteristic(NodeLocalTransformation):
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

    def __init__(self, period, num_workers=None, manual_bypass=False):
        super().__init__(num_workers=num_workers)
        self.period = period
        self.manual_bypass = manual_bypass

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_fpgadataflow_node(node) is True:
            try:
                # lookup op_type in registry of CustomOps
                inst = registry.getCustomOp(node)
                # TODO move into HLSCustomOp?
                # ideally, call execute with rtlsim mode and
                # specify some way of setting up a hook
                # ensure rtlsim is ready
                assert inst.get_nodeattr("rtlsim_so") != "", (
                    "rtlsim not ready for " + node.name
                )
                if inst.get_nodeattr("io_chrc_period") > 0:
                    warnings.warn(
                        "Skipping node %s: already has FIFO characteristic" % node.name
                    )
                    return (node, False)
                # restricted to single input and output nodes for now
                multistream_optypes = [
                    "StreamingConcat",
                ]
                if node.op_type in multistream_optypes:
                    warnings.warn(f"Skipping {node.name} for rtlsim characterization")
                    return (node, False)
                exp_cycles = inst.get_exp_cycles()
                n_inps = np.prod(inst.get_folded_input_shape()[:-1])
                n_outs = np.prod(inst.get_folded_output_shape()[:-1])
                if exp_cycles == 0:
                    # try to come up with an optimistic estimate
                    exp_cycles = min(n_inps, n_outs)
                assert (
                    exp_cycles <= self.period
                ), "Period %d too short to characterize %s : expects min %d cycles" % (
                    self.period,
                    node.name,
                    exp_cycles,
                )
                sim = inst.get_rtlsim()
                # signal name
                sname = "_" + inst.hls_sname() + "_"
                io_dict = {
                    "inputs": {
                        "in0": [0 for i in range(n_inps)],
                        # "weights": wei * num_w_reps
                    },
                    "outputs": {"out": []},
                }
                # override for certain fork/join nodes
                if node.op_type == "DuplicateStreams_Batch":
                    del io_dict["outputs"]["out"]
                    io_dict["outputs"]["out0"] = []
                    io_dict["outputs"]["out1"] = []
                    # n_outs is total of output streams
                    # so multiply expected by 2
                    n_outs *= 2
                elif node.op_type == "AddStreams_Batch":
                    io_dict["inputs"]["in1"] = [0 for i in range(n_inps)]

                try:
                    # fill out weight stream for decoupled-mode components
                    mem_mode = inst.get_nodeattr("mem_mode")
                    if mem_mode in ["decoupled", "external"]:
                        if op_type == "Thresholding_Batch":
                            n_weight_inps = inst.calc_tmem()
                        else:
                            n_weight_inps = inst.calc_wmem()
                        num_w_reps = np.prod(inst.get_nodeattr("numInputVectors"))
                        io_dict["inputs"]["weights"] = [
                            0 for i in range(num_w_reps * n_weight_inps)
                        ]
                except AttributeError:
                    pass

                # extra dicts to keep track of cycle-by-cycle transaction behavior
                # note that we restrict key names to filter out weight streams etc
                txns_in = {
                    key: [] for (key, value) in io_dict["inputs"].items() if "in" in key
                }
                txns_out = {
                    key: []
                    for (key, value) in io_dict["outputs"].items()
                    if "out" in key
                }

                def monitor_txns(sim_obj):
                    for inp in txns_in:
                        in_ready = _read_signal(sim, inp + sname + "TREADY") == 1
                        in_valid = _read_signal(sim, inp + sname + "TVALID") == 1
                        if in_ready and in_valid:
                            txns_in[inp].append(1)
                        else:
                            txns_in[inp].append(0)
                    for outp in txns_out:
                        if (
                            _read_signal(sim, outp + sname + "TREADY") == 1
                            and _read_signal(sim, outp + sname + "TVALID") == 1
                        ):
                            txns_out[outp].append(1)
                        else:
                            txns_out[outp].append(0)

                reset_rtlsim(sim)
                total_cycle_count = rtlsim_multi_io(
                    sim,
                    io_dict,
                    n_outs,
                    sname=sname,
                    liveness_threshold=self.period,
                    hook_preclk=monitor_txns,
                )
                assert total_cycle_count <= self.period
                inst.set_nodeattr("io_chrc_period", self.period)

                def accumulate_char_fxn(chrc):
                    p = len(chrc)
                    ret = []
                    for t in range(2 * p):
                        if t == 0:
                            ret.append(chrc[0])
                        else:
                            ret.append(ret[-1] + chrc[t % p])
                    return np.asarray(ret, dtype=np.int32)

                all_txns_in = np.empty((len(txns_in.keys()), 2 * self.period))
                all_txns_out = np.empty((len(txns_out.keys()), 2 * self.period))
                all_pad_in = []
                all_pad_out = []
                for in_idx, in_strm_nm in enumerate(txns_in.keys()):
                    txn_in = txns_in[in_strm_nm]
                    if len(txn_in) < self.period:
                        pad_in = self.period - len(txn_in)
                        txn_in += [0 for x in range(pad_in)]
                    txn_in = accumulate_char_fxn(txn_in)
                    all_txns_in[in_idx, :] = txn_in
                    all_pad_in.append(pad_in)

                for out_idx, out_strm_nm in enumerate(txns_out.keys()):
                    txn_out = txns_out[out_strm_nm]
                    if len(txn_out) < self.period:
                        pad_out = self.period - len(txn_out)
                        txn_out += [0 for x in range(pad_out)]
                    txn_out = accumulate_char_fxn(txn_out)
                    all_txns_out[out_idx, :] = txn_out
                    all_pad_out.append(pad_out)

                # TODO specialize here for DuplicateStreams and AddStreams
                inst.set_nodeattr("io_chrc_in", all_txns_in)
                inst.set_nodeattr("io_chrc_out", all_txns_out)
                inst.set_nodeattr("io_chrc_pads_in", all_pad_in)
                inst.set_nodeattr("io_chrc_pads_out", all_pad_out)

            except KeyError:
                # exception if op_type is not supported
                raise Exception(
                    "Custom op_type %s is currently not supported." % op_type
                )
        return (node, False)

    def apply(self, model: ModelWrapper):
        (model, run_again) = super().apply(model)
        if not self.manual_bypass:
            return (model, run_again)
        # apply manual fix for DuplicateStreams and AddStreams for
        # simple residual reconvergent paths with bypass
        addstrm_nodes = model.get_nodes_by_op_type("AddStreams_Batch")
        for addstrm_node in addstrm_nodes:
            # we currently only support the case where one branch is
            # a bypass
            b0 = model.find_producer(addstrm_node.input[0])
            b1 = model.find_producer(addstrm_node.input[1])
            if (b0 is None) or (b1 is None):
                warnings.warn("Found unsupported AddStreams, skipping")
                return (model, run_again)
            b0_is_bypass = b0.op_type == "DuplicateStreams_Batch"
            b1_is_bypass = b1.op_type == "DuplicateStreams_Batch"
            if (not b0_is_bypass) and (not b1_is_bypass):
                warnings.warn("Found unsupported AddStreams, skipping")
                return (model, run_again)
            ds_node = b0 if b0_is_bypass else b1
            comp_branch_last = b1 if b0_is_bypass else b0

            ds_comp_bout = ds_node.output[0] if b0_is_bypass else ds_node.output[1]
            comp_branch_first = model.find_consumer(ds_comp_bout)
            if comp_branch_first is None or comp_branch_last is None:
                warnings.warn("Found unsupported DuplicateStreams, skipping")
                return (model, run_again)
            comp_branch_last = registry.getCustomOp(comp_branch_last)
            comp_branch_first = registry.getCustomOp(comp_branch_first)
            # for DuplicateStreams, use comp_branch_first's input characterization
            # for AddStreams, use comp_branch_last's output characterization
            period = comp_branch_first.get_nodeattr("io_chrc_period")
            comp_branch_first_f = comp_branch_first.get_nodeattr("io_characteristic")[
                : 2 * period
            ]
            comp_branch_last_f = comp_branch_last.get_nodeattr("io_characteristic")[
                2 * period :
            ]
            ds_node_inst = registry.getCustomOp(ds_node)
            addstrm_node_inst = registry.getCustomOp(addstrm_node)
            ds_node_inst.set_nodeattr("io_chrc_period", period)
            ds_node_inst.set_nodeattr("io_characteristic", comp_branch_first_f * 2)
            addstrm_node_inst.set_nodeattr("io_chrc_period", period)
            addstrm_node_inst.set_nodeattr("io_characteristic", comp_branch_last_f * 2)
            warnings.warn(
                f"Set {ds_node.name} chrc. from {comp_branch_first.onnx_node.name}"
            )
            warnings.warn(
                f"Set {addstrm_node.name} chrc. from {comp_branch_last.onnx_node.name}"
            )
        return (model, run_again)


class DeriveFIFOSizes(NodeLocalTransformation):
    """Prerequisite: DeriveCharacteristic already called on graph.
    For each node in the graph, use the accumulated I/O characteristic function
    to perform FIFO sizing, setting the in/outFIFODepth attributes of HLSCustomOp
    nodes.

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(self, num_workers=None):
        super().__init__(num_workers=num_workers)

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_fpgadataflow_node(node) is True:
            try:
                # lookup op_type in registry of CustomOps
                prod = registry.getCustomOp(node)
                assert op_type != "StreamingFIFO", "Found existing FIFOs"
                period = prod.get_nodeattr("io_chrc_period")
                prod_chrc = prod.get_nodeattr("io_chrc_out")[0]
                assert (
                    len(prod_chrc) == 2 * period
                ), "Found unexpected characterization attribute"
                if prod.get_nodeattr("outFIFODepth") > 2:
                    # FIFO depth already set, can skip this node
                    return (node, False)

                # find consumers
                model = self.ref_input_model
                out_fifo_depths = []
                for output_name in node.output:
                    cons_node = model.find_consumer(output_name)
                    if cons_node is None:
                        # could be final node, will be overridden if so
                        # need an entry in the list anyway
                        out_fifo_depths.append(2)
                        continue
                    cons = registry.getCustomOp(cons_node)
                    cons_chrc = cons.get_nodeattr("io_chrc_in")[0]
                    # find minimum phase shift satisfying the constraint
                    pshift_min = period - 1
                    for pshift_cand in range(period):
                        prod_chrc_part = prod_chrc[pshift_cand:period]
                        cons_chrc_part = cons_chrc[: period - pshift_cand]
                        if (prod_chrc_part >= cons_chrc_part).all():
                            pshift_min = pshift_cand
                            break
                    prod_chrc_part = prod_chrc[pshift_min : (pshift_min + period)]
                    cons_chrc_part = cons_chrc[:period]
                    fifo_depth = int((prod_chrc_part - cons_chrc_part).max())
                    out_fifo_depths.append(fifo_depth)
                # set output FIFO depth for this (producing) node
                # InsertFIFO looks at the max of (outFIFODepth, inFIFODepth)
                # for each tensor
                if len(out_fifo_depths) > 0:
                    prod.set_nodeattr("outFIFODepth", out_fifo_depths[0])
                # used only for multi-producer nodes
                prod.set_nodeattr("outFIFODepths", out_fifo_depths)

            except KeyError:
                # exception if op_type is not supported
                raise Exception(
                    "Custom op_type %s is currently not supported." % op_type
                )
        return (node, False)
