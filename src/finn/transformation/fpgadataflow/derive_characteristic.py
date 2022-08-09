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
from pyverilator.util.axi_utils import _read_signal, reset_rtlsim, rtlsim_multi_io
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

    def __init__(self, period, num_workers=None):
        super().__init__(num_workers=num_workers)
        self.period = period

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
                # restricted to single input and output nodes for now
                multistream_optypes = [
                    "AddStreams_Batch",
                    "DuplicateStreams_Batch",
                    "StreamingConcat",
                ]
                assert (
                    node.op_type not in multistream_optypes
                ), f"{node.name} unsupported"
                try:
                    mem_mode = inst.get_nodeattr("mem_mode")
                    assert mem_mode == "const", "Only mem_mode=const supported for now"
                except AttributeError:
                    pass
                exp_cycles = inst.get_exp_cycles()
                n_inps = np.prod(inst.get_folded_input_shape()[:-1])
                n_outs = np.prod(inst.get_folded_output_shape()[:-1])
                if exp_cycles == 0:
                    # try to come up with an optimistic estimate
                    exp_cycles = min(n_inps, n_outs)
                assert (
                    exp_cycles < self.period
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

                txns_in = []
                txns_out = []

                def monitor_txns(sim_obj):
                    for inp in io_dict["inputs"]:
                        in_ready = _read_signal(sim, inp + sname + "TREADY") == 1
                        in_valid = _read_signal(sim, inp + sname + "TVALID") == 1
                        if in_ready and in_valid:
                            txns_in.append(1)
                        else:
                            txns_in.append(0)
                    for outp in io_dict["outputs"]:
                        if (
                            _read_signal(sim, outp + sname + "TREADY") == 1
                            and _read_signal(sim, outp + sname + "TVALID") == 1
                        ):
                            txns_out.append(1)
                        else:
                            txns_out.append(0)

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
                if len(txns_in) < self.period:
                    txns_in += [0 for x in range(self.period - len(txns_in))]
                if len(txns_out) < self.period:
                    txns_out += [0 for x in range(self.period - len(txns_out))]
                io_characteristic = txns_in + txns_out
                inst.set_nodeattr("io_characteristic", io_characteristic)
                inst.set_nodeattr("io_characteristic_period", self.period)
            except KeyError:
                # exception if op_type is not supported
                raise Exception(
                    "Custom op_type %s is currently not supported." % op_type
                )
        return (node, False)
