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

from finn.util.create import hls_random_mlp_maker
from finn.core.datatype import DataType
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.custom_op.registry import getCustomOp
from finn.core.rtlsim_exec import rtlsim_exec
from finn.util.basic import gen_finn_dt_tensor
from finn.util.pyverilator import axilite_read, axilite_write
import numpy as np
import os

test_fpga_part = "xc7z020clg400-1"
target_clk_ns = 5


def test_runtime_weights_single_layer():
    idt = DataType.UINT32
    wdt = DataType.UINT8
    act = None
    mw = 4
    mh = 4
    pe = 1
    simd = 1
    layer_spec = {
        "idt": idt,
        "wdt": wdt,
        "mw": mw,
        "mh": mh,
        "act": act,
        "pe": pe,
        "simd": simd,
    }
    layer_spec_list = [layer_spec]
    model = hls_random_mlp_maker(layer_spec_list)
    for fcl in model.get_nodes_by_op_type("StreamingFCLayer_Batch"):
        op_inst = getCustomOp(fcl)
        op_inst.set_nodeattr("mem_mode", "decoupled")
        op_inst.set_nodeattr("runtime_writeable_weights", 1)
    model.set_initializer("W_0", np.eye(mh).astype(np.float32))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareRTLSim())
    model.set_metadata_prop("exec_mode", "rtlsim")
    os.environ["RTLSIM_TRACE_DEPTH"] = "5"
    model.set_metadata_prop("rtlsim_trace", "trace.vcd")
    in_tensor = np.asarray(range(mw), dtype=np.float32).reshape(1, mw)
    exec_ctx = {"act_0": in_tensor}
    extracted_weights = np.zeros((mh, mw))

    def read_weights(sim):
        addr = 0
        for w in range(mw):
            for h in range(mh):
                extracted_weights[h, w] = axilite_read(
                    sim, addr, basename="s_axilite_0_"
                )
                addr += 4

    rtlsim_exec(model, exec_ctx, pre_hook=read_weights)
    assert (extracted_weights == model.get_initializer("W_0")).all()
    y = exec_ctx["act_1"]
    assert (y == np.dot(in_tensor, extracted_weights.T)).all()
    new_weights = gen_finn_dt_tensor(wdt, (mh, mw))

    def write_weights(sim):
        addr = 0
        for w in range(mw):
            for h in range(mh):
                axilite_write(sim, addr, new_weights[h, w], basename="s_axilite_0_")
                addr += 4

    rtlsim_exec(model, exec_ctx, pre_hook=write_weights)
    y = exec_ctx["act_1"]
    assert (y == np.dot(in_tensor, new_weights.T)).all()
