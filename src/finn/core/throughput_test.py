# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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
from qonnx.util.basic import gen_finn_dt_tensor

from finn.core.rtlsim_exec import rtlsim_exec


def throughput_test_rtlsim(model, clk_ns, batchsize=100):
    """Runs a throughput test for the given IP-stitched model. When combined
    with tracing, useful to determine bottlenecks and required FIFO sizes."""

    assert (
        model.get_metadata_prop("exec_mode") == "rtlsim"
    ), """Top-level exec_mode
    metadata_prop must be set to rtlsim"""

    # make empty exec context and insert random inputs
    ctx = model.make_empty_exec_context()
    i_bytes = 0
    for i_vi in model.graph.input:
        # create random input
        iname = i_vi.name
        ishape = model.get_tensor_shape(iname)
        ishape_batch = ishape
        ishape_batch[0] = batchsize
        idt = model.get_tensor_datatype(iname)
        dummy_input = gen_finn_dt_tensor(idt, ishape_batch)
        ctx[iname] = dummy_input
        i_bytes += (np.prod(ishape_batch) * idt.bitwidth()) / 8

    # compute total output size as well
    o_bytes = 0
    for o_vi in model.graph.output:
        oname = o_vi.name
        oshape = model.get_tensor_shape(oname)
        oshape_batch = oshape
        oshape_batch[0] = batchsize
        odt = model.get_tensor_datatype(oname)
        o_bytes += (np.prod(oshape_batch) * odt.bitwidth()) / 8

    rtlsim_exec(model, ctx)
    # extract metrics
    cycles = int(model.get_metadata_prop("cycles_rtlsim"))
    fclk_mhz = 1 / (clk_ns * 0.001)
    runtime_s = (cycles * clk_ns) * (10**-9)
    res = dict()
    res["cycles"] = cycles
    res["runtime[ms]"] = runtime_s * 1000
    res["throughput[images/s]"] = batchsize / runtime_s
    res["DRAM_in_bandwidth[MB/s]"] = i_bytes * 0.000001 / runtime_s
    res["DRAM_out_bandwidth[MB/s]"] = o_bytes * 0.000001 / runtime_s
    res["fclk[mhz]"] = fclk_mhz
    res["N"] = batchsize

    return res
