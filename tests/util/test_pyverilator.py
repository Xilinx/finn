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

import pkg_resources as pk
from pyverilator import PyVerilator
from finn.util.pyverilator import axilite_read, axilite_write, reset_rtlsim


def test_pyverilator_axilite():
    example_root = pk.resource_filename("finn.qnn-data", "verilog/myadd")
    # load example verilog: takes two 32-bit integers as AXI lite mem mapped
    # registers, adds them together and return result
    sim = PyVerilator.build(
        "myadd_myadd.v", verilog_path=[example_root], top_module_name="myadd_myadd",
    )
    ifname = "s_axi_control_"
    expected_signals = [
        "AWVALID",
        "AWREADY",
        "AWADDR",
        "WVALID",
        "WREADY",
        "WDATA",
        "WSTRB",
        "ARVALID",
        "ARREADY",
        "ARADDR",
        "RVALID",
        "RREADY",
        "RDATA",
        "RRESP",
        "BVALID",
        "BREADY",
        "BRESP",
    ]
    for signal_name in expected_signals:
        assert ifname + signal_name in sim.io
    reset_rtlsim(sim)
    # initial values
    sim.io[ifname + "WVALID"] = 0
    sim.io[ifname + "AWVALID"] = 0
    sim.io[ifname + "ARVALID"] = 0
    sim.io[ifname + "BREADY"] = 0
    sim.io[ifname + "RREADY"] = 0
    # write + verify first parameter in AXI lite memory mapped regs
    val_a = 3
    addr_a = 0x18
    axilite_write(sim, addr_a, val_a)
    ret_data = axilite_read(sim, addr_a)
    assert ret_data == val_a
    # write + verify second parameter in AXI lite memory mapped regs
    val_b = 5
    addr_b = 0x20
    axilite_write(sim, addr_b, val_b)
    ret_data = axilite_read(sim, addr_b)
    assert ret_data == val_b
    # launch accelerator and wait for completion
    addr_ctrl_status = 0x00
    # check for ap_idle
    assert axilite_read(sim, addr_ctrl_status) and (1 << 2) != 0
    # set ap_start
    axilite_write(sim, addr_ctrl_status, 1)
    # wait until ap_done
    while 1:
        ap_done = axilite_read(sim, addr_ctrl_status) and (1 << 1)
        if ap_done != 0:
            break
    # read out and verify result
    addr_return = 0x10
    val_ret = axilite_read(sim, addr_return)
    assert val_ret == val_a + val_b
