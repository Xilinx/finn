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


def _find_signal(sim, signal_name):
    # handle both mixed caps and lowercase signal names
    if signal_name in sim.io:
        return signal_name
    elif signal_name.lower() in sim.io:
        return signal_name.lower()
    else:
        raise Exception("Signal not found: " + signal_name)


def _read_signal(sim, signal_name):
    signal_name = _find_signal(sim, signal_name)
    return sim.io[signal_name]


def _write_signal(sim, signal_name, signal_value):
    signal_name = _find_signal(sim, signal_name)
    sim.io[signal_name] = signal_value


def reset_rtlsim(sim, rst_name="ap_rst_n", active_low=True):
    """Sets reset input in pyverilator to zero, toggles the clock and set it
    back to one"""
    _write_signal(sim, rst_name, 0 if active_low else 1)
    toggle_clk(sim)
    toggle_clk(sim)
    _write_signal(sim, rst_name, 1 if active_low else 0)
    toggle_clk(sim)
    toggle_clk(sim)


def toggle_clk(sim, clk_name="ap_clk"):
    """Toggles the clock input in pyverilator once."""
    _write_signal(sim, clk_name, 0)
    sim.eval()
    _write_signal(sim, clk_name, 1)
    sim.eval()


def wait_for_handshake(sim, ifname, basename="s_axi_control_", dataname="DATA"):
    """Wait for handshake (READY and VALID high at the same time) on given
    interface on PyVerilator sim object.

    Arguments:
    - sim : PyVerilator sim object
    - ifname : name for decoupled interface to wait for handshake on
    - basename : prefix for decoupled interface name
    - dataname : interface data sig name, will be return value if it exists

    Returns: value of interface data signal during handshake (if given by dataname),
    None otherwise (e.g. if there is no data signal associated with interface)
    """
    ret = None
    while 1:
        hs = (
            _read_signal(sim, basename + ifname + "READY") == 1
            and _read_signal(sim, basename + ifname + "VALID") == 1
        )
        try:
            ret = _read_signal(sim, basename + ifname + dataname)
        except Exception:
            ret = None
        toggle_clk(sim)
        if hs:
            break
    return ret


def multi_handshake(sim, ifnames, basename="s_axi_control_"):
    """Perform a handshake on list of interfaces given by ifnames. Will assert
    VALID and de-assert after READY observed, in as few cycles as possible."""

    done = []
    for ifname in ifnames:
        _write_signal(sim, basename + ifname + "VALID", 1)
    while len(ifnames) > 0:
        for ifname in ifnames:
            if (
                _read_signal(sim, basename + ifname + "READY") == 1
                and _read_signal(sim, basename + ifname + "VALID") == 1
            ):
                done.append(ifname)
        toggle_clk(sim)
        for ifname in done:
            ifnames.remove(ifname)
            _write_signal(sim, basename + ifname + "VALID", 0)


def axilite_write(
    sim, addr, val, basename="s_axi_control_", wstrb=0xF, sim_addr_and_data=True
):
    """Write val to addr on AXI lite interface given by basename.

    Arguments:
    - sim : PyVerilator sim object
    - addr : address for write
    - val : value to be written at addr
    - basename : prefix for AXI lite interface name
    - wstrb : write strobe value to do partial writes, see AXI protocol reference
    - sim_addr_and_data : handshake AW and W channels simultaneously
    """
    _write_signal(sim, basename + "WSTRB", wstrb)
    _write_signal(sim, basename + "WDATA", val)
    _write_signal(sim, basename + "AWADDR", addr)
    if sim_addr_and_data:
        multi_handshake(sim, ["AW", "W"], basename=basename)
    else:
        _write_signal(sim, basename + "AWVALID", 1)
        wait_for_handshake(sim, "AW", basename=basename)
        # write request done
        _write_signal(sim, basename + "AWVALID", 0)
        # write data
        _write_signal(sim, basename + "WVALID", 1)
        wait_for_handshake(sim, "W", basename=basename)
        # write data OK
        _write_signal(sim, basename + "WVALID", 0)
    # wait for write response
    _write_signal(sim, basename + "BREADY", 1)
    wait_for_handshake(sim, "B", basename=basename)
    # write response OK
    _write_signal(sim, basename + "BREADY", 0)


def axilite_read(sim, addr, basename="s_axi_control_"):
    """Read val from addr on AXI lite interface given by basename.

    Arguments:
    - sim : PyVerilator sim object
    - addr : address for read
    - basename : prefix for AXI lite interface name

    Returns: read value from AXI lite interface at given addr
    """
    _write_signal(sim, basename + "ARADDR", addr)
    _write_signal(sim, basename + "ARVALID", 1)
    wait_for_handshake(sim, "AR", basename=basename)
    # read request OK
    _write_signal(sim, basename + "ARVALID", 0)
    # wait for read response
    _write_signal(sim, basename + "RREADY", 1)
    ret_data = wait_for_handshake(sim, "R", basename=basename)
    _write_signal(sim, basename + "RREADY", 0)
    return ret_data
