#!/usr/bin/python3
#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	SimEngine abstraction for running FINN task in simulated hardware.
# @author	Thomas B. Preußer <thomas.preusser@amd.com>
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################

import xsi


class SimEngine:
    def __init__(self, kernel, design, log=None, wdb=None):
        top = xsi.Design(xsi.Kernel(kernel), design, log, wdb)
        clk = top.getPort("ap_clk")
        clk2x = top.getPort("ap_clk2x")
        for p in top.ports():
            if p.isInput():
                p.clear().write_back()

        self.top = top
        self.cycle = (
            lambda up: (clk.set(up).write_back(), top.run(5000))
            if clk2x is None
            else (
                clk.set(up).write_back(),
                clk2x.set(1).write_back(),
                top.run(2500),
                clk2x.set(0).write_back(),
                top.run(2500),
            )
        )
        self.ticks = 0
        self.tasks = []

    def get_bus_port(self, bus, suffix):
        port = self.top.getPort(bus + '_' + suffix.lower())
        return  port if port is not None else self.top.getPort(bus + '_' + suffix.upper())

    def run(self, cycles=float("inf")):
        timeout = self.ticks + cycles

        "Run all tasks to completion."
        while len(self.tasks) > 0 and self.ticks < timeout:
            # Update Tick Counters
            self.ticks += 1

            # Process Tasks and Collect Updates to Write Back
            tasks = []
            updates = []

            # Active Clock Edge -> read
            self.cycle(0)
            for task in self.tasks:
                ret = task(self)
                if ret is not None:
                    updates.extend(ret)
                    tasks.append(task)

            # Write Back
            self.cycle(1)
            for port in updates:
                port.write_back()

            # Update to Unfinished Tasks
            self.tasks = tasks

        # Return Number of Unfinished Tasks
        return len(self.tasks)

    def enlist(self, task):
        self.tasks.append(task)

    def do_reset(self):
        "Schedule a reset sequence."

        class Reset:
            def __init__(self, top):
                self.cnt = 0
                self.rst_n = top.getPort("ap_rst_n")

            def __call__(self, sim):
                cnt = self.cnt
                self.cnt = cnt + 1

                if cnt == 0:
                    return [self.rst_n.set(0)]
                if cnt < 16:
                    return []
                if cnt == 16:
                    return [self.rst_n.set(1)]
                return None

        self.enlist(Reset(self.top))

    def stream_input(self, istream, values, throttle=(float("inf"), 0)):
        "Stream all values from the passed iterator into the specified stream."

        class InputStreamer:
            def __init__(self, top, istream, values, throttle):
                self.vld = top.get_bus_port(istream, "tvalid")
                self.rdy = top.get_bus_port(istream, "tready")
                self.dat = top.get_bus_port(istream, "tdata")
                self.values = values

                self.throttle = throttle
                self.await_tick = 0
                self.count_txns = throttle[0]

            def __call__(self, sim):
                vld = self.vld.as_bool()
                if vld and not self.rdy.read().as_bool():
                    return []

                # Track Transaction Count
                if vld:
                    self.count_txns += 1

                # Proceed according to Throttling Rate
                if self.count_txns < self.throttle[0] or not sim.ticks < self.await_tick:
                    # Try Feed
                    val = next(self.values, None)
                    if val is None:
                        # Unset vld, then exit
                        return [self.vld.set(0), self.dat.clear()] if vld else None

                    # Feed next Value
                    ret = [self.dat.set_hexstr(val)]
                    if not vld:
                        ret.append(self.vld.set(1))
                    if self.count_txns == self.throttle[0]:
                        self.count_txns = 0
                        self.await_tick = sim.ticks + self.throttle[1]
                    return ret

                # Stall Feed
                return [self.vld.set(0), self.dat.clear()] if vld else []

        self.enlist(InputStreamer(self, istream, values, throttle))

    def collect_output(self, ostream, size):
        "Collect size outputs from the specified stream into the returned iterable buffer."

        class OutputCollector:
            def __init__(self, top, ostream, size):
                self.size = size
                self.vld = top.get_bus_port(ostream, "tvalid")
                self.rdy = top.get_bus_port(ostream, "tready")
                self.dat = top.get_bus_port(ostream, "tdata")
                self.buf = []

            def __iter__(self):
                return iter(self.buf)

            def __call__(self, sim):
                if self.rdy.as_bool():
                    if self.vld.read().as_bool():
                        val = self.dat.read().as_hexstr()
                        self.buf.append(val)
                        if len(self.buf) == size:
                            return [self.rdy.set(0)]
                    return []

                if len(self.buf) < size:
                    return [self.rdy.set(1)]
                return None

        ret = OutputCollector(self, ostream, size)
        self.enlist(ret)
        return ret

    def write_axilite(self, m_axilite, writes):
        "Execute writes specified as a list of (addr, val)-tuples to AXI-lite interface"

        class AxiLiteWriter:
            INIT = 0
            FEED = 1
            COOL = 2

            def __init__(self, top, m_axilite, writes):
                self.awready = top.get_bus_port(m_axilite, "awready")
                self.awvalid = top.get_bus_port(m_axilite, "awvalid")
                self.awaddr  = top.get_bus_port(m_axilite, "awaddr")
                self.wready  = top.get_bus_port(m_axilite, "wready")
                self.wvalid  = top.get_bus_port(m_axilite, "wvalid")
                self.wdata   = top.get_bus_port(m_axilite, "wdata")
                self.bready  = top.get_bus_port(m_axilite, "bready")
                self.bvalid  = top.get_bus_port(m_axilite, "bvalid")
                self.bresp   = top.get_bus_port(m_axilite, "bresp")
                self.writes  = writes
                self.state   = self.INIT
                self.pending = 0

            def __call__(self, sim):

                # Termination
                if(self.state == self.COOL and not self.bready.as_bool()):
                    return  None

                ret = []

                # Always Monitor Completions
                if(self.state == self.INIT):
                    ret.append(self.bready.set(1))
                    self.state = self.FEED

                if(self.bvalid.read().as_bool()):
                    if(self.pending < 1):
                        print("Received spurious completion on", self.bresp.name())
                    else:
                        self.pending -= 1
                        if(self.pending == 0 and self.state == self.COOL):
                            ret.append(self.bready.set(0))

                    if(self.bresp.read().as_unsigned() != 0):
                        print("Received error indication on", self.bresp.name())

                # Transaction Feed
                if(self.state == self.FEED):
                    step = True

                    # Check for busy address feed
                    avld = self.awvalid.as_bool()
                    aclr = False
                    if(avld):
                        if(self.awready.read().as_bool()):
                            aclr = True
                        else:
                            step = False

                    # Check for busy data feed
                    wvld = self.wvalid.as_bool()
                    wclr = False
                    if(wvld):
                        if(self.wready.read().as_bool()):
                            wclr = True
                        else:
                            step = False

                    # Proceed with next Write
                    if(step):
                        addr, val = next(self.writes, (None, None))
                        if addr is not None:
                            ret.extend([self.awaddr.set(addr), self.wdata.set_hexstr(val)])
                            if not avld:
                                ret.append(self.awvalid.set(1))
                            if not wvld:
                                ret.append(self.wvalid.set(1))
                            self.pending += 1
                            return  ret
                        if not self.pending:
                            ret.append(self.bready.set(0))
                        self.state = self.COOL

                    # Deassert completed feed
                    if(aclr):
                        ret.append(self.awvalid.set(0))
                    if(wclr):
                        ret.append(self.wvalid.set(0))

                return ret

        self.enlist(AxiLiteWriter(self, m_axilite, writes))

    def read_axilite(self, m_axilite, reads):
        class AxiLiteReader:
            def __init__(self, top, m_axilite, reads):
                self.arready = top.get_bus_port(m_axilite, "arready")
                self.arvalid = top.get_bus_port(m_axilite, "arvalid")
                self.araddr  = top.get_bus_port(m_axilite, "araddr")
                self.rready  = top.get_bus_port(m_axilite, "rready")
                self.rvalid  = top.get_bus_port(m_axilite, "rvalid")
                self.rdata   = top.get_bus_port(m_axilite, "rdata")
                self.reads   = reads
                self.pending  = []
                self.draining = False
                self.replies  = {}

            def __call__(self, sim):
                ret = []

                # Address Stream Feed: assert self.draining when done
                if not self.draining:
                    if self.arready.read().as_bool() or not self.arvalid.as_bool():
                        addr = next(self.reads, None)
                        if addr is None:
                            ret.append(self.arvalid.set(0))
                            self.draining = True
                        else:
                            ret.extend([self.arvalid.set(1), self.araddr.set(addr)])
                            self.pending.append(addr)

                # Reply Collection
                if not self.rready.as_bool():
                    # Termination
                    if self.draining: return  None
                    # Activation
                    ret.append(self.rready.set(1))
                elif self.rvalid.read().as_bool():
                    assert len(self.pending) > 0, "Spurious reply."
                    self.replies[self.pending.pop(0)] = self.rdata.read().as_hexstr()
                    if self.draining and len(self.pending) == 0:
                        ret.append(self.rready.set(0))

                return  ret

            def __iter__(self):
                return  iter(self.replies)
            def __getitem__(self, addr):
                return  self.replies[addr]

        ret = AxiLiteReader(self, m_axilite, reads)
        self.enlist(ret)
        return  ret
