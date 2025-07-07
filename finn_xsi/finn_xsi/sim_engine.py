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

import numpy as np
import xsi


class SimEngine:
    # ------------------------------------------------------------------------
    # Life Cycle
    def __init__(self, kernel, design, log=None, wdb=None):
        top = xsi.Design(xsi.Kernel(kernel), design, log, wdb)
        clk = top.getPort("ap_clk")
        clk2x = top.getPort("ap_clk2x")
        for p in top.ports():
            if p.isInput():
                p.clear().write_back()

        def cycle(updates):
            # Rising Edge
            clk.set(1).write_back()
            if clk2x is not None:
                clk2x.set(1).write_back()
            # Updates after Active Edge
            top.run(1)
            for port, update in updates.items():
                port.set_hexstr(update).write_back()

            # Edges inactive on interface & finish Cycle
            if clk2x is None:
                top.run(4999)
                clk.set(0).write_back()
                top.run(5000)
            else:
                top.run(2499)
                clk2x.set(0).write_back()
                top.run(2500)
                clk.set(0).write_back()
                clk2x.set(1).write_back()
                top.run(2500)
                clk2x.set(0).write_back()
                top.run(2500)

        self.top = top
        self.cycle = cycle
        self.ticks = 0
        self.tasks = []
        self.watchdogs = []

    # ------------------------------------------------------------------------
    # Utility
    def get_bus_port(self, bus, suffix):
        port = self.top.getPort(bus + "_" + suffix.lower())
        return port if port is not None else self.top.getPort(bus + "_" + suffix.upper())

    # ------------------------------------------------------------------------
    # Simulation Setup

    # Task Scheduling
    def enlist(self, task):
        self.tasks.append(task)

    # Watchdog Generation
    def create_watchdog(self, name, timeout):
        class Watchdog:
            def __init__(self, name, timeout):
                self.name = name
                self.ticks = 0
                self.timeout = timeout

            def __bool__(self):
                return self.ticks < self.timeout

            def __repr__(self):
                return self.name

            def __call__(self):
                self.ticks += 1

            def reset(self):
                self.ticks = 0

        ret = Watchdog(name, timeout)
        self.watchdogs.append(ret)
        return ret

    def remove_watchdog(self, watchdog):
        self.watchdogs.remove(watchdog)

    # ------------------------------------------------------------------------
    # Execution
    def run(self, cycles=None):
        "Run all tasks to completion or until a watchdog triggers."
        timeout = None if cycles is None else self.create_watchdog("Run Timeout", cycles)

        woken = []
        while len(self.tasks) > 0 and len(woken := [w for w in self.watchdogs if not w]) == 0:
            # Process Tasks and Collect Updates to Write Back
            tasks = []
            updates = {}

            # Execute Cycle
            self.ticks += 1
            print(f"Cycle {self.ticks}")
            strong = False
            for task in self.tasks:
                # Tasks read signals and derive updates to schedule for after the clock cycle
                ret = task(self)
                if ret is not None:
                    updates.update(ret)
                    tasks.append(task)
                    strong |= bool(task)
            self.cycle(updates)

            # Step Watchdogs
            for watchdog in self.watchdogs:
                watchdog()

            # Update to Unfinished Tasks
            self.tasks = tasks if strong else []

        # Return List of Woken Watchdogs
        if timeout is not None:
            self.remove_watchdog(timeout)
        return woken

    # ------------------------------------------------------------------------
    # Standard Tasks
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
                    return {self.rst_n: "0"}
                if cnt < 16:
                    return {}
                if cnt == 16:
                    return {self.rst_n: "1"}
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
                    return {}

                # Track Transaction Count
                if vld:
                    self.count_txns += 1

                # Proceed according to Throttling Rate
                if self.count_txns < self.throttle[0] or not sim.ticks < self.await_tick:
                    # Try Feed
                    val = next(self.values, None)
                    if val is None:
                        # Unset vld, then exit
                        return {self.vld: "0", self.dat: "0"} if vld else None

                    # Feed next Value
                    ret = {self.dat: val}
                    if not vld:
                        ret[self.vld] = "1"
                    if self.count_txns == self.throttle[0]:
                        self.count_txns = 0
                        self.await_tick = sim.ticks + self.throttle[1]
                    return ret

                # Stall Feed
                return {self.vld: "0", self.dat: "0"} if vld else {}

        self.enlist(InputStreamer(self, istream, values, throttle))

    def collect_output(self, ostream, size, watchdog=None):
        "Collect size outputs from the specified stream into the returned iterable buffer."

        class OutputCollector:
            def __init__(self, top, ostream, size, watchdog):
                self.size = size
                self.vld = top.get_bus_port(ostream, "tvalid")
                self.rdy = top.get_bus_port(ostream, "tready")
                self.dat = top.get_bus_port(ostream, "tdata")
                self.buf = []
                self.watchdog = watchdog

            def __iter__(self):
                return iter(self.buf)

            def __call__(self, sim):
                if self.rdy.as_bool():
                    if self.vld.read().as_bool():
                        # Have a n Output Transaction
                        if self.watchdog is not None:
                            self.watchdog.reset()
                        val = self.dat.read().as_hexstr()
                        self.buf.append(val)
                        if len(self.buf) == size:
                            return {self.rdy: "0"}
                    return {}

                if len(self.buf) < size:
                    return {self.rdy: "1"}
                return None

        ret = OutputCollector(self, ostream, size, watchdog)
        self.enlist(ret)
        return ret

    def trace_stream(self, stream):
        "Monitor an AXI-Stream and trace its transaction activity"

        class StreamTracer:
            def __init__(self, sim, stream):
                self.vld = sim.get_bus_port(stream, "tvalid")
                self.rdy = sim.get_bus_port(stream, "tready")
                self.trace = ""

            def __call__(self, sim):
                self.trace += (
                    "1" if self.vld.read().as_bool() and self.rdy.read().as_bool() else "0"
                )
                return {}

            def __bool__(self):
                return False

            def __str__(self):
                return self.trace

        ret = StreamTracer(self, stream)
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
                self.awaddr = top.get_bus_port(m_axilite, "awaddr")
                self.wready = top.get_bus_port(m_axilite, "wready")
                self.wvalid = top.get_bus_port(m_axilite, "wvalid")
                self.wdata = top.get_bus_port(m_axilite, "wdata")
                wstrb = top.get_bus_port(m_axilite, "wstrb")
                wstrb.set_binstr("1" * wstrb.width()).write_back()
                self.bready = top.get_bus_port(m_axilite, "bready")
                self.bvalid = top.get_bus_port(m_axilite, "bvalid")
                self.bresp = top.get_bus_port(m_axilite, "bresp")
                self.writes = writes
                self.state = self.INIT
                self.pending = 0

            def __call__(self, sim):
                # Termination
                if self.state == self.COOL and not self.bready.as_bool():
                    return None

                ret = {}

                # Always Monitor Completions
                if self.state == self.INIT:
                    ret[self.bready] = "1"
                    self.state = self.FEED

                if self.bvalid.read().as_bool():
                    if self.pending < 1:
                        print("Received spurious completion on", self.bresp.name())
                    else:
                        self.pending -= 1
                        if self.pending == 0 and self.state == self.COOL:
                            ret[self.bready] = "0"

                    if self.bresp.read().as_unsigned() != 0:
                        print("Received error indication on", self.bresp.name())

                # Transaction Feed
                if self.state == self.FEED:
                    step = True

                    # Check for busy address feed
                    avld = self.awvalid.as_bool()
                    aclr = False
                    if avld:
                        if self.awready.read().as_bool():
                            aclr = True
                        else:
                            step = False

                    # Check for busy data feed
                    wvld = self.wvalid.as_bool()
                    wclr = False
                    if wvld:
                        if self.wready.read().as_bool():
                            wclr = True
                        else:
                            step = False

                    # Proceed with next Write
                    if step:
                        addr, val = next(self.writes, (None, None))
                        if addr is not None:
                            ret[self.awaddr] = f"{addr:x}"
                            ret[self.wdata] = val
                            if not avld:
                                ret[self.awvalid] = "1"
                            if not wvld:
                                ret[self.wvalid] = "1"
                            self.pending += 1
                            return ret
                        if not self.pending:
                            ret[self.bready] = "0"
                        self.state = self.COOL

                    # Deassert completed feed
                    if aclr:
                        ret[self.awvalid] = "0"
                    if wclr:
                        ret[self.wvalid] = "0"

                return ret

        self.enlist(AxiLiteWriter(self, m_axilite, writes))

    def read_axilite(self, m_axilite, reads):
        class AxiLiteReader:
            def __init__(self, top, m_axilite, reads):
                self.arready = top.get_bus_port(m_axilite, "arready")
                self.arvalid = top.get_bus_port(m_axilite, "arvalid")
                self.araddr = top.get_bus_port(m_axilite, "araddr")
                self.rready = top.get_bus_port(m_axilite, "rready")
                self.rvalid = top.get_bus_port(m_axilite, "rvalid")
                self.rdata = top.get_bus_port(m_axilite, "rdata")
                self.reads = reads
                self.pending = []
                self.draining = False
                self.replies = {}

            def __call__(self, sim):
                ret = {}

                # Address Stream Feed: assert self.draining when done
                if not self.draining:
                    if self.arready.read().as_bool() or not self.arvalid.as_bool():
                        addr = next(self.reads, None)
                        if addr is None:
                            ret[self.arvalid] = "0"
                            self.draining = True
                        else:
                            ret[self.arvalid] = "1"
                            ret[self.araddr] = f"{addr:x}"
                            self.pending.append(addr)

                # Reply Collection
                if not self.rready.as_bool():
                    # Termination
                    if self.draining:
                        return None
                    # Activation
                    ret[self.rready] = "1"
                elif self.rvalid.read().as_bool():
                    assert len(self.pending) > 0, "Spurious reply."
                    self.replies[self.pending.pop(0)] = self.rdata.read().as_hexstr()
                    if self.draining and len(self.pending) == 0:
                        ret[self.rready] = "0"

                return ret

            def __iter__(self):
                return iter(self.replies)

            def __getitem__(self, addr):
                return self.replies[addr]

        ret = AxiLiteReader(self, m_axilite, reads)
        self.enlist(ret)
        return ret

    def aximm_ro_image(self, mm_axi, base, img):
        class AximmRoImage:
            def __init__(self, top, mm_axi, base, img):
                self.mm_axi = mm_axi
                self.rd_count = 0
                # Tie off Write Channels
                for tie_off in ("awready", "wready", "bvalid"):
                    port = top.get_bus_port(mm_axi, tie_off)
                    if port is not None:
                        port.set(0).write_back()

                # Collect Ports of Read Channels
                for name in (
                    "arready",
                    "arvalid",
                    "araddr",
                    "arlen",
                    "arburst",
                    "arsize",
                    "rready",
                    "rvalid",
                    "rdata",
                    "rresp",
                    "rlast",
                ):
                    self.__dict__[name] = top.get_bus_port(mm_axi, name)
                self.arready.set(1).write_back()
                self.rvalid.set(0).write_back()
                self.rresp.set(0).write_back()

                # Hold on to Image
                self.base = base
                self.img = [f"{_:02x}" for _ in np.array(img).astype(np.uint8)]
                self.queue = []

            def __bool__(self):
                return False

            def __call__(self, sim):
                ret = {}

                # Push out Read Replies
                if self.rready.read().as_bool() or not self.rvalid.as_bool():
                    if len(self.queue) > 0:
                        # Work on Head of Queue
                        addr, length, size = self.queue.pop(0)
                        data = ""
                        for i in range(size):
                            data = self.img[addr] + data
                            addr += 1
                        ret[self.rdata] = data

                        if length > 1:
                            self.queue.insert(0, (addr, length - 1, size))
                            ret[self.rlast] = "0"
                        else:
                            ret[self.rlast] = "1"
                        ret[self.rvalid] = "1"

                    elif self.rvalid.as_bool():
                        # Silent Reply Interface
                        ret[self.rvalid] = "0"

                # Queue up newly received Read Requests
                if self.arvalid.read().as_bool():
                    assert self.arburst.read().as_unsigned() == 1, "Only INCR bursts supported."

                    addr = int(self.araddr.read().as_hexstr(), 16)
                    #addr = addr - 8*self.rd_count
                    #self.rd_count = self.rd_count + 2

                    assert self.base <= addr, "Read address out of range."
                    addr -= self.base

                    #length = 1 + self.arlen.read().as_unsigned()
                    length = self.arlen.read().as_unsigned()
                    size = 2 ** self.arsize.read().as_unsigned()
                    assert addr + length * size - 1 < len(self.img), "Read extends beyond range."

                    self.queue.append((addr, length, size))

                return ret

        ret = AximmRoImage(self, mm_axi, base, img)
        self.enlist(ret)
        return ret

    def aximm_queue(self, mm_axi):
        "Pick up all write requests to carry them over to complete"
        " a later read request with the same address and size."

        class AximmQueue:
            def __init__(self, top, mm_axi):
                # Collect Ports of Read Channels
                for name in (
                    "awready",
                    "awvalid",
                    "awaddr",
                    "awlen",
                    "awburst",
                    "awsize",
                    "wready",
                    "wvalid",
                    "wdata",
                    "wlast",
                    "bready",
                    "bvalid",
                    "bdata",
                    "bresp",
                    "arready",
                    "arvalid",
                    "araddr",
                    "arlen",
                    "arburst",
                    "arsize",
                    "rready",
                    "rvalid",
                    "rdata",
                    "rresp",
                    "rlast",
                ):
                    self.__dict__[name] = top.get_bus_port(mm_axi, name)
                self.awready.set(1).write_back()
                self.wready.set(1).write_back()
                self.bvalid.set(0).write_back()
                self.bresp.set(0).write_back()
                self.arready.set(1).write_back()
                self.rvalid.set(0).write_back()
                self.rresp.set(0).write_back()

                # Hold on to Contents Map per transfer: addr -> data
                self.map = {}  # addr -> (data, size)

                # Queued transactions
                self.wa_queue = []  # Write Addresses (addr, len, size)
                self.wd_queue = []  # Write Data      (data)
                self.ra_queue = []  # Read Addresses  (addr, len, size)
                self.wr_completion_queue = [] # A queue to track the write completions

            def __call__(self, sim):
                ret = {}

                # Process Write Updates
                while len(self.wa_queue) > 0:
                    addr, length, size = self.wa_queue.pop(0)
                    while length > 0:
                        if len(self.wd_queue) > 0:
                            self.map[addr] = (self.wd_queue.pop(0), size)
                            addr += size
                            length -= 1
                        else:
                            self.wa_queue.insert(0, (addr, length, size))
                            break
                    if len(self.wd_queue) == 0:
                        break

                # Push out Read Replies
                if self.rready.read().as_bool() or not self.rvalid.as_bool():
                    if len(self.ra_queue) > 0:
                        # Work on Head of Queue
                        addr, length, size0 = self.ra_queue.pop(0)
                        assert addr in self.map, "Missing data entry"
                        data, size = self.map[addr]
                        assert size == size0, "Write and read size mismatch."
                        ret[self.rdata] = data
                        if length > 1:
                            self.ra_queue.insert(0, (addr + size, length - 1, size))
                            ret[self.rlast] = "0"
                        else:
                            ret[self.rlast] = "1"
                        ret[self.rvalid] = "1"
                    elif self.rvalid.as_bool():
                        # Silent Reply Interface
                        ret[self.rvalid] = "0"

                # Process write completion queue items
                if len(self.wr_completion_queue) > 0:
                    if self.bready.read().as_bool():
                        ret[self.bvalid] = "1"
                        _ = self.wr_completion_queue.pop(0)
                else:
                    ret[self.bvalid] = "0"

                # Queue new Write Address Requests
                if self.awvalid.read().as_bool():
                    assert self.awburst.read().as_unsigned() == 1, "Only INCR bursts supported."

                    addr = int(self.awaddr.read().as_hexstr(), 16)
                    length = 1 + self.awlen.read().as_unsigned()
                    size = 2 ** self.awsize.read().as_unsigned()
                    self.wa_queue.append((addr, length, size))
                    self.wr_completion_queue.insert(0, 1)

                # Queue received Write Data
                if self.wvalid.read().as_bool():
                    self.wd_queue.append(self.wdata.read().as_hexstr())

                # Queue new Read Requests
                if self.arvalid.read().as_bool():
                    assert self.arburst.read().as_unsigned() == 1, "Only INCR bursts supported."

                    addr = int(self.araddr.read().as_hexstr(), 16)
                    length = 1 + self.arlen.read().as_unsigned()
                    size = 2 ** self.arsize.read().as_unsigned()
                    self.ra_queue.append((addr, length, size))

                return ret

        self.enlist(AximmQueue(self, mm_axi))
