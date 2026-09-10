# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import numpy as np
import os
import sys
import types


class _Signal:
    def __init__(self):
        self.value = 0

    def read(self):
        return self

    def as_bool(self):
        return bool(self.value)

    def as_unsigned(self):
        return int(self.value)

    def as_hexstr(self):
        return f"{int(self.value):x}"

    def set(self, value):
        self.value = value
        return self

    def set_hexstr(self, value):
        self.value = int(value, 16)
        return self

    def write_back(self):
        return self


class _Engine:
    def __init__(self):
        self.ports = {}
        self.tasks = []

    def get_bus_port(self, bus, suffix):
        return self.ports.setdefault(f"{bus}_{suffix}", _Signal())

    def enlist(self, task):
        self.tasks.append(task)


def _apply(updates):
    for port, value in updates.items():
        port.set_hexstr(value).write_back()


def test_aximm_ro_image_backpressures_until_burst_completion(monkeypatch):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.syspath_prepend(os.path.join(repo_root, "finn_xsi"))
    monkeypatch.setitem(sys.modules, "xsi", types.ModuleType("xsi"))
    sim_engine = importlib.import_module("finn_xsi.sim_engine")

    engine = _Engine()
    responder = sim_engine.SimEngine.aximm_ro_image(
        engine, "mem", 0, np.array([0x11, 0x22, 0x33, 0x44], dtype=np.uint8)
    )

    def port(suffix):
        return engine.ports[f"mem_{suffix}"]

    port("arvalid").value = 1
    port("arburst").value = 1
    port("araddr").value = 0
    port("arlen").value = 1
    port("arsize").value = 1
    port("arid").value = 2
    port("rready").value = 1

    updates = responder(engine)
    assert updates[port("arready")] == "0"
    assert len(responder.queue) == 1
    _apply(updates)

    # ARVALID may remain asserted until the master observes ARREADY low. It
    # must not enqueue the same request again while the burst is active.
    updates = responder(engine)
    assert updates[port("rdata")] == "2211"
    assert updates[port("rid")] == "2"
    assert updates[port("rlast")] == "0"
    assert len(responder.queue) == 1
    _apply(updates)

    port("rready").value = 0
    assert responder(engine) == {}
    assert len(responder.queue) == 1

    port("rready").value = 1
    updates = responder(engine)
    assert updates[port("rdata")] == "4433"
    assert updates[port("rid")] == "2"
    assert updates[port("rlast")] == "1"
    assert updates[port("arready")] == "1"
    assert len(responder.queue) == 0
    _apply(updates)

    port("arvalid").value = 0
    updates = responder(engine)
    assert updates[port("rvalid")] == "0"
    assert responder.arready_state


def test_aximm_ro_image_accepts_next_address_while_last_beat_is_stalled(monkeypatch):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.syspath_prepend(os.path.join(repo_root, "finn_xsi"))
    monkeypatch.setitem(sys.modules, "xsi", types.ModuleType("xsi"))
    sim_engine = importlib.import_module("finn_xsi.sim_engine")

    engine = _Engine()
    responder = sim_engine.SimEngine.aximm_ro_image(
        engine, "mem", 0, np.array([0x11, 0x22, 0x33, 0x44], dtype=np.uint8)
    )

    def port(suffix):
        return engine.ports[f"mem_{suffix}"]

    port("arvalid").value = 1
    port("arburst").value = 1
    port("araddr").value = 0
    port("arlen").value = 0
    port("arsize").value = 1
    port("rready").value = 0
    _apply(responder(engine))

    port("arvalid").value = 0
    updates = responder(engine)
    assert updates[port("rvalid")] == "1"
    assert updates[port("rlast")] == "1"
    assert updates[port("arready")] == "1"
    _apply(updates)

    # A master may overlap the next address with the stalled final read beat.
    port("arvalid").value = 1
    port("araddr").value = 2
    port("arid").value = 1
    updates = responder(engine)
    assert updates[port("arready")] == "0"
    assert len(responder.queue) == 1
    assert port("rvalid").as_bool()


def test_aximm_queue_backpressures_each_burst(monkeypatch):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.syspath_prepend(os.path.join(repo_root, "finn_xsi"))
    monkeypatch.setitem(sys.modules, "xsi", types.ModuleType("xsi"))
    sim_engine = importlib.import_module("finn_xsi.sim_engine")

    engine = _Engine()
    sim_engine.SimEngine.aximm_queue(engine, "mem")
    responder = engine.tasks[0]

    def port(suffix):
        return engine.ports[f"mem_{suffix}"]

    port("awvalid").value = 1
    port("awburst").value = 1
    port("awaddr").value = 0
    port("awlen").value = 1
    port("awsize").value = 2
    port("awid").value = 1
    port("wvalid").value = 1
    port("wdata").value = 0x11111111
    port("wlast").value = 0
    port("bready").value = 1

    updates = responder(engine)
    assert updates[port("awready")] == "0"
    assert updates[port("wready")] == "1"
    assert responder.map == {}
    _apply(updates)

    port("awvalid").value = 0
    updates = responder(engine)
    assert responder.map[0] == ("11111111", 4)
    assert responder.write_burst[:3] == (4, 1, 4)
    _apply(updates)

    port("wdata").value = 0x22222222
    port("wlast").value = 1
    updates = responder(engine)
    assert responder.map[4] == ("22222222", 4)
    assert updates[port("wready")] == "0"
    assert updates[port("bid")] == "1"
    assert updates[port("bvalid")] == "1"
    _apply(updates)

    port("wvalid").value = 0
    updates = responder(engine)
    assert updates[port("bvalid")] == "0"
    assert updates[port("awready")] == "1"
    _apply(updates)

    port("arvalid").value = 1
    port("arburst").value = 1
    port("araddr").value = 0
    port("arlen").value = 1
    port("arsize").value = 2
    port("arid").value = 3
    port("rready").value = 1
    updates = responder(engine)
    assert updates[port("arready")] == "0"
    _apply(updates)

    port("arvalid").value = 0
    updates = responder(engine)
    assert updates[port("rdata")] == "11111111"
    assert updates[port("rid")] == "3"
    assert updates[port("rlast")] == "0"
    _apply(updates)

    updates = responder(engine)
    assert updates[port("rdata")] == "22222222"
    assert updates[port("rid")] == "3"
    assert updates[port("rlast")] == "1"
    assert updates[port("arready")] == "1"
    _apply(updates)

    updates = responder(engine)
    assert updates[port("rvalid")] == "0"


def test_aximm_queue_accepts_next_read_while_last_beat_is_stalled(monkeypatch):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.syspath_prepend(os.path.join(repo_root, "finn_xsi"))
    monkeypatch.setitem(sys.modules, "xsi", types.ModuleType("xsi"))
    sim_engine = importlib.import_module("finn_xsi.sim_engine")

    engine = _Engine()
    sim_engine.SimEngine.aximm_queue(engine, "mem")
    responder = engine.tasks[0]

    def port(suffix):
        return engine.ports[f"mem_{suffix}"]

    responder.map[0] = ("11111111", 4)
    responder.map[4] = ("22222222", 4)
    port("arvalid").value = 1
    port("arburst").value = 1
    port("araddr").value = 0
    port("arlen").value = 0
    port("arsize").value = 2
    port("rready").value = 0
    _apply(responder(engine))

    port("arvalid").value = 0
    updates = responder(engine)
    assert updates[port("rvalid")] == "1"
    assert updates[port("rlast")] == "1"
    assert updates[port("arready")] == "1"
    _apply(updates)

    port("arvalid").value = 1
    port("araddr").value = 4
    updates = responder(engine)
    assert updates[port("arready")] == "0"
    assert responder.read_burst[:3] == (4, 1, 4)
    assert port("rvalid").as_bool()
