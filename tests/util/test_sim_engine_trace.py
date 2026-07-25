# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os
import sys
import types


class _Signal:
    def __init__(self):
        self.value = False

    def read(self):
        return self

    def as_bool(self):
        return self.value


class _Engine:
    def __init__(self):
        self.vld = _Signal()
        self.rdy = _Signal()
        self.tasks = []

    def get_bus_port(self, _stream, suffix):
        return self.vld if suffix == "tvalid" else self.rdy

    def enlist(self, task):
        self.tasks.append(task)


def test_stream_trace_uses_linear_byte_storage(monkeypatch):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    monkeypatch.syspath_prepend(os.path.join(repo_root, "finn_xsi"))
    monkeypatch.setitem(sys.modules, "xsi", types.ModuleType("xsi"))
    sim_engine = importlib.import_module("finn_xsi.sim_engine")

    engine = _Engine()
    tracer = sim_engine.SimEngine.trace_stream(engine, "axis")
    for valid, ready in [(False, False), (True, False), (True, True), (False, True)]:
        engine.vld.value = valid
        engine.rdy.value = ready
        assert tracer(engine) == {}

    assert engine.tasks == [tracer]
    assert isinstance(tracer.trace, bytearray)
    assert str(tracer) == "0010"
