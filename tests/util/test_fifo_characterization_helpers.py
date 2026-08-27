# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import finn.builder.build_dataflow_steps as steps


class _FakeFIFO:
    def __init__(self, impl_style):
        self.impl_style = impl_style

    def get_nodeattr(self, name):
        assert name == "impl_style"
        return self.impl_style

    def set_nodeattr(self, name, value):
        assert name == "impl_style"
        self.impl_style = value


class _FakeFIFOModel:
    def __init__(self):
        self.fifos = [_FakeFIFO("vivado"), _FakeFIFO("rtl")]

    def get_nodes_by_op_type(self, op_type):
        assert op_type == "StreamingFIFO_rtl"
        return self.fifos


def test_node_by_node_rtlsim_uses_rtl_fifo_copy(monkeypatch):
    model = _FakeFIFOModel()
    monkeypatch.setattr(steps, "getCustomOp", lambda node: node)

    verify_model = steps.prepare_for_node_by_node_rtlsim(model)

    assert verify_model is not model
    assert [fifo.impl_style for fifo in model.fifos] == ["vivado", "rtl"]
    assert [fifo.impl_style for fifo in verify_model.fifos] == ["rtl", "rtl"]


def test_characterization_period_excludes_skipped_loop(monkeypatch):
    ordinary = SimpleNamespace(name="ordinary", op_type="FakeRTL")
    loop = SimpleNamespace(name="loop", op_type="FINNLoop")
    model = SimpleNamespace(graph=SimpleNamespace(node=[ordinary, loop]))
    cycles = {"ordinary": 100, "loop": 10_000}

    monkeypatch.setattr(steps, "is_hls_node", lambda _node: False)
    monkeypatch.setattr(steps, "is_rtl_node", lambda _node: True)
    monkeypatch.setattr(
        steps,
        "getCustomOp",
        lambda node: SimpleNamespace(get_nodeattr=lambda _name: cycles[node.name]),
    )

    assert steps._get_characterization_period(model) == 10_010
    assert steps._get_characterization_period(model, {loop.name}) == 110
