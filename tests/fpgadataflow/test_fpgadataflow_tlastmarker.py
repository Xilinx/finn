# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn import xsi as finnxsi
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import pynq_part_map

test_fpga_part = pynq_part_map["AUP-ZU3_8GB"]
test_synth_clk_period_ns = 10


def make_tlastmarker_modelwrapper(num_iters, direction, stream_width):
    """Single TLastMarker_rtl node carrying one element per beat."""
    shape = [1, num_iters, 1]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    node = helper.make_node(
        "TLastMarker_rtl",
        ["inp"],
        ["outp"],
        NumIters=num_iters,
        StreamWidth=stream_width,
        ElemWidth=stream_width,
        Direction=direction,
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
    )
    graph = helper.make_graph([node], "tlastmarker_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="tlastmarker-model"))
    return model.transform(GiveUniqueNodeNames())


class BeatCollector:
    """Accept output beats, recording TDATA and (optionally) TLAST per transaction.
    With stall=True, TREADY is deasserted every other cycle to apply backpressure."""

    def __init__(self, sim, size, with_last, stall):
        self.vld = sim.get_bus_port("out0_V", "tvalid")
        self.rdy = sim.get_bus_port("out0_V", "tready")
        self.dat = sim.get_bus_port("out0_V", "tdata")
        self.lst = sim.get_bus_port("out0_V", "tlast") if with_last else None
        self.size = size
        self.stall = stall
        self.ticks = 0
        self.data = []
        self.last = []

    def __call__(self, sim):
        if self.rdy.as_bool() and self.vld.read().as_bool():
            self.data.append(int(self.dat.read().as_hexstr(), 16))
            if self.lst is not None:
                self.last.append(self.lst.read().as_bool())
        if len(self.data) == self.size:
            return {self.rdy: "0"} if self.rdy.as_bool() else None
        self.ticks += 1
        ready = not (self.stall and self.ticks % 2 == 0)
        return {self.rdy: "1" if ready else "0"}


# NumIters, StreamWidth: constant TLAST, 1-bit counter, non-power-of-two wrap;
# non-byte-aligned widths exercise the AXI TDATA padding
@pytest.mark.parametrize("num_iters,stream_width", [(1, 64), (2, 12), (5, 12)])
# Direction: insert ("out") or strip ("in") TLAST
@pytest.mark.parametrize("direction", ["out", "in"])
# traffic: back-to-back, or bursty input combined with output backpressure
@pytest.mark.parametrize("traffic", ["backtoback", "stalled"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_tlastmarker_rtl(num_iters, stream_width, direction, traffic):
    if not finnxsi.is_available():
        pytest.skip("finn_xsi (XSI rtlsim) not available")

    num_frames = 3
    total = num_frames * num_iters
    rng = np.random.default_rng(42)
    values = [int(v) for v in rng.integers(0, 2**stream_width, size=total, dtype=np.uint64)]

    model = make_tlastmarker_modelwrapper(num_iters, direction, stream_width)
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, test_synth_clk_period_ns))
    model = model.transform(PrepareRTLSim())
    inst = getCustomOp(model.graph.node[0])

    stalled = traffic == "stalled"
    sim = inst.get_rtlsim()
    try:
        inst.reset_rtlsim(sim)
        throttle = (1, 3) if stalled else (float("inf"), 0)
        sim.stream_input("in0_V", (f"{v:x}" for v in values), throttle=throttle)
        out = BeatCollector(sim, total, with_last=(direction == "out"), stall=stalled)
        sim.enlist(out)
        trace = sim.trace_stream("out0_V")
        assert not sim.run(100 * total), "rtlsim timed out"
    finally:
        inst.close_rtlsim(sim)

    # data passes through unmodified in both directions
    assert out.data == values
    if direction == "out":
        # TLAST on exactly the last beat of every frame; the counter must only
        # advance on completed transactions, not on idle or stalled cycles
        assert out.last == [i % num_iters == num_iters - 1 for i in range(total)]
    if not stalled:
        # pure passthrough: one beat per cycle, no bubbles
        assert "1" * total in str(trace)


# exec_mode: node execution must not depend on a compiled simulation
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
def test_fpgadataflow_tlastmarker_rtl_execute_passthrough(exec_mode):
    """TLAST is not part of the tensor data, so node execution forwards the input
    unchanged in any exec_mode."""
    model = make_tlastmarker_modelwrapper(4, "out", 8)
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("exec_mode", exec_mode)
    x = np.arange(4, dtype=np.float32).reshape(1, 4, 1)
    context = {"inp": x}
    inst.execute_node(context, model.graph)
    assert np.array_equal(context["outp"], x)
