# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""JustInTimeSynthesize builds IP and rtlsim only for the nodes that are
characterized with rtlsim, and only where the IP does not exist yet.

Code generation, HLS synthesis and rtlsim preparation are replaced by
recorders, so no Vivado is needed."""

import pytest

import numpy as np
import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

import finn.transformation.fpgadataflow.derive_characteristic as derive_characteristic
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveTokenAccessVectors,
    JustInTimeSynthesize,
    needs_rtlsim_characterization,
)
from finn.util.basic import make_build_dir

DT = DataType["INT4"]


def make_fork_join_model(ch=16, pe=2, n_vec=8):
    """``inp -> DuplicateStreams -> (a, b)``, ``a`` scaled by a constant (HLS),
    ``b`` through a 1:4 and a 4:1 width converter (RTL), then
    ``ElementwiseAdd(a, b)``: a fork and a join, which are never characterized,
    and three characterized nodes, one of them an HLS node that needs HLS
    synthesis."""
    shape = [1, n_vec, ch]
    tensors = ["inp", "a", "a_scaled", "b", "b_wide", "b_narrow", "outp"]
    width = pe * DT.bitwidth()
    nodes = [
        helper.make_node(
            "DuplicateStreams_hls",
            ["inp"],
            ["a", "b"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            NumChannels=ch,
            PE=pe,
            NumOutputStreams=2,
            inputDataType=DT.name,
            numInputVectors=[1, n_vec],
        )
    ]
    nodes.append(
        helper.make_node(
            "ElementwiseMul_hls",
            ["a", "scale"],
            ["a_scaled"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            lhs_shape=shape,
            rhs_shape=[1],
            out_shape=shape,
            lhs_dtype=DT.name,
            rhs_dtype=DT.name,
            out_dtype=DT.name,
            lhs_style="input",
            rhs_style="const",
        )
    )
    for inw, outw, src, dst in [
        (width, 4 * width, "b", "b_wide"),
        (4 * width, width, "b_wide", "b_narrow"),
    ]:
        nodes.append(
            helper.make_node(
                "StreamingDataWidthConverter_rtl",
                [src],
                [dst],
                domain="finn.custom_op.fpgadataflow.rtl",
                backend="fpgadataflow",
                shape=shape,
                inWidth=inw,
                outWidth=outw,
                dataType=DT.name,
            )
        )
    nodes.append(
        helper.make_node(
            "ElementwiseAdd_hls",
            ["a_scaled", "b_narrow"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            lhs_shape=shape,
            rhs_shape=shape,
            out_shape=shape,
            lhs_dtype=DT.name,
            rhs_dtype=DT.name,
            out_dtype="INT8",
            lhs_style="input",
            rhs_style="input",
        )
    )
    vi = {t: helper.make_tensor_value_info(t, TensorProto.FLOAT, shape) for t in tensors}
    graph = helper.make_graph(
        nodes,
        "fork_join",
        [vi["inp"]],
        [vi["outp"]],
        value_info=[v for t, v in vi.items() if t not in ("inp", "outp")],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="fork-join"))
    for t in tensors:
        model.set_tensor_datatype(t, DT)
    model.set_initializer("scale", np.ones([1], dtype=np.float32))
    model.set_tensor_datatype("scale", DT)
    return model.transform(GiveUniqueNodeNames())


class _NoOp(Transformation):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def apply(self, model):
        return (model, False)


@pytest.fixture
def recorders(monkeypatch):
    """Replace code generation, HLS synthesis, the Verilog path fix-up and
    rtlsim preparation (HLS and RTL) by recorders. Returns the names of the nodes each step
    ran for."""
    calls = {"codegen": [], "ipgen": [], "rtlsim": []}

    def codegen(node, model, fpgapart, clk):
        calls["codegen"].append(node.name)
        inst = getCustomOp(node)
        if inst.get_nodeattr("code_gen_dir_ipgen") == "":
            inst.set_nodeattr("code_gen_dir_ipgen", make_build_dir("jit_test_codegen_"))

    def ipgen(self, fpgapart=None):
        calls["ipgen"].append(self.onnx_node.name)
        ip = os.path.join(self.get_nodeattr("code_gen_dir_ipgen"), "project_" + self.onnx_node.name)
        os.makedirs(ip, exist_ok=True)
        self.set_nodeattr("ipgen_path", ip)

    def prepare_rtlsim(self, behav=False):
        calls["rtlsim"].append(self.onnx_node.name)
        self.set_nodeattr("rtlsim_so", "fake_rtlsim.so")

    monkeypatch.setattr(derive_characteristic, "_codegen_single_node", codegen)
    monkeypatch.setattr(derive_characteristic, "ReplaceVerilogRelPaths", _NoOp)
    monkeypatch.setattr(HLSBackend, "ipgen_singlenode_code", ipgen)
    monkeypatch.setattr(HLSBackend, "prepare_rtlsim", prepare_rtlsim)
    monkeypatch.setattr(
        "finn.custom_op.fpgadataflow.rtlbackend.RTLBackend.prepare_rtlsim", prepare_rtlsim
    )
    return calls


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("ip_state", ["fresh", "exists", "missing"])
def test_jit_synthesizes_only_what_is_needed(recorders, ip_state):
    """Code is generated and rtlsim prepared for the characterized nodes, HLS IP
    synthesized for the characterized HLS nodes unless it already exists; the
    fork and the join get neither IP nor rtlsim."""
    model = make_fork_join_model()
    hls = ["ElementwiseMul_hls_0"]
    chrc = hls + ["StreamingDataWidthConverter_rtl_0", "StreamingDataWidthConverter_rtl_1"]
    for node in model.graph.node:
        if node.name not in hls or ip_state == "fresh":
            continue
        inst = getCustomOp(node)
        code_gen_dir = make_build_dir("jit_test_existing_")
        inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        ip = os.path.join(code_gen_dir, "project_" + node.name)
        if ip_state == "exists":
            os.makedirs(ip)
        # "missing": recorded in the node, but not on disk
        inst.set_nodeattr("ipgen_path", ip)

    model = model.transform(JustInTimeSynthesize("xczu3eg-sbva484-1-e", 5.0))

    assert sorted(recorders["codegen"]) == chrc
    assert recorders["ipgen"] == ([] if ip_state == "exists" else hls)
    assert sorted(recorders["rtlsim"]) == chrc
    for node in model.graph.node:
        inst = getCustomOp(node)
        if node.name in hls:
            assert os.path.isdir(inst.get_nodeattr("ipgen_path"))
        if node.name in chrc:
            assert inst.get_nodeattr("rtlsim_so") != ""
        else:
            assert inst.get_nodeattr("rtlsim_so") == ""


def make_pwpolyf_model(ch=16, pe=2, n_vec=8):
    """``inp -> PWPolyF -> outp``: an RTL node whose tree model replaces rtlsim
    characterization."""
    shape = [1, n_vec, ch]
    fp32 = DataType["FLOAT32"].name
    node = helper.make_node(
        "PWPolyF_rtl",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        func="gelu",
        NumChannels=ch,
        PE=pe,
        inputDataType=fp32,
        outputDataType=fp32,
        numInputVectors=[1, n_vec],
    )
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    graph = helper.make_graph([node], "pwpolyf", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pwpolyf"))
    for t in ["inp", "outp"]:
        model.set_tensor_datatype(t, DataType["FLOAT32"])
    return model.transform(GiveUniqueNodeNames())


@pytest.mark.fpgadataflow
def test_jit_skips_pwpolyf(recorders):
    """PWPolyF has a tree model: heuristic_analytical neither synthesizes nor
    simulates it, and the tree model characterizes it without rtlsim."""
    model = make_pwpolyf_model()
    (node,) = model.graph.node
    assert not needs_rtlsim_characterization(node, only_without_tree_model=True)

    model = model.transform(JustInTimeSynthesize("xcvc1902-vsva2197-2MP-e-S", 5.0, True))
    assert recorders == {"codegen": [], "ipgen": [], "rtlsim": []}

    model = model.transform(DeriveTokenAccessVectors("tree_model", rtlsim_fallback=False))
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("io_chrc_in") != ""
    assert inst.get_nodeattr("rtlsim_so") == ""
