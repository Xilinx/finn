# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""JustInTimeSynthesize builds IP and rtlsim only for the nodes that are
characterized with rtlsim, and only where the IP does not exist yet.

Code generation, HLS synthesis and rtlsim preparation are replaced by
recorders, so no Vivado is needed."""

import pytest

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
from finn.transformation.fpgadataflow.derive_characteristic import JustInTimeSynthesize
from finn.util.basic import make_build_dir

DT = DataType["INT4"]


def make_fork_join_model(ch=16, pe=2, n_vec=8):
    """``inp -> DuplicateStreams -> (a, b)``, ``b`` through a 1:4 and a 4:1
    width converter, then ``ElementwiseAdd(a, b)``: a fork and a join, which are
    never characterized, and two characterized nodes."""
    shape = [1, n_vec, ch]
    tensors = ["inp", "a", "b", "b_wide", "b_narrow", "outp"]
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
    for inw, outw, src, dst in [
        (width, 4 * width, "b", "b_wide"),
        (4 * width, width, "b_wide", "b_narrow"),
    ]:
        nodes.append(
            helper.make_node(
                "StreamingDataWidthConverter_hls",
                [src],
                [dst],
                domain="finn.custom_op.fpgadataflow.hls",
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
            ["a", "b_narrow"],
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
    return model.transform(GiveUniqueNodeNames())


class _NoOp(Transformation):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def apply(self, model):
        return (model, False)


@pytest.fixture
def recorders(monkeypatch):
    """Replace code generation, HLS synthesis, the Verilog path fix-up and
    rtlsim preparation by recorders. Returns the names of the nodes each step
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
    return calls


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("ip_state", ["fresh", "exists", "missing"])
def test_jit_synthesizes_only_what_is_needed(recorders, ip_state):
    """IP is synthesized for the characterized nodes unless it already exists;
    the fork and the join get neither IP nor rtlsim."""
    model = make_fork_join_model()
    dwcs = [n.name for n in model.graph.node if n.op_type.startswith("StreamingDataWidth")]
    for node in model.graph.node:
        if node.name not in dwcs or ip_state == "fresh":
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

    assert sorted(recorders["codegen"]) == sorted(dwcs)
    assert sorted(recorders["ipgen"]) == ([] if ip_state == "exists" else sorted(dwcs))
    assert sorted(recorders["rtlsim"]) == sorted(dwcs)
    for node in model.graph.node:
        inst = getCustomOp(node)
        if node.name in dwcs:
            assert os.path.isdir(inst.get_nodeattr("ipgen_path"))
            assert inst.get_nodeattr("rtlsim_so") != ""
        else:
            assert inst.get_nodeattr("rtlsim_so") == ""
