# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Double-pumped designs: propagation of the phase-aligned 2x clock ap_clk2x.

Covered here, for a stitched IP with an ap_clk2x input:
- the generic detection of nodes needing ap_clk2x (their "clk2x" interface),
- stitched-IP rtlsim of MVAUs with pumpedCompute and/or pumpedMemory next to
  non-pumped nodes (internal_decoupled and dynamic weights),
- the early rejection of configurations and shell flows that cannot provide
  ap_clk2x.

The internal ap_clk2x generator of the stitched IP and the shell flows built on
it are covered by test_fpgadataflow_clk2x_gen.py.

Pumped MVAUs inside FINNLoop bodies (fetch-weights MVAU with pumpedCompute) are
covered by the MLO end-to-end tests in test_fpgadataflow_finnloop.py.
"""

import pytest

import json
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from types import SimpleNamespace
from unittest.mock import Mock, patch

import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow_steps import check_shell_clk2x_support
from finn.custom_op.fpgadataflow.rtl.finn_loop import FINNLoop
from finn.transformation.fpgadataflow.alveo_build import PrepareForLinking, SlashLink
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.make_pynq_proj import (
    PYNQBuild,
    check_versal_shell_clk2x,
)
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.general import ApplyConfig
from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.fpgadataflow import find_clk2x_nodes

# DSP58 compute pumping needs a Versal part
versal_part = "xcvc1902-vsva2197-2MP-e-S"
clk_ns = 5

RTL_DOMAIN = "finn.custom_op.fpgadataflow.rtl"
HLS_DOMAIN = "finn.custom_op.fpgadataflow.hls"


def make_mvau_node(name, inputs, output, impl, mw, mh, idt, wdt, odt, **attrs):
    return helper.make_node(
        "MVAU_" + impl,
        inputs,
        [output],
        name=name,
        domain=RTL_DOMAIN if impl == "rtl" else HLS_DOMAIN,
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        numInputVectors=[1, 3, 3],
        **attrs,
    )


def make_pumped_chain_model(mw=16, mh=16, pe=4, simd=4):
    """MVAU chain with all pumping variants next to non-pumped nodes:
    MVAU_rtl (pumpedCompute) -> Thresholding_rtl (not pumped) -> MVAU_hls
    (pumpedMemory, fused activation) -> MVAU_rtl (pumpedCompute and pumpedMemory).
    All weights are internal_decoupled (on-chip memstream)."""
    idt = wdt = adt = DataType["INT4"]
    acc = DataType["INT32"]
    n_steps = adt.get_num_possible_values() - 1
    shape_in = [1, 3, 3, mw]
    shape_out = [1, 3, 3, mh]
    common = {"PE": pe, "SIMD": simd, "mem_mode": "internal_decoupled"}
    nodes = [
        make_mvau_node(
            "mvau0",
            ["inp", "w0"],
            "mm0",
            "rtl",
            mw,
            mh,
            idt,
            wdt,
            acc,
            noActivation=1,
            pumpedCompute=1,
            **common,
        ),
        helper.make_node(
            "Thresholding_rtl",
            ["mm0", "thr0"],
            ["act0"],
            name="thr0",
            domain=RTL_DOMAIN,
            backend="fpgadataflow",
            NumChannels=mh,
            PE=pe,
            inputDataType=acc.name,
            weightDataType="INT33",
            outputDataType=adt.name,
            ActVal=int(adt.min()),
            numSteps=n_steps,
            numInputVectors=[1, 3, 3],
        ),
        make_mvau_node(
            "mvau1",
            ["act0", "w1", "thr1"],
            "act1",
            "hls",
            mh,
            mh,
            adt,
            wdt,
            adt,
            noActivation=0,
            ActVal=int(adt.min()),
            pumpedMemory=1,
            **common,
        ),
        make_mvau_node(
            "mvau2",
            ["act1", "w2"],
            "outp",
            "rtl",
            mh,
            mh,
            adt,
            wdt,
            acc,
            noActivation=1,
            pumpedCompute=1,
            pumpedMemory=1,
            **common,
        ),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="pumped_chain",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape_in)],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape_out)],
        value_info=[
            helper.make_tensor_value_info(t, TensorProto.FLOAT, shape_out)
            for t in ["mm0", "act0", "act1"]
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pumped-chain"))
    model.set_initializer("w0", gen_finn_dt_tensor(wdt, (mw, mh)))
    model.set_initializer("w1", gen_finn_dt_tensor(wdt, (mh, mh)))
    model.set_initializer("w2", gen_finn_dt_tensor(wdt, (mh, mh)))
    # thresholds within the accumulator range, so that the activations vary
    acc_max = mw * 64
    thr0 = np.sort(np.random.randint(-acc_max // 4, acc_max // 4, (1, n_steps)), axis=1)
    thr1 = np.sort(np.random.randint(-acc_max // 4, acc_max // 4, (mh, n_steps)), axis=1)
    model.set_initializer("thr0", thr0.astype(np.float32))
    model.set_initializer("thr1", thr1.astype(np.float32))
    for t, dt in [
        ("inp", idt),
        ("w0", wdt),
        ("w1", wdt),
        ("w2", wdt),
        ("mm0", acc),
        ("thr0", acc),
        ("act0", adt),
        ("thr1", acc),
        ("act1", adt),
        ("outp", acc),
    ]:
        model.set_tensor_datatype(t, dt)
    return model


def cppsim_reference(model, x):
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    return oxe.execute_onnx(model, {model.graph.input[0].name: x})[model.graph.output[0].name]


def stitch_and_rtlsim(model, inputs, part):
    """Size FIFOs, stitch and run stitched-IP rtlsim on the graph inputs (in
    order; FIFO sizing renames the tensors)."""
    model = model.transform(InsertAndSetFIFODepths(part, clk_ns))
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(part, clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")
    input_dict = {inp.name: x for inp, x in zip(model.graph.input, inputs)}
    out = oxe.execute_onnx(model, input_dict)[model.graph.output[0].name]
    return model, out


def assert_stitched_clk2x(model):
    """The stitched IP exposes one ap_clk2x port, made external from the first
    pumped node and connected to all others."""
    pumped = find_clk2x_nodes(model)
    assert pumped, "no pumped node in the stitched model"
    assert json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))["clk2x"] == ["ap_clk2x"]
    with open(model.get_metadata_prop("vivado_stitch_proj") + "/make_project.tcl") as f:
        tcl = f.read()
    assert tcl.count("set_property name ap_clk2x [get_bd_ports ap_clk2x_0]") == 1
    for name in pumped:
        assert (
            "make_bd_pins_external [get_bd_pins %s/ap_clk2x]" % name in tcl
            or "connect_bd_net [get_bd_ports ap_clk2x] [get_bd_pins %s/ap_clk2x]" % name in tcl
        ), f"ap_clk2x of {name} not wired"


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_clk2x_stitched_mvau_chain():
    model = make_pumped_chain_model()
    model = model.transform(GiveUniqueNodeNames())
    x = gen_finn_dt_tensor(DataType["INT4"], model.get_tensor_shape("inp"))
    y_ref = cppsim_reference(model, x)

    model, y = stitch_and_rtlsim(model, [x], versal_part)
    # the three pumped MVAUs and the non-pumped thresholding sit in one stitched IP
    assert len(find_clk2x_nodes(model)) == 3
    assert_stitched_clk2x(model)
    assert (y == y_ref).all(), "stitched-IP rtlsim of the pumped MVAU chain failed"


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_clk2x_stitched_dynamic_mvau():
    # MVAU_rtl with dynamic (streamed) weights and pumpedCompute
    n_vectors, mw, mh = 8, 16, 16
    idt = wdt = DataType["INT8"]
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, n_vectors, mw])
    wfm = helper.make_tensor_value_info("wfm", TensorProto.FLOAT, [1, mw, mh])
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, [1, n_vectors, mh])
    graph = helper.make_graph(
        nodes=[helper.make_node("MatMul", ["ifm", "wfm"], ["ofm"])],
        name="dyn_matmul",
        inputs=[ifm, wfm],
        outputs=[ofm],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="dyn-matmul"))
    model.set_tensor_datatype("ifm", idt)
    model.set_tensor_datatype("wfm", wdt)
    model.set_tensor_datatype("ofm", DataType["INT32"])
    x = gen_finn_dt_tensor(idt, [1, n_vectors, mw])
    w = gen_finn_dt_tensor(wdt, [1, mw, mh])
    y_ref = np.matmul(x, w)

    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(GiveUniqueNodeNames())
    getCustomOp(model.graph.node[0]).set_nodeattr("preferred_impl_style", "rtl")
    model = model.transform(SpecializeLayers(versal_part))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "MVAU_rtl"
    assert getCustomOp(model.graph.node[0]).get_nodeattr("mem_mode") == "dynamic"
    model = model.transform(
        ApplyConfig(
            {
                "Defaults": {},
                "MVAU_rtl_0": {"PE": 4, "SIMD": 8, "resType": "dsp", "pumpedCompute": 1},
            }
        )
    )
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(InferDataTypes())

    model, y = stitch_and_rtlsim(model, [x, w], versal_part)
    assert_stitched_clk2x(model)
    assert (y == y_ref).all(), "stitched-IP rtlsim of the pumped dynamic MVAU failed"


def _bd_nets(bd_file):
    """Map each net of a block design file to the set of pins/ports it connects."""
    with open(bd_file) as f:
        design = json.load(f)["design"]
    return {name: set(net["ports"]) for name, net in design.get("nets", {}).items()}


@pytest.mark.parametrize("body_clk2x", [[], ["ap_clk2x"]])
@pytest.mark.fpgadataflow
def test_finnloop_propagates_optional_clk2x(body_clk2x):
    body = Mock()
    body.get_metadata_prop.return_value = repr(
        {
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
            "clk2x": body_clk2x,
            "aximm": [],
        }
    )
    loop_inst = Mock()
    loop_inst.onnx_node.attribute = []
    loop_inst.get_nodeattr.return_value = body
    loop_inst.get_instream_width_padded.return_value = 32
    loop_inst.get_outstream_width_padded.return_value = 32

    intf_names = FINNLoop.get_verilog_top_module_intf_names(loop_inst)

    if body_clk2x:
        assert intf_names["clk2x"] == body_clk2x
    else:
        assert "clk2x" not in intf_names


@pytest.mark.fpgadataflow
def test_stitcher_detects_clk2x_by_interface():
    # op types are deliberately not MVAU/FINNLoop: only the interface counts
    nodes = [
        SimpleNamespace(name="A", op_type="AnyOp_rtl"),
        SimpleNamespace(name="B", op_type="AnyOp_hls"),
        SimpleNamespace(name="C", op_type="FINNLoop"),
    ]
    clk2x = {"A": [], "B": ["ap_clk2x"], "C": ["ap_clk2x"]}

    def get_custom_op(node):
        inst = Mock()
        inst.get_verilog_top_module_intf_names.return_value = {
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
            "clk2x": clk2x[node.name],
        }
        return inst

    stitcher = CreateStitchedIP(versal_part, clk_ns)
    with patch(
        "finn.transformation.fpgadataflow.create_stitched_ip.getCustomOp",
        side_effect=get_custom_op,
    ):
        for node in nodes:
            stitcher.connect_clk_rst(node)

    cmds = stitcher.connect_cmds
    assert not [c for c in cmds if "A/ap_clk2x" in c]
    assert "make_bd_pins_external [get_bd_pins B/ap_clk2x]" in cmds
    assert "connect_bd_net [get_bd_ports ap_clk2x] [get_bd_pins C/ap_clk2x]" in cmds
    assert stitcher.intf_names["clk2x"] == ["ap_clk2x"]

    # with the internal clock generator, its clock domain drives all clocks
    stitcher = CreateStitchedIP(versal_part, clk_ns, clk2x_gen=True)
    stitcher.clk2x_gen_active = True
    with patch(
        "finn.transformation.fpgadataflow.create_stitched_ip.getCustomOp",
        side_effect=get_custom_op,
    ):
        for node in nodes:
            stitcher.connect_clk_rst(node)
    cmds = stitcher.connect_cmds
    assert not [c for c in cmds if "get_bd_ports" in c or "external" in c]
    for name in "ABC":
        assert (
            "connect_bd_net [get_bd_pins clk2x_domain/clk] [get_bd_pins %s/ap_clk]" % name in cmds
        )
        assert (
            "connect_bd_net [get_bd_pins clk2x_domain/rst_n] [get_bd_pins %s/ap_rst_n]" % name
            in cmds
        )
    assert [c for c in cmds if "clk2x_domain/clk2x" in c] == [
        "connect_bd_net [get_bd_pins clk2x_domain/clk2x] [get_bd_pins %s/ap_clk2x]" % name
        for name in "BC"
    ]
    assert "clk2x" not in stitcher.intf_names


def make_single_mvau_model(name="mvau", **attrs):
    mw = mh = 8
    dt = DataType["INT4"]
    node = make_mvau_node(
        name,
        ["inp_" + name, "w_" + name],
        "outp_" + name,
        "rtl",
        mw,
        mh,
        dt,
        dt,
        DataType["INT32"],
        noActivation=1,
        PE=2,
        SIMD=2,
        **attrs,
    )
    graph = helper.make_graph(
        nodes=[node],
        name="g_" + name,
        inputs=[helper.make_tensor_value_info("inp_" + name, TensorProto.FLOAT, [1, 3, 3, mw])],
        outputs=[helper.make_tensor_value_info("outp_" + name, TensorProto.FLOAT, [1, 3, 3, mh])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w_" + name, gen_finn_dt_tensor(dt, (mw, mh)))
    model.set_tensor_datatype("inp_" + name, dt)
    model.set_tensor_datatype("w_" + name, dt)
    model.set_tensor_datatype("outp_" + name, DataType["INT32"])
    return model


@pytest.mark.fpgadataflow
def test_find_clk2x_nodes_recurses_into_loops_and_partitions():
    pumped = make_single_mvau_model("pumped", pumpedCompute=1, mem_mode="internal_decoupled")
    plain = make_single_mvau_model("plain", mem_mode="internal_decoupled")
    assert find_clk2x_nodes(pumped) == ["pumped"]
    assert find_clk2x_nodes(plain) == []

    def wrap_in_loop(body, name):
        # find_clk2x_nodes only looks at the body graph, not at the loop's
        # other attributes, so an otherwise bare FINNLoop node is enough
        loop = helper.make_node(
            "FINNLoop",
            [body.graph.input[0].name],
            [body.graph.output[0].name],
            name=name,
            domain=RTL_DOMAIN,
            backend="fpgadataflow",
            body=body.graph,
        )
        graph = helper.make_graph(
            nodes=[loop],
            name="outer_" + name,
            inputs=list(body.graph.input),
            outputs=list(body.graph.output),
        )
        return ModelWrapper(qonnx_make_model(graph))

    # nested loops: the pumped MVAU two loop levels down
    nested = wrap_in_loop(wrap_in_loop(pumped, "inner"), "outer")
    assert find_clk2x_nodes(nested) == ["outer/inner/pumped"]
    assert find_clk2x_nodes(wrap_in_loop(plain, "loop")) == []

    # dataflow partition
    parent = pumped.transform(CreateDataflowPartition())
    assert parent.graph.node[0].op_type == "StreamingDataflowPartition"
    sdp_name = parent.graph.node[0].name
    assert find_clk2x_nodes(parent) == [sdp_name + "/pumped"]


@pytest.mark.parametrize(
    "impl,attrs",
    [
        # pumpedMemory only pumps the on-chip weight memstream
        ("rtl", {"pumpedMemory": 1, "mem_mode": "internal_embedded"}),
        ("hls", {"pumpedMemory": 1, "mem_mode": "internal_embedded"}),
        ("rtl", {"pumpedMemory": 1, "mem_mode": "external_mem"}),
        # the tiled MVU supports neither pumping mode
        ("rtl", {"pumpedCompute": 1, "TH": 3, "mem_mode": "internal_decoupled"}),
        ("rtl", {"pumpedMemory": 1, "TH": 3, "mem_mode": "internal_decoupled"}),
    ],
)
@pytest.mark.fpgadataflow
def test_mvau_rejects_unsupported_pumping(impl, attrs):
    model = make_single_mvau_model(**attrs)
    if impl == "hls":
        node = model.graph.node[0]
        node.op_type = "MVAU_hls"
        node.domain = HLS_DOMAIN
    with pytest.raises(Exception, match="pumpedMemory requires|not supported by the tiled MVU"):
        model.transform(PrepareIP(versal_part, clk_ns))


@pytest.mark.fpgadataflow
def test_shell_flows_reject_clk2x():
    pumped = make_single_mvau_model(pumpedCompute=1, mem_mode="internal_decoupled")
    plain = make_single_mvau_model(mem_mode="internal_decoupled")

    # SLASH and the Versal golden-reference shell support no double pumping
    with pytest.raises(Exception, match="slash-vrt.*mvau"):
        PrepareForLinking(versal_part, clk_ns, "slash-vrt").apply(pumped)
    with pytest.raises(Exception, match="slash-vrt.*mvau"):
        SlashLink().apply(pumped.transform(CreateDataflowPartition()))
    with pytest.raises(Exception, match="Versal shell.*mvau"):
        check_versal_shell_clk2x(pumped, "VCK190")
    with pytest.raises(Exception, match="Versal shell.*mvau"):
        PYNQBuild("VCK190", 10).apply(pumped)
    check_versal_shell_clk2x(plain, "VCK190")

    # the builder check, per board and ap_clk2x source: with the internal clock
    # generator, the Zynq and Vitis flows support double pumping (Vitis at an
    # integer frequency in MHz); with an ap_clk2x input only Vitis does, given
    # the platform clocks
    gen, ext = build_cfg.Clk2xSource.GENERATOR, build_cfg.Clk2xSource.EXTERNAL
    out_dir = make_build_dir("test_clk2x_shell_")
    for board, period, source, ids, supported in [
        ("AUP-ZU3_8GB", 10.0, gen, None, True),
        ("AUP-ZU3_8GB", 3.0, gen, None, True),
        ("AUP-ZU3_8GB", 10.0, ext, None, False),
        ("U55C", 10.0, gen, None, True),
        ("U55C", 3.0, gen, None, False),
        ("U55C", 10.0, ext, None, False),
        ("U55C", 10.0, ext, [2, 1], True),
        ("VCK190", 10.0, gen, None, False),
        ("V80", 10.0, gen, None, False),
    ]:
        for outputs, requested in [
            ([build_cfg.DataflowOutputType.BITFILE], True),
            ([build_cfg.DataflowOutputType.STITCHED_IP], False),
        ]:
            cfg = build_cfg.DataflowBuildConfig(
                output_dir=out_dir,
                synth_clk_period_ns=period,
                board=board,
                generate_outputs=outputs,
                clk2x_source=source,
                vitis_clk2x_clock_ids=ids,
            )
            check_shell_clk2x_support(plain, cfg)
            if supported or not requested:
                check_shell_clk2x_support(pumped, cfg)
            else:
                with pytest.raises(Exception, match="mvau"):
                    check_shell_clk2x_support(pumped, cfg)
    robust_rmtree(out_dir)
