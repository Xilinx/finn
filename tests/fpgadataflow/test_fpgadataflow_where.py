# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from functools import partial
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferWhereLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.vivado import parse_ooc_synth_results

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10
DEFAULT_SHAPE = (1, 2, 4)
SYNTH_SHAPE = (1, 64, 64)
# X replays a 1024-word row (BRAM); cond/Y broadcast and stay in LUTRAM.
SYNTH_OOC_OUT_SHAPE = (1, 8, 1024)
SYNTH_OOC_X_SHAPE = (1, 1, 1024)
SYNTH_OOC_COND_SHAPE = (1, 1, 1)
SYNTH_OOC_Y_SHAPE = (1, 8, 1)

SHAPE_CASES = [
    pytest.param((None, None, None, None), id="simple"),
    pytest.param(((1, 3, 1), (1, 1, 4), (1, 2, 1, 1), (1, 2, 3, 4)), id="broadcast"),
    pytest.param(((), (2, 3), (1, 3), (2, 3)), id="scalar_broadcast"),
]


def make_where_modelwrapper(finn_dtype, cond_shape, x_shape, y_shape, out_shape):
    """Build a plain ONNX Where model for conversion to the FINN streaming op."""

    node = helper.make_node("Where", ["cond", "xval", "yval"], ["out"], name="where_select")
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, cond_shape)
    xval = helper.make_tensor_value_info("xval", TensorProto.FLOAT, x_shape)
    yval = helper.make_tensor_value_info("yval", TensorProto.FLOAT, y_shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    graph = helper.make_graph([node], "where_test", [cond, xval, yval], [output])
    model = ModelWrapper(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)]))
    model.set_tensor_datatype("cond", DataType["BINARY"])
    for name in ["xval", "yval", "out"]:
        model.set_tensor_datatype(name, finn_dtype)
    return model


def prepare_inputs(finn_dtype, cond_shape, x_shape, y_shape):
    cond = gen_finn_dt_tensor(DataType["BINARY"], cond_shape)
    xval = gen_finn_dt_tensor(finn_dtype, x_shape)
    yval = gen_finn_dt_tensor(finn_dtype, y_shape)
    return cond, xval, yval


def _specialize(model):
    model = model.transform(SpecializeLayers(FPGA_PART))
    return model.transform(GiveUniqueNodeNames())


def _run_sim_style(model, sim_style):
    model = _specialize(model)
    if sim_style == "node_by_node":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    elif sim_style == "stitched_ip":
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = _specialize(model)
        model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(FPGA_PART, CLK_NS))
        model.set_metadata_prop("exec_mode", "rtlsim")
    else:
        raise ValueError("Unknown simulation style: %s" % sim_style)
    return model


@pytest.mark.parametrize(
    "finn_dtype",
    [
        DataType["INT8"],
        DataType["INT4"],
        DataType["UINT4"],
        DataType["BIPOLAR"],
        DataType["FLOAT16"],
        DataType["FLOAT32"],
    ],
)
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.parametrize("shape_case", SHAPE_CASES)
@pytest.mark.parametrize("sim_style", ["node_by_node", "stitched_ip"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_where_execution(sim_style, shape_case, pe, finn_dtype):
    cond_case, x_case, y_case, out_case = shape_case
    cond_shape = DEFAULT_SHAPE if cond_case is None else cond_case
    x_shape = DEFAULT_SHAPE if x_case is None else x_case
    y_shape = DEFAULT_SHAPE if y_case is None else y_case
    out_shape = DEFAULT_SHAPE if out_case is None else out_case
    if out_shape[-1] % pe != 0:
        pytest.skip("PE (%d) must divide innermost output dim (%d)" % (pe, out_shape[-1]))
    if len(cond_shape) == 0 and sim_style == "stitched_ip":
        pytest.skip("A scalar condition has no leading batch dimension for stitched-IP rtlsim")

    model = make_where_modelwrapper(finn_dtype, cond_shape, x_shape, y_shape, out_shape)
    cond, xval, yval = prepare_inputs(finn_dtype, cond_shape, x_shape, y_shape)
    expected = np.where(cond.astype(bool), xval, yval)

    model = model.transform(InferWhereLayer())
    getCustomOp(model.graph.node[0]).set_nodeattr("PE", pe)
    model = _run_sim_style(model, sim_style)

    produced = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})["out"]
    np.testing.assert_array_equal(produced, expected)

    # Stitched-IP execution includes FIFO latency, so compare the cycle model
    # against the node-by-node result only.
    if sim_style == "node_by_node":
        node = model.get_nodes_by_op_type("HWWhere_rtl")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0


@pytest.mark.parametrize(
    "shape, ram_style, expected_bram, expected_lut, expected_uram",
    [
        (DEFAULT_SHAPE, "auto", 0, 114, 0),
        (SYNTH_SHAPE, "auto", 0, 114, 0),
        (DEFAULT_SHAPE, "block", 3, 80, 0),
        (DEFAULT_SHAPE, "distributed", 0, 114, 0),
        (DEFAULT_SHAPE, "ultra", 0, 80, 3),
    ],
)
@pytest.mark.fpgadataflow
def test_where_resource_estimation(shape, ram_style, expected_bram, expected_lut, expected_uram):
    model = make_where_modelwrapper(DataType["INT8"], shape, shape, shape, shape)
    model = model.transform(InferWhereLayer())
    where = getCustomOp(model.graph.node[0])
    where.set_nodeattr("PE", 2)
    where.set_nodeattr("ram_style", ram_style)
    model = _specialize(model)

    resources = model.analysis(partial(res_estimation, fpgapart=FPGA_PART))
    node_resources = resources[model.graph.node[0].name]
    assert node_resources["BRAM_18K"] == expected_bram
    assert node_resources["LUT"] == expected_lut
    assert node_resources["URAM"] == expected_uram
    assert node_resources["DSP"] == 0


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_where_stitched_ip_synth_ooc():
    model = make_where_modelwrapper(
        DataType["INT8"],
        SYNTH_OOC_COND_SHAPE,
        SYNTH_OOC_X_SHAPE,
        SYNTH_OOC_Y_SHAPE,
        SYNTH_OOC_OUT_SHAPE,
    )
    model = model.transform(InferWhereLayer())
    where = getCustomOp(model.graph.node[0])
    where.set_nodeattr("PE", 2)
    # Broadcasting X drives a real replay buffer that Vivado's auto mapping is
    # expected to place in BRAM, so this exercises the BRAM estimate against
    # synthesis (the small functional-test shape stays in LUTRAM).
    where.set_nodeattr("ram_style", "auto")
    model = _run_sim_style(model, "stitched_ip")

    where_rtl = getCustomOp(model.get_nodes_by_op_type("HWWhere_rtl")[0])
    expected_bram = where_rtl.bram_estimation(FPGA_PART)
    assert expected_bram > 0
    model = model.transform(CreateStitchedIP(FPGA_PART, CLK_NS, run_pnr=True))
    ret = parse_ooc_synth_results(model.get_metadata_prop("vivado_stitch_proj"))
    assert ret is not None
    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    synthesized_bram = ret.get("BRAM_18K", 0) + 2 * ret.get("BRAM_36K", 0)
    assert synthesized_bram == expected_bram
    assert ret["WNS"] >= 0
