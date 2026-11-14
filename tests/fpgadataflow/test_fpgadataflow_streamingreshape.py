# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import math
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.test import tree_model_test

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10

# (op_type, input shape, output shape, attributes, constant second input)
SHAPE_OPS = {
    "squeeze": ("Squeeze", [1, 1, 4, 6], [1, 4, 6], {}, [1]),
    "squeeze-last": ("Squeeze", [1, 4, 6, 1], [1, 4, 6], {}, [3]),
    "unsqueeze": ("Unsqueeze", [1, 4, 6], [1, 4, 1, 6], {}, [2]),
    "reshape-regroup": ("Reshape", [1, 4, 6], [1, 8, 3], {}, [1, 8, 3]),
    "flatten": ("Flatten", [1, 2, 3, 4], [1, 24], {"axis": 1}, None),
}


def shape_op_node(name, inp, out, op):
    """The ONNX node ``name`` of the SHAPE_OPS entry ``op``, and its constant."""
    op_type, _, _, attrs, const = op
    if const is None:
        return helper.make_node(op_type, [inp], [out], name=name, **attrs), []
    const_name = name + "_const"
    init = numpy_helper.from_array(np.asarray(const, dtype=np.int64), name=const_name)
    return helper.make_node(op_type, [inp, const_name], [out], name=name, **attrs), [init]


def make_shape_op_modelwrapper(ops, idt, tail=True):
    """``inp -> Add -> ops... [-> Add] -> outp``, the Adds with constants.

    The Adds become hardware layers, so the shape operators sit between two of
    them, or at the end of the graph if ``tail`` is False.
    """
    ishape, oshape = ops[0][1], ops[-1][2]
    nodes, inits = [], []
    c0 = numpy_helper.from_array(gen_finn_dt_tensor(idt, ishape), name="c0")
    nodes.append(helper.make_node("Add", ["inp", "c0"], ["t0"], name="add0"))
    inits.append(c0)
    for i, op in enumerate(ops):
        out = "t%d" % (i + 1) if (tail or i < len(ops) - 1) else "outp"
        node, const = shape_op_node("shape%d" % i, "t%d" % i, out, op)
        nodes.append(node)
        inits += const
    if tail:
        c1 = numpy_helper.from_array(gen_finn_dt_tensor(idt, oshape), name="c1")
        nodes.append(helper.make_node("Add", ["t%d" % len(ops), "c1"], ["outp"], name="add1"))
        inits.append(c1)
    graph = helper.make_graph(
        nodes,
        "streamingreshape-model",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)],
        inits,
    )
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="streamingreshape-model",
            opset_imports=[helper.make_opsetid("", 13)],
        ),
        fix_missing_initializer_valueinfo=True,
    )
    for name in ["inp", "c0", "c1"]:
        if model.get_initializer(name) is not None or name == "inp":
            model.set_tensor_datatype(name, idt)
    model = model.transform(InferShapes())
    return model.transform(InferDataTypes())


def to_hw_layers(model):
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    return model.transform(to_hw.InferStreamingReshape())


def prepare_stitched_ip_model(model):
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    return model.transform(CreateStitchedIP(FPGA_PART, CLK_NS))


# shape operator
@pytest.mark.parametrize("op", list(SHAPE_OPS), ids=list(SHAPE_OPS))
# input datatype, and PE of the reshape layer (its largest or 1), per execution mode
@pytest.mark.parametrize(
    "idt,pe,exec_mode",
    [
        pytest.param(DataType["INT4"], "max", "cppsim", id="INT4-max-cppsim"),
        pytest.param(DataType["INT4"], "max", "rtlsim", id="INT4-max-rtlsim"),
        pytest.param(DataType["UINT7"], 1, "rtlsim", id="UINT7-1-rtlsim"),
        pytest.param(DataType["INT4"], "max", "stitched_rtlsim", id="INT4-max-stitched"),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_streamingreshape(op, idt, pe, exec_mode):
    shape_op = SHAPE_OPS[op]
    model = make_shape_op_modelwrapper([shape_op], idt)
    input_dict = {"inp": gen_finn_dt_tensor(idt, shape_op[1])}
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    model = to_hw_layers(model)
    assert [n.op_type for n in model.graph.node] == [
        "ElementwiseAdd",
        "StreamingReshape",
        "ElementwiseAdd",
    ]
    reshape = getCustomOp(model.get_nodes_by_op_type("StreamingReshape")[0])
    if pe == "max":
        pe = math.gcd(shape_op[1][-1], shape_op[2][-1])
    reshape.set_nodeattr("PE", pe)
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert np.array_equal(y_produced, y_expected), "Execution of the hw layers failed"

    model = model.transform(SpecializeLayers(FPGA_PART))
    assert len(model.get_nodes_by_op_type("StreamingReshape_rtl")) == 1

    if exec_mode == "cppsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    elif exec_mode == "stitched_rtlsim":
        model = prepare_stitched_ip_model(model)
        model.set_metadata_prop("exec_mode", "rtlsim")
    else:
        raise ValueError("Unknown exec_mode")

    y_produced = oxe.execute_onnx(model, input_dict)["outp"].reshape(y_expected.shape)
    assert np.array_equal(y_produced, y_expected), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("StreamingReshape_rtl")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0


# shape operators, whether they are followed by a hardware layer, and the
# number of StreamingReshape layers expected
@pytest.mark.parametrize(
    "ops,tail,expected",
    [
        pytest.param(["squeeze"], True, 1, id="between-hw-layers"),
        pytest.param(["unsqueeze", "squeeze"], True, 1, id="chain-collapses"),
        pytest.param(["squeeze"], False, 0, id="graph-output-stays"),
    ],
)
@pytest.mark.fpgadataflow
def test_infer_streamingreshape(ops, tail, expected):
    ops = [SHAPE_OPS[x] for x in ops]
    if len(ops) == 2:
        # make the squeeze undo the unsqueeze
        ops[1] = ("Squeeze", ops[0][2], ops[0][1], {}, [2])
    idt = DataType["INT4"]
    model = make_shape_op_modelwrapper(ops, idt, tail)
    input_dict = {"inp": gen_finn_dt_tensor(idt, ops[0][1])}
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    model = to_hw_layers(model)
    assert len(model.get_nodes_by_op_type("StreamingReshape")) == expected
    remaining = [n for n in model.graph.node if n.op_type in to_hw.InferStreamingReshape.SHAPE_OPS]
    assert len(remaining) == (0 if expected else len(ops))
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert np.array_equal(y_produced, y_expected)


# shape operator
@pytest.mark.parametrize("op", ["squeeze", "reshape-regroup"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_streamingreshape(op):
    _, ishape, oshape, _, _ = SHAPE_OPS[op]
    idt = DataType["INT8"]
    node = helper.make_node(
        "StreamingReshape_rtl",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        DataType=idt.name,
        in_shape=ishape,
        out_shape=oshape,
        PE=math.gcd(ishape[-1], oshape[-1]),
        name="StreamingReshape_rtl_0",
    )
    graph = helper.make_graph(
        [node],
        "streamingreshape-single",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="streamingreshape-single"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    # The layer is a wire, so the model is exact; the constants allow for the
    # simulation driver's offset only.
    assert tree_model_test(
        model, ("StreamingReshape", op), "xczu7ev-ffvc1156-2-e", 5, 0.0, 0.0, 3, 3
    ), "characterized TAV does not match RTLsim'd one!"
