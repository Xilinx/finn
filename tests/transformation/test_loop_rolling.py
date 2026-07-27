import pytest

import brevitas.onnx as bo
import numpy as np
import onnx
import qonnx.util.basic as util
import torch
from brevitas.nn import QuantLinear
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
from finn.transformation.fpgadataflow.raise_scalar_to_rank1 import RaiseScalarToRank1
from finn.transformation.fpgadataflow.set_loop_boundary import SetLoopBoundary
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline.absorb import AbsorbSignBiasIntoMultiThreshold
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.reorder import (
    MoveAddPastMul,
    MoveScalarAddPastMatMul,
    MoveScalarMulPastMatMul,
)
from finn.util.basic import make_build_dir, robust_rmtree


class SimpleSubModule(torch.nn.Module):
    def __init__(self, in_features, out_features, mul_val=4):
        super(SimpleSubModule, self).__init__()
        self.mul_val = torch.tensor([mul_val])
        self.linear = QuantLinear(
            in_features,
            out_features,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            input_quant=Int8ActPerTensorFloat,
        )

    def forward(self, x):
        return self.mul_val * self.linear(x)


# Simple Torch Module with parameterizable number of linear layers
class SimpleModule(torch.nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=4, mul_val=4, output_size=None):
        super(SimpleModule, self).__init__()
        self.mul_val = mul_val

        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()

        # Create the linear layers
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            out_features = (
                hidden_size if i != (num_layers - 1) or output_size is None else hidden_size
            )
            self.layers.append(SimpleSubModule(in_features, out_features, mul_val=self.mul_val))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# export the model to ONNX format using dynamo
def export_model_to_qonnx(out_dir, input_size=10, hidden_size=20, num_layers=4, output_size=None):
    model = SimpleModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        mul_val=4,
    )
    x = torch.rand((1, input_size))
    model(x)  # Initialise scale factors
    model.eval()

    # per-test out_dir so concurrent workers never share this export path
    onnx_path = str(out_dir / f"simple_module_{num_layers}layers.onnx")
    with torch.no_grad():
        bo.export_qonnx(
            model,
            x,
            onnx_path,
            do_constant_folding=True,
            input_names=["x"],
            opset_version=18,
            dynamo=True,
            optimize=True,
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)

    return onnx_path, model


def check_tensor_shape(model_wrapper, name, expected_shape):
    actual_shape = model_wrapper.get_tensor_shape(name)
    assert (
        actual_shape == expected_shape
    ), f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"


@pytest.mark.transform
def test_loop_extraction_default_paths_are_unique():
    first = LoopExtraction(hierarchy_list=[["", "layers.0"]])
    second = LoopExtraction(hierarchy_list=[["", "layers.0"]])
    first_dir = Path(first.loop_body_template_path).parent
    second_dir = Path(second.loop_body_template_path).parent
    assert first_dir != second_dir
    assert first_dir.parent == second_dir.parent
    robust_rmtree(first_dir)
    robust_rmtree(second_dir)


# input_size == hidden_size to create model that can be rolled
@pytest.mark.parametrize("input_size", [20, 30, 40])
# num_layers
@pytest.mark.parametrize("num_layers", [6, 12, 24])
@pytest.mark.transform
def test_finn_loop(input_size, num_layers):
    out_dir = Path(make_build_dir(prefix="test_finn_loop_"))
    hidden_size = input_size

    onnx_path, model = export_model_to_qonnx(out_dir, input_size, hidden_size, num_layers)

    qonnx_cleanup(onnx_path, out_file=onnx_path)
    model_wrapper = ModelWrapper(onnx_path)
    model_wrapper = model_wrapper.transform(ConvertQONNXtoFINN())

    model_wrapper = model_wrapper.transform(RaiseScalarToRank1())

    # Warning: Running standard streamlining here causes optimizations
    # across loop body boundaries that breaks current loop rolling assumptions.
    # instead of streamlining only apply some transformations and then convert to hw
    model_wrapper = model_wrapper.transform(AbsorbSignBiasIntoMultiThreshold())
    model_wrapper = model_wrapper.transform(ConvertSubToAdd())
    model_wrapper = model_wrapper.transform(ConvertDivToMul())
    model_wrapper = model_wrapper.transform(MoveScalarMulPastMatMul())
    model_wrapper = model_wrapper.transform(MoveScalarAddPastMatMul())
    model_wrapper = model_wrapper.transform(CollapseRepeatedMul())
    model_wrapper = model_wrapper.transform(MoveAddPastMul())
    model_wrapper = model_wrapper.transform(CollapseRepeatedAdd())
    model_wrapper = model_wrapper.transform(CollapseRepeatedMul())
    model_wrapper = model_wrapper.transform(to_hw.InferThresholdingLayer())
    model_wrapper = model_wrapper.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model_wrapper = model_wrapper.transform(to_hw.InferElementwiseBinaryOperation())

    m_input_dt = model_wrapper.get_tensor_datatype(model_wrapper.model.graph.input[0].name)
    m_output_dt = model_wrapper.get_tensor_datatype(model_wrapper.model.graph.output[0].name)

    # Specialize to backend-specific ops before loop extraction/rolling. MLO
    # weight streaming requires the RTL MVAU backend (mem_mode "external_mem" is
    # set by MVAU_rtl.adapt_for_loop_body during rolling), so the abstract MVAU
    # must be specialized first. This must happen before LoopExtraction, since
    # SpecializeLayers only walks top-level graph nodes and would skip the
    # compute nodes once they are moved into the fn_loop-body subgraphs. A Versal
    # part is used so the MVAU specializes to RTL.
    fpga_part = "xcvc1902-vsva2197-2MP-e-S"
    model_wrapper = model_wrapper.transform(SpecializeLayers(fpga_part))

    model_wrapper = model_wrapper.transform(GiveUniqueNodeNames())
    # temporarily set loop boundaries manually
    node_metadata = {
        "pkg.torch.onnx.name_scopes": "['', 'layers.0']",
        "pkg.torch.onnx.class_hierarchy": "['TestModule', 'test']",
    }
    node_range = (model_wrapper.graph.node[0], model_wrapper.graph.node[3])
    model_wrapper = model_wrapper.transform(SetLoopBoundary(node_metadata, node_range=node_range))

    # Loop extraction
    template_path = out_dir / "loop-body-template.onnx"
    loop_extraction = LoopExtraction(
        hierarchy_list=[["", "layers.0"]], loop_body_template_path=template_path
    )
    model_wrapper = model_wrapper.transform(loop_extraction)

    # should be one constant node and one loop-body node per layer
    assert (
        len(model_wrapper.get_nodes_by_op_type("fn_loop-body")) == num_layers
    ), "Loop extraction did not find expected number of loop bodies"

    model_wrapper = model_wrapper.transform(LoopRolling(loop_extraction.loop_body_template))
    model_wrapper = model_wrapper.transform(InferShapes(), apply_to_subgraphs=True)
    assert len(model_wrapper.model.graph.node) == 1, "Should Roll into a Single FinnLoop Node"
    loop_node = model_wrapper.model.graph.node[0]

    assert loop_node.op_type == "FINNLoop", "Node should be op_type FinnLoop"

    assert util.get_by_name(loop_node.attribute, "iteration").i == num_layers
    assert util.get_by_name(loop_node.attribute, "backend").s.decode("utf-8") == "fpgadataflow"
    assert util.get_by_name(loop_node.attribute, "inputDataType").s.decode("utf-8") == m_input_dt
    assert util.get_by_name(loop_node.attribute, "outputDataType").s.decode("utf-8") == m_output_dt

    # Check tensor shapes by name since loop rolling may reorder inputs
    check_tensor_shape(
        model_wrapper, model_wrapper.graph.input[0].name, [1, input_size]
    )  # activation input shape should remain the same
    # commented because name has changed with the additional transformations applied
    check_tensor_shape(
        model_wrapper, model_wrapper.graph.output[0].name, [1, hidden_size]
    )  # activation output shape should remain the same
    assert (
        model_wrapper.get_tensor_shape(loop_node.input[1])[0] == num_layers
    )  # loop iteration count should match number of layers
    assert (
        model_wrapper.get_tensor_shape(loop_node.input[2])[0] == num_layers
    )  # loop condition count should match number of layers

    loop_body_wrapper = model_wrapper.make_subgraph_modelwrapper(
        util.get_by_name(loop_node.attribute, "body").g
    )

    # nodes are specialized (e.g. MVAU_rtl, Thresholding_rtl, ElementwiseAdd_hls)
    # so match on the op_type prefix
    mlo_nodes = ["MVAU", "Thresholding", "ElementwiseAdd", "ElementwiseMul"]
    seen_prefixes = set()
    for node in loop_body_wrapper.model.graph.node:
        for prefix in mlo_nodes:
            if node.op_type.startswith(prefix):
                seen_prefixes.add(prefix)
        if any(node.op_type.startswith(prefix) for prefix in mlo_nodes):
            mlo_attr = util.get_by_name(node.attribute, "mlo_max_iter")
            assert (
                mlo_attr is not None
            ), f"{node.op_type} node in loop body should have mlo_max_iter attribute"
            assert (
                mlo_attr.i == num_layers
            ), "Loop body max iteration count should match number of layers"
        # MVAU_rtl.adapt_for_loop_body should have switched the streamed-weight
        # MVAU to external_mem so its weights are fetched over AXI-MM per iteration
        if node.op_type.startswith("MVAU"):
            mem_mode_attr = util.get_by_name(node.attribute, "mem_mode")
            assert (
                mem_mode_attr is not None and mem_mode_attr.s.decode("utf-8") == "external_mem"
            ), """MVAU node in loop body should have mem_mode
            'external_mem' set by adapt_for_loop_body"""
        # ElementwiseBinary.adapt_for_loop_body should have switched the style of
        # any per-iteration (streamed) operand from "const" to "input". Verify
        # the observable effect: a "const" operand must be backed by an embedded
        # initializer, while a streamed operand (a body input, no initializer)
        # must have style "input".
        if node.op_type.startswith("Elementwise"):
            for side, style_name in ((0, "lhs_style"), (1, "rhs_style")):
                if side >= len(node.input):
                    continue
                style_attr = util.get_by_name(node.attribute, style_name)
                if style_attr is None:
                    continue
                style = style_attr.s.decode("utf-8")
                has_init = loop_body_wrapper.get_initializer(node.input[side]) is not None
                if style == "const":
                    assert has_init, (
                        f"{node.op_type} {style_name} is 'const' but the operand has no "
                        "initializer; adapt_for_loop_body should have set it to 'input'"
                    )
                elif style == "input":
                    assert not has_init, (
                        f"{node.op_type} {style_name} is 'input' but the operand is an "
                        "embedded initializer"
                    )

    # make sure the adapt_for_loop_body coverage above is not vacuous: the loop
    # body must actually contain the node types we assert on
    assert seen_prefixes == set(mlo_nodes), (
        "Loop body is missing expected MLO node types; "
        f"expected {set(mlo_nodes)}, saw {seen_prefixes}"
    )

    # Numeric verification is skipped here: functional/numeric equivalence is
    # covered by the MLO tests. Executing the specialized model would require
    # cppsim (and therefore a Vivado toolchain), which is not available in the
    # lightweight quicktest CI runners.

    # on success drop the per-test scratch dir, kept on failure for debugging
    robust_rmtree(out_dir)


@pytest.mark.transform
def test_inconsistent_initializer_shape():
    out_dir = Path(make_build_dir(prefix="test_inconsistent_initializer_"))
    # test that if the initializer shape is inconsistent with the value info
    # shape, the transformation fails
    input_size = 20
    hidden_size = 20
    num_layers = 6
    output_size = None

    onnx_path, model = export_model_to_qonnx(
        out_dir, input_size, hidden_size, num_layers, output_size
    )

    qonnx_cleanup(onnx_path, out_file=onnx_path)
    model_wrapper = ModelWrapper(onnx_path)

    # manually change the shape of one of the initializers in the loop body to
    # be inconsistent with the value info
    param0 = model_wrapper.get_initializer("Mul_0_param0")
    model_wrapper.set_initializer("Mul_0_param0", np.append(param0, param0))

    template_path = out_dir / "loop-body-template.onnx"
    loop_extraction = LoopExtraction(
        hierarchy_list=[["", "layers.0"]], loop_body_template_path=template_path
    )
    model_wrapper = model_wrapper.transform(loop_extraction)

    # should throw an error because the initializer shape is inconsistent with the value info shape
    with pytest.raises(
        Exception,
        match=(
            "LoopRolling: all loop-body initializers of the same index must have the " "same shape"
        ),
    ):
        model_wrapper = model_wrapper.transform(LoopRolling(loop_extraction.loop_body_template))

    # on success (the expected raise fired) drop the scratch dir, kept otherwise
    robust_rmtree(out_dir)
