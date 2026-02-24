import pytest

import brevitas.onnx as bo
import numpy as np
import onnx
import os
import qonnx.util.basic as util
import torch
from brevitas.nn import QuantLinear
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
from finn.transformation.fpgadataflow.raise_scalar_to_rank1 import RaiseScalarToRank1
from finn.transformation.fpgadataflow.set_loop_boundary import SetLoopBoundary
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
def export_model_to_qonnx(input_size=10, hidden_size=20, num_layers=4, output_size=None):
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

    # Export the model to ONNX format
    onnx_path = os.environ["FINN_BUILD_DIR"] + f"/simple_module_{num_layers}layers.onnx"
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


# input_size == hidden_size to create model that can be rolled
@pytest.mark.parametrize("input_size", [20, 30, 40])
# num_layers
@pytest.mark.parametrize("num_layers", [6, 12, 24])
@pytest.mark.transform
def test_finn_loop(input_size, num_layers):
    hidden_size = input_size

    onnx_path, model = export_model_to_qonnx(input_size, hidden_size, num_layers)

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

    model_wrapper = model_wrapper.transform(GiveUniqueNodeNames())
    # temporarily set loop boundaries manually
    node_metadata = {
        "pkg.torch.onnx.name_scopes": "['', 'layers.0']",
        "pkg.torch.onnx.class_hierarchy": "['TestModule', 'test']",
    }
    node_range = (model_wrapper.graph.node[0], model_wrapper.graph.node[3])
    model_wrapper = model_wrapper.transform(SetLoopBoundary(node_metadata, node_range=node_range))

    # Loop extraction
    loop_extraction = LoopExtraction(hierarchy_list=[["", "layers.0"]])
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

    mlo_nodes = ["MVAU", "Thresholding", "ElementwiseAdd", "ElementwiseMul"]
    for node in loop_body_wrapper.model.graph.node:
        if node.op_type in mlo_nodes:
            mlo_attr = util.get_by_name(node.attribute, "mlo_max_iter")
            assert (
                mlo_attr is not None
            ), f"{node.op_type} node in loop body should have mlo_max_iter attribute"
            assert (
                mlo_attr.i == num_layers
            ), "Loop body max iteration count should match number of layers"

    inp_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], [1, input_size])
    idict = {model_wrapper.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model_wrapper, idict)
    produced = odict[model_wrapper.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    expected = model.forward(inp_tensor).detach().numpy()

    max_diff = np.max(np.abs(produced - expected))
    print(f"Max difference between produced and expected: {max_diff}")

    # compare results within a tolerance
    rtol = 1e-4
    atol = 1e-3
    assert np.allclose(
        produced, expected, rtol=rtol, atol=atol
    ), "Results do not match within tolerance!"

    # when run successfully, delete temp onnx files
    os.remove(onnx_path)


@pytest.mark.transform
def test_inconsistent_initializer_shape():
    # test that if the initializer shape is inconsistent with the value info
    # shape, the transformation fails
    input_size = 20
    hidden_size = 20
    num_layers = 6
    output_size = None

    onnx_path, model = export_model_to_qonnx(input_size, hidden_size, num_layers, output_size)

    qonnx_cleanup(onnx_path, out_file=onnx_path)
    model_wrapper = ModelWrapper(onnx_path)

    # manually change the shape of one of the initializers in the loop body to
    # be inconsistent with the value info
    param0 = model_wrapper.get_initializer("Mul_0_param0")
    model_wrapper.set_initializer("Mul_0_param0", np.append(param0, param0))

    loop_extraction = LoopExtraction(hierarchy_list=[["", "layers.0"]])
    model_wrapper = model_wrapper.transform(loop_extraction)

    # should throw an error because the initializer shape is inconsistent with the value info shape
    with pytest.raises(
        Exception,
        match=(
            "LoopRolling: all loop-body initializers of the same index must have the " "same shape"
        ),
    ):
        model_wrapper = model_wrapper.transform(LoopRolling(loop_extraction.loop_body_template))
