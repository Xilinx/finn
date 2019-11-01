import os
from functools import reduce
from operator import mul
from pkgutil import get_data

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
from models.common import get_act_quant, get_quant_linear, get_quant_type, get_stats_op
from torch.nn import BatchNorm1d, Dropout, Module, ModuleList

import finn.core.onnx_exec as oxe
import finn.transformation.infer_shapes as si
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper

FC_OUT_FEATURES = [1024, 1024, 1024]
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
LAST_FC_PER_OUT_CH_SCALING = False
IN_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.2

export_onnx_path = "test_output_lfc.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_lfc_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/LFC_1W1A/checkpoints/best.tar"
)


class LFC(Module):
    def __init__(
        self,
        num_classes=10,
        weight_bit_width=None,
        act_bit_width=None,
        in_bit_width=None,
        in_ch=1,
        in_features=(28, 28),
    ):
        super(LFC, self).__init__()

        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        stats_op = get_stats_op(weight_quant_type)

        self.features = ModuleList()
        self.features.append(get_act_quant(in_bit_width, in_quant_type))
        self.features.append(Dropout(p=IN_DROPOUT))
        in_features = reduce(mul, in_features)
        for out_features in FC_OUT_FEATURES:
            self.features.append(
                get_quant_linear(
                    in_features=in_features,
                    out_features=out_features,
                    per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                    bit_width=weight_bit_width,
                    quant_type=weight_quant_type,
                    stats_op=stats_op,
                )
            )
            in_features = out_features
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(get_act_quant(act_bit_width, act_quant_type))
            self.features.append(Dropout(p=HIDDEN_DROPOUT))
        self.fc = get_quant_linear(
            in_features=in_features,
            out_features=num_classes,
            per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
            bit_width=weight_bit_width,
            quant_type=weight_quant_type,
            stats_op=stats_op,
        )

    def forward(self, x):
        x = x.view(1, 784)
        # removing the torch.tensor here creates a float64 op for some reason..
        # so explicitly wrapped with torch.tensor to make a float32 one instead
        x = 2.0 * x - torch.tensor([1.0])
        for mod in self.features:
            x = mod(x)
        out = self.fc(x)
        return out


def test_brevitas_to_onnx_export():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    # TODO the following way of testing is highly sensitive to small changes
    # in PyTorch ONNX export: the order, names, count... of nodes could
    # easily change between different versions, and break this test.
    assert len(model.graph.input) == 23
    assert len(model.graph.node) == 24
    assert len(model.graph.output) == 1
    assert model.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 10
    act_node = model.graph.node[3]
    assert act_node.op_type == "Sign"
    matmul_node = model.graph.node[4]
    assert matmul_node.op_type == "MatMul"
    assert act_node.output[0] == matmul_node.input[0]
    assert model.get_tensor_datatype(matmul_node.input[0]) == DataType.BIPOLAR
    W = model.get_initializer(matmul_node.input[1])
    assert W is not None
    assert model.get_tensor_datatype(matmul_node.input[1]) == DataType.BIPOLAR
    int_weights_pytorch = lfc.features[2].int_weight.transpose(1, 0).detach().numpy()
    assert (W == int_weights_pytorch).all()
    os.remove(export_onnx_path)


def test_brevitas_trained_lfc_pytorch():
    # load pretrained weights into LFC-w1a1
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1).eval()
    checkpoint = torch.load(trained_lfc_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_tensor = torch.from_numpy(nph.to_array(input_tensor)).float()
    assert input_tensor.shape == (1, 1, 28, 28)
    # do forward pass in PyTorch/Brevitas
    produced = lfc.forward(input_tensor).detach().numpy()
    expected = [
        [
            3.3253,
            -2.5652,
            9.2157,
            -1.4251,
            1.4251,
            -3.3728,
            0.2850,
            -0.5700,
            7.0781,
            -1.2826,
        ]
    ]
    assert np.isclose(produced, expected, atol=1e-4).all()


def test_brevitas_to_onnx_export_and_exec():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    checkpoint = torch.load(trained_lfc_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform_single(si.infer_shapes)
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {"0": nph.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # run using PyTorch/Brevitas
    input_tensor = torch.from_numpy(nph.to_array(input_tensor)).float()
    assert input_tensor.shape == (1, 1, 28, 28)
    # do forward pass in PyTorch/Brevitas
    expected = lfc.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
    # remove the downloaded model and extracted files
    os.remove(export_onnx_path)
