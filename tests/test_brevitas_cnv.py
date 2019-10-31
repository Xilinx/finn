import os
import pkg_resources as pk

import brevitas.onnx as bo
import numpy as np
import torch
from models.common import (
    get_act_quant,
    get_quant_conv2d,
    get_quant_linear,
    get_quant_type,
    get_stats_op
)
from torch.nn import BatchNorm1d, BatchNorm2d, MaxPool2d, Module, ModuleList, Sequential

import finn.core.onnx_exec as oxe
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper

# QuantConv2d configuration
CNV_OUT_CH_POOL = [
    (0, 64, False),
    (1, 64, True),
    (2, 128, False),
    (3, 128, True),
    (4, 256, False),
    (5, 256, False),
]

# Intermediate QuantLinear configuration
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]

# Last QuantLinear configuration
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False

# MaxPool2d configuration
POOL_SIZE = 2

export_onnx_path = "test_output_cnv.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_cnv_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/CNV_1W1A/checkpoints/best.tar"
)


class CNV(Module):
    def __init__(
        self,
        num_classes=10,
        weight_bit_width=None,
        act_bit_width=None,
        in_bit_width=None,
        in_ch=3,
    ):
        super(CNV, self).__init__()

        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        stats_op = get_stats_op(weight_quant_type)

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()
        self.conv_features.append(get_act_quant(in_bit_width, in_quant_type))

        for i, out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(
                get_quant_conv2d(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    bit_width=weight_bit_width,
                    quant_type=weight_quant_type,
                    stats_op=stats_op,
                )
            )
            in_ch = out_ch
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))
            if i == 5:
                self.conv_features.append(Sequential())
            self.conv_features.append(BatchNorm2d(in_ch))
            self.conv_features.append(get_act_quant(act_bit_width, act_quant_type))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                get_quant_linear(
                    in_features=in_features,
                    out_features=out_features,
                    per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                    bit_width=weight_bit_width,
                    quant_type=weight_quant_type,
                    stats_op=stats_op,
                )
            )
            self.linear_features.append(BatchNorm1d(out_features))
            self.linear_features.append(get_act_quant(act_bit_width, act_quant_type))
        self.fc = get_quant_linear(
            in_features=LAST_FC_IN_FEATURES,
            out_features=num_classes,
            per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
            bit_width=weight_bit_width,
            quant_type=weight_quant_type,
            stats_op=stats_op,
        )

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0])
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(1, 256)
        for mod in self.linear_features:
            x = mod(x)
        out = self.fc(x)
        return out


def test_brevitas_trained_cnv_pytorch():
    # load pretrained weights into CNV-w1a1
    cnv = CNV(weight_bit_width=1, act_bit_width=1, in_bit_width=1, in_ch=3).eval()
    checkpoint = torch.load(trained_cnv_checkpoint, map_location="cpu")
    cnv.load_state_dict(checkpoint["state_dict"])
    fn = pk.resource_filename("finn", "data/cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"]
    input_tensor = torch.from_numpy(input_tensor).float()
    assert input_tensor.shape == (1, 3, 32, 32)
    # do forward pass in PyTorch/Brevitas
    cnv.forward(input_tensor).detach().numpy()
    # TODO verify produced answer


def test_brevitas_cnv_export():
    cnv = CNV(weight_bit_width=1, act_bit_width=1, in_bit_width=1, in_ch=3).eval()
    bo.export_finn_onnx(cnv, (1, 3, 32, 32), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    assert model.graph.node[2].op_type == "Sign"
    assert model.graph.node[3].op_type == "Conv"
    conv0_wname = model.graph.node[3].input[1]
    assert list(model.get_initializer(conv0_wname).shape) == [64, 3, 3, 3]
    assert model.graph.node[4].op_type == "Mul"


def test_brevitas_cnv_export_exec():
    cnv = CNV(weight_bit_width=1, act_bit_width=1, in_bit_width=1, in_ch=3).eval()
    checkpoint = torch.load(trained_cnv_checkpoint, map_location="cpu")
    cnv.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(cnv, (1, 3, 32, 32), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform_single(si.infer_shapes)
    model.save(export_onnx_path)
    fn = pk.resource_filename("finn", "data/cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    assert input_tensor.shape == (1, 3, 32, 32)
    # run using FINN-based execution
    input_dict = {"0": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # do forward pass in PyTorch/Brevitas
    input_tensor = torch.from_numpy(input_tensor).float()
    expected = cnv.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
    os.remove(export_onnx_path)
