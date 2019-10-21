import os
from collections import Counter
from functools import reduce
from operator import mul

import brevitas.onnx as bo
import numpy as np
import torch
from models.common import get_act_quant, get_quant_linear, get_quant_type, get_stats_op
from torch.nn import BatchNorm1d, Dropout, Module, ModuleList

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


def test_modelwrapper():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    checkpoint = torch.load(trained_lfc_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    inp_shape = model.get_tensor_shape("0")
    assert inp_shape == [1, 1, 28, 28]
    l0_weights = model.get_initializer("26")
    assert l0_weights.shape == (784, 1024)
    l0_weights_hist = Counter(l0_weights.flatten())
    assert l0_weights_hist[1.0] == 401311 and l0_weights_hist[-1.0] == 401505
    l0_weights_rand = np.random.randn(784, 1024)
    model.set_initializer("26", l0_weights_rand)
    assert (model.get_initializer("26") == l0_weights_rand).all()
    inp_cons = model.find_consumer("0")
    assert inp_cons.op_type == "Flatten"
    out_prod = model.find_producer("53")
    assert out_prod.op_type == "Mul"
    os.remove(export_onnx_path)
