import os
from functools import reduce
from operator import mul

import brevitas.onnx as bo
import onnx
import onnx.numpy_helper as nph
import torch
import torch.onnx
from models.common import get_act_quant, get_quant_linear, get_quant_type, get_stats_op
from torch.nn import BatchNorm1d, Dropout, Module, ModuleList


def test_brevitas_to_onnx_export():
    FC_OUT_FEATURES = [1024, 1024, 1024]
    INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
    LAST_FC_PER_OUT_CH_SCALING = False
    IN_DROPOUT = 0.2
    HIDDEN_DROPOUT = 0.2

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
            x = 2.0 * x - 1.0
            x = x.view(x.shape[0], -1)
            for mod in self.features:
                x = mod(x)
            out = self.fc(x)
            return out

    export_onnx_path = "test_output_lfc.onnx"
    with torch.no_grad():
        lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
        bo.prepare_for_onnx_export(lfc, True)
        torch.onnx.export(
            lfc, torch.empty(784, dtype=torch.float), export_onnx_path, verbose=True
        )
        model = onnx.load(export_onnx_path)
        # TODO the following way of testing is highly sensitive to small changes
        # in PyTorch ONNX export: the order, names, count... of nodes could
        # easily change between different versions, and break this test.
        assert len(model.graph.input) == 32
        assert len(model.graph.node) == 33
        assert len(model.graph.output) == 1
        assert model.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 10
        assert model.graph.node[12].op_type == "QuantizedHardTanh"
        assert model.graph.node[13].op_type == "Constant"
        assert model.graph.node[14].op_type == "MatMul"
        assert model.graph.node[12].output[0] == model.graph.node[14].input[1]
        assert model.graph.node[13].output[0] == model.graph.node[14].input[0]
        int_weights_pytorch = lfc.features[2].int_weight.detach().numpy()
        int_weights_onnx = nph.to_array(model.graph.node[13].attribute[0].t)
        assert (int_weights_onnx == int_weights_pytorch).all()
        assert model.graph.node[12].attribute[0].name == "activation_qnt"
        assert model.graph.node[12].attribute[0].s.decode("utf-8") == "1"
        assert model.graph.node[14].attribute[1].name == "weight_qnt"
        assert model.graph.node[14].attribute[1].s.decode("utf-8") == "1"
        os.remove(export_onnx_path)
