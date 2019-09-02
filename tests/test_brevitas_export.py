from functools import reduce
from operator import mul

import onnx
import torch
import torch.onnx
from models.common import get_act_quant, get_quant_linear, get_quant_type, get_stats_op
from torch import nn
from torch.autograd import Function
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

    class objdict(dict):
        def __getattr__(self, name):
            if name in self:
                return self[name]
            else:
                raise AttributeError("No such attribute: " + name)

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            if name in self:
                del self[name]
            else:
                raise AttributeError("No such attribute: " + name)

    # TODO: <all this needs to mvoe into Brevitas>
    quantization_annotation = dict()

    class QuantizedLinearPlaceholderFunction(Function):
        @staticmethod
        def symbolic(g, W, x, bw, out_features):
            # import pdb; pdb.set_trace()
            quantization_annotation[W.uniqueName()] = str(bw)
            return g.op("MatMul", W, x, domain_s="finn")

        @staticmethod
        def forward(ctx, W, x, bw, out_features):
            return torch.empty(1, out_features, dtype=torch.float)

    class QuantizedLinearPlaceholder(nn.Module):
        def __init__(self, quantized_linear):
            super(QuantizedLinearPlaceholder, self).__init__()
            self.in_features = quantized_linear.in_features
            self.out_features = quantized_linear.out_features
            # compute the quantized weights
            W, s, bitwidth = quantized_linear.weight_quant(quantized_linear.weight)
            W = W.detach().numpy().reshape(self.out_features, self.in_features)
            s = s.detach().numpy()
            s = s.reshape(s.size, 1)
            W = W / s
            self.W = torch.from_numpy(W)
            self.bitwidth = bitwidth.item()

        def forward(self, x):
            # return linear(self.W, x)
            return QuantizedLinearPlaceholderFunction.apply(
                self.W, x, self.bitwidth, self.out_features
            )

    class QuantizedHardTanhPlaceholderFunction(Function):
        @staticmethod
        def symbolic(g, input):
            ret = g.op("QuantizedHardTanh", input, domain_s="finn")
            # insert quantization annotation for the resulting tensor, TODO fix bitwidth
            quantization_annotation[ret.uniqueName()] = "1"
            return ret

        @staticmethod
        def forward(ctx, input):
            return input.clamp(0)

    class QuantizedHardTanhPlaceholder(nn.Module):
        def __init__(self):
            super(QuantizedHardTanhPlaceholder, self).__init__()

        def forward(self, x):
            return QuantizedHardTanhPlaceholderFunction.apply(x)

    # TODO: </all this needs to mvoe into Brevitas>
    export_onnx_path = "test_output_lfc.onnx"
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    for i in range(len(lfc.features)):
        L = lfc.features[i]
        if type(L).__name__ == "QuantLinear":
            lfc.features[i] = QuantizedLinearPlaceholder(L)
        elif type(L).__name__ == "QuantHardTanh":
            lfc.features[i] = QuantizedHardTanhPlaceholder()
    lfc.fc = QuantizedLinearPlaceholder(lfc.fc)
    torch.onnx.export(lfc, torch.empty(784, dtype=torch.float), export_onnx_path)
    model = onnx.load(export_onnx_path)
    assert len(model.graph.input) == 16
    assert len(model.graph.node) == 33
    assert len(model.graph.output) == 1
    assert model.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 10
    assert model.graph.node[13].op_type == "Constant"
    assert model.graph.node[12].op_type == "QuantizedHardTanh"
