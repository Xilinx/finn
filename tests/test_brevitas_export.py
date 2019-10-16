import os
import shutil
from functools import reduce
from operator import mul

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
import wget
from models.common import get_act_quant, get_quant_linear, get_quant_type, get_stats_op
from torch.nn import BatchNorm1d, Dropout, Module, ModuleList

import finn.core.onnx_exec as oxe

FC_OUT_FEATURES = [1024, 1024, 1024]
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
LAST_FC_PER_OUT_CH_SCALING = False
IN_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.2

mnist_onnx_url_base = "https://onnxzoo.blob.core.windows.net/models/opset_8/mnist"
mnist_onnx_filename = "mnist.tar.gz"
mnist_onnx_local_dir = "/tmp/mnist_onnx"
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
    model = onnx.load(export_onnx_path)
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
    inits = [x.name for x in model.graph.initializer]
    qnt_annotations = {
        a.tensor_name: a.quant_parameter_tensor_names[0].value
        for a in model.graph.quantization_annotation
    }
    assert qnt_annotations[matmul_node.input[0]] == "BIPOLAR"
    assert matmul_node.input[1] in inits
    assert qnt_annotations[matmul_node.input[1]] == "BIPOLAR"
    init_ind = inits.index(matmul_node.input[1])
    int_weights_pytorch = lfc.features[2].int_weight.transpose(1, 0).detach().numpy()
    int_weights_onnx = nph.to_array(model.graph.initializer[init_ind])
    assert (int_weights_onnx == int_weights_pytorch).all()
    os.remove(export_onnx_path)


def test_brevitas_to_onnx_export_and_exec():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    checkpoint = torch.load(trained_lfc_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = onnx.load(export_onnx_path)
    try:
        os.remove("/tmp/" + mnist_onnx_filename)
    except OSError:
        pass
    dl_ret = wget.download(mnist_onnx_url_base + "/" + mnist_onnx_filename, out="/tmp")
    shutil.unpack_archive(dl_ret, mnist_onnx_local_dir)
    # load one of the test vectors
    input_tensor = onnx.TensorProto()
    output_tensor = onnx.TensorProto()
    with open(mnist_onnx_local_dir + "/mnist/test_data_set_0/input_0.pb", "rb") as f:
        input_tensor.ParseFromString(f.read())
    with open(mnist_onnx_local_dir + "/mnist/test_data_set_0/output_0.pb", "rb") as f:
        output_tensor.ParseFromString(f.read())
    # run using FINN-based execution
    input_dict = {"0": nph.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    assert np.isclose(nph.to_array(output_tensor), output_dict["53"], atol=1e-3).all()
    # remove the downloaded model and extracted files
    os.remove(dl_ret)
    shutil.rmtree(mnist_onnx_local_dir)
    os.remove(export_onnx_path)
