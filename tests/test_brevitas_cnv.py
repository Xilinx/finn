import os
import pkg_resources as pk

import brevitas.onnx as bo
import numpy as np
import torch
from models.CNV import CNV

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes

export_onnx_path = "test_output_cnv.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_cnv_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/CNV_1W1A/checkpoints/best.tar"
)


def test_brevitas_cnv_w1a1_export():
    cnv = CNV(weight_bit_width=1, act_bit_width=1, in_bit_width=1, in_ch=3).eval()
    bo.export_finn_onnx(cnv, (1, 3, 32, 32), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    assert model.graph.node[2].op_type == "Sign"
    assert model.graph.node[3].op_type == "Conv"
    conv0_wname = model.graph.node[3].input[1]
    assert list(model.get_initializer(conv0_wname).shape) == [64, 3, 3, 3]
    assert model.graph.node[4].op_type == "Mul"
    os.remove(export_onnx_path)


def test_brevitas_cnv_w1a1_export_exec():
    cnv = CNV(weight_bit_width=1, act_bit_width=1, in_bit_width=1, in_ch=3).eval()
    checkpoint = torch.load(trained_cnv_checkpoint, map_location="cpu")
    cnv.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(cnv, (1, 3, 32, 32), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
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


def test_brevitas_trained_cnv_w1a1_pytorch():
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
