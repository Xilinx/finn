import os
from pkgutil import get_data

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
from models.LFC import LFC

import finn.core.onnx_exec as oxe
import finn.transformation.fold_constants as fc
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper

export_onnx_path = "test_output_lfc.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_lfc_w1a1_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/LFC_1W1A/checkpoints/best.tar"
)


def test_brevitas_trained_lfc_w1a1_pytorch():
    # load pretrained weights into LFC-w1a1
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1).eval()
    checkpoint = torch.load(trained_lfc_w1a1_checkpoint, map_location="cpu")
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


def test_brevitas_to_onnx_export_and_exec_lfc_w1a1():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    checkpoint = torch.load(trained_lfc_w1a1_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform_single(si.infer_shapes)
    model = model.transform_repeated(fc.fold_constants)
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
