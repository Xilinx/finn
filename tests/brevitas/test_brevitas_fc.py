from pkgutil import get_data

import pytest

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
from models.LFC import LFC
from models.SFC import SFC
from models.TFC import TFC

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import make_build_dir

export_onnx_path = make_build_dir("test_brevitas_fc_")


# TODO get from config instead, hardcoded to Docker path for now
def get_trained_fc_checkpoint(size, weight_bit_width, act_bit_width):
    nname = "%s_%dW%dA" % (size, weight_bit_width, act_bit_width)
    root = "/workspace/brevitas_cnv_lfc/pretrained_models/%s/checkpoints/best.tar"
    return root % nname


def get_fc_model_def(size):
    model_def_map = {"LFC": LFC, "SFC": SFC, "TFC": TFC}
    return model_def_map[size]


# activation: None or DataType
@pytest.mark.parametrize("size", ["TFC", "SFC", "LFC"])
# weight bits
@pytest.mark.parametrize("wbits", [1])
# act bits
@pytest.mark.parametrize("abits", [1, 2])
def test_brevitas_fc_onnx_export_and_exec(size, wbits, abits):
    nname = "%s_%dW%dA" % (size, wbits, abits)
    finn_onnx = export_onnx_path + "/%s.onnx" % nname
    checkpoint_loc = get_trained_fc_checkpoint(size, wbits, abits)
    model_def = get_fc_model_def(size)
    fc = model_def(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=abits)
    checkpoint = torch.load(checkpoint_loc, map_location="cpu")
    fc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(fc, (1, 1, 28, 28), finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
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
    expected = fc.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
