import os
from pkgutil import get_data

import brevitas.onnx as bo
import onnx
import onnx.numpy_helper as nph
import torch
from models.LFC import LFC

import finn.core.onnx_exec as oxe
import finn.transformation.fold_constants as fc
import finn.transformation.infer_shapes as si
import finn.transformation.streamline as sl
from finn.core.modelwrapper import ModelWrapper

export_onnx_path = "test_output_lfc.onnx"
transformed_onnx_path = "test_output_lfc_transformed.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_lfc_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/LFC_1W1A/checkpoints/best.tar"
)


def test_sign_to_thres():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    checkpoint = torch.load(trained_lfc_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform_single(si.infer_shapes)
    model = model.transform_repeated(fc.fold_constants)
    new_model = model.transform_single(sl.convert_sign_to_thres)
    assert new_model.graph.node[3].op_type == "MultiThreshold"
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_dict = {"0": nph.to_array(input_tensor)}
    assert oxe.compare_execution(model, new_model, input_dict)
    os.remove(export_onnx_path)
