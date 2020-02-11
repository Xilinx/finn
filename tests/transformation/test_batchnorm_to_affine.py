import os
from pkgutil import get_data

import brevitas.onnx as bo
import onnx
import onnx.numpy_helper as nph

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.batchnorm_to_affine import BatchNormToAffine
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
from finn.util.test import get_fc_model_trained

export_onnx_path = "test_output_lfc.onnx"
transformed_onnx_path = "test_output_lfc_transformed.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_lfc_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/LFC_1W1A/checkpoints/best.tar"
)


def test_batchnorm_to_affine():
    lfc = get_fc_model_trained("LFC", 1, 1)
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    new_model = model.transform(BatchNormToAffine())
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_dict = {"0": nph.to_array(input_tensor)}
    assert oxe.compare_execution(model, new_model, input_dict)
    os.remove(export_onnx_path)
