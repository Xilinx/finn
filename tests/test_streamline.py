import os
from pkgutil import get_data

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
from models.LFC import LFC

import finn.core.onnx_exec as oxe
import finn.transformation.streamline as sl
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.batchnorm_to_affine import BatchNormToAffine
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import (
    ConvertSubToAdd,
    GiveReadableTensorNames,
    GiveUniqueNodeNames
)
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes

export_onnx_path = "test_output_lfc.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_lfc_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/LFC_1W1A/checkpoints/best.tar"
)


def test_streamline_lfc_w1a1():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    checkpoint = torch.load(trained_lfc_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {"global_in": nph.to_array(input_tensor)}
    expected_ctx = oxe.execute_onnx(model, input_dict, True)
    expected = expected_ctx[model.graph.output[0].name]
    transforms = [
        ConvertSubToAdd(),
        BatchNormToAffine(),
        sl.ConvertSignToThres(),
        sl.MoveScalarAddPastMatMul(),
        sl.MoveScalarMulPastMatMul(),
        sl.MoveAddPastMul(),
        sl.CollapseRepeatedAdd(),
        sl.CollapseRepeatedMul(),
        sl.AbsorbAddIntoMultiThreshold(),
        sl.FactorOutMulSignMagnitude(),
        sl.AbsorbMulIntoMultiThreshold(),
        sl.Absorb1BitMulIntoMatMul(),
        sl.RoundThresholds(),
    ]
    trn_ind = 0
    for trn in transforms:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        produced_ctx = oxe.execute_onnx(model, input_dict, True)
        produced = produced_ctx[model.graph.output[0].name]
        # model.save("%d-%s.onnx" % (trn_ind, trn.__name__))
        assert np.isclose(expected, produced, atol=1e-3).all()
        trn_ind += 1
    os.remove(export_onnx_path)
