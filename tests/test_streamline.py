import os
from pkgutil import get_data

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
from models.LFC import LFC

import finn.core.onnx_exec as oxe
import finn.transformation.batchnorm_to_affine as ba
import finn.transformation.fold_constants as fc
import finn.transformation.general as tg
import finn.transformation.infer_datatypes as di
import finn.transformation.infer_shapes as si
import finn.transformation.streamline as sl
from finn.core.modelwrapper import ModelWrapper

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
    model = model.transform_single(si.infer_shapes)
    model = model.transform_repeated(fc.fold_constants)
    model = model.transform_single(tg.give_unique_node_names)
    model = model.transform_single(tg.give_readable_tensor_names)
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {"global_in": nph.to_array(input_tensor)}
    expected_ctx = oxe.execute_onnx(model, input_dict, True)
    expected = expected_ctx[model.graph.output[0].name]
    transforms = [
        tg.convert_sub_to_add,
        ba.batchnorm_to_affine,
        sl.convert_sign_to_thres,
        sl.move_scalar_add_past_matmul,
        sl.move_scalar_mul_past_matmul,
        sl.move_add_past_mul,
        sl.collapse_repeated_add,
        sl.collapse_repeated_mul,
        sl.absorb_add_into_multi_threshold,
        sl.factor_out_mul_sign_magnitude,
        sl.absorb_mul_into_multi_threshold,
        sl.absorb_1bit_mul_into_matmul,
    ]
    trn_ind = 0
    for trn in transforms:
        model = model.transform_repeated(trn)
        model = model.transform_single(tg.give_unique_node_names)
        model = model.transform_single(tg.give_readable_tensor_names)
        model = model.transform_repeated(di.infer_datatypes)
        produced_ctx = oxe.execute_onnx(model, input_dict, True)
        produced = produced_ctx[model.graph.output[0].name]
        # model.save("%d-%s.onnx" % (trn_ind, trn.__name__))
        assert np.isclose(expected, produced, atol=1e-3).all()
        trn_ind += 1
    os.remove(export_onnx_path)
