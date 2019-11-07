import os
from pkgutil import get_data

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as np_helper
from models.LFC import LFC

import finn.core.onnx_exec as oxe
import finn.transformation.fold_constants as fc
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper

export_onnx_path = "test_output_lfc.onnx"


def test_const_folding():
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform_single(si.infer_shapes)
    model = model.transform_single(fc.fold_constants)
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    raw_o = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/output_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    output_tensor = onnx.load_tensor_from_string(raw_o)
    input_dict = {"Input3": np_helper.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    assert np.isclose(
        np_helper.to_array(output_tensor), output_dict["Plus214_Output_0"], atol=1e-3
    ).all()


def test_const_folding_shapes():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform_single(si.infer_shapes)
    model = model.transform_repeated(fc.fold_constants)
    assert model.graph.node[0].op_type == "Reshape"
    assert list(model.get_tensor_shape("0")) == [1, 1, 28, 28]
    assert list(model.get_tensor_shape("27")) == [1, 784]
    os.remove(export_onnx_path)
