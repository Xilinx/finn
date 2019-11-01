from pkgutil import get_data

import numpy as np
import onnx
import onnx.numpy_helper as np_helper

import finn.core.onnx_exec as oxe
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper


def test_infer_shapes():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/mixed-model/mixed_model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform_single(si.infer_shapes)

