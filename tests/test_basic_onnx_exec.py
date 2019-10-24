from pkgutil import get_data

import numpy as np
import onnx
import onnx.numpy_helper as np_helper

import finn.core.onnx_exec as oxe
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper


def test_mnist_onnx_download_extract_run():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform_single(si.infer_shapes)
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    raw_o = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/output_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    output_tensor = onnx.load_tensor_from_string(raw_o)
    # run using FINN-based execution
    input_dict = {"Input3": np_helper.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    assert np.isclose(
        np_helper.to_array(output_tensor), output_dict["Plus214_Output_0"], atol=1e-3
    ).all()
