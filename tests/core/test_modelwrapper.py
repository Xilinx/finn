import os
from collections import Counter

import brevitas.onnx as bo
import numpy as np

from finn.core.modelwrapper import ModelWrapper
from finn.util.test import get_test_model_trained

export_onnx_path = "test_output_lfc.onnx"


def test_modelwrapper():
    lfc = get_test_model_trained("LFC", 1, 1)
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    assert model.check_all_tensor_shapes_specified() is False
    inp_shape = model.get_tensor_shape("0")
    assert inp_shape == [1, 1, 28, 28]
    l0_mat_tensor_name = "33"
    l0_weights = model.get_initializer(l0_mat_tensor_name)
    assert l0_weights.shape == (784, 1024)
    l0_weights_hist = Counter(l0_weights.flatten())
    assert l0_weights_hist[1.0] == 401311 and l0_weights_hist[-1.0] == 401505
    l0_weights_rand = np.random.randn(784, 1024)
    model.set_initializer(l0_mat_tensor_name, l0_weights_rand)
    assert (model.get_initializer(l0_mat_tensor_name) == l0_weights_rand).all()
    l0_inp_tensor_name = "32"
    inp_cons = model.find_consumer(l0_inp_tensor_name)
    assert inp_cons.op_type == "MatMul"
    out_prod = model.find_producer(l0_inp_tensor_name)
    assert out_prod.op_type == "Sign"
    os.remove(export_onnx_path)
