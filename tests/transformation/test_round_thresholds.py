import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.streamline import RoundAndClipThresholds


def test_round_thresholds():
    v = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 4])
    thresholds = helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [4, 1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    node_def = helper.make_node(
        "MultiThreshold", ["v", "thresholds"], ["out"], domain="finn"
    )
    graph_def = helper.make_graph([node_def], "test_model", [v, thresholds], [out])
    model_def = helper.make_model(graph_def)
    model = ModelWrapper(model_def)
    threshold_val = np.asarray([[-1.1], [0.7], [2.3], [5.1]], dtype=np.float32)
    model.set_initializer("thresholds", threshold_val)
    model.set_tensor_datatype("v", DataType.INT8)
    inp_dict_f = {"v": np.floor(threshold_val).T}
    inp_dict_n = {"v": np.round(threshold_val).T}
    inp_dict_c = {"v": np.ceil(threshold_val).T}
    orig_f = oxe.execute_onnx(model, inp_dict_f)["out"]
    orig_n = oxe.execute_onnx(model, inp_dict_n)["out"]
    orig_c = oxe.execute_onnx(model, inp_dict_c)["out"]
    assert model.get_tensor_datatype("thresholds") == DataType.FLOAT32
    new_model = model.transform(RoundAndClipThresholds())
    # rounded up thresholds should have same dtype as input
    assert new_model.get_tensor_datatype("thresholds") == DataType.INT8
    new_f = oxe.execute_onnx(new_model, inp_dict_f)["out"]
    new_n = oxe.execute_onnx(new_model, inp_dict_n)["out"]
    new_c = oxe.execute_onnx(new_model, inp_dict_c)["out"]
    assert np.isclose(orig_f, new_f, atol=1e-3).all()
    assert np.isclose(orig_n, new_n, atol=1e-3).all()
    assert np.isclose(orig_c, new_c, atol=1e-3).all()
