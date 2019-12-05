from pkgutil import get_data

import numpy as np
import onnx
import onnx.numpy_helper as np_helper

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes


def test_renaming():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # do some basic checks
    assert model.graph.input[0].name == "global_in"
    assert model.graph.output[0].name == "global_out"
    assert model.graph.node[1].op_type == "Conv"
    assert model.graph.node[1].name == "Conv_0"
    assert model.graph.node[1].input[1] == "Conv_0_param0"
    assert model.graph.node[6].op_type == "Add"
    assert model.graph.node[6].name == "Add_1"
    assert model.graph.node[6].input[1] == "Add_1_param0"
    # ensure running renaming twice still yields the same names
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    assert model.graph.node[1].op_type == "Conv"
    assert model.graph.node[1].name == "Conv_0"
    assert model.graph.node[1].input[1] == "Conv_0_param0"
    assert model.graph.node[6].op_type == "Add"
    assert model.graph.node[6].name == "Add_1"
    assert model.graph.node[6].input[1] == "Add_1_param0"
    # run renamed model to make sure we did not mess up the topology
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    raw_o = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/output_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    output_tensor = onnx.load_tensor_from_string(raw_o)
    input_dict = {"global_in": np_helper.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    assert np.isclose(
        np_helper.to_array(output_tensor), output_dict["global_out"], atol=1e-3
    ).all()
