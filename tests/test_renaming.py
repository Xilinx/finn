from pkgutil import get_data

import finn.transformation.general as tg
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper


def test_renaming():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform_single(si.infer_shapes)
    model = model.transform_single(tg.give_unique_node_names)
    model = model.transform_single(tg.give_readable_tensor_names)
    assert model.graph.node[1].op_type == "Conv"
    assert model.graph.node[1].name == "Conv_1"
    assert model.graph.node[1].input[1] == "Conv_1_param0"
    assert model.graph.node[6].op_type == "Add"
    assert model.graph.node[6].name == "Add_6"
    assert model.graph.node[6].input[1] == "Add_6_param0"
