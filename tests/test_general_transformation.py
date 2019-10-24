from pkgutil import get_data

import finn.transformation.general as tg
from finn.core.modelwrapper import ModelWrapper


def test_give_unique_node_names():
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform_single(tg.give_unique_node_names)
    assert model.graph.node[0].name == "Reshape_0"
    assert model.graph.node[1].name == "Conv_1"
    assert model.graph.node[11].name == "Add_11"
