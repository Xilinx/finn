from pkgutil import get_data

import numpy as np
import onnx
import onnx.numpy_helper as np_helper
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper


def test_infer_shapes():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    graph = model.graph
    node_ind = 0
    node_dict = {}
    for n in graph.node:
        node_ind += 1
        node_dict[node_ind] = n

    # multi-thresholding node to be inserted between the first Relu and MaxPool node

    # get Relu node to use data to make a new Relu node and delete the old one
    Relu_node = node_dict[4]
    assert Relu_node.op_type == "Relu", "The wrong model was chosen for the check"
    graph.node.remove(Relu_node)

    # create new tensors (thresholds as constant) and add them to the graph info
    mt_v0 = helper.make_tensor_value_info("mt_v0", TensorProto.FLOAT, [1, 8, 28, 28])
    mt_thresh0 = helper.make_tensor_value_info("mt_thresh0", TensorProto.FLOAT, [8, 7])

    graph.value_info.append(mt_v0)
    graph.value_info.append(mt_thresh0)

    # random numbers for the thresholds
    # thresholds for one channel have to be sorted to guarantee the correct behavior
    mt_thresh0_values = np.empty([8, 7], dtype=np.float32)
    for i in range(len(mt_thresh0_values)):
        mt_thresh0_values[i] = np.sort(np.random.random_sample(7,) * 10)

    model.set_initializer(mt_thresh0.name, mt_thresh0_values)

    # create and insert new Relu node and one multi-thresholding node
    new_Relu_node = helper.make_node("Relu", [Relu_node.input[0]], ["mt_v0"])
    mt_node = helper.make_node(
        "MultiThreshold", ["mt_v0", "mt_thresh0"], [Relu_node.output[0]], domain="finn"
    )

    graph.node.insert(4, new_Relu_node)
    graph.node.insert(5, mt_node)

    # test shape inference on mixed model
    model = model.transform_single(si.infer_shapes)

    # execution with input values from mnist-conv model
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)

    # run using FINN-based execution
    input_dict = {"Input3": np_helper.to_array(input_tensor)}
    oxe.execute_onnx(model, input_dict)
