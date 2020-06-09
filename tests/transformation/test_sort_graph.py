from onnx import TensorProto, helper
import numpy as np

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import SortGraph
from finn.transformation.infer_shapes import InferShapes
import pytest
import finn.analysis.topology as ta


def make_randomly_sorted_linear_model(num_of_nodes, seed=None):
    if seed is not None:
        np.random.seed(seed)

    ch = 2
    ifmdim = 16
    input_shape = (1, ch, ifmdim, ifmdim)

    top_in = helper.make_tensor_value_info("t0", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info(
        "t" + str(num_of_nodes), TensorProto.FLOAT, input_shape
    )

    value_info = []
    nodes = []
    for i in range(num_of_nodes):
        nodes += [
            helper.make_node("Add", ["t" + str(i), "p" + str(i)], ["t" + str(i + 1)])
        ]
        value_info += [
            helper.make_tensor_value_info("p" + str(i), TensorProto.FLOAT, input_shape)
        ]

    nodes = np.random.permutation(nodes)

    modelproto = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=nodes,
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())

    for i in range(num_of_nodes):
        model.set_initializer(
            "p" + str(i), np.random.rand(*input_shape).astype(np.float32)
        )

    return model


@pytest.mark.parametrize("num_of_nodes", [64])
def test_sort_linear_graph(num_of_nodes):
    model = make_randomly_sorted_linear_model(num_of_nodes, seed=0)
    new_model = model.transform(SortGraph())

    # Test
    ret = new_model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"], "Nodes are not topologically sorted."


def test_sort_nonlinear_graph():
    ch = 2
    ifmdim = 16
    input_shape = (1, ch, ifmdim, ifmdim)

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, input_shape)

    num_of_params = 8
    value_info = []
    for i in range(num_of_params):
        value_info += [
            helper.make_tensor_value_info("p" + str(i), TensorProto.FLOAT, input_shape)
        ]

    modelproto = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                # Not sorted nodes
                helper.make_node("Mul", ["fork1", "p2"], ["t3"]),
                helper.make_node("Add", ["t4", "p3"], ["t5"]),
                helper.make_node("Add", ["t2", "t3"], ["t4"]),
                helper.make_node("Add", ["t6", "t7"], ["t8"]),
                helper.make_node("Add", ["fork3", "fork3"], ["top_out"]),
                helper.make_node("Mul", ["t5", "p4"], ["fork2"]),
                helper.make_node("Add", ["top_in", "p0"], ["fork1"]),
                helper.make_node("Mul", ["fork1", "p1"], ["t2"]),
                helper.make_node("Add", ["fork2", "p5"], ["t6"]),
                helper.make_node("Add", ["fork2", "p6"], ["t7"]),
                helper.make_node("Mul", ["t8", "p7"], ["fork3"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())

    np.random.seed(0)
    for i in range(num_of_params):
        model.set_initializer(
            "p" + str(i), np.random.rand(*input_shape).astype(np.float32)
        )

    new_model = model.transform(SortGraph())

    # Test
    ret = new_model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"], "Nodes are not topologically sorted."


if __name__ == "__main__":
    import time

    sizes = [10, 50, 100, 500, 1000]
    times = []
    reps = 10

    print("SortGraph performance test:")
    print("Test sizes", sizes)
    print("Repetitions per size:", reps)
    for sz in sizes:
        acc_time = 0
        print(" Testing size ", sz)
        for i in range(reps):
            # it should take the same time even with the sorted one
            # but better new model each time as it is a more general approach
            model = make_randomly_sorted_linear_model(sz)  # new model as seed is None
            bef = time.time()
            new_model = model.transform(SortGraph(), make_deepcopy=False)
            acc_time += time.time() - bef

        times += [acc_time / reps]

    # print csv
    print("\nnum_of_nodes,  seconds")
    for sz, tm in zip(sizes, times):
        print("{:12d}, {:6.4e}".format(sz, tm))

    # plot
    # import matplotlib.pyplot as plt
    # plt.plot(sizes,times,"--o")
    # plt.grid(True)
