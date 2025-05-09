import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds


def generate_random_threshold_values(data_type, num_input_channels, num_steps):
    return np.random.randint(
        data_type.min(),
        data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)


def make_loop_modelwrapper(mw, mh, iter_count):
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, 3, 3, mw])
    mm0_out = helper.make_tensor_value_info("mm0_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mt0_out = helper.make_tensor_value_info("mt0_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mm1_out = helper.make_tensor_value_info("mm1_out", TensorProto.FLOAT, [1, 3, 3, mh])
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, (1, 3, 3, mh))
    dtype = DataType["INT8"]
    W0 = gen_finn_dt_tensor(dtype, (mw, mh))
    W1 = gen_finn_dt_tensor(dtype, (mw, mh))
    thresh0 = generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1)
    thresh0 = np.sort(thresh0, axis=1)
    thresh1 = generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1)
    thresh1 = np.sort(thresh1, axis=1)
    matmul_node0 = helper.make_node("MatMul", ["ifm", "weights0"], ["mm0_out"], name="MatMul0")
    mt_node0 = helper.make_node(
        "MultiThreshold",
        ["mm0_out", "thresh0"],
        ["mt0_out"],
        domain="qonnx.custom_op.general",
        out_dtype="INT8",
        out_bias=float(dtype.min()),
        out_scale=1.0,
        data_layout="NHWC",
        name="MultiThreshold0",
    )
    matmul_node1 = helper.make_node("MatMul", ["mt0_out", "weights1"], ["mm1_out"], name="MatMul1")
    mt_node1 = helper.make_node(
        "MultiThreshold",
        ["mm1_out", "thresh1"],
        ["ofm"],
        domain="qonnx.custom_op.general",
        out_dtype="INT8",
        out_bias=float(dtype.min()),
        out_scale=1.0,
        data_layout="NHWC",
        name="MultiThreshold1",
    )
    nodes = [matmul_node0, mt_node0, matmul_node1, mt_node1]
    loop_body = helper.make_graph(
        nodes=nodes,
        name="matmul_graph",
        inputs=[ifm],
        outputs=[ofm],
        value_info=[mm0_out, mt0_out, mm1_out],
    )
    loop_body_model = qonnx_make_model(loop_body, producer_name="loop-body-model")
    loop_body_model = ModelWrapper(loop_body_model)

    loop_body_model.set_initializer("weights0", W0)
    loop_body_model.set_tensor_datatype("weights0", dtype)
    loop_body_model.set_initializer("weights1", W1)
    loop_body_model.set_tensor_datatype("weights1", dtype)
    loop_body_model.set_initializer("thresh0", thresh0)
    loop_body_model.set_initializer("thresh1", thresh1)
    loop_body_model.set_tensor_datatype("ifm", dtype)
    loop_body_model.set_tensor_datatype("ofm", dtype)
    loop_body_model = loop_body_model.transform(InferShapes())
    loop_body_model = loop_body_model.transform(InferDataTypes())
    loop_body_model = loop_body_model.transform(RoundAndClipThresholds())

    iteration = 3
    x = gen_finn_dt_tensor(DataType["INT8"], [1, 3, 3, mw])
    input_dict = {loop_body_model.graph.input[0].name: x}
    # calculate reference io
    for i in range(iteration):
        y_dict = oxe.execute_onnx(loop_body_model, input_dict)
        y = y_dict[loop_body_model.graph.output[0].name]
        input_dict[loop_body_model.graph.input[0].name] = y
    refio = (x, y)

    # stack according to iteration count
    W0 = np.stack([W0] * iter_count)
    W1 = np.stack([W1] * iter_count)
    thresh0 = np.stack([thresh0] * iter_count)
    thresh1 = np.stack([thresh1] * iter_count)
    loop_node = helper.make_node(
        "FINNLoop",
        domain="finn.custom_op.fpgadataflow",
        inputs=["ifm", "weights0", "thresh0", "weights1", "thresh1"],
        outputs=["ofm"],
        body=loop_body_model.graph,
        iteration=iteration,
        inputDataType="INT8",
        outputDataType="INT8",
        paramNodes=["MatMul0", "MultiThreshold0", "MatMul1", "MultiThreshold1"],
    )
    graph = helper.make_graph(nodes=[loop_node], name="loop_graph", inputs=[ifm], outputs=[ofm])
    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_initializer("weights0", W0)
    model.set_tensor_datatype("weights0", dtype)
    model.set_initializer("weights1", W1)
    model.set_tensor_datatype("weights1", dtype)
    model.set_initializer("thresh0", thresh0)
    model.set_initializer("thresh1", thresh1)
    model.set_tensor_datatype("ifm", dtype)
    model.set_tensor_datatype("ofm", dtype)

    return model, refio


def test_fpgadataflow_loop():
    # model = ModelWrapper("finn_loop.onnx")
    # inst = getCustomOp(model.graph.node[6])
    model, refio = make_loop_modelwrapper(16, 16, 3)
    inst = getCustomOp(model.graph.node[0])
    body = inst.get_nodeattr("body")
    body = body.transform(to_hw.InferThresholdingLayer())
    body = body.transform(to_hw.InferQuantizedMatrixVectorActivation())
    # update loop and loop body
    # get all param nodes and set param stream to external
    param_node_op_types = ["MVAU"]
    param_nodes = []
    for op_type in param_node_op_types:
        param_nodes = body.get_nodes_by_op_type(op_type)
        if param_nodes:
            for param_node in param_nodes:
                getCustomOp(param_node).set_nodeattr("mem_mode", "external")
    inst.set_nodeattr("body", body.graph)
    inst.set_nodeattr("paramNodes", [node.name for node in body.graph.node])
    input_dict = {model.graph.input[0].name: refio[0]}
    y_dict = oxe.execute_onnx(model, input_dict)
    y = y_dict[model.graph.output[0].name]
    assert (y == refio[1]).all()
