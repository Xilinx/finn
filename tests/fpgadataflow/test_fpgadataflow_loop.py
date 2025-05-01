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


def make_single_matmul_loop_modelwrapper(mw, mh, iter_count):
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, 3, 3, mw])
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, (1, 3, 3, mh))
    W = gen_finn_dt_tensor(DataType["INT8"], (mw, mh))
    matmul_node = helper.make_node("MatMul", ["ifm", "weights"], ["ofm"], name="MatMul0")
    loop_body = helper.make_graph(
        nodes=[matmul_node], name="matmul_graph", inputs=[ifm], outputs=[ofm]
    )
    loop_body_model = qonnx_make_model(loop_body, producer_name="loop-body-model")
    loop_body_model = ModelWrapper(loop_body_model)
    loop_body_model = loop_body_model.transform(InferShapes())
    loop_body_model = loop_body_model.transform(InferDataTypes())

    loop_body_model.set_initializer("weights", W)
    loop_body_model.set_tensor_datatype("weights", DataType["INT8"])
    loop_body_model.set_tensor_datatype("ifm", DataType["INT8"])
    loop_body_model.set_tensor_datatype("ofm", DataType["INT32"])

    # stack according to iteration count
    W = np.stack([W] * iter_count)
    loop_node = helper.make_node(
        "FINNLoop",
        domain="finn.custom_op.fpgadataflow",
        inputs=["ifm", "weights"],
        outputs=["ofm"],
        body=loop_body_model.graph,
        iteration=3,
        inputDataType="INT8",
        outputDataType="INT32",
        paramNodes=["MatMul0"],
    )
    graph = helper.make_graph(nodes=[loop_node], name="loop_graph", inputs=[ifm], outputs=[ofm])
    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_initializer("weights", W)
    model.set_tensor_datatype("weights", DataType["INT8"])

    return model


def test_fpgadataflow_loop():
    # model = ModelWrapper("finn_loop.onnx")
    # inst = getCustomOp(model.graph.node[6])
    model = make_single_matmul_loop_modelwrapper(16, 16, 3)
    inst = getCustomOp(model.graph.node[0])
    body = inst.get_nodeattr("body")
    body = body.transform(to_hw.InferQuantizedMatrixVectorActivation())
    # update loop and loop body
    # get all param nodes and set param stream to external
    param_node_op_types = ["MVAU", "Thresholding"]
    param_nodes = []
    for op_type in param_node_op_types:
        param_nodes = body.get_nodes_by_op_type(op_type)
        if param_nodes:
            for param_node in param_nodes:
                getCustomOp(param_node).set_nodeattr("mem_mode", "external")
    inst.set_nodeattr("body", body.graph)
    inst.set_nodeattr("paramNodes", [node.name for node in body.graph.node])
    model.save("test.onnx")
    x = gen_finn_dt_tensor(DataType["INT8"], [1, 3, 3, 16])
    input_dict = {model.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model, input_dict)
    y = y_dict[model.graph.output[0].name]
    print(y)
