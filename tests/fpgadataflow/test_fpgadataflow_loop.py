from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


def test_fpgadataflow_loop():
    model = ModelWrapper("finn_loop.onnx")
    inst = getCustomOp(model.graph.node[6])
    body = inst.get_nodeattr("body")
    print(body)
