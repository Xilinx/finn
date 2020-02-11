import os.path
from pkgutil import get_data

from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)


def test_create_dataflow_partition():
    # load the onnx model
    raw_m = get_data(
        "finn", "data/onnx/finn-hls-model/tfc_w1_a1_after_conv_to_hls.onnx"
    )
    model = ModelWrapper(raw_m)
    model = model.transform(CreateDataflowPartition())
    assert model.graph.node[2].op_type == "StreamingDataflowPartition"
    sdp_node = getCustomOp(model.graph.node[2])
    assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
    assert os.path.isfile(sdp_node.get_nodeattr("model"))
