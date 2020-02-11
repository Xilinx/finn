import os.path
from pkgutil import get_data

import pytest

from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.util.basic import make_build_dir

build_dir = make_build_dir("test_dataflow_partition_")


@pytest.mark.dependency()
def test_dataflow_partition_create():
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
    model.save(build_dir + "/test_dataflow_partition_create.onnx")


@pytest.mark.dependency(depends=["test_dataflow_partition_create"])
def test_dataflow_partition_tlastmarker():
    model = ModelWrapper(build_dir + "/test_dataflow_partition_create.onnx")
    model_path = getCustomOp(model.graph.node[2]).get_nodeattr("model")
    model = ModelWrapper(model_path)
    model = model.transform(InsertTLastMarker())
    assert model.graph.node[-1].op_type == "TLastMarker"
    tl_node = getCustomOp(model.graph.node[-1])
    assert tl_node.get_nodeattr("NumIters") == 1
    assert tl_node.get_nodeattr("StreamWidth") == 320
    model.save(build_dir + "/test_dataflow_partition_tlastmarker.onnx")
    model = model.transform(InsertTLastMarker())
    model.save(build_dir + "/test_dataflow_partition_tlastmarker2.onnx")
