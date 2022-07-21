import numpy as np
import onnx.parser as oprs
import qonnx.core.data_layout as dl
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls


def build_model(dt0, dt1):
    np.random.seed(0)
    shp = [1, 3, 4, 2]
    shp_str = str(shp)
    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{shp_str} in0, float{shp_str} in1) => (float{shp_str} out0)
    {{
        sub_out = Sub(in0, in1)
        out0 = Abs(sub_out)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_tensor_datatype("in0", dt0)
    model.set_tensor_datatype("in1", dt1)
    model.set_tensor_layout("in0", dl.NHWC)
    model.set_tensor_layout("in1", dl.NHWC)
    model = model.transform(InferShapes())
    return model


def test_fpgadataflow_eltwise():
    dt0 = DataType["UINT7"]
    dt1 = DataType["UINT8"]
    model = build_model(dt0, dt1)
    model = model.transform(to_hls.InferStreamingEltwiseAbsDiff())
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "StreamingEltwise"
