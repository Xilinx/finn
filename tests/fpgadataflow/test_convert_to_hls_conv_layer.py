from onnx import TensorProto, helper
import numpy as np
import pytest

from finn.core.datatype import DataType
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import gen_finn_dt_tensor
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls

from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.slow
def test_convert_to_hls_conv_layer(padding, kernel_size):

    assert (
        kernel_size % 2 != 0
    ), """test_convert_to_hls_conv_layer test only
    supports odd kernel_size"""

    np.random.seed(0)
    padding = True
    idt = DataType.UINT4

    in_feature_dim = 7
    in_chn = 3

    stages = 1  # just one convolution

    out_feature_dim = (
        in_feature_dim if padding else in_feature_dim - (kernel_size // 2 * 2) * stages
    )

    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    output_shape = [1, in_chn, out_feature_dim, out_feature_dim]

    conv_param_shape = [in_chn, in_chn, kernel_size, kernel_size]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = 1
    conv_config["kernel_shape"] = [kernel_size, kernel_size]
    if padding:
        pad = kernel_size // 2
        conv_config["pads"] = [pad, pad, pad, pad]
    else:
        conv_config["pads"] = [0, 0, 0, 0]
    conv_config["strides"] = [1, 1]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)
    value_info = [
        helper.make_tensor_value_info("p1", TensorProto.FLOAT, conv_param_shape)
    ]

    modelproto = helper.make_model(
        helper.make_graph(
            name="conv_test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("Conv", ["top_in", "p1"], ["top_out"], **conv_config),
            ],
        ),
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", idt)
    model.set_tensor_datatype("top_out", idt)
    model.set_tensor_datatype("p1", DataType.UINT4)

    model = model.transform(InferShapes())
    model.set_initializer(
        "p1", np.round(np.random.rand(*conv_param_shape).astype(np.float32) * 16)
    )

    model.set_tensor_datatype(model.graph.input[0].name, idt)
    model = model.transform(InferShapes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())

    new_model = model.transform(LowerConvsToMatMul())
    new_model = new_model.transform(to_hls.InferConvInpGen())

    new_model = new_model.transform(PrepareCppSim())
    new_model = new_model.transform(CompileCppSim())
    new_model = new_model.transform(SetExecMode("cppsim"))

    x = gen_finn_dt_tensor(idt, input_shape)
    inp_dict = {model.graph.input[0].name: x}
    assert oxe.compare_execution(model, new_model, inp_dict)
