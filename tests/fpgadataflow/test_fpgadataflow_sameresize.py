import pytest

from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes


def make_single_sameresize_modelwrapper(idim, kdim, stride, num_ch, idt, pad_style):
    assert idim % stride == 0, "Stride must divide input dimension."
    # number of "same" windows over the input data
    same_windows = idim // stride
    odim = kdim + stride * (same_windows - 1)
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, idim, idim, num_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, odim, odim, num_ch]
    )

    SameResize_node = helper.make_node(
        "SameResize_Batch",
        ["inp"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        ImgDim=idim,
        KernelDim=kdim,
        Stride=stride,
        NumChannels=num_ch,
        inputDataType=str(idt.name),
        PaddingStyle=pad_style,
    )

    graph = helper.make_graph(
        nodes=[SameResize_node], name="sameresize_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="sameresize-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    return model


# image dimension
@pytest.mark.parametrize("idim", [4])
# kernel dimension
@pytest.mark.parametrize("kdim", [2])
# stride
@pytest.mark.parametrize("stride", [1])
# number of channels
@pytest.mark.parametrize("num_ch", [2])
# FINN input datatype
@pytest.mark.parametrize("idt", [DataType.INT2])
# PaddingStyle: distribution of added values to achieve "same" padding
@pytest.mark.parametrize("pad_style", [2])
def test_fpgadataflow_sameresize(idim, kdim, stride, num_ch, idt, pad_style):
    model = make_single_sameresize_modelwrapper(
        idim, kdim, stride, num_ch, idt, pad_style
    )
    model = model.transform(InferShapes())
