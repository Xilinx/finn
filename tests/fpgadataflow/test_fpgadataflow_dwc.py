import pytest

from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe


def make_single_dwc_modelwrapper(Shape, INWidth, OUTWidth, finn_dtype):

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, Shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, Shape)

    DWC_node = helper.make_node(
        "StreamingDataWidthConverter_Batch",
        ["inp"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        shape=Shape,
        inWidth=INWidth,
        outWidth=OUTWidth,
        dataType=str(finn_dtype.name),
    )

    graph = helper.make_graph(
        nodes=[DWC_node], name="dwc_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="dwc-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def prepare_inputs(input_tensor, dt):
    return {"inp": input_tensor}


# shape
@pytest.mark.parametrize("Shape", [[1, 4], [1, 2, 8]])
# inWidth
@pytest.mark.parametrize("INWidth", [2, 4])
# outWidth
@pytest.mark.parametrize("OUTWidth", [2, 4])
# finn_dtype
@pytest.mark.parametrize("finn_dtype", [DataType.BIPOLAR, DataType.INT2])
def test_fpgadataflow_dwc_rtlsim(Shape, INWidth, OUTWidth, finn_dtype):

    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, Shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_dwc_modelwrapper(Shape, INWidth, OUTWidth, finn_dtype)

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(CodeGen_ipgen("xc7z020clg400-1", 5))
    model = model.transform(HLSSynth_IPGen())
    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert (
        y == x
    ).all(), """The output values are not the same as the
        input values anymore."""
    assert y.shape == tuple(Shape), """The output shape is incorrect."""
