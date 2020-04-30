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


def make_single_fifo_modelwrapper(Shape, Depth, fld_shape, finn_dtype):

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, Shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, Shape)

    FIFO_node = helper.make_node(
        "StreamingFIFO",
        ["inp"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        depth=Depth,
        folded_shape=fld_shape,
        dataType=str(finn_dtype.name),
    )

    graph = helper.make_graph(
        nodes=[FIFO_node], name="fifo_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fifo-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def prepare_inputs(input_tensor, dt):
    return {"inp": input_tensor}


# shape
@pytest.mark.parametrize("Shape", [[1, 4]])
# inWidth
@pytest.mark.parametrize("folded_shape", [[1, 1, 4]])
# outWidth
@pytest.mark.parametrize("depth", [2])
# finn_dtype
@pytest.mark.parametrize("finn_dtype", [DataType.BIPOLAR, DataType.INT2])
def test_fpgadataflow_fifo_rtlsim(Shape, folded_shape, depth, finn_dtype):

    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, Shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_fifo_modelwrapper(Shape, depth, folded_shape, finn_dtype)

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
