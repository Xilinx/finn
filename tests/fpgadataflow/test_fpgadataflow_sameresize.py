import pytest
import os

from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim

from finn.util.basic import pynq_part_map

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def make_single_sameresize_modelwrapper(
    idim, odim, kdim, stride, num_ch, idt, pad_style
):
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
def test_fpgadataflow_sameresize_cppsim(idim, kdim, stride, num_ch, idt, pad_style):
    assert idim % stride == 0, "Stride must divide input dimension."
    # number of "same" windows over the input data
    same_windows = idim // stride
    odim = kdim + stride * (same_windows - 1)

    # generate input data
    x = gen_finn_dt_tensor(idt, [1, idim, idim, num_ch])
    input_dict = {"inp": x}

    model = make_single_sameresize_modelwrapper(
        idim, odim, kdim, stride, num_ch, idt, pad_style
    )
    model = model.transform(InferShapes())
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    y = oxe.execute_onnx(model, input_dict)["outp"]

    expected_oshape = (1, odim, odim, num_ch)
    assert y.shape == expected_oshape


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
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_sameresize_rtlsim(idim, kdim, stride, num_ch, idt, pad_style):
    assert idim % stride == 0, "Stride must divide input dimension."
    # number of "same" windows over the input data
    same_windows = idim // stride
    odim = kdim + stride * (same_windows - 1)

    # generate input data
    x = gen_finn_dt_tensor(idt, [1, idim, idim, num_ch])
    input_dict = {"inp": x}

    model = make_single_sameresize_modelwrapper(
        idim, odim, kdim, stride, num_ch, idt, pad_style
    )
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y = oxe.execute_onnx(model, input_dict)["outp"]

    expected_oshape = (1, odim, odim, num_ch)
    assert y.shape == expected_oshape
