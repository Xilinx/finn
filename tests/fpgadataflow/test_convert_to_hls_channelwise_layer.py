import pytest

from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor
from finn.transformation.infer_shapes import InferShapes
import numpy as np


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


def make_single_maxpool_modelwrapper(onnx_op_name, ishape, idt, pdt, pshape):

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    p0 = helper.make_tensor_value_info("p0", TensorProto.FLOAT, pshape)

    model = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[inp],
            outputs=[outp],
            value_info=[p0],
            nodes=[helper.make_node(onnx_op_name, ["inp", "p0"], ["outp"])],
        )
    )

    model = ModelWrapper(model)
    model.set_initializer("p0", gen_finn_dt_tensor(pdt, pshape))
    model.set_tensor_datatype("inp", idt)
    model.transform(InferDataLayouts(), make_deepcopy=False)
    model.transform(InferShapes(), make_deepcopy=False)
    return model


# parameter datatype
@pytest.mark.parametrize("pdt", [DataType.BIPOLAR, DataType.UINT4, DataType.INT2])
# input datatype
@pytest.mark.parametrize("idt", [DataType.INT32, DataType.UINT4, DataType.INT4])
# function
@pytest.mark.parametrize("onnx_op_name", ["Add", "Mul"])
# vector parameter or scalar parameter (broadcast)
@pytest.mark.parametrize("scalar_param", [True, False])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.vivado
@pytest.mark.slow
def test_convert_to_hls_channelwise_layer(
    pdt, idt, onnx_op_name, scalar_param, exec_mode
):
    ifm_ch = 16
    ifm_dim = 5
    ishape = (1, ifm_ch, ifm_dim, ifm_dim)
    if scalar_param:
        pshape = (1,)
    else:
        pshape = (1, ifm_ch, 1, 1)

    np.random.seed(0)
    model = make_single_maxpool_modelwrapper(onnx_op_name, ishape, idt, pdt, pshape)

    # Since the aren't Data types with a bit width of a non power of 2,
    # there are cases where the input won't use it full range.
    if idt == DataType.INT32:
        x = gen_finn_dt_tensor(DataType.INT16, (1, ifm_ch, ifm_dim, ifm_dim))
    elif idt == DataType.UINT32:
        x = gen_finn_dt_tensor(DataType.UINT16, (1, ifm_ch, ifm_dim, ifm_dim))
    else:
        x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_dim, ifm_dim))

    input_dict = prepare_inputs(x)
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    new_model = model.transform(to_hls.InferChannelwiseLinearLayer())
    new_model = new_model.transform(GiveUniqueNodeNames())

    if exec_mode == "cppsim":
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
        new_model = new_model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    ctx_produced = oxe.execute_onnx(
        new_model, input_dict, return_full_exec_context=True
    )
    y_produced = ctx_produced["outp"]

    assert (y_produced == y_expected).all()
    assert new_model.graph.node[1].op_type == "ChannelwiseOp_Batch"
