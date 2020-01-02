from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.utils import gen_finn_dt_tensor


def test_im2col():
    idt = odt = DataType.BIPOLAR
    k = 2
    stride = 1
    ifm_ch = 1
    ifm_dim = 4
    ofm_dim = int(((ifm_dim - k) / stride) + 1)
    out_pix = ofm_dim * ofm_dim

    # set up onnx model
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, out_pix, k * k * ifm_ch]
    )

    Im2Col_node = helper.make_node(
        "Im2Col", ["inp"], ["outp"], domain="finn", stride=stride, kernel_size=k,
    )

    graph = helper.make_graph(
        nodes=[Im2Col_node], name="im2col_graph", inputs=[inp], outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_dim, ifm_dim))
    # prepare input data
    input_dict = {"inp": x}

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    print("Input:")
    print(x)
    print("Output:")
    print(y_produced)
