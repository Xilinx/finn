from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.analysis.verify_custom_nodes import verify_nodes
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.utils import gen_finn_dt_tensor
from finn.transformation.infer_datatypes import InferDataTypes


def check_two_dict_for_equality(dict1, dict2):
    for key in dict1:
        assert key in dict2, "Key: {} is not in both dictionaries".format(key)
        assert (
            dict1[key] == dict2[key]
        ), """Values for key {} are not the same
        in both dictionaries""".format(
            key
        )

    return True


def test_im2col():
    idt = DataType.BIPOLAR
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

    # test datatype inference
    assert model.get_tensor_datatype("outp") is DataType.FLOAT32
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("outp") is DataType.BIPOLAR

    # test node verification
    produced = model.analysis(verify_nodes)
    print(produced)
    expected = {
        "Im2Col": [
            "The number of attributes is correct",
            "Attribute domain is set correctly",
            "All necessary attributes exist",
            "The number of inputs is correct",
        ],
    }
    assert check_two_dict_for_equality(
        produced, expected
    ), """The produced output of
    the verification analysis pass is not equal to the expected one"""

    # test execution
    x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_dim, ifm_dim))
    # prepare input data
    input_dict = {"inp": x}

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    print("Input:")
    print(x)
    print("Output:")
    print(y_produced)
