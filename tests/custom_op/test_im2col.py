import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.custom_op.im2col import compute_conv_output_dim


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


def execution_im2col(x, idt, k, stride, ifm_ch, ifm_dim, pad_amt=0, pad_val=0):
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, pad_amt)

    # set up onnx model
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, k * k * ifm_ch]
    )

    Im2Col_node = helper.make_node(
        "Im2Col",
        ["inp"],
        ["outp"],
        domain="finn",
        stride=stride,
        kernel_size=k,
        pad_amount=pad_amt,
        pad_value=pad_val,
        input_shape="(1,{},{},{})".format(ifm_dim, ifm_dim, ifm_ch),
    )

    graph = helper.make_graph(
        nodes=[Im2Col_node], name="im2col_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)

    # test shape inference
    model.transform(InferShapes())
    assert model.get_tensor_shape("outp") == [1, ofm_dim, ofm_dim, k * k * ifm_ch]

    # test datatype inference
    assert model.get_tensor_datatype("outp") is DataType.FLOAT32
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("outp") is idt

    # prepare input data
    input_dict = {"inp": x}

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    return y_produced


def test_im2col():
    # bipolar inputs with following im2col parameters
    idt = DataType.BIPOLAR
    k = 2
    stride = 1
    ifm_ch = 1
    ifm_dim = 4
    pad_amt = 0
    pad_val = 0
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, pad_amt)

    x = np.asarray(
        [
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    ).reshape(1, ifm_dim, ifm_dim, ifm_ch)

    expected = np.asarray(
        [
            -1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    ).reshape(1, ofm_dim, ofm_dim, k * k * ifm_ch)

    produced = execution_im2col(x, idt, k, stride, ifm_ch, ifm_dim, pad_amt, pad_val)
    assert (produced == expected).all()

    idt = DataType.INT8
    k = 2
    stride = 1
    ifm_ch = 2
    ifm_dim = 4
    pad_amt = 0
    pad_val = 0
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, pad_amt)

    x = np.asarray(
        [
            [
                [[1, -1], [2, -2], [3, -3], [4, -4]],
                [[5, -5], [6, -6], [7, -7], [8, -8]],
                [[9, -9], [10, -10], [11, -11], [12, -12]],
                [[13, -13], [14, -14], [15, -15], [16, -16]],
            ]
        ],
        dtype=np.float32,
    )

    expected = np.asarray(
        [
            [
                [
                    [1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 6.0, -6.0],
                    [2.0, -2.0, 3.0, -3.0, 6.0, -6.0, 7.0, -7.0],
                    [3.0, -3.0, 4.0, -4.0, 7.0, -7.0, 8.0, -8.0],
                ],
                [
                    [5.0, -5.0, 6.0, -6.0, 9.0, -9.0, 10.0, -10.0],
                    [6.0, -6.0, 7.0, -7.0, 10.0, -10.0, 11.0, -11.0],
                    [7.0, -7.0, 8.0, -8.0, 11.0, -11.0, 12.0, -12.0],
                ],
                [
                    [9.0, -9.0, 10.0, -10.0, 13.0, -13.0, 14.0, -14.0],
                    [10.0, -10.0, 11.0, -11.0, 14.0, -14.0, 15.0, -15.0],
                    [11.0, -11.0, 12.0, -12.0, 15.0, -15.0, 16.0, -16.0],
                ],
            ]
        ],
        dtype=np.float32,
    )

    produced = execution_im2col(x, idt, k, stride, ifm_ch, ifm_dim, pad_amt, pad_val)
    assert (produced == expected).all()

    idt = DataType.INT8
    k = 2
    stride = 1
    ifm_ch = 2
    ifm_dim = 4
    pad_amt = 1
    pad_val = 0
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, pad_amt)

    x = np.asarray(
        [
            [
                [[1, -1], [2, -2], [3, -3], [4, -4]],
                [[5, -5], [6, -6], [7, -7], [8, -8]],
                [[9, -9], [10, -10], [11, -11], [12, -12]],
                [[13, -13], [14, -14], [15, -15], [16, -16]],
            ]
        ],
        dtype=np.float32,
    )

    expected = np.asarray(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 3.0, -3.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0, -3.0, 4.0, -4.0],
                    [0.0, 0.0, 0.0, 0.0, 4.0, -4.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 5.0, -5.0],
                    [1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 6.0, -6.0],
                    [2.0, -2.0, 3.0, -3.0, 6.0, -6.0, 7.0, -7.0],
                    [3.0, -3.0, 4.0, -4.0, 7.0, -7.0, 8.0, -8.0],
                    [4.0, -4.0, 0.0, 0.0, 8.0, -8.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 5.0, -5.0, 0.0, 0.0, 9.0, -9.0],
                    [5.0, -5.0, 6.0, -6.0, 9.0, -9.0, 10.0, -10.0],
                    [6.0, -6.0, 7.0, -7.0, 10.0, -10.0, 11.0, -11.0],
                    [7.0, -7.0, 8.0, -8.0, 11.0, -11.0, 12.0, -12.0],
                    [8.0, -8.0, 0.0, 0.0, 12.0, -12.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 9.0, -9.0, 0.0, 0.0, 13.0, -13.0],
                    [9.0, -9.0, 10.0, -10.0, 13.0, -13.0, 14.0, -14.0],
                    [10.0, -10.0, 11.0, -11.0, 14.0, -14.0, 15.0, -15.0],
                    [11.0, -11.0, 12.0, -12.0, 15.0, -15.0, 16.0, -16.0],
                    [12.0, -12.0, 0.0, 0.0, 16.0, -16.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 13.0, -13.0, 0.0, 0.0, 0.0, 0.0],
                    [13.0, -13.0, 14.0, -14.0, 0.0, 0.0, 0.0, 0.0],
                    [14.0, -14.0, 15.0, -15.0, 0.0, 0.0, 0.0, 0.0],
                    [15.0, -15.0, 16.0, -16.0, 0.0, 0.0, 0.0, 0.0],
                    [16.0, -16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ],
        dtype=np.float32,
    )

    produced = execution_im2col(x, idt, k, stride, ifm_ch, ifm_dim, pad_amt, pad_val)
    assert (produced == expected).all()


def test_im2col_infer_shapes():
    idt = DataType.BIPOLAR
    k = 2
    stride = 1
    ifm_ch = 1
    ifm_dim = 4
    ofm_dim = int(((ifm_dim - k) / stride) + 1)

    # set up onnx model
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, k * k * ifm_ch]
    )

    abs_node = helper.make_node("Abs", inputs=["inp"], outputs=["abs"])

    Im2Col_node = helper.make_node(
        "Im2Col",
        ["abs"],
        ["im2col"],
        domain="finn",
        stride=stride,
        kernel_size=k,
        input_shape="(1,{},{},{})".format(ifm_dim, ifm_dim, ifm_ch),
    )

    abs1_node = helper.make_node("Abs", inputs=["im2col"], outputs=["outp"])

    graph = helper.make_graph(
        nodes=[abs_node, Im2Col_node, abs1_node],
        name="shape_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[
            helper.make_tensor_value_info(
                "abs", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
            ),
            helper.make_tensor_value_info(
                "im2col", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, k * k * ifm_ch]
            ),
        ],
    )

    model = helper.make_model(graph, producer_name="shape-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)

    # test shape inference
    model.transform(InferShapes())
    assert model.get_tensor_shape("im2col") == [1, ofm_dim, ofm_dim, k * k * ifm_ch]
