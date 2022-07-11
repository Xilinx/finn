import pytest

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MakeScaleResizeNHWC


def create_resize_transpose(ifm_dim, ifm_ch, scales, mode, idt):
    ofm_dim_h = ifm_dim[0] * scales[2]
    ofm_dim_w = ifm_dim[1] * scales[3]
    inp = oh.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim[0], ifm_dim[1]]
    )

    param = oh.make_tensor_value_info("scales", TensorProto.FLOAT, [4])

    # Not actually used, only needed for compliance with the Resize node interface
    roi = oh.make_tensor_value_info("roi", TensorProto.FLOAT, [4])

    outp_up = oh.make_tensor_value_info(
        "outp_up", TensorProto.FLOAT, [1, ifm_ch, ofm_dim_h, ofm_dim_w]
    )
    outp = oh.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, ifm_ch]
    )

    resize_node = oh.make_node(
        "Resize",
        inputs=["inp", "roi", "scales"],
        outputs=["outp_up"],
        name="Resize1",
        mode=mode,
    )

    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=["outp_up"],
        outputs=["outp"],
        name="Transpose1",
        perm=[0, 2, 3, 1],
    )

    graph = oh.make_graph(
        nodes=[resize_node, transpose_node],
        name="resize_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[outp_up, param, roi],
    )

    model = oh.make_model(graph, producer_name="resize_model1")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    model = model.transform(InferShapes())

    return model


def create_transpose_resize(ifm_dim, ifm_ch, scales, mode, idt):
    ofm_dim_h = ifm_dim[0] * scales[2]
    ofm_dim_w = ifm_dim[1] * scales[3]
    inp = oh.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim[0], ifm_dim[1], ifm_ch]
    )

    param = oh.make_tensor_value_info("scales", TensorProto.FLOAT, [4])

    # Not actually used, only needed for compliance with the Resize node interface
    roi = oh.make_tensor_value_info("roi", TensorProto.FLOAT, [4])

    outp = oh.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ifm_ch, ofm_dim_h, ofm_dim_w]
    )
    outp_tr = oh.make_tensor_value_info(
        "outp_tr", TensorProto.FLOAT, [1, ifm_ch, ifm_dim[0], ifm_dim[1]]
    )

    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=["inp"],
        outputs=["outp_tr"],
        name="Transpose1",
        perm=[0, 3, 1, 2],
    )

    resize_node = oh.make_node(
        "Resize",
        inputs=["outp_tr", "roi", "scales"],
        outputs=["outp"],
        name="Resize1",
        mode=mode,
    )

    graph = oh.make_graph(
        nodes=[transpose_node, resize_node],
        name="resize_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[outp_tr, param, roi],
    )

    model = oh.make_model(graph, producer_name="resize_model2")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    model = model.transform(InferShapes())

    return model


def create_transpose_resize_transpose(ifm_dim, ifm_ch, scales, mode, idt):
    ofm_dim_h = ifm_dim[0] * scales[2]
    ofm_dim_w = ifm_dim[1] * scales[3]
    inp = oh.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim[0], ifm_dim[1], ifm_ch]
    )

    param = oh.make_tensor_value_info("scales", TensorProto.FLOAT, scales)

    # Not actually used, only needed for compliance with the Resize node interface
    roi = oh.make_tensor_value_info("roi", TensorProto.FLOAT, [4])

    outp_tr = oh.make_tensor_value_info(
        "outp_tr", TensorProto.FLOAT, [1, ifm_ch, ifm_dim[0], ifm_dim[1]]
    )

    outp_up = oh.make_tensor_value_info(
        "outp_up", TensorProto.FLOAT, [1, ifm_ch, ofm_dim_h, ofm_dim_w]
    )
    outp = oh.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, ifm_ch]
    )

    transpose_node1 = onnx.helper.make_node(
        "Transpose",
        inputs=["inp"],
        outputs=["outp_tr"],
        name="Transpose1",
        perm=[0, 3, 1, 2],
    )

    resize_node = oh.make_node(
        "Resize",
        inputs=["outp_tr", "roi", "scales"],
        outputs=["outp_up"],
        name="Resize1",
        mode=mode,
    )

    transpose_node2 = onnx.helper.make_node(
        "Transpose",
        inputs=["out_up"],
        outputs=["outp"],
        name="Transpose2",
        perm=[0, 2, 3, 1],
    )

    graph = oh.make_graph(
        nodes=[transpose_node1, resize_node, transpose_node2],
        name="resize_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[outp_up, outp_tr, param, roi],
    )

    model = oh.make_model(graph, producer_name="resize_model3")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    model = model.transform(InferShapes())

    return model


@pytest.mark.streamline
# input dimension
@pytest.mark.parametrize("ifm_dim", [[2**i, 2**i] for i in range(3, 6)])
# input channels
@pytest.mark.parametrize("ifm_ch", [3])
# scales
@pytest.mark.parametrize(
    "scales", [[1, 1, i, j] for i in range(2, 5) for j in range(2, 5)]
)
# mode
@pytest.mark.parametrize("mode", ["nearest"])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT4"]])
def test_scale_resize_nhwc(ifm_dim, ifm_ch, scales, mode, idt):
    # create models
    resize_model1 = create_resize_transpose(ifm_dim, ifm_ch, scales, mode, idt)
    resize_model2 = create_transpose_resize(ifm_dim, ifm_ch, scales, mode, idt)
    resize_model3 = create_transpose_resize_transpose(
        ifm_dim, ifm_ch, scales, mode, idt
    )

    # set initializers
    resize_model1.set_initializer("scales", np.array(scales, dtype=np.float32))
    resize_model2.set_initializer("scales", np.array(scales, dtype=np.float32))
    resize_model3.set_initializer("scales", np.array(scales, dtype=np.float32))

    # generate input tensor for testing
    input_tensor_nchw = gen_finn_dt_tensor(idt, [1, ifm_ch, ifm_dim[0], ifm_dim[1]])
    input_tensor_nhwc = gen_finn_dt_tensor(idt, [1, ifm_dim[0], ifm_dim[1], ifm_ch])
    input_dict_nchw = {"inp": input_tensor_nchw}
    input_dict_nhwc = {"inp": input_tensor_nhwc}

    # execute first model
    output_dict1 = oxe.execute_onnx(resize_model1, input_dict_nchw)
    expected1 = output_dict1["outp"]

    # transform Resize into ResizeNHWC
    resize_model1 = resize_model1.transform(MakeScaleResizeNHWC())

    # execute transformed model
    output_node_name1 = resize_model1.graph.output[0].name
    output_dict1 = oxe.execute_onnx(
        resize_model1, input_dict_nchw, return_full_exec_context=False
    )
    output1 = output_dict1[output_node_name1]

    # compare outputs
    assert (expected1 == output1).all()

    # execute second model
    output_dict2 = oxe.execute_onnx(resize_model2, input_dict_nhwc)
    expected2 = output_dict2["outp"]

    # transform Resize into ResizeNHWC
    resize_model2 = resize_model2.transform(MakeScaleResizeNHWC())

    # execute transformed model
    output_node_name2 = resize_model2.graph.output[0].name
    output_dict2 = oxe.execute_onnx(
        resize_model2, input_dict_nhwc, return_full_exec_context=False
    )
    output2 = output_dict2[output_node_name2]

    # compare outputs
    assert (expected2 == output2).all()

    # execute third model
    output_dict3 = oxe.execute_onnx(resize_model3, input_dict_nhwc)
    expected3 = output_dict3["outp"]

    # transform Resize into ResizeNHWC
    resize_model3 = resize_model3.transform(MakeScaleResizeNHWC())

    # execute transformed model
    output_node_name3 = resize_model3.graph.output[0].name
    output_dict3 = oxe.execute_onnx(
        resize_model3, input_dict_nhwc, return_full_exec_context=False
    )
    output3 = output_dict3[output_node_name3]

    # compare outputs
    assert (expected3 == output3).all()
