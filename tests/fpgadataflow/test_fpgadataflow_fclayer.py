import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.multithreshold import MultiThreshold


def test_fpgadataflow_fclayer_all_bipolar():
    mh = 4
    mw = 4
    pe = 4
    simd = 4
    wmem = mw * mh // (pe * simd)
    n_thres_steps = 1
    assert mw * mh == wmem * pe * simd
    nf = mh // pe
    tmem = nf
    sf = mw // simd
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, sf, simd])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, nf, pe])
    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        ["inp", "weights", "thresh"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        resDataType="Recast<XnorMul>",
    )

    graph = helper.make_graph(
        nodes=[FCLayer_node],
        name="fclayer_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[
            helper.make_tensor_value_info(
                "weights", TensorProto.FLOAT, [pe, wmem, simd]
            ),
            helper.make_tensor_value_info(
                "thresh", TensorProto.FLOAT, [pe, tmem, n_thres_steps]
            ),
        ],
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    # set the tensor datatypes (in this case: all to bipolar)
    for tensor in graph.input:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])
    for tensor in graph.output:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])

    # generate input data
    x = np.random.randint(2, size=mw)
    input_tensor = (np.asarray(x, dtype=np.float32)).reshape(1, sf, simd)
    input_dict = {"inp": input_tensor}

    # generate weights
    W = np.random.randint(2, size=(mh, mw))
    weights_tensor = (np.asarray(W, dtype=np.float32)).reshape(pe, wmem, simd)
    input_dict["weights"] = weights_tensor

    # generate threshold activation
    T = np.zeros(mh)
    thresh_tensor = (np.asarray(T, dtype=np.float32)).reshape(pe, tmem, n_thres_steps)
    input_dict["thresh"] = thresh_tensor

    output_dict = oxe.execute_onnx(model, input_dict)

    # convert to bipolar values
    Wb = 2 * W - 1
    xb = 2 * x - 1
    yb = np.dot(Wb, xb)
    thres = MultiThreshold()
    expected = thres._execute(yb.reshape(1, mh), T.reshape((mh, 1)))
    assert (output_dict["outp"] == expected).all()
