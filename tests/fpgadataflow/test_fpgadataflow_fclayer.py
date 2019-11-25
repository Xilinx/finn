import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.utils import interleave_matrix_outer_dim_from_partitions
from finn.custom_op.multithreshold import multithreshold


def make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T=None, tdt=None):
    mh = W.shape[0]
    mw = W.shape[1]
    assert mh % pe == 0
    assert mw % simd == 0
    wmem = mw * mh // (pe * simd)
    assert mw * mh == wmem * pe * simd
    nf = mh // pe
    tmem = nf
    sf = mw // simd
    # distribute rows between PEs
    W_reshaped = interleave_matrix_outer_dim_from_partitions(W, pe)
    # create SIMD as innermost dimension
    W_reshaped = W_reshaped.reshape(pe, wmem, simd)
    if T is not None:
        assert T.shape[0] == 1 or T.shape[0] == mh
        n_thres_steps = T.shape[1]
        # if single global threshold specified, repeat along channels
        if T.shape[0] == 1:
            T = np.tile(T, (mh, 1))
        # distribute T rows between PEs
        T_reshaped = interleave_matrix_outer_dim_from_partitions(T, pe)
        assert T_reshaped.shape[0] == pe
        assert T_reshaped.shape[1] == tmem
        assert T_reshaped.shape[2] == n_thres_steps
    else:
        n_thres_steps = 0
    if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
        rdt = "Recast<XnorMul>"
    else:
        assert "Weight & input datatype combo not yet supported"

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, sf, simd])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, nf, pe])
    if T is not None:
        node_inp_list = ["inp", "weights", "thresh"]
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        WMEM=wmem,
        TMEM=tmem,
        resDataType=rdt,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", W_reshaped)
    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T_reshaped)
    return model


def test_fpgadataflow_fclayer_ibp_wbp_noact():
    mh = 4
    mw = 4
    pe = 4
    simd = 4
    wdt = idt = DataType.BIPOLAR
    odt = DataType.UINT32
    # generate weights
    W = np.random.randint(2, size=(mh, mw))
    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt)
    # generate input data
    x = np.random.randint(2, size=mw)
    ishape = model.get_tensor_shape("inp")
    oshape = model.get_tensor_shape("outp")
    input_tensor = (np.asarray(x, dtype=np.float32)).reshape(*ishape)
    input_dict = {"inp": input_tensor}
    produced = oxe.execute_onnx(model, input_dict)["outp"]
    # convert to bipolar values
    Wb = 2 * W - 1
    xb = 2 * x - 1
    yb = np.dot(Wb, xb).reshape(oshape.shape)
    # XnorMul produces positive outputs only, adjust expectation accordingly
    expected = 2 * yb - mw
    assert (produced == expected).all()


def test_fpgadataflow_fclayer_all_bipolar():
    mh = 4
    mw = 4
    pe = 4
    simd = 4
    wdt = idt = odt = DataType.BIPOLAR
    tdt = DataType.UINT32
    # generate weights
    W = np.random.randint(2, size=(mh, mw))
    # single global threshold at zero
    T = np.zeros((1, 1))
    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    # generate input data
    x = np.random.randint(2, size=mw)
    ishape = model.get_tensor_shape("inp")
    input_tensor = (np.asarray(x, dtype=np.float32)).reshape(*ishape)
    input_dict = {"inp": input_tensor}
    produced = oxe.execute_onnx(model, input_dict)["outp"]
    # convert to bipolar values
    Wb = 2 * W - 1
    xb = 2 * x - 1
    yb = np.dot(Wb, xb)
    expected = multithreshold(yb.reshape(1, mh), T)
    assert (produced == expected).all()
