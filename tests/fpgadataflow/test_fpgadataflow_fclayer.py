import pytest

import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
import finn.custom_op.xnorpopcount as xp
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.utils import calculate_signed_dot_prod_range, gen_finn_dt_tensor
from finn.custom_op.multithreshold import multithreshold
from finn.transformation.fpgadataflow.cleanup import CleanUp
from finn.transformation.fpgadataflow.codegen import CodeGen
from finn.transformation.fpgadataflow.compile import Compile


def make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T=None, tdt=None):
    mw = W.shape[0]
    mh = W.shape[1]
    assert mh % pe == 0
    assert mw % simd == 0
    wmem = mw * mh // (pe * simd)
    assert mw * mh == wmem * pe * simd
    nf = mh // pe
    sf = mw // simd
    if T is not None:
        tmem = nf
    else:
        tmem = 0

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, sf, simd])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, nf, pe])
    if T is not None:
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType.BIPOLAR:
            actval = 0
        else:
            actval = odt.min()
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
        actval = 0
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
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", W)
    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)
    return model


def prepare_inputs(model, input_tensor, idt):
    ishape = model.get_tensor_shape("inp")
    input_tensor = (np.asarray(input_tensor, dtype=np.float32)).reshape(*ishape)
    # flip SIMD (innermost) dimension of input tensor, there's some reversal
    # going on somewhere with a mistmatch between npy and hls...
    input_tensor = np.flip(input_tensor, -1)
    return {"inp": input_tensor}


# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType.BIPOLAR])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType.BIPOLAR, DataType.INT2])
# input datatype
@pytest.mark.parametrize("idt", [DataType.BIPOLAR, DataType.INT2])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 1])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1, 1])
# HLS matrix width (input features)
@pytest.mark.parametrize("mw", [4])
# HLS matrix height (output features)
@pytest.mark.parametrize("mh", [4])
def test_fpgadataflow_fclayer(idt, wdt, act, nf, sf, mw, mh):
    if nf == -1:
        nf = mh
    if sf == -1:
        sf = mw
    pe = mh // nf
    simd = mw // sf
    assert mh % pe == 0
    assert mw % sf == 0
    # generate weights
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    # generate input data
    x = gen_finn_dt_tensor(idt, (1, mw))
    if act is None:
        # no activation, produce accumulators
        T = None
        tdt = None
        if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
            odt = DataType.UINT32
        else:
            odt = DataType.INT32
    else:
        odt = act
        (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
        # generate thresholds for activation
        if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
            tdt = DataType.UINT32
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType.INT32
    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    model = model.transform(CodeGen())
    model = model.transform(Compile())
    # prepare input data
    input_dict = prepare_inputs(model, x, idt)
    if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
        # convert inputs to binary and use xnorpopcountmatmul
        y = xp.xnorpopcountmatmul((x + 1) / 2, (W + 1) / 2)
    else:
        y = np.matmul(x, W)
    if T is not None:
        y = multithreshold(y, T)
        if act == DataType.BIPOLAR:
            # binary to bipolar
            y = 2 * y - 1
        else:
            # signed offset
            y -= act.min()
    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced.reshape(y_expected.shape) == y_expected).all()
    model = model.transform(CleanUp())
