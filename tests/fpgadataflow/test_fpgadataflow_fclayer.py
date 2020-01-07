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
from finn.transformation.fpgadataflow.codegen_npysim import CodeGen_npysim
from finn.transformation.fpgadataflow.compile import Compile


def make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T=None, tdt=None):
    mw = W.shape[0]
    mh = W.shape[1]
    assert mh % pe == 0
    assert mw % simd == 0

    # there are two ways to implement bipolar weights and inputs for
    # StreamingFC:
    # - specify their datatypes as such
    # - specify their datatypes as BINARY as use binaryXnorMode
    if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
        # we'll internally convert weights/inputs to binary and specify the
        # datatypes as such, and also set the binaryXnorMode attribute to 1
        export_wdt = DataType.BINARY
        export_idt = DataType.BINARY
        binary_xnor_mode = 1
    else:
        export_wdt = wdt
        export_idt = idt
        binary_xnor_mode = 0

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType.BIPOLAR:
            actval = 0
        else:
            actval = odt.min()
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
        actval = 0
        no_act = 1
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
        inputDataType=export_idt.name,
        weightDataType=export_wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    if binary_xnor_mode:
        # convert bipolar to binary
        model.set_initializer("weights", (W + 1) / 2)
    else:
        model.set_initializer("weights", W)
    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)
    return model


def prepare_inputs(input_tensor, idt, wdt):
    if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
        # convert bipolar to binary
        return {"inp": (input_tensor + 1) / 2}
    else:
        return {"inp": input_tensor}


# activation: None or DataType
@pytest.mark.parametrize("act", [None, DataType.BIPOLAR, DataType.INT2])
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
        # provide non-decreasing thresholds
        T = np.sort(T, axis=1)
        # generate thresholds for activation
        if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
            tdt = DataType.UINT32
            # bias thresholds to be positive
            T = np.ceil((T + mw) / 2)
            assert (T >= 0).all()
        else:
            tdt = DataType.INT32
    model = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    model = model.transform(CodeGen_npysim())
    model = model.transform(Compile())
    # prepare input data
    input_dict = prepare_inputs(x, idt, wdt)
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
            y += act.min()
    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced.reshape(y_expected.shape) == y_expected).all()
    model = model.transform(CleanUp())
