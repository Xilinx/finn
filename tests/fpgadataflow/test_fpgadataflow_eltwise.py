import pytest

import numpy as np
import onnx.parser as oprs
import qonnx.core.data_layout as dl
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


def build_model(shp, dt0, dt1, do_abs):
    np.random.seed(0)
    shp_str = str(shp)
    if do_abs:
        graph = """
        sub_out = Sub(in0, in1)
        out0 = Abs(sub_out)
        """
    else:
        graph = "out0 = Sub(in0, in1)"

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{shp_str} in0, float{shp_str} in1) => (float{shp_str} out0)
    {{
        {graph}
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_tensor_datatype("in0", dt0)
    model.set_tensor_datatype("in1", dt1)
    model.set_tensor_layout("in0", dl.NHWC)
    model.set_tensor_layout("in1", dl.NHWC)
    model = model.transform(InferShapes())
    return model


# input datatype for one operand
@pytest.mark.parametrize("dt0", [DataType["UINT4"], DataType["UINT7"]])
# channels
@pytest.mark.parametrize("ch", [1, 64])
# folding
@pytest.mark.parametrize("fold", [-1, 2, 1])
# include Abs output node or not
@pytest.mark.parametrize("do_abs", [True, False])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_eltwise(dt0, ch, fold, do_abs, exec_mode):
    if fold == -1:
        pe = 1
    else:
        pe = max(1, ch // fold)
    assert ch % pe == 0
    dt1 = DataType["UINT8"]
    shp = [1, 4, 2, ch]
    model = build_model(shp, dt0, dt1, do_abs)
    in0 = gen_finn_dt_tensor(dt0, shp)
    in1 = gen_finn_dt_tensor(dt1, shp)
    idict = {"in0": in0, "in1": in1}
    y_expected = execute_onnx(model, idict)["out0"]
    model = model.transform(to_hls.InferStreamingEltwise())
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "StreamingEltwise"
    getCustomOp(model.graph.node[0]).set_nodeattr("PE", pe)
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")
    y_produced = execute_onnx(model, idict)["out0"]
    assert (y_produced == y_expected).all(), exec_mode + " failed"
    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("StreamingEltwise")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0
