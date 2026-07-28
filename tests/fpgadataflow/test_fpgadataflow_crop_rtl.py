# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferCrop
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10


def make_crop_modelwrapper(ishape, axis, indices, idt):
    indices = np.asarray(indices, dtype=np.int64)
    oshape = list(ishape)
    oshape[axis] = len(indices)
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)
    gather = helper.make_node("Gather", ["inp", "indices"], ["outp"], axis=axis)
    graph = helper.make_graph(
        [gather],
        "crop-model",
        [inp],
        [outp],
        [numpy_helper.from_array(indices, name="indices")],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="crop-model"))
    model = model.transform(InferShapes())
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    return model


def infer_and_specialize_crop(model, simd):
    model = model.transform(InferCrop())
    crop_nodes = model.get_nodes_by_op_type("Crop")
    assert len(crop_nodes) == 1
    crop = getCustomOp(crop_nodes[0])
    crop.set_nodeattr("preferred_impl_style", "rtl")
    crop.set_nodeattr("SIMD", simd)
    model = model.transform(SpecializeLayers(FPGA_PART))
    assert len(model.get_nodes_by_op_type("Crop_rtl")) == 1
    return model


def prepare_crop_stitched_ip_model(model):
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    return model.transform(CreateStitchedIP(FPGA_PART, CLK_NS))


# Input shape, Gather axis/indices, and SIMD are coupled to cover both spatial axes,
# all channel-folding extremes, and the special 2D input representation.
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(([3, 4, 4], 0, [1, 2], 1), id="3d-height-simd1"),
        pytest.param(([3, 4, 4], 1, [1, 2], 2), id="3d-width-simd2"),
        pytest.param(([4, 4], 0, [0, 1, 2], 4), id="2d-width-simd4"),
    ],
)
@pytest.mark.parametrize(
    "idt",
    [
        pytest.param(DataType["INT8"], id="INT8"),
        pytest.param(DataType["UINT4"], id="UINT4"),
        pytest.param(DataType["INT6"], id="INT6"),
    ],
)
@pytest.mark.parametrize("exec_mode", ["rtlsim", "stitched_rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_crop_rtl(config, idt, exec_mode):
    ishape, axis, indices, simd = config
    input_tensor = gen_finn_dt_tensor(idt, ishape)
    input_dict = {"inp": input_tensor}
    y_expected = np.take(input_tensor, indices, axis=axis)
    model = make_crop_modelwrapper(ishape, axis, indices, idt)

    # Golden reference from the original Gather graph.
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all(), "Execution of Gather model failed"

    model = infer_and_specialize_crop(model, simd)

    if exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    elif exec_mode == "stitched_rtlsim":
        model = prepare_crop_stitched_ip_model(model)
        model.set_metadata_prop("exec_mode", "rtlsim")
    else:
        raise ValueError("Unknown exec_mode")

    y_produced = oxe.execute_onnx(model, input_dict)["outp"].reshape(y_expected.shape)
    assert (y_produced == y_expected).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("Crop_rtl")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0
