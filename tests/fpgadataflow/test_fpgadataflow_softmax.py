############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier:  BSD-3 Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import pytest

import numpy as np
import torch
import torch.nn as nn
from brevitas.export import export_qonnx
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferHWSoftmax
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.test import tree_model_test

test_fpga_part: str = "xcvc1902-vsva2197-2MP-e-S"
target_clk_ns = 5

tree_model_fpga_part = "xczu7ev-ffvc1156-2-e"
tree_model_clk_ns = 5.0


class SoftMaxSimple(nn.Module):
    def __init__(self):
        super(SoftMaxSimple, self).__init__()
        self.softmax = nn.Softmax(dim=-1)  # softmax along the last dimension

    def forward(self, x):
        x = self.softmax(x)
        return x


def create_softmax_model(io_shape, idt, build_dir):
    export_onnx_path = f"{build_dir}/pytest_softmax_dut.onnx"
    dut = SoftMaxSimple()
    input = torch.rand(io_shape)
    export_qonnx(dut, input, export_onnx_path, opset_version=11)
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model.set_tensor_datatype(model.graph.input[0].name, idt)
    return model


@pytest.mark.parametrize("simd", [1, 2, 4])
@pytest.mark.parametrize("sim_style", ["cppsim", "node_by_node", "stitched_ip"])
# Selected (idt, impl_style, ifm_dim) configs to cover key code paths:
# - Both impl_style (hls, rtl)
# - Different data types (HLS: integer only, RTL: integer + FLOAT32 passthrough)
# - Different input shapes (2D, 3D, 4D)
@pytest.mark.parametrize(
    "idt, impl_style, ifm_dim",
    [
        # HLS with integer inputs
        ("INT8", "hls", (1, 32, 96)),
        ("INT9", "hls", (1, 3, 32, 32)),
        # HLS with FLOAT32 (for non-Versal devices where RTL is not available)
        ("FLOAT32", "hls", (1, 32, 96)),
        # RTL with integer inputs (uses int_to_fp32 conversion)
        ("INT8", "rtl", (1, 32, 96)),
        ("INT9", "rtl", (1, 3, 16, 32)),
        ("INT4", "rtl", (4, 32)),
        ("UINT8", "rtl", (1, 3, 32, 32)),
        # RTL with FLOAT32 - passthrough path (no int_to_fp32 conversion)
        ("FLOAT32", "rtl", (1, 32, 96)),
        ("FLOAT32", "rtl", (1, 3, 16, 32)),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_hwsoftmax(simd, idt, impl_style, sim_style, ifm_dim):
    build_dir = make_build_dir(prefix="test_fpgadataflow_hwsoftmax_")
    try:
        _test_fpgadataflow_hwsoftmax(simd, idt, impl_style, sim_style, ifm_dim, build_dir)
    finally:
        robust_rmtree(build_dir)


def _test_fpgadataflow_hwsoftmax(simd, idt, impl_style, sim_style, ifm_dim, build_dir):
    idt = DataType[idt]
    io_shape = ifm_dim
    # tighter tolerance for HLS/cppsim, looser for RTL FP32 numerical drift
    if sim_style == "cppsim":
        tolerance = 1e-5
    elif impl_style == "rtl":
        tolerance = 2**-4
        rtol = 1e-3
    else:
        tolerance = 1e-5

    model = create_softmax_model(io_shape, idt, build_dir)

    input = gen_finn_dt_tensor(idt, io_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Create reference values using the qonnx model
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    # Infer HWSoftmax
    model = model.transform(InferHWSoftmax())

    # request the desired implementation style
    getCustomOp(model.graph.node[0]).set_nodeattr("preferred_impl_style", impl_style)

    # run the model (HWSoftmax base falls through to scipy reference)
    y_out = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_out, atol=1e-5), "Model output does not match expected output"

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())

    expected_op_type = f"HWSoftmax_{impl_style}"
    assert (
        model.graph.node[0].op_type == expected_op_type
    ), f"HWSoftmax wasn't converted to {expected_op_type}"

    # set SIMD post-specialize (matches layernorm test pattern)
    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", int(simd))

    if sim_style == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif sim_style == "node_by_node":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    elif sim_style == "stitched_ip":
        model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
        model.set_metadata_prop("exec_mode", "rtlsim")

    # run the model
    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    # verify expected vs actual cycles for node-by-node rtlsim
    if sim_style == "node_by_node":
        node = model.get_nodes_by_op_type(expected_op_type)[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, rtol=0.10)
        assert exp_cycles != 0

    if impl_style == "rtl" and sim_style != "cppsim":
        assert np.allclose(
            y_ref, y_hw, rtol=rtol, atol=tolerance
        ), "Model output does not match expected output"
    else:
        assert np.allclose(
            y_ref, y_hw, atol=tolerance
        ), "Model output does not match expected output"


def make_softmax_modelwrapper(ishape, simd, idt):
    """A graph holding one HWSoftmax node."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, list(ishape))
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, list(ishape))
    node = helper.make_node(
        "HWSoftmax",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ifm_dim=list(ishape),
        SIMD=simd,
        input_data_type=idt.name,
        NumChannels=ishape[-1],
        preferred_impl_style="hls",
    )
    graph = helper.make_graph(nodes=[node], name="softmax_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="softmax-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", DataType["FLOAT32"])
    return model


@pytest.mark.parametrize(
    "config",
    [
        ((1, 16, 64), 1),
        ((1, 16, 64), 2),
        ((1, 16, 64), 4),
        ((1, 16, 64), 8),
        ((1, 32, 96), 1),
        ((1, 8, 32), 1),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_softmax(config):
    ishape, simd = config

    model = make_softmax_modelwrapper(ishape, simd, DataType["INT8"])
    model = model.transform(SpecializeLayers(tree_model_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "HWSoftmax_hls"

    node_details = ("HWSoftmax", config)

    # The rate is exact, so the two curves stay on top of each other and the
    # volume budget is a floor: what is left is where in the vector the stored
    # window happens to start, which can never be worth more than one stall.
    #
    # The length budget is the pipeline fill, which the period leaves out on
    # purpose -- folding it in would slow the rate below the design's and the
    # curves would drift apart, which is the error that matters. The fill is two
    # vectors of inter-stage FIFO plus the floating-point latency, so it is a
    # tenth of the recorded period at the shortest frames here and a fortieth at
    # the longest, and it is a fraction rather than a floor for that reason.
    max_allowed_volume_frac = 0.0
    volume_const = 5
    max_allowed_length_frac = 0.10
    length_const = 0

    assert tree_model_test(
        model,
        node_details,
        tree_model_fpga_part,
        tree_model_clk_ns,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    ), "characterized TAV does not match RTLsim'd one!"
