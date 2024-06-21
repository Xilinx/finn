# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_single_dwc_modelwrapper(shape, inWidth, outWidth, finn_dtype, impl_style):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    optype = "StreamingDataWidthConverter"

    DWC_node = helper.make_node(
        optype,
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        shape=shape,
        inWidth=inWidth,
        outWidth=outWidth,
        dataType=str(finn_dtype.name),
        preferred_impl_style=impl_style,
    )

    graph = helper.make_graph(nodes=[DWC_node], name="dwc_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="dwc-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def prepare_inputs(input_tensor, dt):
    return {"inp": input_tensor}


@pytest.mark.parametrize(
    "config",
    [
        ([1, 24], 6, 4, DataType["INT2"]),
        ([1, 24], 4, 6, DataType["INT2"]),
        ([1, 4], 2, 4, DataType["BIPOLAR"]),
        ([1, 4], 4, 2, DataType["INT2"]),
        ([1, 2, 8], 4, 4, DataType["INT2"]),
        ([1, 2, 8], 8, 16, DataType["INT2"]),
    ],
)
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_dwc(config, exec_mode, impl_style):
    shape, inWidth, outWidth, finn_dtype = config

    test_fpga_part = "xc7z020clg400-1"
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_dwc_modelwrapper(shape, inWidth, outWidth, finn_dtype, impl_style)
    # verify abstraction level execution
    y = oxe.execute_onnx(model, input_dict)["outp"]
    assert (
        y == x
    ).all(), """The output values are not the same as the
        input values anymore."""
    assert y.shape == tuple(shape), """The output shape is incorrect."""

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert (
        y == x
    ).all(), """The output values are not the same as the
        input values anymore."""
    assert y.shape == tuple(shape), """The output shape is incorrect."""


@pytest.mark.parametrize(
    "config",
    [
        ([1, 4], 2, 4, DataType["BIPOLAR"]),
        ([1, 4], 4, 2, DataType["INT2"]),
        ([1, 2, 8], 4, 4, DataType["INT2"]),
        ([1, 2, 8], 8, 16, DataType["INT2"]),
    ],
)
@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_dwc_stitched_rtlsim(config, impl_style):
    shape, inWidth, outWidth, finn_dtype = config

    test_fpga_part = "xc7z020clg400-1"
    target_clk_ns = 10.0
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_dwc_modelwrapper(shape, inWidth, outWidth, finn_dtype, impl_style)
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")
    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert (
        y == x
    ).all(), """The output values are not the same as the
        input values anymore."""
    assert y.shape == tuple(shape), """The output shape is incorrect."""
