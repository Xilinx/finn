# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

build_dir = os.environ["FINN_BUILD_DIR"]
test_fpga_part = "xc7z020clg400-1"
target_clk_ns = 10


def make_single_fifo_modelwrapper(Shape, Depth, fld_shape, finn_dtype):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, Shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, Shape)

    FIFO_node = helper.make_node(
        "StreamingFIFO",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        depth=Depth,
        folded_shape=fld_shape,
        normal_shape=Shape,
        dataType=str(finn_dtype.name),
    )

    graph = helper.make_graph(nodes=[FIFO_node], name="fifo_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="fifo-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def prepare_inputs(input_tensor, dt):
    return {"inp": input_tensor}


def test_fpgadataflow_fifo_rtl_ipi_deduplicates_helpers_and_retries_module_resolution():
    model = make_single_fifo_modelwrapper([1, 8], 2, [1, 1, 8], DataType["INT2"])
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    fifo = getCustomOp(model.graph.node[0])
    fifo.set_nodeattr("code_gen_dir_ipgen", os.path.join(build_dir, "fifo_ipgen"))
    fifo.set_nodeattr("gen_top_module", "StreamingFIFO_rtl_0")

    ipi_commands = fifo.code_generation_ipi()
    shared_source_command = ipi_commands[0]
    assert "::finn_streamingfifo_rtl_shared_sources_added" in shared_source_command
    assert shared_source_command.count("add_files") == 2
    assert ipi_commands[1].endswith("StreamingFIFO_rtl_0.v")

    create_cell_command = ipi_commands[-1]

    assert "catch {create_bd_cell" in create_cell_command
    assert "update_compile_order -fileset sources_1" in create_cell_command
    assert create_cell_command.count("create_bd_cell") == 2


# shape
@pytest.mark.parametrize("Shape", [[1, 128]])
# inWidth
@pytest.mark.parametrize("folded_shape", [[1, 1, 128]])
# outWidth
@pytest.mark.parametrize("depth", [16])
# finn_dtype
@pytest.mark.parametrize("finn_dtype", [DataType["BIPOLAR"], DataType["INT2"]])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_fifo_rtlsim(Shape, folded_shape, depth, finn_dtype):
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, Shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_fifo_modelwrapper(Shape, depth, folded_shape, finn_dtype)
    model = model.transform(SpecializeLayers(test_fpga_part))

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    y = oxe.execute_onnx(model, input_dict)["outp"]
    assert (
        y == x
    ).all(), """The output values are not the same as the
       input values anymore."""
    assert y.shape == tuple(Shape), """The output shape is incorrect."""
