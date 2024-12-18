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

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import ApplyConfig, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def save_model(model, name):
    model.save("dynMM_debug_" + name + ".onnx")


#
# Init
#


def make_dynamic_matmul_modelwrapper(inp_A, inp_B, out_Y, A_dtype, B_dtype):
    A_vi = helper.make_tensor_value_info("inp_A", TensorProto.FLOAT, inp_A)
    B_vi = helper.make_tensor_value_info("inp_B", TensorProto.FLOAT, inp_B)
    outp_tensor_value_info = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_Y)

    matmul_node = helper.make_node("MatMul", ["inp_A", "inp_B"], ["outp"])
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="matmul_graph_2_inputs",
        inputs=[A_vi, B_vi],
        outputs=[outp_tensor_value_info],
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp_A", A_dtype)
    model.set_tensor_datatype("inp_B", B_dtype)
    model.set_tensor_datatype("outp", DataType["INT32"])

    return model


# matrix size (1xHxMxK * 1xHxKxN) = 1xHxMxN
@pytest.mark.parametrize("inp_A", [(1, 12, 128, 32)])
@pytest.mark.parametrize("inp_B", [(1, 12, 32, 16)])
@pytest.mark.parametrize("A_dtype", [DataType["INT8"]])
@pytest.mark.parametrize("B_dtype", [DataType["INT8"]])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_infer_dyn_mvau(inp_A, inp_B, A_dtype, B_dtype):
    """
    This test generates a MatMul Onnx graph, and then applies transformations
    """
    out_Y = [1, 12, 128, 16]
    model = make_dynamic_matmul_modelwrapper(inp_A, inp_B, out_Y, A_dtype, B_dtype)
    model = model.transform(GiveUniqueNodeNames())
    # Create MatMul & obtain golden reference output
    inpTensor_A = gen_finn_dt_tensor(
        model.get_tensor_datatype("inp_A"), model.get_tensor_shape("inp_A")
    )
    inpTensor_B = gen_finn_dt_tensor(
        model.get_tensor_datatype("inp_B"), model.get_tensor_shape("inp_B")
    )
    input_dict = {"inp_A": inpTensor_A, "inp_B": inpTensor_B}
    # Execute ONNX model
    output_matmul = oxe.execute_onnx(model, input_dict)["outp"]

    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    save_model(model, "infer.onnx")
    output_mvau = oxe.execute_onnx(model, input_dict)["outp"]

    assert np.allclose(
        output_matmul, output_mvau
    ), "Output of ONNX model not matching output of MVAU!"


# matrix size (1xHxMxK * 1xHxKxN) = 1xHxMxN
@pytest.mark.parametrize("inp_A", [(1, 12, 128, 32)])
@pytest.mark.parametrize("inp_B", [(1, 12, 32, 16)])
@pytest.mark.parametrize("pe", [1, 2, 4])
@pytest.mark.parametrize("simd", [1, 4, 8])
@pytest.mark.parametrize("A_dtype", [DataType["INT8"]])
@pytest.mark.parametrize("B_dtype", [DataType["INT8"]])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_dynamic_mvau_rtlsim(inp_A, inp_B, pe, simd, A_dtype, B_dtype):
    """
    This test generates a MatMul Onnx graph, and then applies transformations
    """
    part = "xcvc1902-vsva2197-2MP-e-S"
    clk_ns = 4

    out_Y = [1, 12, 128, 16]
    model = make_dynamic_matmul_modelwrapper(inp_A, inp_B, out_Y, A_dtype, B_dtype)
    model = model.transform(GiveUniqueNodeNames())
    # Create MatMul & obtain golden reference output
    inpTensor_A = gen_finn_dt_tensor(
        model.get_tensor_datatype("inp_A"), model.get_tensor_shape("inp_A")
    )
    inpTensor_B = gen_finn_dt_tensor(
        model.get_tensor_datatype("inp_B"), model.get_tensor_shape("inp_B")
    )
    input_dict = {"inp_A": inpTensor_A, "inp_B": inpTensor_B}
    # Execute ONNX model for reference values
    output_matmul = oxe.execute_onnx(model, input_dict)["outp"]

    # Begin FINN Transformations
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(GiveUniqueNodeNames())

    for node in model.graph.node:
        # lookup op_type in registry of CustomOps
        inst = getCustomOp(node)
        inst.set_nodeattr("preferred_impl_style", "rtl")
        inst.set_nodeattr("mem_mode", "external")
        inst.set_nodeattr("rtlsim_backend", "pyverilator")
        inst.set_nodeattr("inFIFODepths", [16, 16])
    # Apply convert-to-rtl step
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())

    # Apply folding (i.e. specify to use DSPs)
    folding_config = {
        "Defaults": {},
        "MVAU_rtl_0": {
            "PE": pe,
            "SIMD": simd,
            "resType": "dsp",
        },
    }
    model = model.transform(ApplyConfig(folding_config))
    model = model.transform(InferDataTypes())
    # Run node-by-node RTLsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    model.set_metadata_prop("rtlsim_backend", "pyverilator")

    output_mvau_rtl = oxe.execute_onnx(model, input_dict)["outp"]
    assert (
        output_matmul == output_mvau_rtl
    ).all(), "Output of ONNX model not matching output of node-by-node RTLsim!"


# matrix size (1xHxMxK * 1xHxKxN) = 1xHxMxN
@pytest.mark.parametrize("inp_A", [(1, 12, 128, 32)])
@pytest.mark.parametrize("inp_B", [(1, 12, 32, 16)])
@pytest.mark.parametrize("pe", [1, 2, 4])
@pytest.mark.parametrize("simd", [1, 4, 8])
@pytest.mark.parametrize("A_dtype", [DataType["INT8"]])
@pytest.mark.parametrize("B_dtype", [DataType["INT8"]])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_rtl_dynamic_mvau(inp_A, inp_B, pe, simd, A_dtype, B_dtype):
    """
    This test generates a MatMul Onnx graph, and then applies transformations
    """
    part = "xcvc1902-vsva2197-2MP-e-S"
    clk_ns = 4

    out_Y = [1, 12, 128, 16]
    model = make_dynamic_matmul_modelwrapper(inp_A, inp_B, out_Y, A_dtype, B_dtype)
    model = model.transform(GiveUniqueNodeNames())
    # Create MatMul & obtain golden reference output
    inpTensor_A = gen_finn_dt_tensor(
        model.get_tensor_datatype("inp_A"), model.get_tensor_shape("inp_A")
    )
    inpTensor_B = gen_finn_dt_tensor(
        model.get_tensor_datatype("inp_B"), model.get_tensor_shape("inp_B")
    )
    input_dict = {"inp_A": inpTensor_A, "inp_B": inpTensor_B}
    # Execute ONNX model for reference values
    output_matmul = oxe.execute_onnx(model, input_dict)["outp"]

    # Begin FINN Transformations
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(GiveUniqueNodeNames())

    for node in model.graph.node:
        # lookup op_type in registry of CustomOps
        inst = getCustomOp(node)
        inst.set_nodeattr("preferred_impl_style", "rtl")
        inst.set_nodeattr("mem_mode", "external")
        inst.set_nodeattr("rtlsim_backend", "pyverilator")
        inst.set_nodeattr("inFIFODepths", [16, 16])
    # Apply convert-to-rtl step
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())

    # Apply folding (i.e. specify to use DSPs)
    folding_config = {
        "Defaults": {},
        "MVAU_rtl_0": {
            "PE": pe,
            "SIMD": simd,
            "resType": "dsp",
        },
    }
    model = model.transform(ApplyConfig(folding_config))
    model = model.transform(InferDataTypes())

    # Run stitched-ip RTLsim
    model = model.transform(InsertFIFO(True))
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(part, clk_ns))
    model = model.transform(PrepareRTLSim())
    model = model.transform(SetExecMode("rtlsim"))
    model.set_metadata_prop("exec_mode", "rtlsim")
    model.set_metadata_prop("rtlsim_backend", "pyverilator")

    output_mvau_rtl_stitch = oxe.execute_onnx(model, input_dict)["outp"]

    assert (
        output_matmul == output_mvau_rtl_stitch
    ).all(), "Output of ONNX model not matching output of stitched-IP RTL model!"

    return 0
