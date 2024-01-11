# Copyright (C) 2022, Advanced Micro Devices, Inc.
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
import os
import pickle
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import ApplyConfig, GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.fpgadataflow.specialize_to_rtl_layers as to_rtl
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths


from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

build_dir = os.environ["FINN_BUILD_DIR"]


def make_single_matmul_modelwrapper(ifm, ofm, idt, wdt, W):
    matmul_node = helper.make_node("MatMul", ["ifm", "weights"], ["ofm"])
    graph = helper.make_graph(nodes=[matmul_node], name="matmul_graph", inputs=[ifm], outputs=[ofm])

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("ifm", idt)
    model.set_tensor_datatype("weights", wdt)
    model.set_tensor_datatype(
        "ofm", DataType["INT32"]
    )  # At this step, the MatMul layer does not optimize the bit-width of the output datatype
    model.set_initializer("weights", W)
    # model.set_tensor_layout("ifm", DataLayout.NHWC)

    return model


def prepare_inputs(input_tensor):
    return {"global_in": input_tensor}


# @pytest.mark.parametrize("mh", [36])
# @pytest.mark.parametrize("mw", [256])
@pytest.mark.parametrize("mh", [9])
@pytest.mark.parametrize("mw", [36])
# @pytest.mark.parametrize("pe", [1, 4, 9, 36])
# @pytest.mark.parametrize("simd", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("pe", [1, 3, 9])
@pytest.mark.parametrize("simd", [1, 3, 6, 18, 36])
@pytest.mark.parametrize("idt", [DataType["UINT4"], DataType["UINT8"]])
@pytest.mark.parametrize("wdt", [DataType["INT4"], DataType["INT8"]])
# @pytest.mark.parametrize("part", ["xcvc1902-vsva2197-2MP-e-S", "xcku3p-ffva676-1-e"])
@pytest.mark.parametrize("part", ["xcvc1902-vsva2197-2MP-e-S"])
@pytest.mark.parametrize("clk_ns", [1.66, 4])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_mvau_rtl(
    mh, mw, pe, simd, idt, wdt, part, clk_ns
):
    if part == "xcku3p-ffva676-1-e" and clk_ns != 1.66:
        pytest.skip("Skip test for varying clk for devices other than Versal, since this variable doesn't change anything for this test")

    # Create test input vector (produced by SWG)
    ofm_shape = (5, 5)
    ofm_h, ofm_w = ofm_shape
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, ofm_h, ofm_w, mw])
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, (1, ofm_h, ofm_w, mh))
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    model = make_single_matmul_modelwrapper(ifm, ofm, idt, wdt, W)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    model.save(build_dir + "/matmul.onnx")

    # Create MatMul & obtain golden reference output
    A = gen_finn_dt_tensor(model.get_tensor_datatype("global_in"), model.get_tensor_shape("global_in"))
    input_dict = prepare_inputs(A)

    # Execute ONNX model
    output_matmul = oxe.execute_onnx(model, input_dict)["global_out"]

    with open(build_dir + "/onnx_output.pkl", "wb") as f:
        pickle.dump(output_matmul, f)

    # Create MVAU (HLS)
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode="decoupled"))
    model = model.transform(GiveUniqueNodeNames())

    # Apply folding (i.e. specify to use DSPs)
    folding_config = {
        "Defaults": {},
        "MatrixVectorActivation_0": {
            "PE": pe,
            "SIMD": simd,
            "mem_mode": "decoupled",
            "ram_style": "auto",
            "resType": "dsp",
            "preferred_backend" : "rtl"
        },
    }
    model = model.transform(ApplyConfig(folding_config))
    model.save(build_dir + "/mvau_hls.onnx")

    # Apply convert-to-rtl step
    model = model.transform(to_rtl.InferRTLMatrixVectorActivation())
    model = model.transform(GiveUniqueNodeNames())
    model.save(build_dir + "/mvau_rtl.onnx")

    # Reset rtlsim_so and ip-related paths such that new Pyverilator SO and IP is generated
    for n in model.graph.node:
        getCustomOp(n).set_nodeattr("rtlsim_trace", build_dir + "/mvu_trace_rtl_nodebynode.vcd")
    
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    output_mvau_rtl = oxe.execute_onnx(model, input_dict)["global_out"]

    with open(build_dir + "/mvau_rtl_output.pkl", "wb") as f:
        pickle.dump(output_mvau_rtl, f)

    model.save(build_dir + "/mvau_rtl_sim.onnx")
    assert (output_matmul == output_mvau_rtl).all(), "Output of ONNX model not matching output of node-by-node sim!"

    model = model.transform(InsertAndSetFIFODepths(part, clk_ns))
    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(part, clk_ns))

    os.environ["RTLSIM_TRACE_DEPTH"] = "3"
    model.set_metadata_prop("rtlsim_so", "")
    model.set_metadata_prop("exec_mode", "rtlsim")
    model.set_metadata_prop("rtlsim_trace", build_dir + "/mvu_trace_rtl_stitch.vcd")
    model.save(build_dir + "/stitched_ip.onnx")
    output_mvau_rtl_stitch = oxe.execute_onnx(model, input_dict)["global_out"]

    assert (output_matmul == output_mvau_rtl_stitch).all(), "Output of ONNX model not matching output of stitched-IP RTL model!"