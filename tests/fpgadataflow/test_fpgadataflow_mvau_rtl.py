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
from qonnx.transformation.general import ApplyConfig, GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.fpgadataflow.specialize_to_rtl_layers as to_rtl
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

build_dir = os.environ["FINN_BUILD_DIR"]


def make_single_matmul_modelwrapper(W, ofm_shape, mh, ifm, weights, idt, wdt):
    (ofm_h, ofm_w) = ofm_shape
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, (1, ofm_h, ofm_w, mh))

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
    return {"ifm": input_tensor}


@pytest.mark.parametrize("mh", [4])
# @pytest.mark.parametrize("mw", [36])
@pytest.mark.parametrize("mw", [18])
# @pytest.mark.parametrize("pe", [1,2,4,8])
@pytest.mark.parametrize("pe", [2])
# @pytest.mark.parametrize("simd", [1,3,6,9,18,36])
@pytest.mark.parametrize("simd", [6])
# @pytest.mark.parametrize("idt", [DataType["UINT4"], DataType["UINT8"]])
@pytest.mark.parametrize("idt", [DataType["UINT8"]])
# @pytest.mark.parametrize("wdt", [DataType["INT4"], DataType["INT6"]])
@pytest.mark.parametrize("wdt", [DataType["INT8"]])
# @pytest.mark.parametrize("part", ["xcvm1802-vsvd1760-2MP-e-S", "xcku3p-ffva676-1-e"])
# @pytest.mark.parametrize("part", ["xcvm1802-vsvd1760-2MP-e-S"])
@pytest.mark.parametrize("part", ["xcvc1902-vsva2197-2MP-e-S"])
@pytest.mark.parametrize("segmentlen", [1])
@pytest.mark.parametrize("double_pumped", [1, 0])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_mvau_rtl(mh, mw, pe, simd, idt, wdt, part, segmentlen, double_pumped):
    # Synthesis constants
    clk_ns = 5
    # Create test input vector (produced by SWG)
    ofm_shape = (5, 5)
    ofm_h, ofm_w = ofm_shape
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, ofm_h, ofm_w, mw])
    weights = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [mw, mh])
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    # np.save("weights.npy", W)
    ##
    # W = np.load("weights.npy")
    model = make_single_matmul_modelwrapper(W, ofm_shape, mh, ifm, weights, idt, wdt)
    model = model.transform(GiveUniqueNodeNames())

    model.save(build_dir + "/matmul.onnx")

    # Create MatMul & obtain golden reference output
    A = gen_finn_dt_tensor(model.get_tensor_datatype("ifm"), model.get_tensor_shape("ifm"))
    # np.save("activations.npy", A)
    ##
    # A = np.load("activations.npy")
    input_dict = prepare_inputs(A)

    # Execute ONNX model
    output_matmul = oxe.execute_onnx(model, input_dict)

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
            # "preferred_backend" : "rtl"
        },
    }
    model = model.transform(ApplyConfig(folding_config))
    model.save(build_dir + "/mvau_hls.onnx")

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    for n in model.graph.node:
        getCustomOp(n).set_nodeattr("rtlsim_trace", "mvu_trace_hls.vcd")
    output_mvau_hls = oxe.execute_onnx(model, input_dict)["ofm"]

    # Apply convert-to-rtl step
    model = model.transform(to_rtl.InferRTLMatrixVectorActivation())
    model = model.transform(GiveUniqueNodeNames())
    for n in model.graph.node:
        if n.op_type == "MatrixVectorActivation_rtl":
            getCustomOp(n).set_nodeattr("pumpedCompute", double_pumped)
    model.save(build_dir + "/mvau_rtl.onnx")

    # Reset rtlsim_so and ip-related paths such that new Pyverilator SO and IP is generated
    for n in model.graph.node:
        getCustomOp(n).set_nodeattr("rtlsim_so", "")
        getCustomOp(n).set_nodeattr("code_gen_dir_ipgen", "")
        getCustomOp(n).set_nodeattr("ipgen_path", "")
        getCustomOp(n).set_nodeattr("ip_path", "")
        getCustomOp(n).set_nodeattr("rtlsim_trace", "mvu_trace_rtl.vcd")
    #    model = model.transform(SetExecMode("rtlsim"))
    #    model = model.transform(PrepareIP(part, clk_ns))
    #    model = model.transform(HLSSynthIP())
    #    model = model.transform(PrepareRTLSim())
    #    output_mvau_rtl = oxe.execute_onnx(model, input_dict)["ofm"]

    model.save(build_dir + "/mvau_rtl_sim.onnx")

    #    with open(build_dir + "/hls_output.pkl", "wb") as f:
    #        pickle.dump(output_mvau_hls, f)

    #    with open(build_dir + "/rtl_output.pkl", "wb") as f:
    #        pickle.dump(output_mvau_rtl, f)

    model = model.transform(PrepareIP(part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(fpgapart=part, clk_ns=clk_ns, vitis=True))

    model.set_metadata_prop("exec_mode", "rtlsim")
    model.save(build_dir + "/stitched_ip.onnx")
    np.save("input.npy", A)
    output_mvau_rtl = oxe.execute_onnx(model, input_dict)["ofm"]
    # model.save(build_dir+"/stitched_ip.onnx")

    assert (output_mvau_hls == output_mvau_rtl).all()


#    assert (output_matmul['ofm'] == output_mvau_rtl).all()
# assert (output_mvau_hls.size > 0)
