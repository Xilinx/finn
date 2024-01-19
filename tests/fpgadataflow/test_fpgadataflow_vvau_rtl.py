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
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.fpgadataflow.specialize_to_rtl_layers as to_rtl
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths

# import qonnx.core.data_layout as DataLayout

build_dir = os.environ["FINN_BUILD_DIR"]


def make_single_dw_conv_modelwrapper(conv_config, idt, wdt):
    kernel_size, in_feature_dim, in_chn = conv_config
    stride = 1
    pad = 0

    out_feature_dim = compute_conv_output_dim(in_feature_dim, kernel_size, stride, pad)
    group = out_chn = in_chn

    conv_param_shape = [out_chn, 1, kernel_size, kernel_size]
    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    output_shape = [1, out_chn, out_feature_dim, out_feature_dim]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = group
    conv_config["kernel_shape"] = [kernel_size, kernel_size]
    conv_config["pads"] = [pad, pad, pad, pad]
    conv_config["strides"] = [stride, stride]

    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, input_shape)
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, output_shape)
    weights = [helper.make_tensor_value_info("weights", TensorProto.FLOAT, conv_param_shape)]

    modelproto = qonnx_make_model(
        helper.make_graph(
            name="conv_test",
            inputs=[ifm],
            outputs=[ofm],
            value_info=weights,
            nodes=[helper.make_node("Conv", ["ifm", "weights"], ["ofm"], **conv_config)],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("ifm", idt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", gen_finn_dt_tensor(wdt, conv_param_shape))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def prepare_inputs(input_tensor):
    return {"global_in": input_tensor}


@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("in_feature_dim", [5])
@pytest.mark.parametrize("in_chn", [4])
@pytest.mark.parametrize("idt", [DataType["UINT8"]])
# @pytest.mark.parametrize("idt", [DataType["UINT8"]])
@pytest.mark.parametrize("wdt", [DataType["INT8"]])
@pytest.mark.parametrize("part", ["xcvc1902-vsva2197-2MP-e-S"])
@pytest.mark.parametrize("segmentlen", [1])
@pytest.mark.parametrize("pe", [2])
@pytest.mark.parametrize("simd", [3])
@pytest.mark.parametrize("double_pumped_compute", [1])
@pytest.mark.parametrize("double_pumped_memory", [0])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_vvau_rtl(
    kernel_size,
    in_feature_dim,
    in_chn,
    idt,
    wdt,
    part,
    segmentlen,
    pe,
    simd,
    double_pumped_compute,
    double_pumped_memory,
):
    # Create depthwise-separable convolution
    conv_config = (kernel_size, in_feature_dim, in_chn)
    np.random.seed(0)
    model = make_single_dw_conv_modelwrapper(conv_config, idt, wdt)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/dw_conv.onnx")

    # Obtain golden reference output
    golden_in = gen_finn_dt_tensor(
        model.get_tensor_datatype("global_in"), model.get_tensor_shape("global_in")
    )
    input_dict = prepare_inputs(golden_in)

    # Convert to HLS custom-op first
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/lowered_conv.onnx")
    golden_out = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
    np.savez(build_dir + "/lowered_conv_out.npz", **golden_out)
    golden_ret = golden_out["global_out"].transpose(0, 2, 3, 1)

    model = model.transform(to_hls.InferConvInpGen(use_rtl_variant=True))
    model = model.transform(to_hls.InferVectorVectorActivation())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/hls_vvau.onnx")

    # Apply folding (i.e. specify to use DSPs)
    folding_config = {
        "Defaults": {},
        "ConvolutionInputGenerator_rtl_0": {
            "SIMD": pe,
            "parallel_window": 1,
            "rtlsim_trace": build_dir + "/swg_trace.vcd",
        },
        "VectorVectorActivation_0": {
            "PE": pe,
            "SIMD": simd,
            "mem_mode": "decoupled",
            "ram_style": "auto",
            "resType": "dsp",
            "preferred_backend": "rtl",
            "rtlsim_trace": build_dir + "/vvu_trace.vcd",
        },
    }
    model = model.transform(ApplyConfig(folding_config))
    model.save(build_dir + "/hls_vvau_folded.onnx")

    # Stitched-IP RTLsim
    model = model.transform(CreateDataflowPartition(partition_model_dir=build_dir))
    model.save(build_dir + "/ip-stitched.onnx")
    partition_model_path = getCustomOp(
        model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    ).get_nodeattr("model")
    partitioned_model = ModelWrapper(partition_model_path)
    # FIFOs needed for stitched-ip RTLsim, DWC needed for VVU operating on SIMD parallelism

    partitioned_model = partitioned_model.transform(PrepareIP(part, 5))

    partitioned_model.save(build_dir + "/partitioned_model.onnx")

    # Apply convert-to-rtl step
    partitioned_model = partitioned_model.transform(to_rtl.InferRTLVectorVectorActivation())
    partitioned_model = partitioned_model.transform(GiveUniqueNodeNames())
    partitioned_model = partitioned_model.transform(GiveReadableTensorNames())
    for n in partitioned_model.graph.node:
        if n.op_type == "VectorVectorActivation_rtl":
            getCustomOp(n).set_nodeattr("pumpedCompute", double_pumped_compute)
            getCustomOp(n).set_nodeattr("pumpedMemory", double_pumped_memory)
            getCustomOp(n).set_nodeattr("rtlsim_trace", build_dir + "/vvu_trace.vcd")
    partitioned_model = partitioned_model.transform(PrepareIP(part, 5))
    partitioned_model = partitioned_model.transform(HLSSynthIP())
    partitioned_model = partitioned_model.transform(PrepareRTLSim())
    partitioned_model = partitioned_model.transform(SetExecMode("rtlsim"))
    input_dict = {"global_in": golden_in.transpose(0, 2, 3, 1)}
    vvu_rtl_nodebynode_out = oxe.execute_onnx(
        partitioned_model, input_dict, return_full_exec_context=True
    )
    np.savez(build_dir + "/vvurtl_nodebynode_out.npz", **vvu_rtl_nodebynode_out)
    partitioned_model.save(build_dir + "/vvurtl_nodebynode.onnx")
    vvu_rtl_nodebynode_ret = vvu_rtl_nodebynode_out["global_out"]

    assert (
        vvu_rtl_nodebynode_ret == golden_ret
    ).all(), "Output of ONNX model not matching output of node-by-node sim!"

    partitioned_model = partitioned_model.transform(InsertAndSetFIFODepths(part, 5))
    partitioned_model = partitioned_model.transform(PrepareIP(part, 5))
    partitioned_model = partitioned_model.transform(HLSSynthIP())
    partitioned_model = partitioned_model.transform(CreateStitchedIP(part, 5))
    partitioned_model.save(build_dir + "/partition_rtl_vvau.onnx")
    partitioned_model.set_metadata_prop("rtlsim_trace", build_dir + "/rtl-vvu-stitched.vcd")
    # Reset rtlsim_so path to re-generate Pyverilator sim object
    partitioned_model.set_metadata_prop("rtlsim_so", "")
    # set top-level prop for stitched-ip rtlsim and launch
    partitioned_model.set_metadata_prop("exec_mode", "rtlsim")

    vvu_rtl_out = oxe.execute_onnx(partitioned_model, input_dict, return_full_exec_context=True)
    with open(build_dir + "/rtl_vvau_output.pkl", "wb") as f:
        pickle.dump(vvu_rtl_out, f)

    vvu_rtl_ret = vvu_rtl_out["global_out"]

    assert (
        vvu_rtl_ret == golden_ret
    ).all(), "Output of ONNX model not matching output of stitched-IP RTL model!"
