# Copyright (c) 2020, Xilinx
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
import tempfile
import torch
from brevitas.export import FINNManager
from torch import nn

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.base import Transformation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hls_layers import InferUpsample
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes


class ForceDataTypeForTensors(Transformation):
    """
    Forces a certain datatype for all tensors in a model.
    """

    def __init__(self, dType=DataType.INT8):
        super().__init__()
        self._dType = dType

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            for inp in n.input:
                model.set_tensor_datatype(inp, self._dType)
            for inp in n.output:
                model.set_tensor_datatype(inp, self._dType)

        return model, False


_to_chan_last_args = (0, 2, 3, 1)
_to_chan_first_args = (0, 3, 1, 2)


class TransposeUpsampleIO(Transformation):
    """
    Converts the inputs outputs for all Upsample and Resize nodes
    from NCHW to NHWC.
    """

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            if n.op_type == "Upsample" or n.op_type == "Resize":
                # Set input shape
                inp = n.input[0]
                NCHW_shape = model.get_tensor_shape(inp)
                NHWC_shape = [NCHW_shape[idx] for idx in _to_chan_last_args]
                model.set_tensor_shape(inp, NHWC_shape)
                # Set output shape
                out = n.output[0]
                NCHW_shape = model.get_tensor_shape(out)
                NHWC_shape = [NCHW_shape[idx] for idx in _to_chan_last_args]
                model.set_tensor_shape(out, NHWC_shape)
        return model, False


class PyTorchTestModel(nn.Module):
    def __init__(self, upscale_factor=2):
        super(PyTorchTestModel, self).__init__()
        self.m = nn.Upsample(
            scale_factor=upscale_factor,
            mode="nearest",
        )

    def forward(self, x):
        x = self.m(x)
        return x


# param datatype
@pytest.mark.parametrize("dt", [DataType.INT8])
# Width/height of square input feature map
@pytest.mark.parametrize("IFMDim", [3, 5])
# Width/height of square output feature map
@pytest.mark.parametrize("OFMDim", [6, 7, 14])
# Number of input/output channels
@pytest.mark.parametrize("NumChannels", [1, 4])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_upsampler(dt, IFMDim, OFMDim, NumChannels, exec_mode):
    tmpdir = tempfile.mkdtemp()
    atol = 1e-3
    # Create the test model and inputs for it
    torch_model = PyTorchTestModel(upscale_factor=OFMDim / IFMDim)
    input_shape = (1, NumChannels, IFMDim, IFMDim)
    test_in = torch.arange(0, np.prod(np.asarray(input_shape)))
    # Limit the input to values valid for the given datatype
    test_in %= dt.max() - dt.min() + 1
    test_in += dt.min()
    # Additionally make sure we always start with 0, for convenience purposes.
    test_in = torch.roll(test_in, dt.min())
    test_in = test_in.view(*input_shape).type(torch.float32)

    # Get golden PyTorch and ONNX inputs
    golden_torch_float = torch_model(test_in)
    export_path = f"{tmpdir}/Upsample_exported.onnx"
    FINNManager.export(
        torch_model, input_shape=input_shape, export_path=export_path, opset_version=11
    )
    model = ModelWrapper(export_path)
    input_dict = {model.graph.input[0].name: test_in.numpy().astype(np.int)}
    input_dict = {model.graph.input[0].name: test_in.numpy()}
    golden_output_dict = oxe.execute_onnx(model, input_dict, True)
    golden_result = golden_output_dict[model.graph.output[0].name]

    # Make sure PyTorch and ONNX match
    pyTorch_onnx_match = np.isclose(golden_result, golden_torch_float).all()
    assert pyTorch_onnx_match, "ONNX and PyTorch upsampling output don't match."

    # Prep model for execution
    model = ModelWrapper(export_path)
    transpose_path = f"{tmpdir}/Upsample_transposed.onnx"
    model = model.transform(TransposeUpsampleIO())
    model.save(transpose_path)

    hls_upsample_path = f"{tmpdir}/Upsample_hls.onnx"
    model = ModelWrapper(transpose_path)
    model = model.transform(ForceDataTypeForTensors(dType=dt))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferUpsample())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model.save(hls_upsample_path)

    # Check that all nodes are UpsampleNearestNeighbour_Batch nodes
    for n in model.get_finn_nodes():
        node_check = n.op_type == "UpsampleNearestNeighbour_Batch"
        assert node_check, "All nodes should be UpsampleNearestNeighbour_Batch nodes."

    # Prep sim
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 10))
        model = model.transform(HLSSynthIP())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # Run sim
    test_in_transposed = test_in.numpy().transpose(_to_chan_last_args)
    input_dict = {model.graph.input[0].name: test_in_transposed}
    output_dict = oxe.execute_onnx(model, input_dict, True)
    test_result = output_dict[model.graph.output[0].name]

    test_result_transposed = test_result.transpose(_to_chan_first_args)
    output_matches = np.isclose(golden_result, test_result_transposed, atol=atol).all()

    if exec_mode == "cppsim":
        assert output_matches, "Cppsim output doesn't match ONNX/PyTorch."
    elif exec_mode == "rtlsim":
        assert output_matches, "Rtlsim output doesn't match ONNX/PyTorch."

    # ToDo: Should this be done / implemented as well?
    # if exec_mode == "rtlsim":
    #     hls_synt_res_est = model.analysis(hls_synth_res_estimation)
    #     assert "ChannelwiseOp_Batch_0" in hls_synt_res_est
    #
    #     node = model.get_nodes_by_op_type("ChannelwiseOp_Batch")[0]
    #     inst = getCustomOp(node)
    #     cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    #     exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    #     exp_cycles = exp_cycles_dict[node.name]
    #     assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
    #     assert exp_cycles != 0
