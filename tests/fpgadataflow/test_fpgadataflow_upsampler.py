# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
import onnx.helper as oh
import qonnx.core.data_layout as DataLayout
import torch
from brevitas.export import export_qonnx
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from torch import nn

import finn.core.onnx_exec as oxe
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferUpsample
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline.reorder import MakeScaleResizeNHWC
from finn.util.basic import make_build_dir, robust_rmtree


class ForceDataTypeForTensors(Transformation):
    """
    Forces a certain datatype for all tensors in a model.
    """

    def __init__(self, dType=DataType["INT8"]):
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


def make_resize_sizes_modelwrapper(ifm_dim, num_ch, sizes, idt):
    """Build a Resize node using the opset-11+ 4-input signature that specifies
    the target output shape via a constant ``sizes`` input (with an empty
    ``scales`` input), in NCHW layout followed by a Transpose to NHWC.

    PyTorch export cannot reliably produce a clean, constant sizes-based Resize,
    so the graph is constructed by hand to deterministically exercise the
    ``is_sizes=True`` path of InferUpsample and the sizes handling in
    MakeScaleResizeNHWC."""
    idim_h, idim_w = ifm_dim
    ofm_dim_h, ofm_dim_w = sizes[2], sizes[3]
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, num_ch, idim_h, idim_w])
    # scales is present but empty; sizes carries the target output shape
    scales = oh.make_tensor_value_info("scales", TensorProto.FLOAT, [])
    # roi is unused, only needed for compliance with the Resize node interface
    roi = oh.make_tensor_value_info("roi", TensorProto.FLOAT, [4])
    size_param = oh.make_tensor_value_info("sizes", TensorProto.INT64, [4])
    outp_up = oh.make_tensor_value_info(
        "outp_up", TensorProto.FLOAT, [1, num_ch, ofm_dim_h, ofm_dim_w]
    )
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, num_ch])

    resize_node = oh.make_node(
        "Resize",
        inputs=["inp", "roi", "scales", "sizes"],
        outputs=["outp_up"],
        name="Resize1",
        mode="nearest",
    )
    transpose_node = oh.make_node(
        "Transpose",
        inputs=["outp_up"],
        outputs=["outp"],
        name="Transpose1",
        perm=_to_chan_last_args,
    )

    graph = oh.make_graph(
        nodes=[resize_node, transpose_node],
        name="resize_sizes_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[outp_up, roi, scales, size_param],
    )
    model = qonnx_make_model(graph, producer_name="resize_sizes_model")
    model = ModelWrapper(model)
    model.set_initializer("scales", np.array([], dtype=np.float32))
    model.set_initializer("sizes", np.array(sizes, dtype=np.int64))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_layout("inp", DataLayout.NCHW)
    model = model.transform(InferShapes())
    model = model.transform(InferDataLayouts())
    return model


# param datatype
@pytest.mark.parametrize("dt", [DataType["INT8"]])
# spatial dim input feature map
@pytest.mark.parametrize("IFMDim", [[3, 3], [3, 5], [3, 1]])
# upscaling factor
@pytest.mark.parametrize("scale", [2, 3])
# Number of input/output channels
@pytest.mark.parametrize("NumChannels", [4])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# parallelization level
@pytest.mark.parametrize("SIMD", [1, 2, 4])
# ONNX export opset: opset 10 emits a 2-input Resize (X, scales), opset 11+ emits
# a 3-input Resize (X, roi, scales), exercising both param-input signatures
@pytest.mark.parametrize("opset_version", [10, 11, 13])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_upsampler(dt, IFMDim, scale, NumChannels, exec_mode, SIMD, opset_version):
    tmpdir = make_build_dir("upsample_export_")
    atol = 1e-3
    idim0, idim1 = IFMDim
    input_shape = (1, NumChannels, idim0, idim1)
    if idim1 == 1:
        upscale_factor = (scale, 1)
    else:
        upscale_factor = (scale, scale)
    # Create the test model and inputs for it
    torch_model = PyTorchTestModel(upscale_factor=upscale_factor)
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
    export_qonnx(torch_model, torch.randn(input_shape), export_path, opset_version=opset_version)
    qonnx_cleanup(export_path, out_file=export_path)
    model = ModelWrapper(export_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    input_dict = {model.get_first_global_in(): test_in.numpy().astype(np.int32)}
    input_dict = {model.get_first_global_in(): test_in.numpy()}
    golden_output_dict = oxe.execute_onnx(model, input_dict, True)
    golden_result = golden_output_dict[model.get_first_global_out()]

    # Make sure PyTorch and ONNX match
    pyTorch_onnx_match = np.isclose(golden_result, golden_torch_float).all()
    assert pyTorch_onnx_match, "ONNX and PyTorch upsampling output don't match."

    # Prep model for execution
    model = ModelWrapper(export_path)
    model = model.transform(MakeInputChannelsLast())
    model = model.transform(InferDataLayouts())
    model = model.transform(absorb.AbsorbTransposeIntoResize())
    model = model.transform(InferShapes())
    model = model.transform(ForceDataTypeForTensors(dType=dt))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferUpsample())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Check that all nodes are UpsampleNearestNeighbour_Batch nodes
    for n in model.get_finn_nodes():
        node_check = n.op_type == "UpsampleNearestNeighbour"
        assert node_check, "All nodes should be UpsampleNearestNeighbour nodes."
        inst = getCustomOp(n)
        inst.set_nodeattr("SIMD", SIMD)

    test_in_transposed = test_in.numpy().transpose(_to_chan_last_args)
    input_dict = {model.get_first_global_in(): test_in_transposed}

    # Run sim
    output_dict = oxe.execute_onnx(model, input_dict, True)
    test_result = output_dict[model.get_first_global_out()]
    output_matches = np.isclose(golden_result, test_result, atol=atol).all()

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

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
    output_dict = oxe.execute_onnx(model, input_dict, True)
    test_result = output_dict[model.get_first_global_out()]
    output_matches = np.isclose(golden_result, test_result, atol=atol).all()

    if exec_mode == "cppsim":
        assert output_matches, "Cppsim output doesn't match ONNX/PyTorch."
    elif exec_mode == "rtlsim":
        assert output_matches, "Rtlsim output doesn't match ONNX/PyTorch."
    robust_rmtree(tmpdir)


# param datatype
@pytest.mark.parametrize("dt", [DataType["INT8"]])
# spatial dim input feature map
@pytest.mark.parametrize("IFMDim", [[4, 4], [2, 4]])
# upscaling factor
@pytest.mark.parametrize("scale", [2])
# Number of input/output channels
@pytest.mark.parametrize("NumChannels", [4])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# parallelization level
@pytest.mark.parametrize("SIMD", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_upsampler_sizes(dt, IFMDim, scale, NumChannels, exec_mode, SIMD):
    # Exercises the sizes-based (4-input) Resize path: the target output shape is
    # given as an explicit sizes input rather than as scales. This is the
    # is_sizes=True branch of InferUpsample, which PyTorch export cannot produce.
    tmpdir = make_build_dir("upsample_sizes_")
    atol = 1e-3
    idim_h, idim_w = IFMDim
    input_shape = (1, NumChannels, idim_h, idim_w)
    # target output sizes in NCHW order; only integer-multiple upsampling is supported
    sizes = [1, NumChannels, idim_h * scale, idim_w * scale]

    # Build a sizes-based Resize model and a matching integer input
    model = make_resize_sizes_modelwrapper(IFMDim, NumChannels, sizes, dt)
    test_in = gen_finn_dt_tensor(dt, input_shape)
    input_dict = {model.get_first_global_in(): test_in}

    # Golden reference from the untransformed ONNX Resize
    golden_output_dict = oxe.execute_onnx(model, input_dict, True)
    golden_result = golden_output_dict[model.get_first_global_out()]

    # Move the Resize into NHWC layout and infer the UpsampleNearestNeighbour HW op
    model = model.transform(MakeScaleResizeNHWC())
    model = model.transform(InferDataLayouts())
    model = model.transform(ForceDataTypeForTensors(dType=dt))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferUpsample())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Check that the Resize was converted to a single UpsampleNearestNeighbour node
    upsample_nodes = model.get_nodes_by_op_type("UpsampleNearestNeighbour")
    assert len(upsample_nodes) == 1, "Expected exactly one UpsampleNearestNeighbour node."
    getCustomOp(upsample_nodes[0]).set_nodeattr("SIMD", SIMD)

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

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

    # Run sim and compare against the golden ONNX output
    output_dict = oxe.execute_onnx(model, input_dict, True)
    test_result = output_dict[model.get_first_global_out()]
    output_matches = np.isclose(golden_result, test_result, atol=atol).all()
    assert output_matches, "%s output doesn't match golden ONNX." % exec_mode
    robust_rmtree(tmpdir)
