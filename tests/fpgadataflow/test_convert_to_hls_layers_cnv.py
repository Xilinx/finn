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

import pkg_resources as pk

import pytest

import brevitas.onnx as bo
import numpy as np
import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.util.test import get_test_model_trained

export_onnx_path_cnv = "test_convert_to_hls_layers_cnv.onnx"


@pytest.mark.fpgadataflow
@pytest.mark.vivado
# Standalone or fused thresholding-based activation
@pytest.mark.parametrize("fused_activation", [True, False])
def test_convert_to_hls_layers_cnv_w1a1(fused_activation):
    cnv = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(cnv, (1, 3, 32, 32), export_onnx_path_cnv)
    model = ModelWrapper(export_onnx_path_cnv)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model = model.transform(InferDataLayouts())
    # model.save("golden.onnx")
    # load one of the test vectors
    fn = pk.resource_filename("finn.qnn-data", "cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    input_tensor = input_tensor / 255
    assert input_tensor.shape == (1, 3, 32, 32)
    # generate expected value from streamlined net
    input_dict = {"global_in": input_tensor}
    expected_ctx = oxe.execute_onnx(model, input_dict, True)
    expected = expected_ctx[model.graph.output[0].name]

    # if we infer thresholding first, all MultiThresholds get converted to HLS
    # subsequently, the FC inference will generate passthrough MVAUs
    if not fused_activation:
        model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(to_hls.InferBinaryMatrixVectorActivation())
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation())
    for node in model.graph.node:
        if node.op_type == "MatrixVectorActivation":
            inst = getCustomOp(node)
            inst.set_nodeattr("mem_mode", "decoupled")
            mw = inst.get_nodeattr("MW")
            mh = inst.get_nodeattr("MH")
            if mh % 4 == 0:
                pe = mh // 4
            else:
                pe = mh
            inst.set_nodeattr("PE", pe)
            if mw % 16 == 0:
                simd = mw // 16
            else:
                simd = mw
            inst.set_nodeattr("SIMD", simd)
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    # check topology status
    finn_nodes = model.get_finn_nodes()
    if fused_activation:
        assert len(finn_nodes) == 18
    else:
        assert len(finn_nodes) == 26
        thr_nodes = model.get_nodes_by_op_type("Thresholding_Batch")
        assert len(thr_nodes) == 8
    non_finn_nodes = model.get_non_finn_nodes()
    assert len(non_finn_nodes) == 5
    exp_non_finn_nodes = ["Transpose", "Transpose", "Reshape", "Mul", "Add"]
    assert [x.op_type for x in non_finn_nodes] == exp_non_finn_nodes
    fc_nodes = model.get_nodes_by_op_type("MatrixVectorActivation")
    assert len(fc_nodes) == 9
    swg_nodes = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    assert len(swg_nodes) == 6
    mp_nodes = model.get_nodes_by_op_type("StreamingMaxPool_Batch")
    assert len(mp_nodes) == 2
    # model.save("cnv-pre-compile.onnx")
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))
    # model.save("cnv-post-compile.onnx")
    produced_ctx = oxe.execute_onnx(model, input_dict, True)
    produced = produced_ctx[model.graph.output[0].name]
    assert np.isclose(expected, produced, atol=1e-3).all()
    assert np.argmax(produced) == 3
    os.remove(export_onnx_path_cnv)
