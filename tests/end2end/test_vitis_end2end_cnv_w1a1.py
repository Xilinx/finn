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

import os
import pytest
import numpy as np

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.registry import getCustomOp
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ
from finn.transformation.general import (
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.util.basic import alveo_part_map, alveo_default_platform
from finn.util.test import get_test_model_trained, load_test_checkpoint_or_skip
from finn.transformation.fpgadataflow.vitis_build import VitisBuild, VitisOptStrategy
import pkg_resources as pk
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
import warnings

build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_alveo_board = os.getenv("ALVEO_BOARD", default="U250")
test_fpga_part = alveo_part_map[test_alveo_board]
test_platform = alveo_default_platform[test_alveo_board]
target_clk_ns = 10
mem_mode = "decoupled"


def test_end2end_vitis_cnv_w1a1_export():
    import brevitas.onnx as bo

    tfc = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(
        tfc, (1, 3, 32, 32), build_dir + "/end2end_vitis_cnv_w1a1_export.onnx"
    )


def test_end2end_vitis_cnv_w1a1_import_and_tidy():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_export.onnx"
    )
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(build_dir + "/end2end_vitis_cnv_w1a1_tidy.onnx")


def test_end2end_vitis_cnv_w1a1_streamline():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_tidy.onnx"
    )
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model = model.transform(RemoveUnusedTensors())
    model.save(build_dir + "/end2end_vitis_cnv_w1a1_streamlined.onnx")


def test_end2end_vitis_cnv_w1a1_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_streamlined.onnx"
    )
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(InferDataLayouts())
    model.save(build_dir + "/end2end_vitis_cnv_w1a1_hls_layers.onnx")


def test_end2end_vitis_cnv_w1a1_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_hls_layers.onnx"
    )
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_vitis_cnv_w1a1_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_vitis_cnv_w1a1_dataflow_model.onnx")


def test_end2end_vitis_cnv_w1a1_fold():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_dataflow_model.onnx"
    )
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # each tuple is (PE, SIMD, in_fifo_depth) for a layer
    folding = [
        (16, 3, 256),
        (32, 32, 256),
        (16, 32, 256),
        (16, 32, 256),
        (4, 32, 214),
        (1, 32, 2),
        (1, 4, 126),
        (1, 8, 62),
        (5, 1, 6),
    ]
    for fcl, (pe, simd, ififodepth) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififodepth)

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    swg_idepth = [2, 51, 9, 106, 2, 2]
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
        swg_inst.set_nodeattr("inFIFODepth", swg_idepth[i])
    model = model.transform(AnnotateResources("estimate"))
    model = model.transform(AnnotateCycles())
    model.save(build_dir + "/end2end_vitis_cnv_w1a1_folded.onnx")


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_vitis_cnv_w1a1_build():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_folded.onnx"
    )
    model = model.transform(
        VitisBuild(
            test_fpga_part,
            target_clk_ns,
            test_platform,
            strategy=VitisOptStrategy.BUILD_SPEED,
        )
    )
    model.save(build_dir + "/end2end_vitis_cnv_w1a1_build.onnx")


def test_end2end_vitis_cnv_w1a1_annotate_resources():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_build.onnx"
    )
    model = model.transform(AnnotateResources("synth"))
    warnings.warn(
        "Post-synthesis resources (excluding shell): "
        + model.get_metadata_prop("res_total_synth")
    )
    warnings.warn(
        "Post-synthesis resources (all inclusive): "
        + model.get_metadata_prop("res_total_top_synth")
    )
    model.save(build_dir + "/end2end_vitis_cnv_w1a1_annotate_resources.onnx")


def test_end2end_vitis_cnv_w1a1_deploy_on_pynq():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_build.onnx"
    )
    try:
        ip = os.environ["ALVEO_IP"]  # no fault for this one; skip if not defined
        if ip == "":
            pytest.skip("Alveo host IP address not specified")
        username = os.getenv("ALVEO_USERNAME", "xilinx")
        password = os.getenv("ALVEO_PASSWORD", "xilinx")
        port = os.getenv("ALVEO_PORT", 22)
        target_dir = os.getenv("ALVEO_TARGET_DIR", "/home/xilinx/finn")
        model = model.transform(DeployToPYNQ(ip, port, username, password, target_dir))
        # save the model to be able to link it to the parent
        model.save(build_dir + "/end2end_vitis_cnv_w1a1_pynq_deploy.onnx")
    except KeyError:
        pytest.skip("Alveo host IP address not specified")


def test_end2end_vitis_cnv_w1a1_run_on_pynq():
    # use the streamlined model as the "golden" model for right answers
    golden = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_streamlined.onnx"
    )
    iname = golden.graph.input[0].name
    oname = golden.graph.output[0].name
    # load one of the test vectors
    fn = pk.resource_filename("finn", "data/cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    input_tensor = input_tensor / 255
    assert input_tensor.shape == (1, 3, 32, 32)
    x = input_tensor
    # x = np.zeros(ishape, dtype=np.float32)
    # run using FINN-based execution
    ret_golden = execute_onnx(golden, {iname: x}, True)
    y_golden = ret_golden[oname]
    # set up parent+child graph to test
    # we'll use models from the previous step as the child model
    parent_model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_cnv_w1a1_dataflow_parent.onnx"
    )
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    try:
        ip = os.environ["ALVEO_IP"]  # NOQA
        if ip == "":
            pytest.skip("Alveo host IP address not specified")
        # produce results with cppsim
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        load_test_checkpoint_or_skip(
            build_dir + "/end2end_vitis_cnv_w1a1_pynq_deploy.onnx"
        )
        sdp_node.set_nodeattr(
            "model", build_dir + "/end2end_vitis_cnv_w1a1_pynq_deploy.onnx"
        )
        ret = execute_onnx(parent_model, {iname: x}, True)
        y = ret[oname]
        assert np.isclose(y, y_golden).all()
        assert np.argmax(y) == 3

    except KeyError:
        pytest.skip("Alveo host IP address not specified")
