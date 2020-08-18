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


# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA


import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb

from finn.custom_op.registry import getCustomOp
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)

from finn.transformation.general import (
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import alveo_part_map, alveo_default_platform
from finn.util.test import get_test_model_trained, load_test_checkpoint_or_skip
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.infer_data_layouts import InferDataLayouts


build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_alveo_board = os.getenv("ALVEO_BOARD", default="U250")
test_fpga_part = alveo_part_map[test_alveo_board]
test_platform = alveo_default_platform[test_alveo_board]
target_clk_ns = 10
mem_mode = "decoupled"


def test_end2end_vitis_tfc_w1a1_export():
    import brevitas.onnx as bo

    tfc = get_test_model_trained("TFC", 1, 1)
    bo.export_finn_onnx(
        tfc, (1, 1, 28, 28), build_dir + "/end2end_vitis_tfc_w1a1_export.onnx"
    )


def test_end2end_vitis_tfc_w1a1_import_and_tidy():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_tfc_w1a1_export.onnx"
    )
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(build_dir + "/end2end_vitis_tfc_w1a1_tidy.onnx")


def test_end2end_vitis_tfc_w1a1_streamline():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_tfc_w1a1_tidy.onnx"
    )
    model = model.transform(Streamline())
    model = model.transform(RemoveUnusedTensors())
    model.save(build_dir + "/end2end_vitis_tfc_w1a1_streamlined.onnx")


def test_end2end_vitis_tfc_w1a1_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_tfc_w1a1_streamlined.onnx"
    )
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    model = model.transform(InferDataLayouts())
    model.save(build_dir + "/end2end_vitis_tfc_w1a1_hls_layers.onnx")


def test_end2end_vitis_tfc_w1a1_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_tfc_w1a1_hls_layers.onnx"
    )
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_vitis_tfc_w1a1_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_vitis_tfc_w1a1_dataflow_model.onnx")


def test_end2end_vitis_tfc_w1a1_fold():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_tfc_w1a1_dataflow_model.onnx"
    )
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # (PE, SIMD, in_fifo_depth, out_fifo_depth, ramstyle) for each layer
    config = [
        (16, 49, 16, 64, "block"),
        (8, 8, 64, 64, "auto"),
        (8, 8, 64, 64, "auto"),
        (10, 8, 64, 10, "distributed"),
    ]
    for fcl, (pe, simd, ififo, ofifo, ramstyle) in zip(fc_layers, config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififo)
        fcl_inst.set_nodeattr("outFIFODepth", ofifo)
        fcl_inst.set_nodeattr("ram_style", ramstyle)

    model.save(build_dir + "/end2end_vitis_tfc_w1a1_folded.onnx")


@pytest.mark.slow
@pytest.mark.vitis
def test_end2end_vitis_tfc_w1a1_build():
    if "VITIS_PATH" not in os.environ:
        pytest.skip("VITIS_PATH not set")
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_vitis_tfc_w1a1_folded.onnx"
    )
    model = model.transform(VitisBuild(test_fpga_part, target_clk_ns, test_platform))
    # TODO post-synth resources
    model.save(build_dir + "/end2end_vitis_tfc_w1a1_build.onnx")
