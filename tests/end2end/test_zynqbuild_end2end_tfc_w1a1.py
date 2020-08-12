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
from pkgutil import get_data

import pytest

import numpy as np

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA
import onnx.numpy_helper as nph

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
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
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
from finn.util.basic import pynq_part_map
from finn.util.test import get_test_model_trained, load_test_checkpoint_or_skip
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources

from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild

build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10
mem_mode = "decoupled"


def test_end2end_zynqbuild_tfc_w1a1_export():
    import brevitas.onnx as bo

    tfc = get_test_model_trained("TFC", 1, 1)
    bo.export_finn_onnx(
        tfc, (1, 1, 28, 28), build_dir + "/end2end_zynqbuild_tfc_w1a1_export.onnx"
    )


def test_end2end_zynqbuild_tfc_w1a1_import_and_tidy():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_export.onnx"
    )
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_tidy.onnx")


def test_end2end_zynqbuild_tfc_w1a1_streamline():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_tidy.onnx"
    )
    model = model.transform(Streamline())
    model = model.transform(RemoveUnusedTensors())
    model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_streamlined.onnx")


def test_end2end_zynqbuild_tfc_w1a1_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_streamlined.onnx"
    )
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_hls_layers.onnx")


def test_end2end_zynqbuild_tfc_w1a1_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_hls_layers.onnx"
    )
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_dataflow_model.onnx")


def test_end2end_zynqbuild_tfc_w1a1_fold():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_dataflow_model.onnx"
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

    model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_folded.onnx")


def test_end2end_zynqbuild_tfc_w1a1_make_driver():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_folded.onnx"
    )
    model = model.transform(MakePYNQDriver(platform="zynq-iodma"))
    model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_pynq_driver.onnx")


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_zynqbuild_tfc_w1a1_build():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_pynq_driver.onnx"
    )
    model = model.transform(ZynqBuild(test_pynq_board, target_clk_ns))
    model = model.transform(AnnotateResources("synth"))
    model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_build.onnx")


def test_end2end_zynqbuild_tfc_w1a1_deploy_on_pynq():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_build.onnx"
    )
    try:
        ip = os.environ["PYNQ_IP"]  # no fault for this one; skip if not defined
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
        username = os.getenv("PYNQ_USERNAME", "xilinx")
        password = os.getenv("PYNQ_PASSWORD", "xilinx")
        port = os.getenv("PYNQ_PORT", 22)
        target_dir = os.getenv("PYNQ_TARGET_DIR", "/home/xilinx/finn")
        model = model.transform(DeployToPYNQ(ip, port, username, password, target_dir))
        # save the model to be able to link it to the parent
        model.save(build_dir + "/end2end_zynqbuild_tfc_w1a1_pynq_deploy.onnx")
    except KeyError:
        pytest.skip("PYNQ board IP address not specified")


def test_end2end_zynqbuild_tfc_w1a1_run_on_pynq():
    # use the streamlined model as the "golden" model for right answers
    golden = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_streamlined.onnx"
    )
    iname = golden.graph.input[0].name
    oname = golden.graph.output[0].name
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    x = nph.to_array(input_tensor)
    # x = np.zeros(ishape, dtype=np.float32)
    # run using FINN-based execution
    ret_golden = execute_onnx(golden, {iname: x}, True)
    y_golden = ret_golden[oname]
    # set up parent+child graph to test
    # we'll use models from the previous step as the child model
    parent_model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_zynqbuild_tfc_w1a1_dataflow_parent.onnx"
    )
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    try:
        ip = os.environ["PYNQ_IP"]  # NOQA
        if ip == "":
            pytest.skip("PYNQ board IP address not specified")
        # produce results with cppsim
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        load_test_checkpoint_or_skip(
            build_dir + "/end2end_zynqbuild_tfc_w1a1_pynq_deploy.onnx"
        )
        sdp_node.set_nodeattr(
            "model", build_dir + "/end2end_zynqbuild_tfc_w1a1_pynq_deploy.onnx"
        )
        ret = execute_onnx(parent_model, {iname: x}, True)
        y = ret[oname]
        assert np.isclose(y, y_golden).all()

    except KeyError:
        pytest.skip("PYNQ board IP address not specified")
