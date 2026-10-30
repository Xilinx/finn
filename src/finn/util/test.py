# Copyright (c) 2020, Xilinx
# Copyright (C) 2026, Advanced Micro Devices, Inc.
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

import importlib_resources as importlib
import numpy as np
import onnx
import onnx.numpy_helper as nph
import os
import shutil
import torchvision.transforms.functional as torchvision_util
import warnings
import xml.etree.ElementTree as ET
from brevitas_examples import bnn_pynq, imagenet_classification
from pkgutil import get_data
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.alveo_build import VitisLink, VitisOptStrategy
from finn.transformation.fpgadataflow.make_pynq_proj import PYNQBuild
from finn.util.basic import (
    make_build_dir,
    pynq_part_map,
    robust_rmtree,
    vitis_default_platform,
    vitis_part_map,
)

# map of (wbits,abits) -> model
example_map = {
    ("CNV", 1, 1): bnn_pynq.cnv_1w1a,
    ("CNV", 1, 2): bnn_pynq.cnv_1w2a,
    ("CNV", 2, 2): bnn_pynq.cnv_2w2a,
    ("LFC", 1, 1): bnn_pynq.lfc_1w1a,
    ("LFC", 1, 2): bnn_pynq.lfc_1w2a,
    ("SFC", 1, 1): bnn_pynq.sfc_1w1a,
    ("SFC", 1, 2): bnn_pynq.sfc_1w2a,
    ("SFC", 2, 2): bnn_pynq.sfc_2w2a,
    ("TFC", 1, 1): bnn_pynq.tfc_1w1a,
    ("TFC", 1, 2): bnn_pynq.tfc_1w2a,
    ("TFC", 2, 2): bnn_pynq.tfc_2w2a,
    ("mobilenet", 4, 4): imagenet_classification.quant_mobilenet_v1_4b,
}


def get_test_model(netname, wbits, abits, pretrained):
    """Returns the model specified by input arguments from the Brevitas BNN-PYNQ
    test networks. Pretrained weights loaded if pretrained is True."""
    model_cfg = (netname, wbits, abits)
    model_def_fxn = example_map[model_cfg]
    fc = model_def_fxn(pretrained)
    return fc.eval()


def get_test_model_trained(netname, wbits, abits):
    "get_test_model with pretrained=True"
    return get_test_model(netname, wbits, abits, pretrained=True)


def get_test_model_untrained(netname, wbits, abits):
    "get_test_model with pretrained=False"
    return get_test_model(netname, wbits, abits, pretrained=False)


def get_topk(vec, k):
    "Return indices of the top-k values in given array vec (treated as 1D)."
    return np.flip(vec.flatten().argsort())[:k]


def soft_verify_topk(invec, idxvec, k):
    """Check that the topK indices provided actually point to the topK largest
    values in the input vector"""
    np_topk = np.flip(invec.flatten().argsort())[:k]
    soft_expected = invec.flatten()[np_topk.astype(np.int_).flatten()]
    soft_produced = invec.flatten()[idxvec.astype(np.int_).flatten()]
    return (soft_expected == soft_produced).all()


def load_test_checkpoint_or_skip(filename):
    "Try to load given .onnx and return ModelWrapper, else skip current test."
    if os.path.isfile(filename):
        model = ModelWrapper(filename)
        return model
    else:
        warnings.warn(filename + " not found from previous test step, skipping")
        pytest.skip(filename + " not found from previous test step, skipping")


def get_hls_synth_interval(op_inst):
    """Return the initiation interval of an HLS node's synthesized top function as
    reported by Vitis HLS. PrepareIP and HLSSynthIP must have been run."""
    node_name = op_inst.onnx_node.name
    code_gen_dir = op_inst.get_nodeattr("code_gen_dir_ipgen")
    report = f"{code_gen_dir}/project_{node_name}/sol1/syn/report/{node_name}_csynth.xml"
    latency = ET.parse(report).getroot().find("PerformanceEstimates/SummaryOfOverallLatency")
    # pipelined tops report PipelineInitiationInterval, dataflow tops only Interval-min
    pii = latency.find("PipelineInitiationInterval")
    return int(pii.text if pii is not None else latency.find("Interval-min").text)


def make_runtime_weight_stream(op_inst, weights):
    """Write runtime weights in FINN's standard format and return the parsed stream."""
    weight_dir = make_build_dir("test_runtime_weights_")
    weight_path = os.path.join(weight_dir, "weights.dat")
    op_inst.make_weight_file(weights, "decoupled_runtime", weight_path)
    with open(weight_path, "r") as f:
        weight_stream = [int(x, 16) for x in f.read().strip().split("\n")]
    robust_rmtree(weight_dir)
    return weight_stream


def get_build_env(board, target_clk_ns):
    """Get board-related build environment for testing. Only relevant for bnn_pynq tests
    - board = any from pynq_part_map, vitis_part_map
    """
    ret = {}
    if board in pynq_part_map:
        ret["toolchain"] = "pynq"
        ret["part"] = pynq_part_map[board]
        ret["build_fxn"] = PYNQBuild(board, target_clk_ns)
    elif board in vitis_part_map:
        ret["toolchain"] = "vitis-xrt"
        ret["part"] = vitis_part_map[board]
        ret["vitis_platform"] = vitis_default_platform[board]
        ret["build_fxn"] = VitisLink(
            vitis_default_platform[board], target_clk_ns, strategy=VitisOptStrategy.BUILD_SPEED
        )
    else:
        raise Exception("Unknown board specified")
    return ret


def get_example_input(topology):
    "Get example numpy input tensor for given topology."

    if "fc" in topology:
        raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
        onnx_tensor = onnx.load_tensor_from_string(raw_i)
        return nph.to_array(onnx_tensor)
    elif topology == "cnv":
        ref = importlib.files("finn.qnn-data") / "cifar10/cifar10-test-data-class3.npz"
        with importlib.as_file(ref) as fn:
            input_tensor = np.load(fn)["arr_0"].astype(np.float32)
        return input_tensor
    else:
        raise Exception("Unknown topology, can't return example input")


def get_trained_network_and_ishape(topology, wbits, abits):
    "Return (trained_model, shape) for given BNN-PYNQ test config."

    topology_to_ishape = {
        "tfc": (1, 1, 28, 28),
        "lfc": (1, 1, 28, 28),
        "cnv": (1, 3, 32, 32),
    }
    ishape = topology_to_ishape[topology]
    model = get_test_model_trained(topology.upper(), wbits, abits)
    return (model, ishape)


def execute_parent(
    parent_path,
    child_path,
    input_tensor_npy,
    return_full_ctx=False,
    batch_child=False,
):
    """Execute a parent model containing one StreamingDataflowPartition.

    With ``batch_child=True``, parent processing keeps its compiled batch-one
    shapes while the partition child receives all preprocessed frames at once.
    A full-context batched call returns one parent context per frame as a list.
    """

    parent_model = load_test_checkpoint_or_skip(parent_path)
    iname = parent_model.get_first_global_in()
    oname = parent_model.get_first_global_out()
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_inst = getCustomOp(sdp_node)
    sdp_inst.set_nodeattr("model", child_path)
    sdp_inst.set_nodeattr("return_full_exec_context", 1 if return_full_ctx else 0)

    batch_size = input_tensor_npy.shape[0]
    if not batch_child or batch_size == 1:
        ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
        if return_full_ctx:
            return ret
        else:
            return ret[oname]

    sdp_index = parent_model.get_node_index(sdp_node)
    frame_contexts = []
    for batch_index in range(batch_size):
        frame_input = input_tensor_npy[batch_index : batch_index + 1]
        if sdp_index == 0:
            frame_context = parent_model.make_empty_exec_context()
            frame_context[iname] = frame_input
        else:
            frame_context = execute_onnx(
                parent_model,
                {iname: frame_input},
                True,
                end_node=parent_model.graph.node[sdp_index - 1],
            )
        frame_contexts.append(frame_context)

    # The partition only reads its declared inputs, so it does not need the full
    # parent execution contexts.
    partition_context = {
        name: np.concatenate([context[name] for context in frame_contexts], axis=0)
        for name in sdp_node.input
    }
    sdp_inst.execute_node(partition_context, parent_model.graph)

    for output_name in sdp_node.output:
        batched_output = partition_context[output_name]
        for batch_index, frame_context in enumerate(frame_contexts):
            frame_context[output_name] = batched_output[batch_index : batch_index + 1]

    if sdp_index + 1 < len(parent_model.graph.node):
        suffix_start = parent_model.graph.node[sdp_index + 1]
        for batch_index, frame_context in enumerate(frame_contexts):
            resume_context = {
                name: value for name, value in frame_context.items() if value is not None
            }
            frame_contexts[batch_index] = execute_onnx(
                parent_model,
                resume_context,
                True,
                start_node=suffix_start,
            )

    if return_full_ctx:
        return frame_contexts
    return np.concatenate([context[oname] for context in frame_contexts], axis=0)


def stage_hw_deployment(deploy_dir, board, model_title, input_npy, reference_npy):
    """Stage a build_dataflow deployment package for the CI hardware pipeline.

    The CI build pipeline zips every hw_deployment_*/<board>/<model_title>
    directory under FINN_BUILD_DIR into the board's zip, and the HW pipeline runs
    driver.py in each on input.npy and compares the result to
    output_reference.npy. deploy_dir is the build's deploy/ output, whose bitfile
    and driver (with its weights/ folder) are copied flat into that directory.
    model_title must be unique per board within one CI row."""

    deployment_dir = os.path.join(
        make_build_dir("hw_deployment_" + board + "_"), board, model_title
    )
    os.makedirs(deployment_dir)
    shutil.copytree(os.path.join(deploy_dir, "bitfile"), deployment_dir, dirs_exist_ok=True)
    shutil.copytree(os.path.join(deploy_dir, "driver"), deployment_dir, dirs_exist_ok=True)
    np.save(os.path.join(deployment_dir, "input.npy"), input_npy)
    np.save(os.path.join(deployment_dir, "output_reference.npy"), reference_npy)
    return deployment_dir


def resize_smaller_side(target_pixels, img):
    """Resizes smallest side of image to target pixels and resizes larger side with
    same ratio. Expects a PIL image."""
    return torchvision_util.resize(img, target_pixels)


def crop_center(size, img):
    """Crop central size*size window out of a PIL image."""
    return torchvision_util.center_crop(img, size)
