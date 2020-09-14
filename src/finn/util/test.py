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

import onnx
import onnx.numpy_helper as nph
import pkg_resources as pk
from pkgutil import get_data
from brevitas_examples import bnn_pynq
import numpy as np
import pytest
import warnings
from finn.core.modelwrapper import ModelWrapper
import os
from finn.util.basic import pynq_part_map, alveo_part_map, alveo_default_platform
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.vitis_build import VitisBuild, VitisOptStrategy
from finn.custom_op.registry import getCustomOp
from finn.core.onnx_exec import execute_onnx

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
    soft_expected = invec.flatten()[np_topk.astype(np.int).flatten()]
    soft_produced = invec.flatten()[idxvec.astype(np.int).flatten()]
    return (soft_expected == soft_produced).all()


def load_test_checkpoint_or_skip(filename):
    "Try to load given .onnx and return ModelWrapper, else skip current test."
    if os.path.isfile(filename):
        model = ModelWrapper(filename)
        return model
    else:
        warnings.warn(filename + " not found from previous test step, skipping")
        pytest.skip(filename + " not found from previous test step, skipping")


def get_build_env(kind, target_clk_ns):
    """Get board-related build environment for testing.
    - kind = either zynq or alveo.
    """
    ret = {}
    if kind == "zynq":
        ret["board"] = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
        ret["part"] = pynq_part_map[ret["board"]]
        ret["ip"] = os.getenv("PYNQ_IP", "")
        ret["username"] = os.getenv("PYNQ_USERNAME", "xilinx")
        ret["password"] = os.getenv("PYNQ_PASSWORD", "xilinx")
        ret["port"] = os.getenv("PYNQ_PORT", 22)
        ret["target_dir"] = os.getenv("PYNQ_TARGET_DIR", "/home/xilinx/finn")
        ret["build_fxn"] = ZynqBuild(ret["board"], target_clk_ns)
    elif kind == "alveo":
        ret["board"] = os.getenv("ALVEO_BOARD", default="U250")
        ret["part"] = alveo_part_map[ret["board"]]
        ret["platform"] = alveo_default_platform[ret["board"]]
        ret["ip"] = os.getenv("ALVEO_IP", "")
        ret["username"] = os.getenv("ALVEO_USERNAME", "")
        ret["password"] = os.getenv("ALVEO_PASSWORD", "")
        ret["port"] = os.getenv("ALVEO_PORT", 22)
        ret["target_dir"] = os.getenv("ALVEO_TARGET_DIR", "/tmp/finn_alveo_deploy")
        ret["build_fxn"] = VitisBuild(
            ret["part"],
            target_clk_ns,
            ret["platform"],
            strategy=VitisOptStrategy.BUILD_SPEED,
        )
    else:
        raise Exception("Unknown test build environment spec")
    return ret


def get_example_input(topology):
    "Get example numpy input tensor for given topology."

    if "fc" in topology:
        raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
        onnx_tensor = onnx.load_tensor_from_string(raw_i)
        return nph.to_array(onnx_tensor)
    elif topology == "cnv":
        fn = pk.resource_filename("finn", "data/cifar10/cifar10-test-data-class3.npz")
        input_tensor = np.load(fn)["arr_0"].astype(np.float32)
        return input_tensor
    else:
        raise Exception("Unknown topology, can't return example input")


def get_trained_network_and_ishape(topology, wbits, abits):
    "Return (trained_model, shape) for given BNN-PYNQ test config."

    topology_to_ishape = {
        "tfc": (1, 1, 28, 28),
        "cnv": (1, 3, 32, 32),
    }
    ishape = topology_to_ishape[topology]
    model = get_test_model_trained(topology.upper(), wbits, abits)
    return (model, ishape)


def execute_parent(parent_path, child_path, input_tensor_npy):
    """Execute parent model containing a single StreamingDataflowPartition by
    replacing it with the model at child_path and return result."""

    parent_model = load_test_checkpoint_or_skip(parent_path)
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    sdp_node.set_nodeattr("model", child_path)
    ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
    y = ret[oname]
    return y
