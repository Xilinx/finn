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

import importlib_resources as importlib
import numpy as np
import onnx
import onnx.numpy_helper as nph
import os
import qonnx.custom_op.registry as registry
import torchvision.transforms.functional as torchvision_util
import warnings
from brevitas_examples import bnn_pynq, imagenet_classification
from pkgutil import get_data
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.derive_characteristic import DeriveCharacteristic
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.prepare_ip import _codegen_single_node

# def _codegen_single_node(node, model, fpgapart, clk):
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.vitis_build import VitisBuild, VitisOptStrategy

# generate each node's files as necessary
from finn.util.basic import (
    alveo_default_platform,
    alveo_part_map,
    make_build_dir,
    pynq_part_map,
)
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

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


def get_build_env(board, target_clk_ns):
    """Get board-related build environment for testing.
    - board = any from pynq_part_map or alveo_part_map
    """
    ret = {}
    if board in pynq_part_map:
        ret["kind"] = "zynq"
        ret["part"] = pynq_part_map[board]
        ret["build_fxn"] = ZynqBuild(board, target_clk_ns)
    elif board in alveo_part_map:
        ret["kind"] = "alveo"
        ret["part"] = alveo_part_map[board]
        ret["build_fxn"] = VitisBuild(
            ret["part"],
            target_clk_ns,
            alveo_default_platform[board],
            strategy=VitisOptStrategy.BUILD_SPEED,
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


def execute_parent(parent_path, child_path, input_tensor_npy, return_full_ctx=False):
    """Execute parent model containing a single StreamingDataflowPartition by
    replacing it with the model at child_path and return result."""

    parent_model = load_test_checkpoint_or_skip(parent_path)
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    sdp_node.set_nodeattr("model", child_path)
    sdp_node.set_nodeattr("return_full_exec_context", 1 if return_full_ctx else 0)
    ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
    if return_full_ctx:
        return ret
    else:
        return ret[oname]


def resize_smaller_side(target_pixels, img):
    """Resizes smallest side of image to target pixels and resizes larger side with
    same ratio. Expects a PIL image."""
    return torchvision_util.resize(img, target_pixels)


def crop_center(size, img):
    """Crop central size*size window out of a PIL image."""
    return torchvision_util.center_crop(img, size)


def compare_two_chr_funcs(a, b, relaxation):
    # relaxation determines how much leeway we allow for the
    # analytical implementation to be off from RTL ground truth

    # a = characteristic
    # b = ground truth
    for inp in range(len(a)):
        for i in range(len(a[inp])):
            start_internal_relaxation = min([relaxation, i])
            end_internal_relaxation = min([relaxation, abs(len(a[inp]) - i)])
            if a[inp][i] not in b[inp][i - start_internal_relaxation : i + end_internal_relaxation]:
                return False
    return True


DEBUGGING = False
CACHING = False


def get_characteristic_fnc(model, node0, part, target_clk_ns, strategy):
    # If set to True: attempt to cache a pre-existing variant of the model
    # this is to avoid generating RTL multiple times during
    # test debugging

    model_cache = None
    if strategy == "rtlsim" and CACHING:
        build_dir = os.environ["FINN_BUILD_DIR"]
        for x in os.listdir(build_dir):
            if x.startswith(str(node0)):
                model_cache = f"{build_dir}/{x}/model.onnx"
        if model_cache is not None:
            model = ModelWrapper(model_cache)

    if model_cache is None:
        model = model.transform(SpecializeLayers(part))
        # model = model.transform(MinimizeWeightBitWidth())
        #  model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(GiveUniqueNodeNames())
        # if strategy == "rtlsim":
        #     model = model.transform(PrepareIP(part, target_clk_ns))

        for node in model.graph.node:
            inst = registry.getCustomOp(node)
            if (is_hls_node(node) or is_rtl_node(node)) and (
                inst.prepare_kwargs_for_characteristic_fx() is None or strategy == "rtlsim"
            ):
                _codegen_single_node(node, model, part, target_clk_ns)

                op_type = node.op_type
                if is_hls_node(node):
                    try:
                        # lookup op_type in registry of CustomOps

                        # ensure that code is generated
                        assert (
                            inst.get_nodeattr("code_gen_dir_ipgen") != ""
                        ), """Node
                        attribute "code_gen_dir_ipgen" is empty. Please run
                        transformation PrepareIP first."""
                        if not os.path.isdir(
                            inst.get_nodeattr("ipgen_path")
                        ) or not inst.get_nodeattr("code_gen_dir_ipgen") in inst.get_nodeattr(
                            "ipgen_path"
                        ):
                            # call the compilation function for this node
                            inst.ipgen_singlenode_code()
                        else:
                            warnings.warn("Using pre-existing IP for %s" % node.name)
                        # ensure that executable path is now set
                        assert (
                            inst.get_nodeattr("ipgen_path") != ""
                        ), """Transformation
                        HLSSynthIP was not successful. Node attribute "ipgen_path"
                        is empty."""
                    except KeyError:
                        # exception if op_type is not supported
                        raise Exception("Custom op_type %s is currently not supported." % op_type)

        model = model.transform(ReplaceVerilogRelPaths())

        for node in model.graph.node:
            inst = registry.getCustomOp(node)
            if (is_hls_node(node) or is_rtl_node(node)) and (
                inst.prepare_kwargs_for_characteristic_fx() is None or strategy == "rtlsim"
            ):
                try:
                    # lookup op_type in registry of CustomOps
                    # inst = registry.getCustomOp(node)
                    inst.prepare_rtlsim()
                    # ensure that executable path is now set
                    assert (
                        inst.get_nodeattr("rtlsim_so") != ""
                    ), "Failed to prepare RTLSim, no rtlsim_so attribute found."
                except KeyError:
                    # exception if op_type is not supported
                    raise Exception("Custom op_type %s is currently not supported." % op_type)

        model = model.transform(AnnotateCycles())

        period = int(model.analysis(dataflow_performance)["max_cycles"] + 12)

        model = model.transform(
            DeriveCharacteristic(
                model,
                period,
                strategy,
                part,
                target_clk_ns,
            )
        )
        if CACHING:
            tmp_caching_output_dir = make_build_dir(str(node0))
            model.save(tmp_caching_output_dir + "/model.onnx")

    return getCustomOp(model.graph.node[0])


def debug_chr_funcs(chr_in, chr_out, rtlsim_in, rtlsim_out, direction):
    if DEBUGGING:

        def concat_list(a):
            b = []
            current = a[0]
            b.append(1)
            for i in a[1:]:
                if i == current:
                    b[-1] += 1
                else:
                    b.append(1)
                    current = i
            return b

        chr_in_concat = concat_list(chr_in[0])
        chr_out_concat = concat_list(chr_out[0])
        rtlsim_in_concat = concat_list(rtlsim_in[0])
        rtlsim_out_concat = concat_list(rtlsim_out[0])

        np.set_printoptions(threshold=np.inf)
        if direction == "input":
            print(f"\nchr IN:    {chr_in[:100]}, {len(chr_in[0])}")
            print(f"rtlsim IN: {rtlsim_in[:100]}, {len(rtlsim_in[0])}")

            print(f"chr IN CONCAT:    {chr_in_concat[:100]}, {len(chr_in_concat)}")
            print(f"rtlsim IN CONCAT: {rtlsim_in_concat[:100]}, {len(rtlsim_in_concat)}")

        elif direction == "output":
            print(f"\nchr OUT:    {chr_out[:100]}, {len(chr_out[0])}")
            print(f"rtlsim OUT: {rtlsim_out[:100]}, {len(rtlsim_out[0])}")

            print(f"chr OUT CONCAT:    {chr_out_concat[:100]}, {len(chr_out_concat)}")
            print(f"rtlsim OUT CONCAT: {rtlsim_out_concat[:100]}, {len(rtlsim_out_concat)}")
    else:
        return True
