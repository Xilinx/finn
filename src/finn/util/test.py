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

import copy
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
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.alveo_build import VitisLink, VitisOptStrategy
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveTokenAccessVectors,
)
from finn.transformation.fpgadataflow.fifo_sizing.tav import load_tav_npy
from finn.transformation.fpgadataflow.fifo_sizing.tree_models.base import get_tree_model
from finn.transformation.fpgadataflow.make_pynq_proj import PYNQBuild
from finn.transformation.fpgadataflow.prepare_ip import _codegen_single_node
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import (
    make_build_dir,
    pynq_part_map,
    robust_rmtree,
    vitis_default_platform,
    vitis_part_map,
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


def execute_parent(parent_path, child_path, input_tensor_npy, return_full_ctx=False):
    """Execute parent model containing a single StreamingDataflowPartition by
    replacing it with the model at child_path and return result."""

    parent_model = load_test_checkpoint_or_skip(parent_path)
    iname = parent_model.get_first_global_in()
    oname = parent_model.get_first_global_out()
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    sdp_node.set_nodeattr("model", child_path)
    sdp_node.set_nodeattr("return_full_exec_context", 1 if return_full_ctx else 0)
    ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
    if return_full_ctx:
        return ret
    else:
        return ret[oname]


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


def tav_tolerance(frac, scale, const):
    """``const + frac * scale`` -- the two terms a TAV error actually has.

    The proportional term is the one that belongs in a budget: a period
    reproduced to within four cycles is excellent on a 100k-cycle frame and
    useless on a 40-cycle one, and a flat cycle count says the same thing about
    both.

    The constant term is not a fudge, it is the fixed part of the error:
    pipeline wind-up, an adder tree's depth, the cycle rtlsim's
    ``cycles_rtlsim // periods_to_simulate`` rounds away. Those cost the same
    few cycles whatever the frame length, which is why they add rather than
    compete -- a long frame carries the same fixed wind-up as a short one *and*
    a proportional error on top, and ``max`` of the two would silently drop
    whichever term happened to be smaller.
    """
    return float(const) + float(frac) * float(scale)


def compare_two_chr_funcs(
    a,
    b,
    max_allowed_volume_frac,
    max_allowed_length_frac,
    volume_const=0,
    length_const=0,
):
    """Compare an analytical TAV ``a`` against the rtlsim ground truth ``b``.

    ``max_allowed_volume_frac`` is a fraction of the tokens the reference moves
    in the window, and ``max_allowed_length_frac`` a fraction of the reference's
    length; ``*_const`` are the fixed cycle/token counts that add to the
    proportional term. See ``tav_tolerance``.

    The leeway this grants may produce larger FIFOs. Output delays due to long
    pipelines generally do not affect FIFO sizes, so a larger relaxation on the
    output side is expected.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    lower_len = min(len(a), len(b))

    len_allowed = tav_tolerance(max_allowed_length_frac, len(b), length_const)
    if len(a) != len(b):
        len_dif = abs(len(a) - len(b))
        print(
            "TAV length delta: %d (%.2f%% of %d, allowed %.0f)"
            % (len_dif, 100.0 * len_dif / max(1, len(b)), len(b), len_allowed)
        )
        if len_dif > len_allowed:
            return False

    # scale the volume budget by the tokens actually moved, not by the cycle
    # count: a peak cumulative delta is a token count and the two differ by the
    # folding factor
    total_tokens = int(b[lower_len - 1]) if lower_len else 0
    vol_allowed = tav_tolerance(max_allowed_volume_frac, total_tokens, volume_const)
    peak_volume_delta = np.max(np.abs(a[:lower_len] - b[:lower_len])) if lower_len else 0
    print(
        "TAV peak volume delta: %d (%.2f%% of %d tokens, allowed %.0f)"
        % (
            peak_volume_delta,
            100.0 * peak_volume_delta / max(1, total_tokens),
            total_tokens,
            vol_allowed,
        )
    )
    if peak_volume_delta > vol_allowed:
        return False
    return True


def get_characteristic_fnc(model, part, target_clk_ns, strategy):
    """Characterize the single node of ``model`` with ``strategy`` ("tree_model"
    or "rtlsim") and return its custom op, for comparing the token access
    vectors of the two strategies. A node without a tree model falls back to
    rtlsim."""
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    if (is_hls_node(node) or is_rtl_node(node)) and (
        get_tree_model(inst) is None or strategy == "rtlsim"
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
                if os.path.isdir(inst.get_nodeattr("ipgen_path")) or inst.get_nodeattr(
                    "code_gen_dir_ipgen"
                ) not in inst.get_nodeattr("ipgen_path"):
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

    node = model.graph.node[0]
    inst = getCustomOp(node)
    if (is_hls_node(node) or is_rtl_node(node)) and (
        get_tree_model(inst) is None or strategy == "rtlsim"
    ):
        try:
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
        DeriveTokenAccessVectors(
            model,
            period,
            strategy,
            part,
            target_clk_ns,
        )
    )

    return getCustomOp(model.graph.node[0])


def tree_model_test(
    model,
    node_details,
    part,
    target_clk_ns,
    max_allowed_volume_frac,
    max_allowed_length_frac,
    volume_const=0,
    length_const=0,
):
    """Characterize ``model``'s single node both ways and compare the two TAVs.

    The tolerances are fractions -- of the tokens moved and of the frame length
    respectively -- with an absolute floor for the fixed part of the error. See
    ``compare_two_chr_funcs``.
    """
    print("TAV comparison for", node_details)
    # ground truth model to rtlsim
    model_rtl = copy.deepcopy(model)
    node_analytical = get_characteristic_fnc(model, part, target_clk_ns, "tree_model")
    node_rtlsim = get_characteristic_fnc(model_rtl, part, target_clk_ns, "rtlsim")

    chr_in = load_tav_npy(node_analytical.get_nodeattr("io_chrc_in"))
    chr_out = load_tav_npy(node_analytical.get_nodeattr("io_chrc_out"))

    rtlsim_in = load_tav_npy(node_rtlsim.get_nodeattr("io_chrc_in"))
    rtlsim_out = load_tav_npy(node_rtlsim.get_nodeattr("io_chrc_out"))

    # test input port
    input_check = compare_two_chr_funcs(
        chr_in[0],
        rtlsim_in[0],
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    )

    # test output port
    output_check = compare_two_chr_funcs(
        chr_out[0],
        rtlsim_out[0],
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    )

    return input_check and output_check


def compare_chr_funcs_up_to_rotation(
    chr_in,
    chr_out,
    rtlsim_in,
    rtlsim_out,
    max_allowed_volume_frac,
    max_allowed_length_frac,
    volume_const=0,
    length_const=0,
):
    """Compare a tree model against rtlsim allowing one *shared* phase offset.

    Weaker than ``compare_two_chr_funcs`` in exactly one way and no other: the
    tree model may place the frame anywhere within the period, as long as it
    places the input and the output row at the *same* offset. That is the right
    assertion for a node whose steady-state schedule is known exactly but whose
    wind-up is not, and it is not a licence to be sloppy -- a model that got the
    read-to-write relationship wrong, or the period, or the token count, still
    fails, and those are what FIFO sizing integrates. Only the absolute
    placement of the frame inside rtlsim's two-period window is forgiven, and
    that placement is an artefact of where the window was cut.

    Tolerances are fractions with an absolute floor, as in
    ``compare_two_chr_funcs``. The length excess is one-sided here -- rtlsim's
    period carries a share of the wind-up, so a model that reproduces the steady
    state exactly is *short* by that share and never long -- and running long is
    rejected outright, being a real error rather than a wind-up artefact.

    Returns ``(ok, rotation, in_delta, out_delta)``.
    """
    a_in, b_in = np.asarray(chr_in), np.asarray(rtlsim_in)
    a_out, b_out = np.asarray(chr_out), np.asarray(rtlsim_out)

    len_dif = len(b_in) - len(a_in)
    len_allowed = tav_tolerance(max_allowed_length_frac, len(b_in), length_const)
    if len_dif < 0 or len_dif > len_allowed:
        print(
            "TAV length delta: %d (%.2f%% of %d, allowed %.0f)"
            % (len_dif, 100.0 * len_dif / max(1, len(b_in)), len(b_in), len_allowed)
        )
        return False, 0, None, None

    # one period of per-cycle deltas, which is what a rotation acts on
    def one_period(cum):
        n = len(cum) // 2
        d = np.diff(np.concatenate(([0], np.asarray(cum, dtype=np.int64))))
        return d[:n]

    d_in, d_out = one_period(a_in), one_period(a_out)
    r_in, r_out = one_period(b_in), one_period(b_out)

    # Deliberately no equal-token-count assertion between the two. rtlsim's
    # period is cycles_rtlsim // periods_to_simulate and rounds *up* off the
    # wind-up, so on a short period its window holds a little more than one
    # frame -- 41 reads against the node's 36 on SF=2, TH=9 -- and demanding
    # equality would fail the model for the reference's rounding. That the
    # model itself moves exactly one folded frame per period is an absolute
    # property with no reference in it, and is asserted on its own.
    n = min(len(d_in), len(r_in))
    ref_in = np.cumsum(r_in[:n])
    ref_out = np.cumsum(r_out[:n])
    best = None
    for rot in range(len(d_in)):
        v = int(np.max(np.abs(np.cumsum(np.roll(d_in, rot)[:n]) - ref_in)))
        if best is None or v < best[0]:
            best = (v, rot)
    in_delta, rot = best
    out_delta = int(np.max(np.abs(np.cumsum(np.roll(d_out, rot)[:n]) - ref_out)))
    vol_allowed = tav_tolerance(
        max_allowed_volume_frac, max(int(ref_in[-1]), int(ref_out[-1])) if n else 0, volume_const
    )
    print(
        "TAV rotation: %d, peak volume delta in/out: %d/%d (allowed %.0f)"
        % (rot, in_delta, out_delta, vol_allowed)
    )
    ok = in_delta <= vol_allowed and out_delta <= vol_allowed
    return ok, rot, in_delta, out_delta


def tree_model_test_up_to_rotation(
    model,
    node_details,
    part,
    target_clk_ns,
    max_allowed_volume_frac,
    max_allowed_length_frac,
    volume_const=0,
    length_const=0,
):
    """``tree_model_test`` with ``compare_chr_funcs_up_to_rotation`` as the check."""
    print("TAV comparison for", node_details)
    model_rtl = copy.deepcopy(model)
    node_analytical = get_characteristic_fnc(model, part, target_clk_ns, "tree_model")
    node_rtlsim = get_characteristic_fnc(model_rtl, part, target_clk_ns, "rtlsim")
    chr_in = load_tav_npy(node_analytical.get_nodeattr("io_chrc_in"))
    chr_out = load_tav_npy(node_analytical.get_nodeattr("io_chrc_out"))
    rtlsim_in = load_tav_npy(node_rtlsim.get_nodeattr("io_chrc_in"))
    rtlsim_out = load_tav_npy(node_rtlsim.get_nodeattr("io_chrc_out"))
    ok, _, _, _ = compare_chr_funcs_up_to_rotation(
        chr_in[0],
        chr_out[0],
        rtlsim_in[0],
        rtlsim_out[0],
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    )
    return ok
