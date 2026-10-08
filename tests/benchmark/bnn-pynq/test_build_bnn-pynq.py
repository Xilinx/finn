# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
import torch
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    benchmark_config_paths,
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    get_verify_steps,
    make_benchmark_cfg,
    verification_io_dir,
)
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.builder.build_dataflow as build
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir
from finn.util.pytorch import ToTensor


# model
def get_model_file(model):
    return os.path.join(benchmark_root(), "models", model + ".onnx")


# The BNN-PYNQ nets are trained on ToTensor-normalized images (raw uint8 / 255).
# Merge that division onto the front of the graph so the build runs inference on
# raw uint8 input, and annotate the global input as UINT8.
def custom_step_add_preproc(model, cfg):
    global_inp_name = model.get_first_global_in()
    ishape = model.get_tensor_shape(global_inp_name)
    preproc_file = os.path.join(make_build_dir("bnn_preproc_"), "preproc.onnx")
    export_qonnx(ToTensor(), torch.randn(ishape), preproc_file, opset_version=13)
    qonnx_cleanup(preproc_file, out_file=preproc_file)
    pre_model = ModelWrapper(preproc_file)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    pre_model = pre_model.transform(InferShapes())
    pre_model = pre_model.transform(FoldConstants())
    model = model.transform(MergeONNXModels(pre_model))
    model.set_tensor_datatype(model.get_first_global_in(), DataType["UINT8"])
    return model


# The Brevitas BNN-PYNQ exports end at the final MatMul (raw N-class vector);
# the verification golden and the LabelSelect-keyed folding/specialize configs
# both expect a top-1 class index. Append a TopK(k=1) so the graph matches --
# convert_to_hw turns it into the configured LabelSelect node.
def custom_step_add_postproc(model, cfg):
    model = model.transform(InsertTopK(k=1))
    return model


verif_steps = [
    "finn_onnx_python",
    "initial_python",
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]


# verification parameters
# tfc and lfc are both MNIST MLPs; cnv is CIFAR-10. The verification I/O is a
# (input image, top-1 class label) pair, and the label is robust to the network /
# quantization (both tfc and lfc classify the shared MNIST sample to the same
# digit), so lfc reuses the tfc MNIST I/O directly -- no lfc-specific golden file.
def get_verify_input_npy(model):
    stem = "tfc_mnist" if model.startswith(("tfc-", "lfc-")) else "cnv_cifar10"
    return os.path.join(verification_io_dir(), stem + "_input.npy")


def get_verify_output_npy(model):
    stem = "tfc_mnist" if model.startswith(("tfc-", "lfc-")) else "cnv_cifar10"
    return os.path.join(verification_io_dir(), stem + "_output.npy")


def configure_build(board, model, output_dir, **overrides):
    # The folding/specialize configs are board-agnostic: they are tuned for the
    # smallest target in the suite (AUP-ZU3_8GB, ZU3EG, no URAM), so they also fit
    # the larger KV260_SOM and U55C parts. One config file per (model) serves every
    # board.
    folding, specialize = benchmark_config_paths(
        "bnn-pynq",
        f"{model}_folding_config",
        f"{model}_specialize_layers",
    )
    cfg = dict(
        folding_config_file=folding,
        specialize_layers_config_file=specialize,
        synth_clk_period_ns=10.0,
        stitched_ip_gen_dcp=False,
        inject_steps_before={
            "step_qonnx_to_finn": [custom_step_add_preproc, custom_step_add_postproc]
        },
        verify_steps=get_verify_steps(verif_steps, board_enabled=(board == BASELINE_BOARD)),
        verify_input_npy=get_verify_input_npy(model),
        verify_expected_output_npy=get_verify_output_npy(model),
        default_swg_exception=True,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


# Baseline-board scheme: correctness (streamline/convert/fold + numeric
# verification) is board/part-independent across the suite's UltraScale+ targets,
# so the baseline board builds every (model, datatype) and is the only one that
# verifies. Every other board builds a single representative that exercises its
# distinct shell flow + fabric fit, with verification OFF. The representatives are
# chosen to equal the retired end2end bnn_pynq sanity configs so coverage is
# preserved: KV260_SOM -> cnv-w1a2 == (w1,a2,cnv,KV260_SOM); U55C -> cnv-w2a2 ==
# (w2,a2,cnv,U55C). The baseline additionally covers lfc-w1a1 == (w1,a1,lfc,
# AUP-ZU3_8GB), the last end2end sanity config.
BASELINE_BOARD = "AUP-ZU3_8GB"
_BASELINE_MODELS = [
    "tfc-w1a1",
    "tfc-w1a2",
    "tfc-w2a2",
    "cnv-w1a1",
    "cnv-w1a2",
    "cnv-w2a2",
    "lfc-w1a1",
    "lfc-w1a2",
]
_EXTRA_BUILDS = [
    ("KV260_SOM", "cnv-w1a2"),
    ("U55C", "cnv-w2a2"),
]

BNN_BUILDS = [(BASELINE_BOARD, m) for m in _BASELINE_MODELS] + _EXTRA_BUILDS


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board,model", BNN_BUILDS)
def test_bnnpynq(board, model, bench_recorder):
    output_dir = make_build_dir(f"build_bnn-pynq_{model}_{board}_")

    # Run build flow
    cfg = configure_build(board, model, output_dir)
    model_file = get_model_file(model)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one. The bitfile
    # artifacts depend on the shell flow (Zynq boards -> .bit/.hwh, U55C -> .xclbin).
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)
