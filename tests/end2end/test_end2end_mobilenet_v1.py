# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
import os
import time
import torch
from brevitas.export import export_qonnx
from PIL import Image
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    RemoveShallowFIFOs,
    SplitLargeFIFOs,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import get_finn_root
from finn.util.pytorch import NormalizePreProc
from finn.util.pyverilator import verilator_fifosim
from finn.util.test import (
    crop_center,
    get_test_model_trained,
    load_test_checkpoint_or_skip,
    resize_smaller_side,
)

build_dir = os.environ["FINN_BUILD_DIR"]

# Select Versal device such that RTL VVU (i.e. DSP58) can be enabled
fpga_part = "xcvm1802-vsvd1760-2MP-e-S"
target_clk_ns = 3
extra_fold = 1
first_layer_res_type = "dsp"


@pytest.mark.end2end
def test_end2end_mobilenet_export():
    # export preprocessing
    preproc_onnx = build_dir + "/end2end_mobilenet_preproc.onnx"
    mean = [0.485, 0.456, 0.406]
    std = 0.226
    ch = 3
    preproc = NormalizePreProc(mean, std, ch)
    export_qonnx(preproc, torch.randn(1, 3, 224, 224), preproc_onnx)
    qonnx_cleanup(preproc_onnx, out_file=preproc_onnx)
    preproc_model = ModelWrapper(preproc_onnx)
    preproc_model = preproc_model.transform(ConvertQONNXtoFINN())
    # set input finn datatype to UINT8
    preproc_model.set_tensor_datatype(preproc_model.graph.input[0].name, DataType["UINT8"])
    preproc_model = preproc_model.transform(InferShapes())
    preproc_model = preproc_model.transform(FoldConstants())
    preproc_model = preproc_model.transform(GiveUniqueNodeNames())
    preproc_model = preproc_model.transform(GiveUniqueParameterTensors())
    preproc_model = preproc_model.transform(GiveReadableTensorNames())
    preproc_model.save(build_dir + "/end2end_mobilenet_preproc.onnx")

    # export mobilenet
    finn_onnx = build_dir + "/end2end_mobilenet_export.onnx"
    mobilenet = get_test_model_trained("mobilenet", 4, 4)
    export_qonnx(mobilenet, torch.randn(1, 3, 224, 224), finn_onnx)
    qonnx_cleanup(finn_onnx, out_file=finn_onnx)

    # calculate golden output with pytorch/brevitas and save as .npy
    # get single image as input and prepare image
    img = Image.open(get_finn_root() + "/tests/brevitas/king_charles.jpg")
    # resize smallest side of the image to 256 pixels and resize larger side
    # with same ratio
    img = resize_smaller_side(256, img)
    # crop central 224*224 window
    img = crop_center(224, img)
    # save image as numpy array and as torch tensor to enable testing in
    # brevitas/pytorch and finn and transpose from (H, W, C) to (C, H, W)
    img_np = np.asarray(img).copy().astype(np.float32).transpose(2, 0, 1)
    img_np = img_np.reshape(1, 3, 224, 224)
    np.save(build_dir + "/end2end_mobilenet_input.npy", img_np)
    img_torch = torch.from_numpy(img_np).float()
    # do forward pass in PyTorch/Brevitas
    input_tensor = preproc.forward(img_torch)
    golden = mobilenet.forward(input_tensor).detach().numpy()
    golden_topk = golden.flatten()
    golden_top5 = np.argsort(golden_topk)[-5:]
    golden_top5 = np.flip(golden_top5)
    golden_top5_prob = []
    for index in golden_top5:
        golden_top5_prob.append(golden_topk[index])
    # save golden output values
    np.save(build_dir + "/end2end_mobilenet_golden_top5.npy", golden_top5)
    np.save(build_dir + "/end2end_mobilenet_golden_top5_prob.npy", golden_top5_prob)
    assert os.path.isfile(finn_onnx)
    assert os.path.isfile(build_dir + "/end2end_mobilenet_preproc.onnx")


@pytest.mark.end2end
def test_end2end_mobilenet_tidy_and_merge_with_preproc():
    preproc_model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_preproc.onnx")
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_export.onnx")
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(InsertTopK())
    # get initializer from Mul that will be absorbed into topk
    a0 = model.get_initializer(model.graph.node[-2].input[1])
    np.save(build_dir + "/end2end_mobilenet_topk_scale.npy", a0)
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(MergeONNXModels(preproc_model))
    model.save(build_dir + "/end2end_mobilenet_tidy.onnx")


@pytest.mark.end2end
def test_end2end_mobilenet_streamline():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_tidy.onnx")
    model = model.transform(Streamline())
    additional_streamline_transformations = [
        DoubleToSingleFloat(),
        reorder.MoveMulPastDWConv(),
        absorb.AbsorbMulIntoMultiThreshold(),
        ChangeDataLayoutQuantAvgPool2d(),
        InferDataLayouts(),
        reorder.MoveTransposePastScalarMul(),
        absorb.AbsorbTransposeIntoFlatten(),
        reorder.MoveFlattenPastAffine(),
        reorder.MoveFlattenPastTopK(),
        reorder.MoveScalarMulPastMatMul(),
        CollapseRepeatedMul(),
        RemoveIdentityOps(),
        RoundAndClipThresholds(),
    ]
    for trn in additional_streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
    model.save(build_dir + "/end2end_mobilenet_streamlined.onnx")
    assert len(model.get_nodes_by_op_type("Add")) == 1  # only final quantized bias Add op remains
    assert len(model.get_nodes_by_op_type("Mul")) == 0  # no Mul ops remain


@pytest.mark.end2end
def test_end2end_mobilenet_lowering():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_streamlined.onnx")
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())
    model.save(build_dir + "/end2end_mobilenet_lowered.onnx")


@pytest.mark.end2end
def test_end2end_mobilenet_convert_to_hw_layers():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_lowered.onnx")
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/end2end_mobilenet_hw_layers.onnx")


@pytest.mark.end2end
def test_end2end_mobilenet_specialize_layers():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_hw_layers.onnx")
    model = model.transform(SpecializeLayers(fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/end2end_mobilenet_specialize_layers.onnx")


@pytest.mark.end2end
def test_end2end_mobilenet_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_specialize_layers.onnx")
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_mobilenet_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
    dataflow_model = dataflow_model.transform(RemoveUnusedTensors())
    dataflow_model.save(build_dir + "/end2end_mobilenet_dataflow_model.onnx")


@pytest.mark.end2end
def test_end2end_mobilenet_folding():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_dataflow_model.onnx")
    # optional extra folding to use fewer resources
    # applied while setting the attributes on each node
    assert extra_fold in [1, 2, 4]
    # set up folding for the conv layers impl'd by MVAUs
    # each value is PE for a layer
    fc_layers = model.get_nodes_by_op_type("MVAU_rtl")
    # each tuple is (PE, SIMD, ram_style) for a layer
    folding = [
        (16, 3, "block"),
        (8, 16, "distributed"),
        (8, 16, "distributed"),
        (16, 16, "distributed"),
        (8, 16, "distributed"),
        (16, 16, "distributed"),
        (8, 16, "block"),
        (16, 16, "block"),
        (16, 16, "block"),
        (16, 16, "block"),
        (16, 16, "block"),
        (16, 16, "block"),
        (8, 16, "block"),
        (16, 16, "block"),
        (4, 4, "block"),
    ]
    for fcl, (pe, simd, ramstyle) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe // extra_fold)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
    # set up folding for the depthwise conv layers impl'd by VVAUs
    # each value is PE for a layer
    vvau_layers = model.get_nodes_by_op_type("VVAU_rtl")
    pe_simd_fold = [
        [16, 3],
        [8, 3],
        [16, 3],
        [4, 3],
        [8, 3],
        [2, 3],
        [4, 3],
        [4, 3],
        [4, 3],
        [4, 3],
        [4, 3],
        [1, 3],
        [2, 3],
    ]
    for vvau, pe_simd in zip(vvau_layers, pe_simd_fold):
        pe, simd = pe_simd
        vvau_inst = getCustomOp(vvau)
        vvau_inst.set_nodeattr("PE", pe // extra_fold)
        vvau_inst.set_nodeattr("SIMD", simd)
        # set SIMD in preceeding ConvInputGen to same value
        convinputgen = model.find_direct_predecessors(vvau)[0]
        convinputgen_inst = getCustomOp(convinputgen)
        convinputgen_inst.set_nodeattr("SIMD", pe // extra_fold)
        # Enable parallel_window mode for SIMD parallelism VVU
        convinputgen_inst.set_nodeattr("parallel_window", 1)
        # set SIMD in preceeding FMPadding to same value
        padding = model.find_direct_predecessors(convinputgen)[0]
        if padding.op_type == "FMPadding_rtl":
            padding_inst = getCustomOp(padding)
            padding_inst.set_nodeattr("SIMD", pe // extra_fold)
    # Set folding Thresholding layers
    thresholding_layers = model.get_nodes_by_op_type("Thresholding_rtl")
    folding = [2, 2, 4, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for thresholding, pe in zip(thresholding_layers, folding):
        thresholding_inst = getCustomOp(thresholding)
        thresholding_inst.set_nodeattr("PE", pe)
    # adjust final pooling layer + its inpgen
    pool_node = model.get_nodes_by_op_type("Pool_hls")[0]
    pool_inst = getCustomOp(pool_node)
    pool_inst.set_nodeattr("PE", 4 // extra_fold)
    pool_inpgen = model.find_direct_predecessors(pool_node)[0]
    pool_inpgen_inst = getCustomOp(pool_inpgen)
    pool_inpgen_inst.set_nodeattr("SIMD", 4 // extra_fold)
    model = model.transform(InferDataLayouts())
    model.save(build_dir + "/end2end_mobilenet_folded.onnx")


@pytest.mark.end2end
def test_end2end_mobilenet_minimize_bit_width():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_folded.onnx")
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(RoundAndClipThresholds())
    model.save(build_dir + "/end2end_mobilenet_minimize_bitwidth.onnx")


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_mobilenet_cppsim():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_minimize_bitwidth.onnx")
    x = np.load(build_dir + "/end2end_mobilenet_input.npy")
    x = x.transpose(0, 2, 3, 1)  # Convert NCHW to NHWC
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp_dict = {inp_name: x}
    start = time.time()
    # cppsim
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))
    end = time.time()
    elapsed_time = end - start
    f = open(build_dir + "/end2end_mobilenet_compile_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()
    model.save(build_dir + "/end2end_mobilenet_cppsim.onnx")
    ret_cppsim = execute_onnx(model, inp_dict, True)
    res_cppsim = ret_cppsim[out_name]
    np.save(build_dir + "/end2end_mobilenet_result_cppsim.npy", res_cppsim)
    a0 = np.load(build_dir + "/end2end_mobilenet_topk_scale.npy")
    res_cppsim_prob = ret_cppsim[model.graph.node[-2].output[0]] * a0
    np.save(build_dir + "/end2end_mobilenet_result_cppsim_prob.npy", res_cppsim_prob)

    # check result with golden values
    golden = np.load(build_dir + "/end2end_mobilenet_golden_top5.npy")
    # golden_prob = np.load(build_dir + "/end2end_mobilenet_golden_top5_prob.npy")

    assert (golden == res_cppsim).all()
    # assert np.isclose(golden_prob, res_cppsim_prob[0, 0, 0, :5]).all()


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_mobilenet_ipgen():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_minimize_bitwidth.onnx")
    model = model.transform(PrepareIP(fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model.save(build_dir + "/end2end_mobilenet_hw_ipgen.onnx")


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_mobilenet_rtlsim():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_hw_ipgen.onnx")
    # use critical path estimate to set rtlsim liveness threshold
    # (very conservative)
    model = model.transform(AnnotateCycles())
    estimate_network_performance = model.analysis(dataflow_performance)
    os.environ["LIVENESS_THRESHOLD"] = str(
        int(estimate_network_performance["critical_path_cycles"])
    )
    x = np.load(build_dir + "/end2end_mobilenet_input.npy")
    x = x.transpose(0, 2, 3, 1)  # Convert NCHW to NHWC
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp_dict = {inp_name: x}
    # rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    model.save(build_dir + "/end2end_mobilenet_rtlsim.onnx")
    ret_rtlsim = execute_onnx(model, inp_dict, True)
    res_rtlsim = ret_rtlsim[out_name]
    np.save(build_dir + "/end2end_mobilenet_result_rtlsim.npy", res_rtlsim)
    a0 = np.load(build_dir + "/end2end_mobilenet_topk_scale.npy")
    res_rtlsim_prob = ret_rtlsim[model.graph.node[-2].output[0]] * a0
    np.save(build_dir + "/end2end_mobilenet_result_rtlsim_prob.npy", res_rtlsim_prob)

    # check result with golden values
    golden = np.load(build_dir + "/end2end_mobilenet_golden_top5.npy")
    # golden_prob = np.load(build_dir + "/end2end_mobilenet_golden_top5_prob.npy")

    assert (golden == res_rtlsim).all()
    # assert np.isclose(golden_prob, res_rtlsim_prob[0, 0, 0, :5]).all()


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_mobilenet_set_fifo_depths():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_hw_ipgen.onnx")
    model = model.transform(
        InsertAndSetFIFODepths(
            fpga_part,
            target_clk_ns,
            swg_exception=False,
            vivado_ram_style="auto",
            force_python_sim=False,
        )
    )
    # perform FIFO splitting and shallow FIFO removal only after the final config
    # json file has been written. otherwise, since these transforms may add/remove
    # FIFOs, we get name mismatch problems when trying to reuse the final config.
    model = model.transform(SplitLargeFIFOs())
    model = model.transform(RemoveShallowFIFOs())
    # after FIFOs are ready to go, call PrepareIP and HLSSynthIP again
    # this will only run for the new nodes (e.g. FIFOs and DWCs)
    model = model.transform(PrepareIP(fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model.save(build_dir + "/end2end_mobilenet_set_fifo_depths.onnx")


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_mobilenet_stitched_ip():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_set_fifo_depths.onnx")
    model = model.transform(
        CreateStitchedIP(
            fpga_part,
            target_clk_ns,
            vitis=False,
            signature=None,
        )
    )
    model.save(build_dir + "/end2end_mobilenet_stitched_ip.onnx")


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_mobilenet_stitched_ip_rtlsim():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_stitched_ip.onnx")
    # use critical path estimate to set rtlsim liveness threshold
    # (very conservative)
    model = model.transform(AnnotateCycles())
    estimate_network_performance = model.analysis(dataflow_performance)
    os.environ["LIVENESS_THRESHOLD"] = str(
        int(estimate_network_performance["critical_path_cycles"])
    )
    # Prepare input
    x = np.load(build_dir + "/end2end_mobilenet_input.npy")
    x = x.transpose(0, 2, 3, 1)  # Convert NCHW to NHWC
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp_dict = {inp_name: x}

    # set top-level prop for stitched-ip rtlsim and launch
    model.set_metadata_prop("exec_mode", "rtlsim")
    ret_rtlsim_ip = execute_onnx(model, inp_dict, True)
    res_rtlsim_ip = ret_rtlsim_ip[out_name]
    np.save(build_dir + "/end2end_mobilenet_result_rtlsim_ip.npy", res_rtlsim_ip)
    a0 = np.load(build_dir + "/end2end_mobilenet_topk_scale.npy")
    res_rtlsim_ip_prob = ret_rtlsim_ip[model.graph.node[-2].output[0]] * a0
    np.save(build_dir + "/end2end_mobilenet_result_cppsim_prob.npy", res_rtlsim_ip_prob)

    # check result with golden values
    golden = np.load(build_dir + "/end2end_mobilenet_golden_top5.npy")
    # golden_prob = np.load(build_dir + "/end2end_mobilenet_golden_top5_prob.npy")

    assert (golden == res_rtlsim_ip).all()
    # assert np.isclose(golden_prob, res_rtlsim_ip_prob[0, 0, 0, :5]).all()


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_mobilenet_rtlsim_performance():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_stitched_ip.onnx")
    report_dir = build_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)
    # multi-in/out streams currently not supported in our C++ verilator driver
    rtlsim_bs = 1

    rtlsim_perf_dict = verilator_fifosim(model, rtlsim_bs)
    # keep keys consistent between the Python and C++-styles
    cycles = rtlsim_perf_dict["cycles"]
    clk_ns = float(model.get_metadata_prop("clk_ns"))
    fclk_mhz = 1 / (clk_ns * 0.001)
    runtime_s = (cycles * clk_ns) * (10**-9)
    rtlsim_perf_dict["runtime[ms]"] = runtime_s * 1000
    rtlsim_perf_dict["throughput[images/s]"] = rtlsim_bs / runtime_s
    rtlsim_perf_dict["fclk[mhz]"] = fclk_mhz
    for key, val in rtlsim_perf_dict.items():
        if "max_count" in key:
            del rtlsim_perf_dict[key]
    # estimate stable-state throughput based on latency+throughput
    rtlsim_perf_dict["stable_throughput[images/s]"] = rtlsim_perf_dict["throughput[images/s]"]

    model.save(build_dir + "/end2end_mobilenet_rtlsim_performance.onnx")
