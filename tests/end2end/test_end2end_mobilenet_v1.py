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
import time
import pytest

from PIL import Image
import os
import numpy as np
import brevitas.onnx as bo
import torch

from finn.custom_op.registry import getCustomOp
from finn.util.pytorch import NormalizePreProc
from finn.util.test import (
    get_test_model_trained,
    load_test_checkpoint_or_skip,
    resize_smaller_side,
    crop_center,
)

from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType

from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
)
from finn.transformation.merge_onnx_models import MergeONNXModels
from finn.transformation.insert_topk import InsertTopK
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline import Streamline
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.streamline.remove import RemoveIdentityOps
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from finn.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.core.onnx_exec import execute_onnx
from finn.util.basic import alveo_part_map, alveo_default_platform
from finn.util.config import extract_model_config_to_json
from finn.transformation.fpgadataflow.vitis_build import VitisBuild, VitisOptStrategy

build_dir = os.environ["FINN_BUILD_DIR"]

test_board = "U250"
test_platform = alveo_default_platform[test_board]
test_fpga_part = alveo_part_map[test_board]
target_clk_ns = 3
mem_mode = "decoupled"
large_fifo_ram_style = "ultra"
extra_fold = 1
first_layer_res_type = "dsp"


def test_end2end_mobilenet_export():
    # export preprocessing
    preproc_onnx = build_dir + "/end2end_mobilenet_preproc.onnx"
    mean = [0.485, 0.456, 0.406]
    std = 0.226
    ch = 3
    preproc = NormalizePreProc(mean, std, ch)
    bo.export_finn_onnx(preproc, (1, 3, 224, 224), preproc_onnx)
    preproc_model = ModelWrapper(preproc_onnx)
    # set input finn datatype to UINT8
    preproc_model.set_tensor_datatype(preproc_model.graph.input[0].name, DataType.UINT8)
    preproc_model = preproc_model.transform(InferShapes())
    preproc_model = preproc_model.transform(GiveUniqueNodeNames())
    preproc_model = preproc_model.transform(GiveUniqueParameterTensors())
    preproc_model = preproc_model.transform(GiveReadableTensorNames())
    preproc_model.save(build_dir + "/end2end_mobilenet_preproc.onnx")

    # export mobilenet
    finn_onnx = build_dir + "/end2end_mobilenet_export.onnx"
    mobilenet = get_test_model_trained("mobilenet", 4, 4)
    bo.export_finn_onnx(mobilenet, (1, 3, 224, 224), finn_onnx)

    # calculate golden output with pytorch/brevitas and save as .npy
    # get single image as input and prepare image
    img = Image.open("/workspace/finn/tests/brevitas/king_charles.jpg")
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


def test_end2end_mobilenet_tidy_and_merge_with_preproc():
    preproc_model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_mobilenet_preproc.onnx"
    )
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_export.onnx")
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


def test_end2end_mobilenet_lowering():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_mobilenet_streamlined.onnx"
    )
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())
    model.save(build_dir + "/end2end_mobilenet_lowered.onnx")


def test_end2end_mobilenet_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_lowered.onnx")
    model = model.transform(to_hls.InferPool_Batch())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferVVAU())
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferChannelwiseLinearLayer())
    model = model.transform(to_hls.InferLabelSelectLayer())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(build_dir + "/end2end_mobilenet_hls_layers.onnx")


def test_end2end_mobilenet_folding():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_mobilenet_hls_layers.onnx"
    )
    # optional extra folding to use fewer resources
    # applied while setting the attributes on each node
    assert extra_fold in [1, 2, 4]
    # set up folding for the depthwise conv layers impl'd by VVAUs
    # each value is PE for a layer
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # each tuple is (PE, SIMD, ram_style) for a layer
    folding = [
        (32, 3, "block"),
        (16, 16, "block"),
        (16, 16, "block"),
        (32, 16, "block"),
        (16, 16, "block"),
        (32, 16, "block"),
        (16, 16, "block"),
        (32, 16, "block"),
        (32, 16, "block"),
        (32, 16, "block"),
        (32, 16, "block"),
        (32, 16, "block"),
        (16, 16, "block"),
        (32, 16, "block"),
        (4, 4, "block"),
    ]
    for fcl, (pe, simd, ramstyle) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe // extra_fold)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
    # first layer uses 8-bit weights & activations
    # control its compute resource type explicitly
    getCustomOp(fc_layers[0]).set_nodeattr("resType", first_layer_res_type)
    # set up folding for the depthwise conv layers impl'd by VVAUs
    # each value is PE for a layer
    vvau_layers = model.get_nodes_by_op_type("Vector_Vector_Activate_Batch")
    folding = [32, 32, 64, 16, 32, 8, 16, 16, 16, 16, 16, 4, 8]
    for vvau, pe in zip(vvau_layers, folding):
        vvau_inst = getCustomOp(vvau)
        vvau_inst.set_nodeattr("PE", pe // extra_fold)
        # set SIMD in preceeding ConvInputGen to same value
        convinputgen = model.find_direct_predecessors(vvau)[0]
        convinputgen_inst = getCustomOp(convinputgen)
        convinputgen_inst.set_nodeattr("SIMD", pe // extra_fold)
        # set SIMD in preceeding FMPadding to same value
        padding = model.find_direct_predecessors(convinputgen)[0]
        if padding.op_type == "FMPadding_Batch":
            padding_inst = getCustomOp(padding)
            padding_inst.set_nodeattr("SIMD", pe // extra_fold)
    # adjust final pooling layer + its inpgen
    pool_node = model.get_nodes_by_op_type("Pool_Batch")[0]
    pool_inst = getCustomOp(pool_node)
    pool_inst.set_nodeattr("PE", 4 // extra_fold)
    pool_inpgen = model.find_direct_predecessors(pool_node)[0]
    pool_inpgen_inst = getCustomOp(pool_inpgen)
    pool_inpgen_inst.set_nodeattr("SIMD", 4 // extra_fold)
    model = model.transform(InferDataLayouts())
    model.save(build_dir + "/end2end_mobilenet_folded.onnx")


def test_end2end_mobilenet_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_folded.onnx")
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_mobilenet_dataflow_parent.onnx")
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
    dataflow_model = dataflow_model.transform(RemoveUnusedTensors())
    dataflow_model.save(build_dir + "/end2end_mobilenet_dataflow_model.onnx")


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.xfail
def test_end2end_mobilenet_cppsim():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_folded.onnx")
    x = np.load(build_dir + "/end2end_mobilenet_input.npy")
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
    golden_prob = np.load(build_dir + "/end2end_mobilenet_golden_top5_prob.npy")

    assert (golden == res_cppsim).all()
    assert np.isclose(golden_prob, res_cppsim_prob).all()


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_mobilenet_gen_hls_ip():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_mobilenet_dataflow_model.onnx"
    )
    start = time.time()
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(ReplaceVerilogRelPaths())
    end = time.time()
    elapsed_time = end - start
    f = open(build_dir + "/end2end_mobilenet_ipgen_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()

    model = model.transform(AnnotateResources("hls"))
    model.save(build_dir + "/end2end_mobilenet_ipgen.onnx")


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.xfail
def test_end2end_mobilenet_rtlsim():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_ipgen.onnx")
    x = np.load(build_dir + "/end2end_mobilenet_input.npy")
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp_dict = {inp_name: x}
    # node-by-node rtlsim
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())
    model.save(build_dir + "/end2end_mobilenet_ipgen_nodebynode_rtlsim.onnx")
    ret_rtlsim_nodebynode = execute_onnx(model, inp_dict, True)
    res_rtlsim_nodebynode = ret_rtlsim_nodebynode[out_name]
    np.save(
        build_dir + "/end2end_mobilenet_result_rtlsim_nodebynode.npy",
        res_rtlsim_nodebynode,
    )
    a0 = np.load(build_dir + "/end2end_mobilenet_topk_scale.npy")
    res_rtlsim_nodebynode_prob = (
        ret_rtlsim_nodebynode[model.graph.node[-2].output[0]] * a0
    )
    np.save(
        build_dir + "/end2end_mobilenet_result_rtlsim_nodebynode_prob.npy",
        res_rtlsim_nodebynode_prob,
    )

    # check result with golden values
    golden = np.load(build_dir + "/end2end_mobilenet_golden_top5.npy")
    golden_prob = np.load(build_dir + "/end2end_mobilenet_golden_top5_prob.npy")

    assert (golden == res_rtlsim_nodebynode).all()
    assert np.isclose(golden_prob, res_rtlsim_nodebynode_prob).all()


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_mobilenet_set_fifo_depths():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_mobilenet_ipgen.onnx")
    start = time.time()
    model = model.transform(
        InsertAndSetFIFODepths(
            test_fpga_part, target_clk_ns, vivado_ram_style=large_fifo_ram_style
        )
    )
    end = time.time()
    elapsed_time = end - start
    f = open(build_dir + "/end2end_mobilenet_fifoset_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()
    extract_model_config_to_json(
        model,
        build_dir + "/end2end_mobilenet_folded_and_fifo_config.json",
        ["PE", "SIMD", "impl_style", "ram_style", "depth"],
    )
    model.save(build_dir + "/end2end_mobilenet_fifodepth.onnx")


@pytest.mark.slow
@pytest.mark.vitis
def test_end2end_mobilenet_build():
    model = load_test_checkpoint_or_skip(
        build_dir + "/end2end_mobilenet_fifodepth.onnx"
    )
    model = model.transform(
        VitisBuild(
            test_fpga_part,
            target_clk_ns,
            test_platform,
            strategy=VitisOptStrategy.PERFORMANCE_BEST,
        )
    )
    model.save(build_dir + "/end2end_mobilenet_build.onnx")
    model = model.transform(AnnotateResources("synth"))
    model.save(build_dir + "/end2end_mobilenet_final.onnx")
