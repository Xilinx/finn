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

import itertools
import numpy as np

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA
import os
import torch
import warnings
from brevitas.export import export_qonnx
from dataset_loading import cifar, mnist
from distutils.dir_util import copy_tree
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from shutil import copy

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.core.onnx_exec import execute_onnx
from finn.core.throughput_test import throughput_test_rtlsim
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import get_finn_root, make_build_dir, test_board_map
from finn.util.pytorch import ToTensor
from finn.util.test import (
    execute_parent,
    get_build_env,
    get_example_input,
    get_topk,
    get_trained_network_and_ishape,
    load_test_checkpoint_or_skip,
)

build_dir = os.environ["FINN_BUILD_DIR"]
target_clk_ns = 20
mem_mode = "internal_decoupled"
rtlsim_trace = False


def get_checkpoint_name(topology, wbits, abits, step):
    return build_dir + "/end2end_%s_w%da%d_%s.onnx" % (
        topology,
        wbits,
        abits,
        step,
    )


def fold_tfc(model):
    fc_layers = model.get_nodes_by_op_type("MVAU_hls")
    # (PE, SIMD, ramstyle) for each layer
    config = [(16, 49, "block"), (8, 8, "auto"), (8, 8, "auto"), (10, 8, "distributed")]
    for fcl, (pe, simd, ramstyle) in zip(fc_layers, config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
        fcl_inst.set_nodeattr("mem_mode", "internal_decoupled")
        fcl_inst.set_nodeattr("resType", "lut")
    # set parallelism for input quantizer to be same as first layer's SIMD
    inp_qnt_node = model.get_nodes_by_op_type("Thresholding_rtl")[0]
    inp_qnt = getCustomOp(inp_qnt_node)
    inp_qnt.set_nodeattr("PE", 49)
    inp_qnt.set_nodeattr("runtime_writeable_weights", 1)
    return model


def fold_lfc(model):
    fc_layers = model.get_nodes_by_op_type("MVAU_hls")
    # (PE, SIMD, ramstyle) for each layer
    config = [
        (32, 49, "block"),
        (64, 32, "auto"),
        (32, 64, "auto"),
        (10, 8, "distributed"),
    ]
    for fcl, (pe, simd, ramstyle) in zip(fc_layers, config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
        fcl_inst.set_nodeattr("runtime_writeable_weights", 1)
        fcl_inst.set_nodeattr("mem_mode", "internal_decoupled")
        fcl_inst.set_nodeattr("resType", "lut")
    # set parallelism for input quantizer to be same as first layer's SIMD
    inp_qnt_node = model.get_nodes_by_op_type("Thresholding_rtl")[0]
    inp_qnt = getCustomOp(inp_qnt_node)
    inp_qnt.set_nodeattr("PE", 49)
    return model


def fold_cnv_large(model):
    fc_layers = model.get_nodes_by_op_type("MVAU_hls")
    # each tuple is (PE, SIMD) for a layer
    folding = [
        (16, 3),
        (32, 32),
        (16, 32),
        (16, 32),
        (4, 32),
        (1, 32),
        (1, 4),
        (1, 8),
        (5, 1),
    ]
    for fcl, (pe, simd) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("mem_mode", "internal_decoupled")
        fcl_inst.set_nodeattr("resType", "lut")

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
        swg_inst.set_nodeattr("ram_style", "distributed")
    return model


def fold_cnv_small(model):
    fc_layers = model.get_nodes_by_op_type("MVAU_hls")
    # each tuple is (PE, SIMD) for a layer
    folding = [
        (8, 3, "distributed"),
        (16, 16, "distributed"),
        (8, 16, "auto"),
        (8, 16, "distributed"),
        (4, 8, "auto"),
        (1, 8, "auto"),
        (1, 2, "block"),
        (2, 2, "auto"),
        (5, 1, "distributed"),
    ]
    for fcl, (pe, simd, ramstyle) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
        fcl_inst.set_nodeattr("mem_mode", "internal_decoupled")
        fcl_inst.set_nodeattr("resType", "lut")

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
        swg_inst.set_nodeattr("ram_style", "distributed")
    inp_qnt_node = model.get_nodes_by_op_type("Thresholding_rtl")[0]
    inp_qnt = getCustomOp(inp_qnt_node)
    inp_qnt.set_nodeattr("depth_trigger_uram", 32000)
    inp_qnt.set_nodeattr("depth_trigger_bram", 32000)
    return model


def get_folding_function(topology, wbits, abits):
    if "tfc" in topology:
        return fold_tfc
    elif "lfc" in topology:
        return fold_lfc
    elif "cnv" in topology:
        if wbits == 1 and abits == 1:
            return fold_cnv_large
        else:
            return fold_cnv_small
    else:
        raise Exception("Unknown topology/quantization combo for predefined folding")


def get_golden_io_pair(topology, wbits, abits, preproc=ToTensor(), return_topk=None):
    (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    input_tensor_npy = get_example_input(topology)
    input_tensor_torch = torch.from_numpy(input_tensor_npy).float()
    if preproc is not None:
        input_tensor_torch = preproc.forward(input_tensor_torch).detach()
    output_tensor_npy = model.forward(input_tensor_torch).detach().numpy()
    if return_topk is not None:
        output_tensor_npy = get_topk(output_tensor_npy, k=return_topk)
    return (input_tensor_npy, output_tensor_npy)


def measure_top1_accuracy(model_chkpt, dataset, parent_chkpt=None):
    if dataset == "cifar10":
        trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
            get_finn_root() + "/dataset", download=True, one_hot=False
        )
    elif dataset == "mnist":
        trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data(
            get_finn_root() + "/dataset", download=True, one_hot=False
        )
    else:
        raise Exception("Unrecognized dataset")
    # move from dataset_loader layout to ONNX layout: NHWC -> NCHW
    testx = testx.transpose(0, 3, 1, 2)
    model = load_test_checkpoint_or_skip(model_chkpt)
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    if parent_chkpt is None:
        ishape = model.get_tensor_shape(iname)
    else:
        parent_model = ModelWrapper(parent_chkpt)
        parent_iname = parent_model.graph.input[0].name
        ishape = parent_model.get_tensor_shape(parent_iname)
    ok = 0
    nok = 0
    n_batches = testx.shape[0]
    for i in range(n_batches):
        tdata = testx[i].reshape(ishape).astype(np.float32)
        exp = testy[i].item()
        if parent_chkpt is not None:
            y = execute_parent(parent_chkpt, model_chkpt, tdata)
        else:
            y = execute_onnx(model, {iname: tdata}, False)[oname]
        ret = y.item()
        if ret == exp:
            ok += 1
        else:
            nok += 1
        if i % 10 == 0:
            print("%d : OK %d NOK %d " % (i, ok, nok))
    acc_top1 = ok * 100.0 / (ok + nok)
    warnings.warn("Final OK %d NOK %d top-1 %f" % (ok, nok, acc_top1))
    return acc_top1


def topology2dataset(topology):
    if "fc" in topology:
        return "mnist"
    elif "cnv" in topology:
        return "cifar10"
    else:
        raise Exception("Unrecognized topology")


def deploy_based_on_board(model, model_title, topology, wbits, abits, board):
    # Check if a deployment directory for this board type already exists
    if ("FINN_DEPLOY_DIR" in os.environ) and (board in os.environ["FINN_DEPLOY_DIR"]):
        deploy_dir_root = os.environ["FINN_DEPLOY_DIR"]
    else:
        deploy_dir_root = make_build_dir(prefix="hw_deployment_" + board + "_")
        # Set it for the next round if multiple bitstreams are selected for generation
        os.environ["FINN_DEPLOY_DIR"] = deploy_dir_root

    # create directory for deployment files
    deployment_dir = deploy_dir_root + "/" + board + "/" + model_title
    os.makedirs(deployment_dir)
    model.set_metadata_prop("pynq_deployment_dir", deployment_dir)

    # get and copy necessary files
    # .bit and .hwh file
    bitfile = model.get_metadata_prop("bitfile")
    hwh_file = model.get_metadata_prop("hw_handoff")
    deploy_files = [bitfile, hwh_file]

    for dfile in deploy_files:
        if dfile is not None:
            copy(dfile, deployment_dir)

    # create input and output test files
    (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
        topology, wbits, abits, return_topk=1
    )

    # Some changes are required in order to prepare the input tensor data for hardware
    # testing. The ONNX graphs for these models contain nodes that manipulate the input
    # tensor shape which FINN considers when creating the model. The same input tensor
    # shaping needs to be done here on the input data.
    # For the convolutional models, the graph contains the Transpose node. The Brevitas
    # model works in NCHW layout but the FINN kernels are optimized for NHWC.
    # The FC models contain a Reshape node, which FINN uses, so we therefore have to
    # reshape the input tensor data to match the reshaping in the model
    if topology == "cnv":
        input_tensor_npy = input_tensor_npy.transpose(0, 2, 3, 1)
    else:
        input_shape = input_tensor_npy.shape
        new_input_shape = (input_shape[0], np.prod(input_shape[1:]))
        input_tensor_npy = input_tensor_npy.reshape(new_input_shape)

    np.save(os.path.join(deployment_dir, "input.npy"), input_tensor_npy.copy())
    np.save(os.path.join(deployment_dir, "output_reference.npy"), output_tensor_npy)

    # driver.py and python libraries
    pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
    copy_tree(pynq_driver_dir, deployment_dir)
    model.set_metadata_prop("pynq_deploy_dir", deployment_dir)


# parameters that make up inputs to test case(s)
def get_full_parameterized_test_list(marker, wbits_list, abits_list, topology_list, board_list):
    test_cases = [
        (
            f"{marker}_w{param1}_a{param2}_{param3}_{param4}",
            {
                "wbits": param1,
                "abits": param2,
                "topology": param3,
                "board": param4,
            },
        )
        for param1, param2, param3, param4 in itertools.product(
            wbits_list,
            abits_list,
            topology_list,
            board_list,
        )
    ]
    return test_cases


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    scenarios = []

    # Full set of test parameters
    wbits = [1, 2]
    abits = [1, 2]
    topology = ["lfc", "tfc", "cnv"]

    # Separate the full list of markers used on command line.
    # This allows a user to select multiple markers
    all_markers_used = metafunc.config.getoption("-m").split(" ")

    for marker in all_markers_used:
        if "sanity_bnn" in marker:
            # Define a set of sanity tests that target each of
            # the supported boards with fixed parameters
            scenarios.extend(
                get_full_parameterized_test_list(
                    "sanity_bnn",
                    wbits_list=[1],
                    abits_list=[1],
                    topology_list=["lfc"],
                    board_list=[test_board_map[0]],
                )
            )
            scenarios.extend(
                get_full_parameterized_test_list(
                    "sanity_bnn",
                    wbits_list=[1],
                    abits_list=[2],
                    topology_list=["cnv"],
                    board_list=[test_board_map[1]],
                )
            )
            scenarios.extend(
                get_full_parameterized_test_list(
                    "sanity_bnn",
                    wbits_list=[2],
                    abits_list=[2],
                    topology_list=["tfc"],
                    board_list=[test_board_map[2]],
                )
            )
            scenarios.extend(
                get_full_parameterized_test_list(
                    "sanity_bnn",
                    wbits_list=[2],
                    abits_list=[2],
                    topology_list=["cnv"],
                    board_list=[test_board_map[3]],
                )
            )

        if "bnn_" in marker:
            # Target the full set of parameters for a single board
            # Extract the board name from the marker used, as it is in the form of 'bnn_<board>'
            bnn_board = next(
                (element for element in test_board_map if marker.split("_")[1] in element.lower()),
                None,
            )
            test_cases = get_full_parameterized_test_list(
                "bnn", wbits, abits, topology, [bnn_board]
            )
            scenarios.extend(test_cases)

    if len(scenarios) > 0:
        for scenario in scenarios:
            idlist.append(scenario[0])
            items = scenario[1].items()
            argnames = [x[0] for x in items]
            argvalues.append([x[1] for x in items])
        metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


@pytest.mark.sanity_bnn
@pytest.mark.bnn_pynq
@pytest.mark.bnn_zcu104
@pytest.mark.bnn_kv260
@pytest.mark.bnn_u250
class TestEnd2End:
    def test_export(self, topology, wbits, abits, board):
        if wbits > abits:
            pytest.skip("No wbits > abits end2end network configs for now")
        if topology == "lfc" and not (wbits == 1 and abits == 1):
            pytest.skip("Skipping certain lfc configs")
        (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
        chkpt_name = get_checkpoint_name(topology, wbits, abits, "export")
        export_qonnx(model, torch.randn(ishape), chkpt_name, opset_version=13)
        qonnx_cleanup(chkpt_name, out_file=chkpt_name)
        model = ModelWrapper(chkpt_name)
        model = model.transform(ConvertQONNXtoFINN())
        model.save(chkpt_name)
        assert os.path.isfile(chkpt_name)

    def test_import_and_tidy(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "export")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(RemoveStaticGraphInputs())
        chkpt = get_checkpoint_name(topology, wbits, abits, "import_and_tidy")
        model.save(chkpt)

    def test_add_pre_and_postproc(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "import_and_tidy")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        global_inp_name = model.graph.input[0].name
        ishape = model.get_tensor_shape(global_inp_name)
        # preprocessing: torchvision's ToTensor divides uint8 inputs by 255
        totensor_pyt = ToTensor()
        chkpt_preproc_name = get_checkpoint_name(topology, wbits, abits, "preproc")
        export_qonnx(totensor_pyt, torch.randn(ishape), chkpt_preproc_name, opset_version=13)
        qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)
        pre_model = ModelWrapper(chkpt_preproc_name)
        pre_model = pre_model.transform(ConvertQONNXtoFINN())
        pre_model.save(chkpt_preproc_name)
        assert os.path.isfile(chkpt_preproc_name)
        # join preprocessing and core model
        pre_model = ModelWrapper(chkpt_preproc_name)
        pre_model = pre_model.transform(InferShapes())
        pre_model = pre_model.transform(FoldConstants())
        model = model.transform(MergeONNXModels(pre_model))
        # add input quantization annotation: UINT8 for all BNN-PYNQ models
        global_inp_name = model.graph.input[0].name
        model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
        # postprocessing: insert Top-1 node at the end
        model = model.transform(InsertTopK(k=1))
        chkpt_name = get_checkpoint_name(topology, wbits, abits, "pre_post")
        # tidy-up again
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(RemoveStaticGraphInputs())
        model.save(chkpt_name)
        assert os.path.isfile(chkpt_name)

    def test_streamline(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "pre_post")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
        # move past any reshapes to be able to streamline input scaling
        model = model.transform(MoveScalarLinearPastInvariants())
        model = model.transform(Streamline())
        if "fc" not in topology:
            model = model.transform(LowerConvsToMatMul())
            model = model.transform(MakeMaxPoolNHWC())
            model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(ConvertBipolarMatMulToXnorPopcount())
        model = model.transform(Streamline())
        # absorb final add-mul nodes into TopK
        model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
        model = model.transform(InferDataLayouts())
        model = model.transform(RemoveUnusedTensors())
        model.save(get_checkpoint_name(topology, wbits, abits, "streamline"))

    def test_convert_to_hw_layers(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "streamline")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        if topology == "tfc" and wbits == 1 and abits == 1:
            # use standalone thresholds for tfc-w1a1 to also exercise that option
            model = model.transform(to_hw.InferThresholdingLayer())
        # needed for bipolar MatMul layers
        model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
        # needed for non-bipolar MatMul layers
        model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
        # TopK to LabelSelect
        model = model.transform(to_hw.InferLabelSelectLayer())
        # input quantization (if any) to standalone thresholding
        model = model.transform(to_hw.InferThresholdingLayer())
        # needed for convolutions
        if "fc" not in topology:
            model = model.transform(to_hw.InferConvInpGen())
            model = model.transform(to_hw.InferStreamingMaxPool())
            model = model.transform(RemoveCNVtoFCFlatten())
        # get rid of Tranpose -> Tranpose identity seq
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataLayouts())
        model.save(get_checkpoint_name(topology, wbits, abits, "convert_to_hw_layers"))
        exp_layer_counts = {
            "tfc": [
                ("Reshape", 1),
                ("Thresholding", 1),
                ("MVAU", 4),
                ("LabelSelect", 1),
            ],
            "tfc-1-1": [
                ("Reshape", 1),
                ("Thresholding", 4),
                ("MVAU", 4),
                ("LabelSelect", 1),
            ],
            "lfc": [
                ("Reshape", 1),
                ("Thresholding", 1),
                ("MVAU", 4),
                ("LabelSelect", 1),
            ],
            "cnv": [
                ("Transpose", 1),
                ("Thresholding", 1),
                ("ConvolutionInputGenerator", 6),
                ("MVAU", 9),
                ("StreamingMaxPool", 2),
                ("LabelSelect", 1),
            ],
        }
        if topology == "tfc" and wbits == 1 and abits == 1:
            exp_key = "tfc-1-1"
        else:
            exp_key = topology
        exp_layer_counts = exp_layer_counts[exp_key]
        for op_type, exp_count in exp_layer_counts:
            assert len(model.get_nodes_by_op_type(op_type)) == exp_count

    def test_specialize_layers(self, topology, wbits, abits, board):
        build_data = get_build_env(board, target_clk_ns)
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "convert_to_hw_layers")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(SpecializeLayers(build_data["part"]))
        model = model.transform(GiveUniqueNodeNames())
        model.save(get_checkpoint_name(topology, wbits, abits, "specialize_layers"))
        exp_layer_counts = {
            "tfc": [
                ("Reshape", 1),
                ("Thresholding_rtl", 1),
                ("MVAU_hls", 4),
                ("LabelSelect_hls", 1),
            ],
            "tfc-1-1": [
                ("Reshape", 1),
                ("Thresholding_rtl", 4),
                ("MVAU_hls", 4),
                ("LabelSelect_hls", 1),
            ],
            "lfc": [
                ("Reshape", 1),
                ("Thresholding_rtl", 1),
                ("MVAU_hls", 4),
                ("LabelSelect_hls", 1),
            ],
            "cnv": [
                ("Transpose", 1),
                ("Thresholding_rtl", 1),
                ("ConvolutionInputGenerator_rtl", 6),
                ("MVAU_hls", 9),
                ("StreamingMaxPool_hls", 2),
                ("LabelSelect_hls", 1),
            ],
        }
        if topology == "tfc" and wbits == 1 and abits == 1:
            exp_key = "tfc-1-1"
        else:
            exp_key = topology
        exp_layer_counts = exp_layer_counts[exp_key]
        for op_type, exp_count in exp_layer_counts:
            assert len(model.get_nodes_by_op_type(op_type)) == exp_count

    def test_create_dataflow_partition(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "specialize_layers")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        parent_model = model.transform(CreateDataflowPartition())
        parent_model_chkpt = get_checkpoint_name(topology, wbits, abits, "dataflow_parent")
        parent_model.save(parent_model_chkpt)
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
        dataflow_model_chkpt = get_checkpoint_name(topology, wbits, abits, "dataflow_model")
        dataflow_model.save(dataflow_model_chkpt)

    def test_fold(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "dataflow_model")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        folding_fxn = get_folding_function(topology, wbits, abits)
        model = folding_fxn(model)
        model.save(get_checkpoint_name(topology, wbits, abits, "fold"))

    def test_minimize_bit_width(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "fold")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(RoundAndClipThresholds())
        curr_chkpt_name = get_checkpoint_name(topology, wbits, abits, "minimize_bit_width")
        model.save(curr_chkpt_name)

    @pytest.mark.slow
    @pytest.mark.vivado
    def test_cppsim(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "minimize_bit_width")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        cppsim_chkpt = get_checkpoint_name(topology, wbits, abits, "cppsim")
        model.save(cppsim_chkpt)
        parent_chkpt = get_checkpoint_name(topology, wbits, abits, "dataflow_parent")
        (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
            topology, wbits, abits, return_topk=1
        )
        y = execute_parent(parent_chkpt, cppsim_chkpt, input_tensor_npy)
        assert np.isclose(y, output_tensor_npy).all()

    @pytest.mark.slow
    @pytest.mark.vivado
    def test_ipgen(self, topology, wbits, abits, board):
        build_data = get_build_env(board, target_clk_ns)
        if build_data["kind"] == "alveo" and ("VITIS_PATH" not in os.environ):
            pytest.skip("VITIS_PATH not set")
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "minimize_bit_width")
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(build_data["part"], target_clk_ns))
        model = model.transform(HLSSynthIP())
        model.save(get_checkpoint_name(topology, wbits, abits, "ipgen_" + board))

    @pytest.mark.slow
    @pytest.mark.vivado
    def test_set_fifo_depths(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "ipgen_" + board)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        test_fpga_part = get_build_env(board, target_clk_ns)["part"]
        if topology == "cnv" and abits == 2 and board == "Pynq-Z1":
            # Enabling swg_exception for these test cases. Disabling the exception results in
            # a design that exceeds the resources of the Pynq-Z1 board. In future this should be
            # revisited and handled correctly as the swg_exception is poorly justified.
            model = model.transform(
                InsertAndSetFIFODepths(test_fpga_part, target_clk_ns, swg_exception=True)
            )
        else:
            model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))

        fifo_layers = model.get_nodes_by_op_type("StreamingFIFO_rtl")
        assert len(fifo_layers) > 0
        model.save(get_checkpoint_name(topology, wbits, abits, "fifodepth_" + board))

    @pytest.mark.slow
    @pytest.mark.vivado
    def test_ipstitch_rtlsim(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "fifodepth_" + board)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        test_fpga_part = get_build_env(board, target_clk_ns)["part"]
        model = model.transform(InsertDWC())
        model = model.transform(SpecializeLayers(test_fpga_part))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        perf = model.analysis(dataflow_performance)
        latency = perf["critical_path_cycles"]
        # rtlsim only supports impl_style=rtl for StreamingFIFO, ensure that
        for fifo_layer in model.get_nodes_by_op_type("StreamingFIFO_rtl"):
            getCustomOp(fifo_layer).set_nodeattr("impl_style", "rtl")
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
        model.set_metadata_prop("exec_mode", "rtlsim")
        os.environ["LIVENESS_THRESHOLD"] = str(int(latency * 1.1))
        if rtlsim_trace:
            model.set_metadata_prop("rtlsim_trace", "%s_w%da%d.vcd" % (topology, wbits, abits))
            os.environ["RTLSIM_TRACE_DEPTH"] = "3"
        rtlsim_chkpt = get_checkpoint_name(topology, wbits, abits, "ipstitch_rtlsim_" + board)
        model.save(rtlsim_chkpt)
        parent_chkpt = get_checkpoint_name(topology, wbits, abits, "dataflow_parent")
        (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
            topology, wbits, abits, return_topk=1
        )
        y = execute_parent(parent_chkpt, rtlsim_chkpt, input_tensor_npy)
        assert np.isclose(y, output_tensor_npy).all()

    @pytest.mark.slow
    @pytest.mark.vivado
    def test_throughput_rtlsim(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "ipstitch_rtlsim_" + board)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        n_nodes = len(model.graph.node)
        perf_est = model.analysis(dataflow_performance)
        ret_b1 = throughput_test_rtlsim(model, batchsize=1)
        latency = int(ret_b1["cycles"])
        cycles_per_sample_est = perf_est["max_cycles"]
        batchsize = 2 * n_nodes
        ret = throughput_test_rtlsim(model, batchsize=batchsize)
        res_cycles = ret["cycles"]
        est_cycles = latency + cycles_per_sample_est * batchsize
        assert (abs(res_cycles - est_cycles) / res_cycles) < 0.15

    @pytest.mark.slow
    @pytest.mark.vivado
    def test_validate_top1(self, topology, wbits, abits, board):
        if "TEST_END2END_VALIDATE_TOP1" not in os.environ:
            pytest.skip("TEST_END2END_VALIDATE_TOP1 not set")
        prepostproc_chkpt = get_checkpoint_name(topology, wbits, abits, "pre_post")
        streamline_chkpt = get_checkpoint_name(topology, wbits, abits, "streamline")
        parent_chkpt = get_checkpoint_name(topology, wbits, abits, "dataflow_parent")
        cppsim_chkpt = get_checkpoint_name(topology, wbits, abits, "cppsim")
        rtlsim_chkpt = get_checkpoint_name(topology, wbits, abits, "ipstitch_rtlsim_" + board)
        dataset = topology2dataset(topology)
        assert measure_top1_accuracy(prepostproc_chkpt, dataset) > 80
        assert measure_top1_accuracy(streamline_chkpt, dataset) > 80
        assert measure_top1_accuracy(cppsim_chkpt, dataset, parent_chkpt) > 80
        assert measure_top1_accuracy(rtlsim_chkpt, dataset, parent_chkpt) > 80

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.vitis
    def test_build(self, topology, wbits, abits, board):
        build_data = get_build_env(board, target_clk_ns)
        if build_data["kind"] == "alveo" and ("VITIS_PATH" not in os.environ):
            pytest.skip("VITIS_PATH not set")
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "fifodepth_" + board)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(build_data["build_fxn"])
        model = model.transform(AnnotateResources("synth", build_data["part"]))
        model.save(get_checkpoint_name(topology, wbits, abits, "build_" + board))

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.vitis
    def test_make_pynq_driver(self, topology, wbits, abits, board):
        build_data = get_build_env(board, target_clk_ns)
        if build_data["kind"] == "alveo" and ("VITIS_PATH" not in os.environ):
            pytest.skip("VITIS_PATH not set")
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "build_" + board)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        board_to_driver_platform = "alveo" if build_data["kind"] == "alveo" else "zynq-iodma"
        model = model.transform(MakePYNQDriver(board_to_driver_platform))
        model.save(get_checkpoint_name(topology, wbits, abits, "driver_" + board))

    def test_deploy(self, topology, wbits, abits, board):
        prev_chkpt_name = get_checkpoint_name(topology, wbits, abits, "driver_" + board)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model_title = "%s_w%d_a%d_%s" % ("bnn", wbits, abits, topology)
        deploy_based_on_board(model, model_title, topology, wbits, abits, board)
        # save the model to be able to link it to the parent
        model.save(get_checkpoint_name(topology, wbits, abits, "deploy_" + board))
