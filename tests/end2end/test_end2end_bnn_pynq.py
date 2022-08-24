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

import brevitas.onnx as bo
import numpy as np

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA
import os
import subprocess
import torch
import warnings
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from collections import OrderedDict
from dataset_loading import cifar, mnist
from datetime import datetime
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
from scipy.stats import linregress

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.core.onnx_exec import execute_onnx
from finn.core.throughput_test import throughput_test_remote, throughput_test_rtlsim
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)
from finn.util.basic import get_finn_root
from finn.util.gdrive import upload_to_end2end_dashboard
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
target_clk_ns = 10
mem_mode = "decoupled"
rtlsim_trace = False


def get_checkpoint_name(topology, wbits, abits, QONNX_export, step):
    return build_dir + "/end2end_%s_w%da%d_QONNX-%d_%s.onnx" % (
        topology,
        wbits,
        abits,
        QONNX_export,
        step,
    )


def get_dashboard_data(topology, wbits, abits):
    stats_file = build_dir + "/end2end_%s_w%da%d.txt" % (topology, wbits, abits)
    stats_dict = OrderedDict()
    if os.path.isfile(stats_file):
        with open(stats_file, "r") as f:
            stats_dict_txt = f.read()
        stats_dict = eval(stats_dict_txt)
    return stats_dict


def update_dashboard_data(topology, wbits, abits, key, val):
    stats_dict = get_dashboard_data(topology, wbits, abits)
    stats_dict[key] = val
    stats_file = build_dir + "/end2end_%s_w%da%d.txt" % (topology, wbits, abits)
    with open(stats_file, "w") as f:
        f.write(str(stats_dict))


def fold_tfc(model):
    fc_layers = model.get_nodes_by_op_type("MatrixVectorActivation")
    # (PE, SIMD, ramstyle) for each layer
    config = [(16, 49, "block"), (8, 8, "auto"), (8, 8, "auto"), (10, 8, "distributed")]
    for fcl, (pe, simd, ramstyle) in zip(fc_layers, config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
    # set parallelism for input quantizer to be same as first layer's SIMD
    inp_qnt_node = model.get_nodes_by_op_type("Thresholding_Batch")[0]
    inp_qnt = getCustomOp(inp_qnt_node)
    inp_qnt.set_nodeattr("PE", 49)
    inp_qnt.set_nodeattr("mem_mode", "decoupled")
    inp_qnt.set_nodeattr("runtime_writeable_weights", 1)
    return model


def fold_lfc(model):
    fc_layers = model.get_nodes_by_op_type("MatrixVectorActivation")
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
    # set parallelism for input quantizer to be same as first layer's SIMD
    inp_qnt_node = model.get_nodes_by_op_type("Thresholding_Batch")[0]
    inp_qnt = getCustomOp(inp_qnt_node)
    inp_qnt.set_nodeattr("PE", 49)
    return model


def fold_cnv_large(model):
    fc_layers = model.get_nodes_by_op_type("MatrixVectorActivation")
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

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
    return model


def fold_cnv_small(model):
    fc_layers = model.get_nodes_by_op_type("MatrixVectorActivation")
    # each tuple is (PE, SIMD) for a layer
    folding = [
        (8, 3, "distributed"),
        (16, 16, "distributed"),
        (8, 16, "auto"),
        (8, 16, "block"),
        (4, 8, "auto"),
        (1, 8, "auto"),
        (1, 2, "distributed"),
        (2, 2, "block"),
        (5, 1, "distributed"),
    ]
    for fcl, (pe, simd, ramstyle) in zip(fc_layers, folding):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("ram_style", ramstyle)

    swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
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
    model = ModelWrapper(model_chkpt)
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


@pytest.mark.parametrize("wbits", [1, 2])
@pytest.mark.parametrize("abits", [1, 2])
@pytest.mark.parametrize("topology", ["lfc", "tfc", "cnv"])
@pytest.mark.parametrize("QONNX_export", [False, True])
@pytest.mark.end2end
class TestEnd2End:
    def test_export(self, topology, wbits, abits, QONNX_export):
        if wbits > abits:
            pytest.skip("No wbits > abits end2end network configs for now")
        if topology == "lfc" and not (wbits == 1 and abits == 1):
            pytest.skip("Skipping certain lfc configs")
        (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
        chkpt_name = get_checkpoint_name(topology, wbits, abits, QONNX_export, "export")
        if QONNX_export:
            BrevitasONNXManager.export(model, ishape, chkpt_name)
            qonnx_cleanup(chkpt_name, out_file=chkpt_name)
            model = ModelWrapper(chkpt_name)
            model = model.transform(ConvertQONNXtoFINN())
            model.save(chkpt_name)
        else:
            bo.export_finn_onnx(model, ishape, chkpt_name)
        nname = "%s_w%da%d" % (topology, wbits, abits)
        update_dashboard_data(topology, wbits, abits, "network", nname)
        dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_dashboard_data(topology, wbits, abits, "datetime", dtstr)
        finn_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=get_finn_root()
        )
        finn_commit = finn_commit.decode("utf-8").strip()
        update_dashboard_data(topology, wbits, abits, "finn-commit", finn_commit)
        assert os.path.isfile(chkpt_name)

    def test_import_and_tidy(self, topology, wbits, abits, QONNX_export):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "export"
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(RemoveStaticGraphInputs())
        chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "import_and_tidy"
        )
        model.save(chkpt)

    def test_add_pre_and_postproc(self, topology, wbits, abits, QONNX_export):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "import_and_tidy"
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        global_inp_name = model.graph.input[0].name
        ishape = model.get_tensor_shape(global_inp_name)
        # preprocessing: torchvision's ToTensor divides uint8 inputs by 255
        totensor_pyt = ToTensor()
        chkpt_preproc_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "preproc"
        )
        bo.export_finn_onnx(totensor_pyt, ishape, chkpt_preproc_name)
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
        chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "pre_post"
        )
        # tidy-up again
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(RemoveStaticGraphInputs())
        model.save(chkpt_name)
        assert os.path.isfile(chkpt_name)

    def test_streamline(self, topology, wbits, abits, QONNX_export):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "pre_post"
        )
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
        model.save(
            get_checkpoint_name(topology, wbits, abits, QONNX_export, "streamline")
        )

    def test_convert_to_hls_layers(self, topology, wbits, abits, QONNX_export):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "streamline"
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        if topology == "tfc" and wbits == 1 and abits == 1:
            # use standalone thresholds for tfc-w1a1 to also exercise that option
            model = model.transform(to_hls.InferThresholdingLayer())
        # needed for bipolar MatMul layers
        model = model.transform(to_hls.InferBinaryMatrixVectorActivation(mem_mode))
        # needed for non-bipolar MatMul layers
        model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
        # TopK to LabelSelect
        model = model.transform(to_hls.InferLabelSelectLayer())
        # input quantization (if any) to standalone thresholding
        model = model.transform(to_hls.InferThresholdingLayer())
        # needed for convolutions
        if "fc" not in topology:
            model = model.transform(to_hls.InferConvInpGen())
            model = model.transform(to_hls.InferStreamingMaxPool())
            model = model.transform(RemoveCNVtoFCFlatten())
        # get rid of Tranpose -> Tranpose identity seq
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataLayouts())
        model.save(
            get_checkpoint_name(
                topology, wbits, abits, QONNX_export, "convert_to_hls_layers"
            )
        )
        exp_layer_counts = {
            "tfc": [
                ("Reshape", 1),
                ("Thresholding_Batch", 1),
                ("MatrixVectorActivation", 4),
                ("LabelSelect_Batch", 1),
            ],
            "tfc-1-1": [
                ("Reshape", 1),
                ("Thresholding_Batch", 4),
                ("MatrixVectorActivation", 4),
                ("LabelSelect_Batch", 1),
            ],
            "lfc": [
                ("Reshape", 1),
                ("Thresholding_Batch", 1),
                ("MatrixVectorActivation", 4),
                ("LabelSelect_Batch", 1),
            ],
            "cnv": [
                ("Transpose", 1),
                ("Thresholding_Batch", 1),
                ("ConvolutionInputGenerator", 6),
                ("MatrixVectorActivation", 9),
                ("StreamingMaxPool_Batch", 2),
                ("LabelSelect_Batch", 1),
            ],
        }
        if topology == "tfc" and wbits == 1 and abits == 1:
            exp_key = "tfc-1-1"
        else:
            exp_key = topology
        exp_layer_counts = exp_layer_counts[exp_key]
        for (op_type, exp_count) in exp_layer_counts:
            assert len(model.get_nodes_by_op_type(op_type)) == exp_count

    def test_create_dataflow_partition(self, topology, wbits, abits, QONNX_export):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "convert_to_hls_layers"
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        parent_model = model.transform(CreateDataflowPartition())
        parent_model_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "dataflow_parent"
        )
        parent_model.save(parent_model_chkpt)
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)
        dataflow_model_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "dataflow_model"
        )
        dataflow_model.save(dataflow_model_chkpt)

    def test_fold(self, topology, wbits, abits, QONNX_export):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "dataflow_model"
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        folding_fxn = get_folding_function(topology, wbits, abits)
        model = folding_fxn(model)
        model.save(get_checkpoint_name(topology, wbits, abits, QONNX_export, "fold"))

    @pytest.mark.slow
    @pytest.mark.vivado
    def test_cppsim(self, topology, wbits, abits, QONNX_export):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "fold"
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        cppsim_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "cppsim"
        )
        model.save(cppsim_chkpt)
        parent_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "dataflow_parent"
        )
        (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
            topology, wbits, abits, return_topk=1
        )
        y = execute_parent(parent_chkpt, cppsim_chkpt, input_tensor_npy)
        assert np.isclose(y, output_tensor_npy).all()

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_ipgen(self, topology, wbits, abits, QONNX_export, kind):
        if kind == "alveo" and ("VITIS_PATH" not in os.environ):
            pytest.skip("VITIS_PATH not set")
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "fold"
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        test_fpga_part = get_build_env(kind, target_clk_ns)["part"]
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model.save(
            get_checkpoint_name(topology, wbits, abits, QONNX_export, "ipgen_" + kind)
        )

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_set_fifo_depths(self, topology, wbits, abits, QONNX_export, kind):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "ipgen_" + kind
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        test_fpga_part = get_build_env(kind, target_clk_ns)["part"]
        model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
        fifo_layers = model.get_nodes_by_op_type("StreamingFIFO")
        assert len(fifo_layers) > 0
        hls_layers = model.get_finn_nodes()
        for node in hls_layers:
            if node.op_type != "StreamingFIFO":
                op_inst = getCustomOp(node)
                assert op_inst.get_nodeattr("inFIFODepth") == 0
                assert op_inst.get_nodeattr("outFIFODepth") == 0
        model.save(
            get_checkpoint_name(
                topology, wbits, abits, QONNX_export, "fifodepth_" + kind
            )
        )

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.parametrize("kind", ["zynq"])
    def test_ipstitch_rtlsim(self, topology, wbits, abits, QONNX_export, kind):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "fifodepth_" + kind
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        test_fpga_part = get_build_env(kind, target_clk_ns)["part"]
        model = model.transform(InsertDWC())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        perf = model.analysis(dataflow_performance)
        latency = perf["critical_path_cycles"]
        # rtlsim only supports impl_style=rtl for StreamingFIFO, ensure that
        for fifo_layer in model.get_nodes_by_op_type("StreamingFIFO"):
            getCustomOp(fifo_layer).set_nodeattr("impl_style", "rtl")
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
        model = model.transform(PrepareRTLSim())
        model.set_metadata_prop("exec_mode", "rtlsim")
        os.environ["LIVENESS_THRESHOLD"] = str(int(latency * 1.1))
        if rtlsim_trace:
            model.set_metadata_prop(
                "rtlsim_trace", "%s_w%da%d.vcd" % (topology, wbits, abits)
            )
            os.environ["RTLSIM_TRACE_DEPTH"] = "3"
        rtlsim_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "ipstitch_rtlsim_" + kind
        )
        model.save(rtlsim_chkpt)
        parent_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "dataflow_parent"
        )
        (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
            topology, wbits, abits, return_topk=1
        )
        y = execute_parent(parent_chkpt, rtlsim_chkpt, input_tensor_npy)
        assert np.isclose(y, output_tensor_npy).all()

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.parametrize("kind", ["zynq"])
    def test_throughput_rtlsim(self, topology, wbits, abits, QONNX_export, kind):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "ipstitch_rtlsim_" + kind
        )
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
        # warnings.warn("Estimated & rtlsim performance: " + str(perf))
        # for (k, v) in perf.items():
        #    update_dashboard_data(topology, wbits, abits, k, v)
        update_dashboard_data(topology, wbits, abits, "cycles_rtlsim", latency)
        assert (abs(res_cycles - est_cycles) / res_cycles) < 0.15

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.parametrize("kind", ["zynq"])
    def test_validate_top1(self, topology, wbits, abits, QONNX_export, kind):
        if "TEST_END2END_VALIDATE_TOP1" not in os.environ:
            pytest.skip("TEST_END2END_VALIDATE_TOP1 not set")
        prepostproc_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "pre_post"
        )
        streamline_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "streamline"
        )
        parent_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "dataflow_parent"
        )
        cppsim_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "cppsim"
        )
        rtlsim_chkpt = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "ipstitch_rtlsim_" + kind
        )
        dataset = topology2dataset(topology)
        assert measure_top1_accuracy(prepostproc_chkpt, dataset) > 80
        assert measure_top1_accuracy(streamline_chkpt, dataset) > 80
        assert measure_top1_accuracy(cppsim_chkpt, dataset, parent_chkpt) > 80
        assert measure_top1_accuracy(rtlsim_chkpt, dataset, parent_chkpt) > 80

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.vitis
    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_build(self, topology, wbits, abits, QONNX_export, kind):
        if kind == "alveo" and ("VITIS_PATH" not in os.environ):
            pytest.skip("VITIS_PATH not set")
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "fifodepth_" + kind
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        cfg = get_build_env(kind, target_clk_ns)
        model = model.transform(cfg["build_fxn"])
        model = model.transform(AnnotateResources("synth"))
        synth_dct = eval(model.get_metadata_prop("res_total_top_synth"))
        for (k, v) in synth_dct.items():
            update_dashboard_data(topology, wbits, abits, k, v)
        update_dashboard_data(topology, wbits, abits, "board", cfg["board"])
        model.save(
            get_checkpoint_name(topology, wbits, abits, QONNX_export, "build_" + kind)
        )

    @pytest.mark.slow
    @pytest.mark.vivado
    @pytest.mark.vitis
    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_make_pynq_driver(self, topology, wbits, abits, QONNX_export, kind):
        if kind == "alveo" and ("VITIS_PATH" not in os.environ):
            pytest.skip("VITIS_PATH not set")
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "build_" + kind
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        kind_to_driver_platform = {"zynq": "zynq-iodma", "alveo": "alveo"}
        model = model.transform(MakePYNQDriver(kind_to_driver_platform[kind]))
        model.save(
            get_checkpoint_name(topology, wbits, abits, QONNX_export, "driver_" + kind)
        )

    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_deploy(self, topology, wbits, abits, QONNX_export, kind):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "driver_" + kind
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)
        cfg = get_build_env(kind, target_clk_ns)
        if cfg["ip"] == "":
            pytest.skip("PYNQ board IP address not specified")
        model = model.transform(
            DeployToPYNQ(
                cfg["ip"],
                cfg["port"],
                cfg["username"],
                cfg["password"],
                cfg["target_dir"],
            )
        )
        # save the model to be able to link it to the parent
        model.save(
            get_checkpoint_name(topology, wbits, abits, QONNX_export, "deploy_" + kind)
        )

    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_run_on_hw(self, topology, wbits, abits, QONNX_export, kind):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "deploy_" + kind
        )
        model = load_test_checkpoint_or_skip(prev_chkpt_name)  # NOQA
        cfg = get_build_env(kind, target_clk_ns)
        if cfg["ip"] == "":
            pytest.skip("PYNQ board IP address not specified")
        (input_tensor_npy, output_tensor_npy) = get_golden_io_pair(
            topology, wbits, abits, return_topk=1
        )
        parent_model = load_test_checkpoint_or_skip(
            get_checkpoint_name(topology, wbits, abits, QONNX_export, "dataflow_parent")
        )
        iname = parent_model.graph.input[0].name
        oname = parent_model.graph.output[0].name
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        sdp_node.set_nodeattr("model", prev_chkpt_name)
        ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
        y = ret[oname]
        assert np.isclose(y, output_tensor_npy).all()

    @pytest.mark.parametrize("kind", ["zynq", "alveo"])
    def test_throughput_hw(self, topology, wbits, abits, QONNX_export, kind):
        prev_chkpt_name = get_checkpoint_name(
            topology, wbits, abits, QONNX_export, "deploy_" + kind
        )
        end2end_example = "%s_w%da%d_%s" % (topology, wbits, abits, kind)
        model = load_test_checkpoint_or_skip(prev_chkpt_name)  # NOQA
        cfg = get_build_env(kind, target_clk_ns)
        if cfg["ip"] == "":
            pytest.skip("PYNQ board IP address not specified")
        ret = dict()
        # try a range of batch sizes, some may fail due to insufficient DMA
        # buffers
        bsize_range_in = [8**i for i in range(5)]
        bsize_range = []
        for bsize in bsize_range_in:
            res = throughput_test_remote(model, bsize)
            if res is not None:
                ret[bsize] = res
                bsize_range.append(bsize)
            else:
                # assume we reached largest possible N
                break
        y = [ret[key]["runtime[ms]"] for key in bsize_range]
        lrret = linregress(bsize_range, y)
        ret_str = ""
        ret_str += "\n" + "%s Throughput Test Results" % end2end_example
        ret_str += "\n" + "-----------------------------"
        ret_str += "\n" + "From linear regression:"
        ret_str += "\n" + "Invocation overhead: %f ms" % lrret.intercept
        ret_str += "\n" + "Time per sample: %f ms" % lrret.slope
        ret_str += "\n" + "Raw data:"

        ret_str += "\n" + "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
            "N", "runtime[ms]", "fclk[mhz]", "fps", "DRAM rd[MB/s]", "DRAM wr[MB/s]"
        )
        for k in bsize_range:
            v = ret[k]
            ret_str += "\n" + "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
                k,
                np.round(v["runtime[ms]"], 4),
                v["fclk[mhz]"],
                np.round(v["throughput[images/s]"], 2),
                np.round(v["DRAM_in_bandwidth[MB/s]"], 2),
                np.round(v["DRAM_out_bandwidth[MB/s]"], 2),
            )
        ret_str += "\n" + "-----------------------------"
        warnings.warn(ret_str)
        largest_bsize = bsize_range[-1]
        update_dashboard_data(
            topology, wbits, abits, "fclk[mhz]", ret[largest_bsize]["fclk[mhz]"]
        )
        update_dashboard_data(
            topology,
            wbits,
            abits,
            "throughput[images/s]",
            ret[largest_bsize]["throughput[images/s]"],
        )

    def test_upload_results_to_dashboard(self, topology, wbits, abits, QONNX_export):
        # ToDo: Extend the dashboard to also upload QONNX exported models?
        if QONNX_export:
            pytest.skip("Dashboard data upload is disabled for QONNX exported models.")
        else:
            dashboard_data = get_dashboard_data(topology, wbits, abits)
            if len(dashboard_data.keys()) > 0:
                upload_to_end2end_dashboard(dashboard_data)
            else:
                pytest.skip("No data to upload to dashboard")
