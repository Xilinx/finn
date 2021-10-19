# Copyright (c) 2021, Xilinx
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

import pkg_resources as pk

import pytest

import brevitas.onnx as bo
import json
import numpy as np
import os
import shutil
import subprocess
import torch
import torch.nn as nn
import wget
from brevitas.core.quant import QuantType
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant_tensor import QuantTensor
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir
from finn.util.test import get_build_env, load_test_checkpoint_or_skip

target_clk_ns = 10
build_kind = "zynq"
build_dir = os.environ["FINN_BUILD_DIR"]


def get_checkpoint_name(step, QONNX_export):
    if step == "build":
        # checkpoint for build step is an entire dir
        return build_dir + "/end2end_cybsecmlp_build_QONNX-%d" % (QONNX_export)
    else:
        # other checkpoints are onnx files
        return build_dir + "/end2end_cybsecmlp_QONNX-%d_%s.onnx" % (QONNX_export, step)


class CybSecMLPForExport(nn.Module):
    def __init__(self, my_pretrained_model):
        super(CybSecMLPForExport, self).__init__()
        self.pretrained = my_pretrained_model
        self.qnt_output = QuantIdentity(
            quant_type=QuantType.BINARY, bit_width=1, min_val=-1.0, max_val=1.0
        )

    def forward(self, x):
        # assume x contains bipolar {-1,1} elems
        # shift from {-1,1} -> {0,1} since that is the
        # input range for the trained network
        x = (x + torch.tensor([1.0])) / 2.0
        out_original = self.pretrained(x)
        out_final = self.qnt_output(out_original)  # output as {-1,1}
        return out_final


@pytest.mark.parametrize("QONNX_export", [False, True])
def test_end2end_cybsec_mlp_export(QONNX_export):
    assets_dir = pk.resource_filename("finn.qnn-data", "cybsec-mlp/")
    # load up trained net in Brevitas
    input_size = 593
    hidden1 = 64
    hidden2 = 64
    hidden3 = 64
    weight_bit_width = 2
    act_bit_width = 2
    num_classes = 1
    model = nn.Sequential(
        QuantLinear(input_size, hidden1, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden1),
        nn.Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden1, hidden2, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden2),
        nn.Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden2, hidden3, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden3),
        nn.Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden3, num_classes, bias=True, weight_bit_width=weight_bit_width),
    )
    trained_state_dict = torch.load(assets_dir + "/state_dict.pth")[
        "models_state_dict"
    ][0]
    model.load_state_dict(trained_state_dict, strict=False)
    W_orig = model[0].weight.data.detach().numpy()
    # pad the second (593-sized) dimensions with 7 zeroes at the end
    W_new = np.pad(W_orig, [(0, 0), (0, 7)])
    model[0].weight.data = torch.from_numpy(W_new)
    model_for_export = CybSecMLPForExport(model)
    export_onnx_path = get_checkpoint_name("export", QONNX_export)
    input_shape = (1, 600)
    # create a QuantTensor instance to mark the input as bipolar during export
    input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
    input_a = 2 * input_a - 1
    scale = 1.0
    input_t = torch.from_numpy(input_a * scale)
    input_qt = QuantTensor(
        input_t, scale=torch.tensor(scale), bit_width=torch.tensor(1.0), signed=True
    )

    if QONNX_export:
        # With the BrevitasONNXManager we need to manually set
        # the FINN DataType at the input
        BrevitasONNXManager.export(
            model_for_export, input_shape, export_path=export_onnx_path
        )
        model = ModelWrapper(export_onnx_path)
        model.set_tensor_datatype(model.graph.input[0].name, DataType["BIPOLAR"])
        model.save(export_onnx_path)
        qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
        model = ModelWrapper(export_onnx_path)
        model = model.transform(ConvertQONNXtoFINN())
        model.save(export_onnx_path)
    else:
        bo.export_finn_onnx(
            model_for_export, export_path=export_onnx_path, input_t=input_qt
        )
    assert os.path.isfile(export_onnx_path)
    # fix input datatype
    finn_model = ModelWrapper(export_onnx_path)
    finnonnx_in_tensor_name = finn_model.graph.input[0].name
    assert tuple(finn_model.get_tensor_shape(finnonnx_in_tensor_name)) == (1, 600)
    # verify a few exported ops
    if QONNX_export:
        # The first "Mul" node dosen't exist in the QONNX export,
        # because the QuantTensor scale is not exported.
        # However, this node would have been unity scale anyways and
        # the models are still equivalent.
        assert finn_model.graph.node[0].op_type == "Add"
        assert finn_model.graph.node[1].op_type == "Div"
        assert finn_model.graph.node[2].op_type == "MatMul"
        assert finn_model.graph.node[-1].op_type == "MultiThreshold"
    else:
        assert finn_model.graph.node[0].op_type == "Mul"
        assert finn_model.get_initializer(finn_model.graph.node[0].input[1]) == 1.0
        assert finn_model.graph.node[1].op_type == "Add"
        assert finn_model.graph.node[2].op_type == "Div"
        assert finn_model.graph.node[3].op_type == "MatMul"
        assert finn_model.graph.node[-1].op_type == "MultiThreshold"
    # verify datatypes on some tensors
    assert (
        finn_model.get_tensor_datatype(finnonnx_in_tensor_name) == DataType["BIPOLAR"]
    )
    first_matmul_w_name = finn_model.get_nodes_by_op_type("MatMul")[0].input[1]
    assert finn_model.get_tensor_datatype(first_matmul_w_name) == DataType["INT2"]


@pytest.mark.slow
@pytest.mark.vivado
# @pytest.mark.parametrize("QONNX_export", [False, True])
@pytest.mark.parametrize("QONNX_export", [True])
def test_end2end_cybsec_mlp_build(QONNX_export):
    model_file = get_checkpoint_name("export", QONNX_export)
    load_test_checkpoint_or_skip(model_file)
    build_env = get_build_env(build_kind, target_clk_ns)
    output_dir = make_build_dir(f"test_end2end_cybsec_mlp_build_QONNX-{QONNX_export}")

    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        target_fps=1000000,
        synth_clk_period_ns=target_clk_ns,
        board=build_env["board"],
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(model_file, cfg)
    # check the generated files
    assert os.path.isfile(output_dir + "/time_per_step.json")
    assert os.path.isfile(output_dir + "/final_hw_config.json")
    assert os.path.isfile(output_dir + "/driver/driver.py")
    est_cycles_report = output_dir + "/report/estimate_layer_cycles.json"
    assert os.path.isfile(est_cycles_report)
    est_res_report = output_dir + "/report/estimate_layer_resources.json"
    assert os.path.isfile(est_res_report)
    assert os.path.isfile(output_dir + "/report/estimate_network_performance.json")
    assert os.path.isfile(output_dir + "/bitfile/finn-accel.bit")
    assert os.path.isfile(output_dir + "/bitfile/finn-accel.hwh")
    assert os.path.isfile(output_dir + "/report/post_synth_resources.xml")
    assert os.path.isfile(output_dir + "/report/post_route_timing.rpt")
    # examine the report contents
    with open(est_cycles_report, "r") as f:
        est_cycles_dict = json.load(f)
        assert est_cycles_dict["StreamingFCLayer_Batch_0"] == 80
        assert est_cycles_dict["StreamingFCLayer_Batch_1"] == 64
    with open(est_res_report, "r") as f:
        est_res_dict = json.load(f)
        assert est_res_dict["total"]["LUT"] == 11360.0
        assert est_res_dict["total"]["BRAM_18K"] == 36.0
    shutil.copytree(output_dir + "/deploy", get_checkpoint_name("build", QONNX_export))


@pytest.mark.parametrize("QONNX_export", [False, True])
def test_end2end_cybsec_mlp_run_on_hw(QONNX_export):
    build_env = get_build_env(build_kind, target_clk_ns)
    assets_dir = pk.resource_filename("finn.qnn-data", "cybsec-mlp/")
    deploy_dir = get_checkpoint_name("build", QONNX_export)
    if not os.path.isdir(deploy_dir):
        pytest.skip(deploy_dir + " not found from previous test step, skipping")
    driver_dir = deploy_dir + "/driver"
    assert os.path.isdir(driver_dir)
    # put all assets into driver dir
    shutil.copy(assets_dir + "/validate-unsw-nb15.py", driver_dir)
    # put a copy of binarized dataset into driver dir
    dataset_url = (
        "https://zenodo.org/record/4519767/files/unsw_nb15_binarized.npz?download=1"
    )
    dataset_local = driver_dir + "/unsw_nb15_binarized.npz"
    if not os.path.isfile(dataset_local):
        wget.download(dataset_url, out=dataset_local)
    assert os.path.isfile(dataset_local)
    # create a shell script for running validation: 10 batches x 10 imgs
    with open(driver_dir + "/validate.sh", "w") as f:
        f.write(
            """#!/bin/bash
cd %s/driver
echo %s | sudo -S python3.6 validate-unsw-nb15.py --batchsize=10 --limit_batches=10
        """
            % (
                build_env["target_dir"] + "/end2end_cybsecmlp_build",
                build_env["password"],
            )
        )
    # set up rsync command
    remote_target = "%s@%s:%s" % (
        build_env["username"],
        build_env["ip"],
        build_env["target_dir"],
    )
    rsync_res = subprocess.run(["rsync", "-avz", deploy_dir, remote_target])
    assert rsync_res.returncode == 0
    remote_verif_cmd = [
        "ssh",
        "%s@%s" % (build_env["username"], build_env["ip"]),
        "sh",
        build_env["target_dir"] + "/end2end_cybsecmlp_build/driver/validate.sh",
    ]
    verif_res = subprocess.run(
        remote_verif_cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        input=build_env["password"],
    )
    assert verif_res.returncode == 0
    log_output = verif_res.stdout.split("\n")
    assert log_output[-3] == "batch 10 / 10 : total OK 93 NOK 7"
    assert log_output[-2] == "Final accuracy: 93.000000"
