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

import pytest

import json
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
from brevitas.core.quant import QuantType
from brevitas.export import export_qonnx
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir
from finn.util.test import load_test_checkpoint_or_skip

target_clk_ns = 10
build_board = "Pynq-Z1"
build_dir = os.environ["FINN_BUILD_DIR"]


def get_checkpoint_name(step):
    if step == "build":
        # checkpoint for build step is an entire dir
        return build_dir + "/end2end_cybsecmlp_build"
    else:
        # other checkpoints are onnx files
        return build_dir + "/end2end_cybsecmlp_%s.onnx" % step


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


@pytest.mark.end2end
def test_end2end_cybsec_mlp_export():
    assets_dir = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/cybsec-mlp"
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
    trained_state_dict = torch.load(assets_dir + "/state_dict.pth")["models_state_dict"][0]
    model.load_state_dict(trained_state_dict, strict=False)
    W_orig = model[0].weight.data.detach().numpy()
    # pad the second (593-sized) dimensions with 7 zeroes at the end
    W_new = np.pad(W_orig, [(0, 0), (0, 7)])
    model[0].weight.data = torch.from_numpy(W_new)
    model_for_export = CybSecMLPForExport(model)
    export_onnx_path = get_checkpoint_name("export")
    input_shape = (1, 600)

    # With the onnx export from Brevitas we need to manually set
    # the FINN DataType at the input
    export_qonnx(model_for_export, torch.randn(input_shape), export_path=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model.set_tensor_datatype(model.graph.input[0].name, DataType["BIPOLAR"])
    model.save(export_onnx_path)
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(ConvertQONNXtoFINN())
    assert os.path.isfile(export_onnx_path)
    # fix input datatype
    finnonnx_in_tensor_name = model.graph.input[0].name
    assert tuple(model.get_tensor_shape(finnonnx_in_tensor_name)) == (1, 600)
    # verify a few exported ops
    # The first "Mul" node doesn't exist in the QONNX export,
    # because the QuantTensor scale is not exported.
    # However, this node would have been unity scale anyways and
    # the models are still equivalent.
    assert model.graph.node[0].op_type == "Add"
    assert model.graph.node[1].op_type == "Div"
    assert model.graph.node[2].op_type == "MatMul"
    assert model.graph.node[-1].op_type == "MultiThreshold"
    # verify datatypes on some tensors
    assert model.get_tensor_datatype(finnonnx_in_tensor_name) == DataType["BIPOLAR"]
    first_matmul_w_name = model.get_nodes_by_op_type("MatMul")[0].input[1]
    assert model.get_tensor_datatype(first_matmul_w_name) == DataType["INT2"]


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_end2end_cybsec_mlp_build():
    model_file = get_checkpoint_name("export")
    load_test_checkpoint_or_skip(model_file)
    output_dir = make_build_dir("test_end2end_cybsec_mlp_build")

    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        target_fps=1000000,
        synth_clk_period_ns=target_clk_ns,
        board=build_board,
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
        assert est_cycles_dict["MatrixVectorActivation_0"] == 80
        assert est_cycles_dict["MatrixVectorActivation_1"] == 64
    with open(est_res_report, "r") as f:
        est_res_dict = json.load(f)
        assert est_res_dict["total"]["LUT"] == 7904.0
        assert est_res_dict["total"]["BRAM_18K"] == 36.0
    shutil.copytree(output_dir + "/deploy", get_checkpoint_name("build"))
    shutil.rmtree(get_checkpoint_name("build"))
