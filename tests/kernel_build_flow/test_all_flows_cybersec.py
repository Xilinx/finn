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

import json
import numpy as np
import os
import shutil
import time
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

from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    step_convert_to_hw,
    step_create_dataflow_partition,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_set_fifo_depths,
    step_create_stitched_ip,
    step_measure_rtlsim_performance,
    step_out_of_context_synthesis,
    step_synthesize_bitfile,
    step_make_driver,
    step_deployment_package,
)

from pathlib import Path

import finn.core.onnx_exec as oxe
from qonnx.util.basic import gen_finn_dt_tensor

def custom_step_generate_reference_io(model, cfg):
    """
    This step is to generate a reference IO pair for the 
    onnx model where the head and the tail have been 
    chopped off.
    """
    input_m = model.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = gen_finn_dt_tensor(DataType["BIPOLAR"], in_shape)
    np.save(cfg.output_dir + "/input.npy", in_tensor)

    input_t = { input_m.name : in_tensor}
    out_name = model.graph.output[0].name

    y_ref = oxe.execute_onnx(model, input_t, True)
    np.save(cfg.output_dir + "/expected_output.npy", y_ref[out_name])
    np.savez(cfg.output_dir + "/expected_context.npz", **y_ref)
    return model


output_dir = make_build_dir("test_all_flows_cybersec")
output_dir_deprecated_steps = make_build_dir("test_deprecated_steps")
output_dir_characterize_fifo = make_build_dir("test_characterize_fifo")
output_dir_vitis_flow = make_build_dir("test_vitis_flow")
output_dir_full_flow = make_build_dir("test_full_flow")


def get_checkpoint_name(step):
    if step == "build":
        # checkpoint for build step is an entire dir
        return output_dir + "/end2end_cybsecmlp_build"
    else:
        # other checkpoints are onnx files
        return output_dir + "/end2end_cybsecmlp_%s.onnx" % step


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
    trained_state_dict = torch.load(assets_dir + "/state_dict.pth", weights_only=False)[
        "models_state_dict"
    ][0]
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


def test_full_flow():
    model_file = get_checkpoint_name("export")
    load_test_checkpoint_or_skip(model_file)

    steps = [
        custom_step_generate_reference_io,
        step_qonnx_to_finn,
        step_tidy_up,
        step_streamline,
        step_convert_to_hw,
        step_create_dataflow_partition,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_set_fifo_depths,
        step_create_stitched_ip,
        step_measure_rtlsim_performance,
        step_out_of_context_synthesis,
        step_synthesize_bitfile,
        step_make_driver,
        step_deployment_package,
    ]

    generate_outputs = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.OOC_SYNTH,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        # build_cfg.DataflowOutputType.CPP_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]

    verify_steps = [
        build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON, # step_qonnx_to_finn
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,       # step_tidy_up
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,   # step_streamline
        build_cfg.VerificationStepType.NODE_BY_NODE_RTLSIM,  # step_hw_ipgen
        build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,   # step_create_stitched_ip
    ]

    cfg = build.DataflowBuildConfig(
        steps=steps,
        output_dir=output_dir_full_flow,
        target_fps=1000000,
        synth_clk_period_ns=10,
        board="Pynq-Z1",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=generate_outputs,
        verify_input_npy=output_dir_full_flow + "/input.npy",
        verify_expected_output_npy=output_dir_full_flow + "/expected_output.npy",
        verify_save_full_context=True,
        verify_steps=verify_steps,
    )
    start_time = time.time()
    build.build_dataflow_cfg(model_file, cfg)
    total_time = time.time() - start_time
    print(f"Build took {total_time:.2f}s")
    # check the generated files
    assert os.path.isfile(output_dir_full_flow + "/time_per_step.json")
    assert os.path.isfile(output_dir_full_flow + "/final_hw_config.json")
    assert os.path.isfile(output_dir_full_flow + "/template_specialize_layers_config.json")
    assert os.path.isfile(output_dir_full_flow + "/driver/driver.py")
    est_cycles_report = output_dir_full_flow + "/report/estimate_layer_cycles.json"
    assert os.path.isfile(est_cycles_report)
    est_res_report = output_dir_full_flow + "/report/estimate_layer_resources.json"
    assert os.path.isfile(est_res_report)
    assert os.path.isfile(output_dir_full_flow + "/report/estimate_network_performance.json")
    assert os.path.isfile(output_dir_full_flow + "/bitfile/finn-accel.bit")
    assert os.path.isfile(output_dir_full_flow + "/bitfile/finn-accel.hwh")
    assert os.path.isfile(output_dir_full_flow + "/report/post_synth_resources.xml")
    assert os.path.isfile(output_dir_full_flow + "/report/post_route_timing.rpt")
    # examine the report contents
    with open(est_cycles_report, "r") as f:
        est_cycles_dict = json.load(f)
        assert est_cycles_dict["MVAU_0"] == 80
        assert est_cycles_dict["MVAU_1"] == 64
    with open(est_res_report, "r") as f:
        est_res_dict = json.load(f)
        assert est_res_dict["total"]["LUT"] == 7899.0
        assert est_res_dict["total"]["BRAM_18K"] == 36.0
    shutil.copytree(output_dir_full_flow + "/deploy", get_checkpoint_name("build"))
    shutil.rmtree(get_checkpoint_name("build"))

def test_deprecated_steps():
    """ Import steps that are no longer used by the flow and
        make sure that they have no effect. Test the flow up
        to IP stitching to perform stitched IP rtlsim. """

    from finn.builder.build_dataflow_steps import (
        step_specialize_layers,
        step_hw_codegen,
        step_hw_ipgen,
    )

    model_file = get_checkpoint_name("export")
    load_test_checkpoint_or_skip(model_file)

    steps = [
        custom_step_generate_reference_io,
        step_qonnx_to_finn,
        step_tidy_up,
        step_streamline,
        step_convert_to_hw,
        step_create_dataflow_partition,
        step_specialize_layers,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_hw_codegen,
        step_hw_ipgen,
        step_set_fifo_depths,
        step_create_stitched_ip,
    ]

    generate_outputs = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    ]

    verify_steps = [
        build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON, # step_qonnx_to_finn
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,       # step_tidy_up
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,   # step_streamline
        build_cfg.VerificationStepType.NODE_BY_NODE_RTLSIM,  # step_hw_ipgen
        build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,   # step_create_stitched_ip
    ]

    cfg = build.DataflowBuildConfig(
        steps=steps,
        output_dir=output_dir_deprecated_steps,
        target_fps=1000000,
        synth_clk_period_ns=10,
        board="Pynq-Z1",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=generate_outputs,
        verify_input_npy=output_dir_deprecated_steps + "/input.npy",
        verify_expected_output_npy=output_dir_deprecated_steps + "/expected_output.npy",
        verify_save_full_context=True,
        verify_steps=verify_steps,
    )
    build.build_dataflow_cfg(model_file, cfg)
    # check the generated files
    assert os.path.isfile(output_dir_deprecated_steps + "/time_per_step.json")
    assert os.path.isfile(output_dir_deprecated_steps + "/final_hw_config.json")
    est_cycles_report = output_dir_deprecated_steps + "/report/estimate_layer_cycles.json"
    assert os.path.isfile(est_cycles_report)
    est_res_report = output_dir_deprecated_steps + "/report/estimate_layer_resources.json"
    assert os.path.isfile(est_res_report)
    assert os.path.isfile(output_dir_deprecated_steps + "/report/estimate_network_performance.json")
    # examine the report contents
    with open(est_cycles_report, "r") as f:
        est_cycles_dict = json.load(f)
        assert est_cycles_dict["MVAU_0"] == 80
        assert est_cycles_dict["MVAU_1"] == 64
    with open(est_res_report, "r") as f:
        est_res_dict = json.load(f)
        assert est_res_dict["total"]["LUT"] == 7899.0
        assert est_res_dict["total"]["BRAM_18K"] == 36.0

def test_characterize_fifo():
    model_file = get_checkpoint_name("export")
    load_test_checkpoint_or_skip(model_file)

    steps = [
        custom_step_generate_reference_io,
        step_qonnx_to_finn,
        step_tidy_up,
        step_streamline,
        step_convert_to_hw,
        step_create_dataflow_partition,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_set_fifo_depths,
        step_create_stitched_ip,
    ]

    generate_outputs = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
    ]

    verify_steps = [
        build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON, # step_qonnx_to_finn
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,       # step_tidy_up
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,   # step_streamline
        build_cfg.VerificationStepType.NODE_BY_NODE_RTLSIM,  # step_hw_ipgen
        build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,   # step_create_stitched_ip
    ]

    cfg = build.DataflowBuildConfig(
        steps=steps,
        output_dir=output_dir_characterize_fifo,
        target_fps=1000000,
        synth_clk_period_ns=10,
        board="Pynq-Z1",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=generate_outputs,
        verify_input_npy=output_dir_characterize_fifo + "/input.npy",
        verify_expected_output_npy=output_dir_characterize_fifo + "/expected_output.npy",
        verify_save_full_context=True,
        verify_steps=verify_steps,
        auto_fifo_strategy="characterize",
    )
    build.build_dataflow_cfg(model_file, cfg)
    # check the generated files
    assert os.path.isfile(output_dir_characterize_fifo + "/time_per_step.json")
    assert os.path.isfile(output_dir_characterize_fifo + "/final_hw_config.json")
    est_cycles_report = output_dir_characterize_fifo + "/report/estimate_layer_cycles.json"
    assert os.path.isfile(est_cycles_report)
    est_res_report = output_dir_characterize_fifo + "/report/estimate_layer_resources.json"
    assert os.path.isfile(est_res_report)
    assert os.path.isfile(output_dir_characterize_fifo + "/report/estimate_network_performance.json")
    # examine the report contents
    with open(est_cycles_report, "r") as f:
        est_cycles_dict = json.load(f)
        assert est_cycles_dict["MVAU_0"] == 80
        assert est_cycles_dict["MVAU_1"] == 64
    with open(est_res_report, "r") as f:
        est_res_dict = json.load(f)
        assert est_res_dict["total"]["LUT"] == 7899.0
        assert est_res_dict["total"]["BRAM_18K"] == 36.0
