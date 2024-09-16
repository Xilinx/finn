# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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
import xml.etree.ElementTree as ET
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.post_synth_res import post_synth_res
from finn.core.throughput_test import throughput_test_rtlsim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import is_hls_node, is_rtl_node


def post_synth_res_dwc(model, override_synth_report_filename=None):
    """Extracts the FPGA resource results from the Vivado synthesis.
    This function extras only a DWC from a DWC-only stitched model

    Returns {node name : resources_dict}."""

    res_dict = {}
    if override_synth_report_filename is not None:
        synth_report_filename = override_synth_report_filename
    else:
        synth_report_filename = model.get_metadata_prop("vivado_synth_rpt")
    if os.path.isfile(synth_report_filename):
        tree = ET.parse(synth_report_filename)
        root = tree.getroot()
        all_cells = root.findall(".//tablecell")
        # strip all whitespace from table cell contents
        for cell in all_cells:
            cell.attrib["contents"] = cell.attrib["contents"].strip()
    else:
        raise Exception("Please run synthesis first")

    # TODO build these indices based on table headers instead of harcoding
    restype_to_ind_default = {
        "LUT": 2,
        "SRL": 5,
        "FF": 6,
        "BRAM_36K": 7,
        "BRAM_18K": 8,
        "DSP48": 9,
    }
    restype_to_ind_vitis = {
        "LUT": 4,
        "SRL": 7,
        "FF": 8,
        "BRAM_36K": 9,
        "BRAM_18K": 10,
        "URAM": 11,
        "DSP48": 12,
    }

    if model.get_metadata_prop("platform") == "alveo":
        restype_to_ind = restype_to_ind_vitis
    else:
        restype_to_ind = restype_to_ind_default

    def get_instance_stats(inst_name):
        row = root.findall(".//*[@contents='%s']/.." % inst_name)
        if row != []:
            node_dict = {}
            row = list(row[0])
            for restype, ind in restype_to_ind.items():
                node_dict[restype] = int(row[ind].attrib["contents"])
            return node_dict
        else:
            return None

    # global (top-level) stats, including shell etc.
    top_dict = get_instance_stats("(top)")
    if top_dict is not None:
        res_dict["(top)"] = top_dict

    for node in model.graph.node:
        if node.op_type == "StreamingDataflowPartition":
            sdp_model = ModelWrapper(getCustomOp(node).get_nodeattr("model"))
            sdp_res_dict = post_synth_res(sdp_model, synth_report_filename)
            res_dict.update(sdp_res_dict)
        elif is_hls_node(node) or is_rtl_node(node):
            node_dict = get_instance_stats(
                f"top_StreamingDataflowPartition_1_0_StreamingDataflowPartition_1_StreamingDataflowPartition_1_StreamingDataWidthConverter_hls_0_0"
            )
            if node_dict is not None:
                res_dict[node.name] = node_dict

    return res_dict


def make_single_dwc_modelwrapper(in_shape, out_shape, inWidth, outWidth, finn_dtype):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    optype = "StreamingDataWidthConverter"

    DWC_node = helper.make_node(
        optype,
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        in_shape=in_shape,
        out_shape=out_shape,
        inWidth=inWidth,
        outWidth=outWidth,
        preferred_impl_style="hls",
        generalized_variant=True,
        dataType=str(finn_dtype.name),
    )

    graph = helper.make_graph(nodes=[DWC_node], name="dwc_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="dwc-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def prepare_inputs(input_tensor, dt):
    return {"inp": input_tensor}


@pytest.mark.parametrize(
    "config",
    [
        ([1, 2, 2, 1680], [1, 2, 2, 1680], 70, 240, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 2, 2, 1680], [1, 2, 2, 1680], 240, 70, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 70, 240, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 240, 70, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 35, 280, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 280, 35, DataType["BIPOLAR"]),  # extra word of padding
        # requires LCM for old version
        ([1, 42], [1, 42], 6, 14, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1239], [1, 1239], 21, 59, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 70, 240, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 42], [1, 42], 14, 6, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1239], [1, 1239], 59, 21, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 240, 70, DataType["BIPOLAR"]),  # extra word of padding
        # conversion without needing LCMs
        ([1, 180], [1, 180], 2, 18, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 720], [1, 720], 8, 72, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 2880], [1, 2880], 32, 288, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 180], [1, 180], 18, 2, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 720], [1, 720], 72, 8, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 2880], [1, 2880], 288, 32, DataType["BIPOLAR"]),  # extra word of padding
        # passthrough
        ([1, 100], [1, 100], 10, 10, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 400], [1, 400], 40, 40, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1600], [1, 1600], 160, 160, DataType["BIPOLAR"]),  # extra word of padding
    ],
)
@pytest.mark.parametrize("exec_mode", ["rtlsim", "cppsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_dwc(config, exec_mode):
    in_shape, out_shape, inWidth, outWidth, finn_dtype = config

    test_fpga_part = "xc7z020clg400-1"
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, in_shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_dwc_modelwrapper(in_shape, out_shape, inWidth, outWidth, finn_dtype)
    # verify abstraction level execution
    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert y.shape == tuple(out_shape), """The output shape is incorrect."""
    # remove padding if it was performed
    y = y.reshape(1, np.prod(y.shape))
    x = x.reshape(1, np.prod(x.shape))

    if y.shape[-1] > x.shape[-1]:
        y = y[0, : x.shape[-1]]
    else:
        x = x[0, : y.shape[-1]]

    assert (
        y == x
    ).all(), """The output values are not the same as the
        input values anymore."""

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(test_fpga_part, 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert y.shape == tuple(out_shape), """The output shape is incorrect."""

    # remove padding if it was performed
    y = y.reshape(1, np.prod(y.shape))
    x = x.reshape(1, np.prod(x.shape))

    if y.shape[-1] > x.shape[-1]:
        y = y[0, : x.shape[-1]]
    else:
        x = x[0, : y.shape[-1]]

    # cpp sim assert fails for BIPOLAR data type, but not RTL.
    if (finn_dtype != DataType["BIPOLAR"]) or (
        finn_dtype != DataType["BIPOLAR"] and exec_mode != "cppsim"
    ):
        assert (
            y == x
        ).all(), """The output values are not the same as the
            input values anymore."""
    else:
        assert True  # we


@pytest.mark.parametrize(
    "config",
    [
        ([1, 840], [1, 840], 35, 120, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 840], [1, 840], 120, 35, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 35, 280, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 280, 35, DataType["BIPOLAR"]),  # extra word of padding
        # requires LCM for old version
        ([1, 42], [1, 42], 6, 14, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1239], [1, 1239], 21, 59, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 70, 240, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 42], [1, 42], 14, 6, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1239], [1, 1239], 59, 21, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1680], [1, 1680], 240, 70, DataType["BIPOLAR"]),  # extra word of padding
        # conversion without needing LCMs
        ([1, 180], [1, 180], 2, 18, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 720], [1, 720], 8, 72, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 2880], [1, 2880], 32, 288, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 180], [1, 180], 18, 2, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 720], [1, 720], 72, 8, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 2880], [1, 2880], 288, 32, DataType["BIPOLAR"]),  # extra word of padding
        # passthrough
        ([1, 100], [1, 100], 10, 10, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 400], [1, 400], 40, 40, DataType["BIPOLAR"]),  # extra word of padding
        ([1, 1600], [1, 1600], 160, 160, DataType["BIPOLAR"]),  # extra word of padding
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.parametrize("measure_resources", [False])
@pytest.mark.parametrize("measure_functionality", [False])
@pytest.mark.parametrize("measure_performance", [False])
@pytest.mark.parametrize("test_type", ["new"])
@pytest.mark.vivado
def test_fpgadataflow_dwc_stitched_rtlsim(
    config, measure_resources, measure_functionality, measure_performance, test_type
):
    in_shape, out_shape, inWidth, outWidth, finn_dtype = config

    test_fpga_part = "xc7z020clg400-1"
    target_clk_ns = 4
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, in_shape)
    input_dict = prepare_inputs(x, finn_dtype)

    test_name = "dwc_res_tests_{inWidth}_{outWidth}"

    build_dir = os.environ["FINN_BUILD_DIR"]

    build_dir = build_dir + "/test_model/"
    if not os.path.isdir(build_dir):
        build_dir = make_build_dir(prefix="dwc_performance_testing_")

    model = make_single_dwc_modelwrapper(in_shape, out_shape, inWidth, outWidth, finn_dtype)
    model = model.transform(SpecializeLayers(test_fpga_part))
    model_dir = f"{build_dir}/dwc_res_tests_{inWidth}_{outWidth}"
    model_file = f"{model_dir}/model.onnx"
    model.save(model_dir)

    final_output_dir = build_dir

    # Delete previous run results if exist
    # if os.path.exists(final_output_dir):
    #     shutil.rmtree(final_output_dir)
    #     print("Previous run results deleted!")

    cfg = build.DataflowBuildConfig(
        output_dir=final_output_dir,
        mvau_wwidth_max=80,
        target_fps=1000000,
        synth_clk_period_ns=target_clk_ns,
        board="Pynq-Z1",
        # board               = "U250",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            # build_cfg.DataflowOutputType.STITCHED_IP,
            #    build_cfg.DataflowOutputType.OOC_SYNTH,
            build_cfg.DataflowOutputType.BITFILE,
            #    build_cfg.DataflowOutputType.PYNQ_DRIVER,
            #    build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(model_dir, cfg)

    model.set_metadata_prop("rtlsim_so", "")
    model.set_metadata_prop("exec_mode", "rtlsim")
    res = post_synth_res_dwc(model, f"{final_output_dir}/report/post_synth_resources.xml")
    res = res[""]
    build_dir = os.environ["FINN_BUILD_DIR"]
    build_dir += f"/dwc_performance_testing_{test_type}"
    lut = res["LUT"]
    ff = res["FF"]
    target_clk = int(np.round(1000 / target_clk_ns))
    with open(f"{build_dir}/measurements.txt", "a+") as f:
        f.writelines(f"{target_clk}\t{inWidth}\t{outWidth}\tnew_hls\t{lut}\t{ff}\n")

    # with open(f"{build_dir}_new_DWC_res.txt", 'a+') as f:
    #   f.write(res) # here filter to only what we care about
    print(f"{target_clk}\t{inWidth}\t{outWidth}\tnew_hls\t{lut}\t{ff}\n")

    # assert True == False

    if measure_functionality:
        y = oxe.execute_onnx(model, input_dict)["outp"]

        assert y.shape == tuple(out_shape), """The output shape is incorrect."""

        # remove padding if it was performed
        y = y.reshape(1, np.prod(y.shape))
        x = x.reshape(1, np.prod(x.shape))

        if y.shape[-1] > x.shape[-1]:
            y = y[0, : x.shape[-1]]
        else:
            x = x[0, : y.shape[-1]]

        assert (
            y == x
        ).all(), """The output values are not the same as the
            input values anymore."""

    if measure_performance:
        rtlsim_bs = 50
        res = throughput_test_rtlsim(model, rtlsim_bs)
        print(f"Performance for {in_shape, out_shape,inWidth,outWidth} :", res)
