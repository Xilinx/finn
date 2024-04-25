##
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
##

import numpy as np
import os
import shutil
import subprocess
from qonnx.custom_op.registry import getCustomOp

import finn.util.data_packing as dpk
from finn.custom_op.fpgadataflow.templates import ipgentcl_template
from finn.transformation.fpgadataflow.vitis_build import CreateVitisXO
from finn.util.hls import CallHLS
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP


def test_step_gen_vitis_xo(model, cfg):
    xo_dir = cfg.output_dir + "/xo"
    xo_dir = str(os.path.abspath(xo_dir))
    os.makedirs(xo_dir, exist_ok=True)
    model = model.transform(CreateVitisXO())
    xo_path = model.get_metadata_prop("vitis_xo")
    shutil.copy(xo_path, xo_dir)
    return model


def test_step_gen_instrumentation_wrapper(model, cfg):
    xo_dir = cfg.output_dir + "/xo"
    xo_dir = str(os.path.abspath(xo_dir))
    os.makedirs(xo_dir, exist_ok=True)
    wrapper_output_dir = cfg.output_dir + "/instrumentation_wrapper"
    wrapper_output_dir = str(os.path.abspath(wrapper_output_dir))
    os.makedirs(wrapper_output_dir, exist_ok=True)
    # conservative max for pending feature maps: number of layers
    pending = len(model.graph.node)
    # query the parallelism-dependent folded input shape from the
    # node consuming the graph input
    inp_name = model.graph.input[0].name
    inp_node = getCustomOp(model.find_consumer(inp_name))
    inp_shape_folded = list(inp_node.get_folded_input_shape())
    inp_stream_width = inp_node.get_instream_width_padded()
    # number of beats per input is given by product of folded input
    # shape except the last dim (which is the stream width)
    ilen = np.prod(inp_shape_folded[:-1])
    ti = "ap_uint<%d>" % inp_stream_width
    # perform the same for the output
    out_name = model.graph.output[0].name
    out_node = getCustomOp(model.find_producer(out_name))
    out_shape_folded = list(out_node.get_folded_output_shape())
    out_stream_width = out_node.get_outstream_width_padded()
    olen = np.prod(out_shape_folded[:-1])
    to = "ap_uint<%d>" % out_stream_width
    ko = out_shape_folded[-1]
    # fill out instrumentation wrapper template
    with open("templates/instrumentation_wrapper.template.cpp", "r") as f:
        instrwrp_cpp = f.read()
    instrwrp_cpp = instrwrp_cpp.replace("@PENDING@", str(pending))
    instrwrp_cpp = instrwrp_cpp.replace("@ILEN@", str(ilen))
    instrwrp_cpp = instrwrp_cpp.replace("@OLEN@", str(olen))
    instrwrp_cpp = instrwrp_cpp.replace("@TI@", str(ti))
    instrwrp_cpp = instrwrp_cpp.replace("@TO@", str(to))
    instrwrp_cpp = instrwrp_cpp.replace("@KO@", str(ko))
    with open(wrapper_output_dir + "/top_instrumentation_wrapper.cpp", "w") as f:
        f.write(instrwrp_cpp)
    # fill out HLS synthesis tcl template
    prjname = "project_instrwrap"
    ipgentcl = ipgentcl_template
    ipgentcl = ipgentcl.replace("$PROJECTNAME$", prjname)
    ipgentcl = ipgentcl.replace("$HWSRCDIR$", wrapper_output_dir)
    ipgentcl = ipgentcl.replace("$TOPFXN$", "instrumentation_wrapper")
    ipgentcl = ipgentcl.replace("$FPGAPART$", cfg._resolve_fpga_part())
    ipgentcl = ipgentcl.replace("$CLKPERIOD$", str(cfg.synth_clk_period_ns))
    ipgentcl = ipgentcl.replace("$DEFAULT_DIRECTIVES$", "")
    ipgentcl = ipgentcl.replace("$EXTRA_DIRECTIVES$", "config_export -format xo")
    # use Vitis RTL kernel (.xo) output instead of IP-XACT
    ipgentcl = ipgentcl.replace("export_design -format ip_catalog", "export_design -format xo")
    with open(wrapper_output_dir + "/hls_syn.tcl", "w") as f:
        f.write(ipgentcl)
    # build bash script to launch HLS synth and call it
    code_gen_dir = wrapper_output_dir
    builder = CallHLS()
    builder.append_tcl(code_gen_dir + "/hls_syn.tcl")
    builder.set_ipgen_path(code_gen_dir + "/{}".format(prjname))
    builder.build(code_gen_dir)
    ipgen_path = builder.ipgen_path
    assert os.path.isdir(ipgen_path), "HLS IPGen failed: %s not found" % (ipgen_path)
    ip_path = ipgen_path + "/sol1/impl/ip"
    assert os.path.isdir(ip_path), "HLS IPGen failed: %s not found. Check log under %s" % (
        ip_path,
        code_gen_dir,
    )
    xo_path = code_gen_dir + "/{}/sol1/impl/export.xo".format(prjname)
    xo_instr_path = xo_dir + "/instrumentation_wrapper.xo"
    shutil.copy(xo_path, xo_instr_path)
    return model


def test_step_export_xo(model, cfg):
    # Copy the generated .xo files to their respective Vitis IP directory
    result = subprocess.call(['cp', cfg.output_dir+"/xo/finn_design.xo", 'instr_wrap_platform/vitis/ip/finn_design/src'])
    result = subprocess.call(['cp', cfg.output_dir+"/xo/instrumentation_wrapper.xo", 'instr_wrap_platform/vitis/ip/instrumentation_wrapper/src'])
    return model


def test_step_build_platform(model, cfg):
    # Clean any previous/partial builds and then build full platform
    result = subprocess.call("cd instr_wrap_platform && make clean && make all", shell=True)
    return model
