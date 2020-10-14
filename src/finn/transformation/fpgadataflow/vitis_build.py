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

import os
import subprocess

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.base import Transformation
from finn.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.util.basic import make_build_dir
from finn.transformation.infer_data_layouts import InferDataLayouts
from . import templates
from enum import Enum


def _check_vitis_envvars():
    assert "VITIS_PATH" in os.environ, "VITIS_PATH must be set for Vitis"
    assert (
        "PLATFORM_REPO_PATHS" in os.environ
    ), "PLATFORM_REPO_PATHS must be set for Vitis"
    assert (
        "XILINX_XRT" in os.environ
    ), "XILINX_XRT must be set for Vitis, ensure the XRT env is sourced"


class VitisOptStrategy(Enum):
    "Values applicable to VitisBuild optimization strategy."

    DEFAULT = "0"
    POWER = "1"
    PERFORMANCE = "2"
    PERFORMANCE_BEST = "3"
    SIZE = "s"
    BUILD_SPEED = "quick"


class CreateVitisXO(Transformation):
    """Create a Vitis object file from a stitched FINN ip.

    Outcome if successful: sets the vitis_xo attribute in the ONNX
    ModelProto's metadata_props field with the name of the object file as value.
    The object file can be found under the ip subdirectory.
    """

    def __init__(self, ip_name="finn_design"):
        super().__init__()
        self.ip_name = ip_name

    def apply(self, model):
        _check_vitis_envvars()
        vivado_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        stitched_ip_dir = vivado_proj_dir + "/ip"
        args_string = []
        m_axis_idx = 0
        s_axis_idx = 0
        # NOTE: this assumes the graph is Vitis-compatible: max one axi lite interface
        # developed from instructions in UG1393 (v2019.2) and package_xo documentation
        # package_xo is responsible for generating the kernel xml
        for node in model.graph.node:
            node_inst = getCustomOp(node)
            arg_id = 0
            if node.op_type == "TLastMarker":
                stream_width = node_inst.get_nodeattr("StreamWidth")
                # add a stream input or output port, based on direction
                if node_inst.get_nodeattr("Direction") == "in":
                    args_string.append(
                        "{in:4:%s:s_axis_%d:0x0:0x0:ap_uint&lt;%s>:0}"
                        % (str(arg_id), s_axis_idx, str(stream_width))
                    )
                    s_axis_idx += 1
                else:
                    args_string.append(
                        "{out:4:%s:m_axis_%d:0x0:0x0:ap_uint&lt;%s>:0}"
                        % (str(arg_id), m_axis_idx, str(stream_width))
                    )
                    m_axis_idx += 1
                arg_id += 1
                # add a axilite port if dynamic
                # add a count parameter if dynamic
                if node_inst.get_nodeattr("DynIters") == 1:
                    args_string.append(
                        "{numReps:0:%s:s_axi_control:0x4:0x10:uint:0}" % str(arg_id)
                    )
                    arg_id += 1
            elif node.op_type == "IODMA":
                port_width = node_inst.get_nodeattr("intfWidth")
                # add an address parameter
                # add a count parameter
                args_string.append(
                    "{addr:1:%s:m_axi_gmem0:0x8:0x10:ap_uint&lt;%s>*:0}"
                    % (str(arg_id), str(port_width))
                )
                arg_id += 1
                args_string.append(
                    "{numReps:0:%s:s_axi_control:0x4:0x1C:uint:0}" % str(arg_id)
                )
                arg_id += 1

        # save kernel xml then run package_xo
        xo_name = self.ip_name + ".xo"
        xo_path = vivado_proj_dir + "/" + xo_name
        model.set_metadata_prop("vitis_xo", xo_path)

        # generate the package_xo command in a tcl script
        package_xo_string = (
            "package_xo -force -xo_path %s -kernel_name %s -ip_directory %s"
            % (xo_path, self.ip_name, stitched_ip_dir)
        )
        for arg in args_string:
            package_xo_string += " -kernel_xml_args " + arg
        with open(vivado_proj_dir + "/gen_xo.tcl", "w") as f:
            f.write(package_xo_string)

        # create a shell script and call Vivado
        package_xo_sh = vivado_proj_dir + "/gen_xo.sh"
        working_dir = os.environ["PWD"]
        with open(package_xo_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_proj_dir))
            f.write("vivado -mode batch -source gen_xo.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", package_xo_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        assert os.path.isfile(xo_path), (
            "Vitis .xo file not created, check logs under %s" % vivado_proj_dir
        )
        return (model, False)


class VitisLink(Transformation):
    """Create an XCLBIN with Vitis.

    Outcome if successful: sets the bitfile attribute in the ONNX
    ModelProto's metadata_props field with the XCLBIN full path as value.
    """

    def __init__(
        self, platform, f_mhz=200, strategy=VitisOptStrategy.PERFORMANCE,
        enable_debug=False
    ):
        super().__init__()
        self.platform = platform
        self.f_mhz = f_mhz
        self.strategy = strategy
        self.enable_debug = enable_debug

    def apply(self, model):
        _check_vitis_envvars()
        # create a config file and empty list of xo files
        config = ["[connectivity]"]
        object_files = []
        idma_idx = 0
        odma_idx = 0
        instance_names = {}
        for node in model.graph.node:
            assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_xo = kernel_model.get_metadata_prop("vitis_xo")
            object_files.append(kernel_xo)
            # gather info on connectivity
            # assume each node connected to outputs/inputs is DMA:
            # has axis, aximm and axilite
            # everything else is axis-only
            # assume only one connection from each ip to the next
            # all aximm allocated to DDR[0]
            # all kernels allocated to SLR0
            producer = model.find_producer(node.input[0])
            consumer = model.find_consumers(node.output[0])
            # define kernel instances
            # name kernels connected to graph inputs as idmaxx
            # name kernels connected to graph inputs as odmaxx
            if producer is None:
                instance_names[node.name] = "idma" + str(idma_idx)
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
                idma_idx += 1
            elif consumer is None:
                instance_names[node.name] = "odma" + str(odma_idx)
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
                odma_idx += 1
            else:
                instance_names[node.name] = node.name
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
            # assign SLRs
            config.append("slr=%s:SLR0" % instance_names[node.name])
            # assign memory banks
            if producer is None or consumer is None:
                config.append(
                    "sp=%s.m_axi_gmem0:DDR[%d]" % (instance_names[node.name], 0)
                )
            # connect streams
            if producer is not None:
                for i in range(len(node.input)):
                    producer = model.find_producer(node.input[i])
                    if producer is not None:
                        j = list(producer.output).index(node.input[i])
                        config.append(
                            "stream_connect=%s.m_axis_%d:%s.s_axis_%d"
                            % (
                                instance_names[producer.name],
                                j,
                                instance_names[node.name],
                                i,
                            )
                        )

        # create a temporary folder for the project
        link_dir = make_build_dir(prefix="vitis_link_proj_")
        model.set_metadata_prop("vitis_link_proj", link_dir)

        config = "\n".join(config) + "\n"
        with open(link_dir + "/config.txt", "w") as f:
            f.write(config)

        # create tcl script to generate resource report in XML format
        gen_rep_xml = templates.vitis_gen_xml_report_tcl_template
        gen_rep_xml = gen_rep_xml.replace("$VITIS_PROJ_PATH$", link_dir)
        with open(link_dir + "/gen_report_xml.tcl", "w") as f:
            f.write(gen_rep_xml)

        debug_commands = []
        if self.enable_debug:
            for inst in list(instance_names.values()):
                debug_commands.append("--dk chipscope:%s" % inst)

        # create a shell script and call Vitis
        script = link_dir + "/run_vitis_link.sh"
        working_dir = os.environ["PWD"]
        with open(script, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(link_dir))
            f.write(
                "v++ -t hw --platform %s --link %s"
                " --kernel_frequency %d --config config.txt --optimize %s"
                " --save-temps -R2 %s\n"
                % (
                    self.platform,
                    " ".join(object_files),
                    self.f_mhz,
                    self.strategy.value,
                    " ".join(debug_commands),
                )
            )
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", script]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        # TODO rename xclbin appropriately here?
        xclbin = link_dir + "/a.xclbin"
        assert os.path.isfile(xclbin), (
            "Vitis .xclbin file not created, check logs under %s" % link_dir
        )
        model.set_metadata_prop("bitfile", xclbin)

        # run Vivado to gen xml report
        gen_rep_xml_sh = link_dir + "/gen_report_xml.sh"
        working_dir = os.environ["PWD"]
        with open(gen_rep_xml_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(link_dir))
            f.write(
                "vivado -mode tcl -source %s\n" % (link_dir + "/gen_report_xml.tcl")
            )
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", gen_rep_xml_sh]
        process_genxml = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_genxml.communicate()
        # filename for the synth utilization report
        synth_report_filename = link_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        return (model, False)


class VitisBuild(Transformation):
    """Best-effort attempt at building the accelerator with Vitis."""

    def __init__(
        self, fpga_part, period_ns, platform,
        strategy=VitisOptStrategy.PERFORMANCE,
        enable_debug=False
    ):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.platform = platform
        self.strategy = strategy
        self.enable_debug = enable_debug

    def apply(self, model):
        _check_vitis_envvars()
        # first infer layouts
        model = model.transform(InferDataLayouts())
        # prepare at global level, then break up into kernels
        prep_transforms = [
            MakePYNQDriver(platform="alveo"),
            InsertIODMA(512),
            InsertDWC(),
            Floorplan(),
            CreateDataflowPartition(),
        ]
        for trn in prep_transforms:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
        # Build each kernel individually
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        for sdp_node in sdp_nodes:
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_model = kernel_model.transform(InsertFIFO())
            kernel_model = kernel_model.transform(
                InsertTLastMarker(both=True, external=False, dynamic=False)
            )
            kernel_model = kernel_model.transform(GiveUniqueNodeNames())
            kernel_model.save(dataflow_model_filename)
            kernel_model = kernel_model.transform(
                PrepareIP(self.fpga_part, self.period_ns)
            )
            kernel_model = kernel_model.transform(HLSSynthIP())
            kernel_model = kernel_model.transform(
                CreateStitchedIP(
                    self.fpga_part, self.period_ns, sdp_node.onnx_node.name, True
                )
            )
            kernel_model = kernel_model.transform(
                CreateVitisXO(sdp_node.onnx_node.name)
            )
            kernel_model.set_metadata_prop("platform", "alveo")
            kernel_model.save(dataflow_model_filename)
        # Assemble design from kernels
        model = model.transform(
            VitisLink(
                self.platform, round(1000 / self.period_ns), strategy=self.strategy,
                enable_debug=self.enable_debug
            )
        )
        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "alveo")

        return (model, False)
