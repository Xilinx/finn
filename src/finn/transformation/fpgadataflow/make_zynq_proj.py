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
from shutil import copy

from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.util.basic import make_build_dir, pynq_part_map

from . import templates


def collect_ip_dirs(model, ipstitch_path):
    # collect list of all IP dirs
    ip_dirs = []
    need_memstreamer = False
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        ip_dir_value = node_inst.get_nodeattr("ip_path")
        assert os.path.isdir(
            ip_dir_value
        ), """The directory that should
        contain the generated ip blocks doesn't exist."""
        ip_dirs += [ip_dir_value]
        if node.op_type in ["StreamingFCLayer_Batch", "Thresholding_Batch"]:
            if node_inst.get_nodeattr("mem_mode") == "decoupled":
                need_memstreamer = True
    ip_dirs += [ipstitch_path + "/ip"]
    if need_memstreamer:
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/memstream")
    return ip_dirs


class MakeZYNQProject(Transformation):
    """Create a Vivado overlay project (including the shell infrastructure)
    from the already-stitched IP block for this graph.
    All nodes in the graph must have the fpgadataflow backend attribute,
    and the CreateStitchedIP transformation must have been previously run on
    the graph. This is functionally equivalent with MakePYNQProject but does
    not use Pynq infrastructure and instead creates a fully custom block design.
    However, this transform requires DMAs in the accelerator design.

    Outcome if successful: sets the vivado_pynq_proj attribute in the ONNX
    ModelProto's metadata_props field, with the created project dir as the
    value.
    """

    def __init__(self, platform, enable_debug=False):
        super().__init__()
        self.platform = platform
        self.enable_debug = 1 if enable_debug else 0

    def apply(self, model):

        # create a config file and empty list of xo files
        config = []
        idma_idx = 0
        odma_idx = 0
        aximm_idx = 0
        axilite_idx = 0
        global_clk_ns = 0
        instance_names = {}
        for node in model.graph.node:
            assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)

            ipstitch_path = kernel_model.get_metadata_prop("vivado_stitch_proj")
            if ipstitch_path is None or (not os.path.isdir(ipstitch_path)):
                raise Exception(
                    "No stitched IPI design found for %s, apply CreateStitchedIP first."
                    % node.name
                )

            vivado_stitch_vlnv = kernel_model.get_metadata_prop("vivado_stitch_vlnv")
            if vivado_stitch_vlnv is None:
                raise Exception(
                    "No vlnv found for %s, apply CreateStitchedIP first." % node.name
                )

            ip_dirs = ["list"]
            ip_dirs += collect_ip_dirs(kernel_model, ipstitch_path)
            ip_dirs_str = "[%s]" % (" ".join(ip_dirs))
            config.append(
                "set_property ip_repo_paths "
                "[concat [get_property ip_repo_paths [current_project]] %s] "
                "[current_project]" % ip_dirs_str
            )
            config.append("update_ip_catalog -rebuild -scan_changes")

            # get metadata property clk_ns to calculate clock frequency
            clk_ns = float(kernel_model.get_metadata_prop("clk_ns"))
            if clk_ns > global_clk_ns:
                global_clk_ns = clk_ns

            ifnames = eval(kernel_model.get_metadata_prop("vivado_stitch_ifnames"))

            # gather info on connectivity
            # assume each node connected to outputs/inputs is DMA:
            # has axis, aximm and axilite
            # everything else is axis-only
            # assume only one connection from each ip to the next
            # all aximm allocated to DDR[0]
            # all kernels allocated to SLR0
            if len(node.input) == 0:
                producer = None
            else:
                producer = model.find_producer(node.input[0])
            consumer = model.find_consumers(node.output[0])
            # define kernel instances
            # name kernels connected to graph inputs as idmaxx
            # name kernels connected to graph outputs as odmaxx
            if (producer is None) or (consumer == []):
                # TODO not a good way of checking for external inp&out
                # should look at the list of top-level in/out instead
                if producer is None:
                    instance_names[node.name] = "idma" + str(idma_idx)
                    idma_idx += 1
                elif consumer == []:
                    instance_names[node.name] = "odma" + str(odma_idx)
                    odma_idx += 1
                config.append(
                    "create_bd_cell -type ip -vlnv %s %s"
                    % (vivado_stitch_vlnv, instance_names[node.name])
                )
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/m_axi_gmem0] "
                    "[get_bd_intf_pins smartconnect_0/S%02d_AXI]"
                    % (instance_names[node.name], aximm_idx)
                )
                assert (
                    len(ifnames["axilite"]) == 1
                ), "Must have 1 AXI lite interface on IODMA nodes"
                axilite_intf_name = ifnames["axilite"][0]
                assert axilite_intf_name is not None
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]"
                    % (instance_names[node.name], axilite_intf_name, axilite_idx)
                )
                # assign_bd_address with appropriate range/offset
                config.append(
                    "assign_axi_addr_proc %s/%s"
                    % (instance_names[node.name], axilite_intf_name)
                )

                aximm_idx += 1
                axilite_idx += 1
            else:
                instance_names[node.name] = node.name
                config.append(
                    "create_bd_cell -type ip -vlnv %s %s"
                    % (vivado_stitch_vlnv, instance_names[node.name])
                )
                for axilite_intf_name in ifnames["axilite"]:
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                        "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]"
                        % (instance_names[node.name], axilite_intf_name, axilite_idx)
                    )
                    # assign_bd_address with appropriate range/offset
                    config.append(
                        "assign_axi_addr_proc %s/%s"
                        % (instance_names[node.name], axilite_intf_name)
                    )
                    axilite_idx += 1
            sdp_node.set_nodeattr("instance_name", instance_names[node.name])

            config.append(
                "connect_bd_net [get_bd_pins %s/ap_clk] "
                "[get_bd_pins smartconnect_0/aclk]" % instance_names[node.name]
            )
            config.append(
                "connect_bd_net [get_bd_pins %s/ap_rst_n] "
                "[get_bd_pins smartconnect_0/aresetn]" % instance_names[node.name]
            )
            # connect streams
            if producer is not None:
                for i in range(len(node.input)):
                    producer = model.find_producer(node.input[i])
                    if producer is not None:
                        j = list(producer.output).index(node.input[i])
                        config.append(
                            "connect_bd_intf_net [get_bd_intf_pins %s/s_axis_%d] "
                            "[get_bd_intf_pins %s/m_axis_%d]"
                            % (
                                instance_names[node.name],
                                i,
                                instance_names[producer.name],
                                j,
                            )
                        )

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_zynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        fclk_mhz = int(1 / (global_clk_ns * 0.001))

        # create a TCL recipe for the project
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        config = "\n".join(config) + "\n"
        with open(ipcfg, "w") as f:
            f.write(
                templates.custom_zynq_shell_template
                % (
                    fclk_mhz,
                    axilite_idx,
                    aximm_idx,
                    self.platform,
                    pynq_part_map[self.platform],
                    config,
                    self.enable_debug,
                )
            )

        # create a TCL recipe for the project
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        working_dir = os.environ["PWD"]
        with open(synth_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_pynq_proj_dir))
            f.write("vivado -mode batch -source %s\n" % ipcfg)
            f.write("cd {}\n".format(working_dir))

        # call the synthesis script
        bash_command = ["bash", synth_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        bitfile_name = (
            vivado_pynq_proj_dir + "/finn_zynq_link.runs/impl_1/top_wrapper.bit"
        )
        if not os.path.isfile(bitfile_name):
            raise Exception(
                "Synthesis failed, no bitfile found. Check logs under %s"
                % vivado_pynq_proj_dir
            )
        deploy_bitfile_name = vivado_pynq_proj_dir + "/resizer.bit"
        copy(bitfile_name, deploy_bitfile_name)
        # set bitfile attribute
        model.set_metadata_prop("bitfile", deploy_bitfile_name)
        hwh_name_alts = [
            vivado_pynq_proj_dir
            + "/finn_zynq_link.srcs/sources_1/bd/top/hw_handoff/top.hwh",
            vivado_pynq_proj_dir
            + "/finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh",
        ]
        hwh_name = None
        for hwh_name_cand in hwh_name_alts:
            if os.path.isfile(hwh_name_cand):
                hwh_name = hwh_name_cand
        if not os.path.isfile(hwh_name):
            raise Exception(
                "Synthesis failed, no bitfile found. Check logs under %s"
                % vivado_pynq_proj_dir
            )
        deploy_hwh_name = vivado_pynq_proj_dir + "/resizer.hwh"
        copy(hwh_name, deploy_hwh_name)
        model.set_metadata_prop("hw_handoff", deploy_hwh_name)
        # filename for the synth utilization report
        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        return (model, False)


class ZynqBuild(Transformation):
    """Best-effort attempt at building the accelerator for Zynq.
    It assumes the model has only fpgadataflow nodes

    """

    def __init__(
        self,
        platform,
        period_ns,
        enable_debug=False,
        partition_model_dir=None,
    ):
        super().__init__()
        self.fpga_part = pynq_part_map[platform]
        self.period_ns = period_ns
        self.platform = platform
        self.enable_debug = enable_debug
        self.partition_model_dir = partition_model_dir

    def apply(self, model):
        # first infer layouts
        model = model.transform(InferDataLayouts())
        # prepare at global level, then break up into kernels
        prep_transforms = [
            InsertIODMA(64),
            InsertDWC(),
            Floorplan(),
            CreateDataflowPartition(partition_model_dir=self.partition_model_dir),
        ]
        for trn in prep_transforms:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
        # Build each kernel individually
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        for sdp_node in sdp_nodes:
            prefix = sdp_node.name + "_"
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_model = kernel_model.transform(InsertFIFO())
            kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
            kernel_model.save(dataflow_model_filename)
            kernel_model = kernel_model.transform(
                PrepareIP(self.fpga_part, self.period_ns)
            )
            kernel_model = kernel_model.transform(HLSSynthIP())
            kernel_model = kernel_model.transform(
                CreateStitchedIP(
                    self.fpga_part, self.period_ns, sdp_node.onnx_node.name, False
                )
            )
            kernel_model.set_metadata_prop("platform", "zynq-iodma")
            kernel_model.save(dataflow_model_filename)
        # Assemble design from IPs
        model = model.transform(
            MakeZYNQProject(self.platform, enable_debug=self.enable_debug)
        )

        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "zynq-iodma")

        return (model, False)
