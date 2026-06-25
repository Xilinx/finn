# Copyright (C) 2020, Xilinx, Inc.
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

import multiprocessing as mp
import os
import subprocess
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import get_num_default_workers
from shutil import copy

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.make_zynq_proj import collect_ip_dirs
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir, part_map, pynq_native_port_width

from . import templates

# golden reference dependencies (golden_ref.tcl, golden_noc.ncr,
versal_golden_dir = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "pynq_deps")
)


class MakeVersalProject(Transformation):
    """Create a Vivado overlay for the already-stitched IP, built on top of a
    Versal *golden reference design* (e.g. VCK190).

    Unlike MakeZYNQProject, which builds a custom block design from scratch and
    instantiates the PS itself, this transformation sources the golden
    reference (golden_ref.tcl) that provides the static CIPS + NoC + DDR
    "contract", adds the FINN accelerator as a PL overlay, locks the NoC
    solution to golden_noc.ncr, and verifies the result against
    golden_routed.dcp with pr_verify. This keeps the generated PL PDI
    compatible with the golden boot PDI.

    All nodes in the graph must have the fpgadataflow backend attribute, and
    CreateStitchedIP must have been previously run on the graph. Requires DMAs
    in the accelerator design.

    Outcome if successful: sets the vivado_pynq_proj/bitfile/hw_handoff
    metadata_props in the ONNX ModelProto.
    """

    def __init__(self, platform, period_ns, golden_dir=None, enable_debug=False):
        super().__init__()
        self.platform = platform
        self.fpga_part = part_map[platform]
        self.period_ns = period_ns
        self.golden_dir = golden_dir if golden_dir is not None else versal_golden_dir
        self.enable_debug = 1 if enable_debug else 0

    def apply(self, model):
        config = []
        idma_idx = 0
        odma_idx = 0
        aximm_idx = 0
        axilite_idx = 0
        instance_names = {}
        for node in model.graph.node:
            assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)

            ipstitch_path = kernel_model.get_metadata_prop("vivado_stitch_proj")
            if ipstitch_path is None or (not os.path.isdir(ipstitch_path)):
                raise Exception(
                    "No stitched IPI design found for %s, apply CreateStitchedIP first." % node.name
                )

            vivado_stitch_vlnv = kernel_model.get_metadata_prop("vivado_stitch_vlnv")
            if vivado_stitch_vlnv is None:
                raise Exception("No vlnv found for %s, apply CreateStitchedIP first." % node.name)

            ip_dirs = ["list"]
            ip_dirs += collect_ip_dirs(kernel_model, ipstitch_path)
            ip_dirs_str = "[%s]" % (" ".join(ip_dirs))
            config.append(
                "set_property ip_repo_paths "
                "[concat [get_property ip_repo_paths [current_project]] %s] "
                "[current_project]" % ip_dirs_str
            )
            config.append("update_ip_catalog -rebuild -scan_changes")

            ifnames = eval(kernel_model.get_metadata_prop("vivado_stitch_ifnames"))

            # gather connectivity info (same assumptions as MakeZYNQProject):
            # kernels connected to graph inputs/outputs are DMAs (axis+aximm+axilite),
            # everything else is axis-only; one connection from each ip to the next.
            if len(node.input) == 0:
                producer = None
            else:
                producer = model.find_producer(node.input[0])
            consumer = model.find_consumers(node.output[0])
            if (producer is None) or (consumer == []):
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
                # aximm master -> DDR SmartConnect -> axi_noc_pl/S00_AXI
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/m_axi_gmem0] "
                    "[get_bd_intf_pins smartconnect_0/S%02d_AXI]"
                    % (instance_names[node.name], aximm_idx)
                )
                # map this master onto DDR through the PS NoC inter-NoC port
                config.append("assign_ddr_addr_proc %s/m_axi_gmem0" % instance_names[node.name])
                assert len(ifnames["axilite"]) == 1, "Must have 1 AXI lite interface on IODMA nodes"
                axilite_intf_name = ifnames["axilite"][0]
                assert axilite_intf_name is not None
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]"
                    % (instance_names[node.name], axilite_intf_name, axilite_idx)
                )
                config.append(
                    "assign_axi_addr_proc %s/%s" % (instance_names[node.name], axilite_intf_name)
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
                    config.append(
                        "assign_axi_addr_proc %s/%s"
                        % (instance_names[node.name], axilite_intf_name)
                    )
                    axilite_idx += 1
            sdp_node.set_nodeattr("instance_name", instance_names[node.name])

            # PL clock/reset from the golden reference
            config.append(
                "connect_bd_net [get_bd_pins %s/ap_clk] "
                "[get_bd_pins versal_cips_0/pl0_ref_clk]" % instance_names[node.name]
            )
            config.append(
                "connect_bd_net [get_bd_pins %s/ap_rst_n] "
                "[get_bd_pins rst_pl0/peripheral_aresetn]" % instance_names[node.name]
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
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_versal_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        fclk_mhz = int(1 / (self.period_ns * 0.001))

        # create a TCL recipe for the project
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        config = "\n".join(config) + "\n"
        num_workers = get_num_default_workers()
        assert num_workers >= 0, "Number of workers must be nonnegative."
        if num_workers == 0:
            num_workers = mp.cpu_count()
        with open(ipcfg, "w") as f:
            f.write(
                templates.custom_versal_shell_template
                % (
                    fclk_mhz,
                    axilite_idx,
                    aximm_idx,
                    self.platform,
                    self.fpga_part,
                    self.golden_dir,
                    config,
                    self.enable_debug,
                    num_workers,
                )
            )

        # create a shell script to launch the synthesis
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

        # the Versal overlay deliverable is the PL PDI (loaded at runtime via PYNQ)
        pdi_name = vivado_pynq_proj_dir + "/finn_versal.pdi"
        if not os.path.isfile(pdi_name):
            raise Exception(
                "Synthesis failed, no PDI found. Check logs under %s" % vivado_pynq_proj_dir
            )
        deploy_pdi_name = vivado_pynq_proj_dir + "/resizer.pdi"
        copy(pdi_name, deploy_pdi_name)
        model.set_metadata_prop("bitfile", deploy_pdi_name)

        hwh_name = vivado_pynq_proj_dir + "/finn_versal.hwh"
        if not os.path.isfile(hwh_name):
            raise Exception(
                "Synthesis failed, no hwh found. Check logs under %s" % vivado_pynq_proj_dir
            )
        deploy_hwh_name = vivado_pynq_proj_dir + "/resizer.hwh"
        copy(hwh_name, deploy_hwh_name)
        model.set_metadata_prop("hw_handoff", deploy_hwh_name)

        # filename for the synth utilization report
        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        return (model, False)


class VersalBuild(Transformation):
    """Best-effort attempt at building the accelerator for embedded Versal
    (e.g. VCK190) on top of a golden reference design.
    It assumes the model has only fpgadataflow nodes.
    """

    def __init__(
        self,
        platform,
        period_ns,
        enable_debug=False,
        partition_model_dir=None,
        golden_dir=None,
    ):
        super().__init__()
        self.fpga_part = part_map[platform]
        self.axi_port_width = pynq_native_port_width[platform]
        self.period_ns = period_ns
        self.platform = platform
        self.enable_debug = enable_debug
        self.partition_model_dir = partition_model_dir
        self.golden_dir = golden_dir

    def apply(self, model):
        # first infer layouts
        model = model.transform(InferDataLayouts())
        # prepare at global level, then break up into kernels
        prep_transforms = [
            InsertIODMA(self.axi_port_width),
            InsertDWC(),
            SpecializeLayers(self.fpga_part),
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
            kernel_model = kernel_model.transform(SpecializeLayers(self.fpga_part))
            kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
            kernel_model.save(dataflow_model_filename)
            kernel_model = kernel_model.transform(PrepareIP(self.fpga_part, self.period_ns))
            kernel_model = kernel_model.transform(HLSSynthIP())
            kernel_model = kernel_model.transform(
                CreateStitchedIP(self.fpga_part, self.period_ns, sdp_node.onnx_node.name)
            )
            kernel_model.set_metadata_prop("platform", "zynq-iodma")
            kernel_model.save(dataflow_model_filename)
        # Assemble design from IPs on top of the golden reference
        model = model.transform(
            MakeVersalProject(
                self.platform,
                self.period_ns,
                golden_dir=self.golden_dir,
                enable_debug=self.enable_debug,
            )
        )

        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "zynq-iodma")

        return (model, False)
