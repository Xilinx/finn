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
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import get_num_default_workers
from shutil import copy

from finn.transformation.fpgadataflow.assign_ddr_weight_offsets import mlo_regions
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
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import (
    get_pynq_golden_dir,
    is_versal,
    make_build_dir,
    pynq_golden_files,
    pynq_native_port_width,
    pynq_part_map,
)
from finn.util.fpgadataflow import find_clk2x_nodes
from finn.util.launcher import run_tool

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
        if node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls":
            if node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
                need_memstreamer = True
        if node.op_type == "FINNLoop":
            loop_body = node_inst.get_nodeattr("body")
            loop_body_ipstitch_path = loop_body.get_metadata_prop("vivado_stitch_proj")
            assert loop_body_ipstitch_path is not None, (
                "No stitched IPI design found for the body of %s, " % node.name
            )
            ip_dirs += collect_ip_dirs(loop_body, loop_body_ipstitch_path)
    ip_dirs += [ipstitch_path + "/ip"]
    if need_memstreamer:
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/memstream")
    return ip_dirs


def check_versal_shell_clk2x(model: ModelWrapper, platform: str | None):
    """Reject double-pumped designs for the Versal (golden reference) shell.

    That shell clocks the whole overlay from pl0_ref_clk and has no phase-aligned
    2x clock for ap_clk2x. The internal clock generator of the stitched IP
    (CreateStitchedIP clk2x_gen) is the intended path, but has not been verified
    with this shell.
    """
    clk2x_nodes = find_clk2x_nodes(model)
    if clk2x_nodes:
        raise Exception(
            "The Versal shell for %s does not support double-pumped kernels yet (a "
            "kernel-internal ap_clk2x generator is the intended path, but unverified), "
            "but these nodes need ap_clk2x (pumpedCompute/pumpedMemory): %s. Disable "
            "clock pumping for this board." % (platform, ", ".join(clk2x_nodes))
        )


class MakePYNQProject(Transformation):
    """Create a Vivado overlay project (including the shell infrastructure)
    from the already-stitched IP block for this graph.
    All nodes in the graph must have the fpgadataflow backend attribute,
    and the CreateStitchedIP transformation must have been previously run on
    the graph. For Zynq this creates a fully custom block design. For Versal
    it builds on the PYNQ golden reference (requires FINN_VCK190_GOLDEN=1).
    However, this transform requires DMAs in the accelerator design.

    The shells provide no double-speed clock ap_clk2x: kernels with
    double-pumped nodes must derive it internally (CreateStitchedIP clk2x_gen).
    The Versal shell rejects double-pumped designs altogether.

    Outcome if successful: sets the vivado_pynq_proj attribute in the ONNX
    ModelProto's metadata_props field, with the created project dir as the
    value.
    """

    def __init__(self, platform, period_ns, enable_debug=False):
        super().__init__()
        self.platform = platform
        self.fpga_part = pynq_part_map[platform]
        self.period_ns = period_ns
        self.enable_debug = 1 if enable_debug else 0

    def apply(self, model):
        versal = is_versal(self.fpga_part)
        golden_dir = None
        if versal:
            clk_pin = "versal_cips_0/pl0_ref_clk"
            rst_pin = "rst_pl0/peripheral_aresetn"
            golden_dir = get_pynq_golden_dir(self.platform)
            missing = [
                f for f in pynq_golden_files if not os.path.isfile(os.path.join(golden_dir, f))
            ]
            if missing:
                raise Exception(
                    "Golden reference files missing from %s: %s." % (golden_dir, ", ".join(missing))
                )
            template = templates.custom_versal_shell_template
            check_versal_shell_clk2x(model, self.platform)
        else:
            clk_pin = "smartconnect_0/aclk"
            rst_pin = "smartconnect_0/aresetn"
            template = templates.custom_zynq_shell_template

        # create a config file and empty list of xo files
        config = []
        idma_idx = 0
        odma_idx = 0
        aximm_idx = 0
        axilite_idx = 0
        instance_names = {}
        # AXI-MM masters of the kernels other than the I/O DMAs (MLO weights and frames),
        # as the pins on every memory port
        masters = {}
        need_cdma = False
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
            if ifnames.get("clk2x"):
                raise Exception(
                    "Kernel %s has an ap_clk2x input, which the shell cannot drive: "
                    "stitch it with the internal clock generator (clk2x_gen)." % node.name
                )

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
                assert len(ifnames["axilite"]) == 1, "Must have 1 AXI lite interface on IODMA nodes"
                axilite_intf_name = ifnames["axilite"][0]
                assert axilite_intf_name is not None
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]"
                    % (instance_names[node.name], axilite_intf_name, axilite_idx)
                )
                # assign_bd_address with appropriate range/offset
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
                    # assign_bd_address with appropriate range/offset
                    config.append(
                        "assign_axi_addr_proc %s/%s"
                        % (instance_names[node.name], axilite_intf_name)
                    )
                    axilite_idx += 1
                # the masters are connected below, once all of them are known. The port
                # is the one chosen by AssignMemory (in mem_port_id), a master that is not
                # an MLO region (or has none assigned) uses port 0
                regions = mlo_regions(kernel_model)
                port_of_intf = {
                    r["intf_name"]: max(r["inst"].get_nodeattr("mem_port_id"), 0) for r in regions
                }
                for intf_name, _ in ifnames["aximm"]:
                    masters.setdefault(port_of_intf.get(intf_name, 0), []).append(
                        "%s/%s" % (instance_names[node.name], intf_name)
                    )
                # weights in a fixed-address bank are loaded by an AXI CDMA
                need_cdma |= any(
                    "body" not in r and r["inst"].get_nodeattr("mem_bank") > 0 for r in regions
                )
            sdp_node.set_nodeattr("instance_name", instance_names[node.name])

            config.append(
                "connect_bd_net [get_bd_pins %s/ap_clk] "
                "[get_bd_pins %s]" % (instance_names[node.name], clk_pin)
            )
            config.append(
                "connect_bd_net [get_bd_pins %s/ap_rst_n] "
                "[get_bd_pins %s]" % (instance_names[node.name], rst_pin)
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

        for port, pins in sorted(masters.items()):
            if port == 0:
                for pin in pins:
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s] "
                        "[get_bd_intf_pins smartconnect_0/S%02d_AXI]" % (pin, aximm_idx)
                    )
                    aximm_idx += 1
                continue
            sc = "smartconnect_mlo%d" % port
            config.append("create_bd_cell -type ip -vlnv $smartconnect_vlnv %s" % sc)
            config.append(
                "set_property -dict [list CONFIG.NUM_SI {%d} CONFIG.NUM_MI {1}] "
                "[get_bd_cells %s]" % (len(pins), sc)
            )
            config.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/M00_AXI] [mlo_port %d]" % (sc, port)
            )
            for i, pin in enumerate(pins):
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s] "
                    "[get_bd_intf_pins %s/S%02d_AXI]" % (pin, sc, i)
                )
            config.append("connect_bd_net [get_bd_pins %s] [get_bd_pins %s/aclk]" % (clk_pin, sc))
            config.append(
                "connect_bd_net [get_bd_pins %s] [get_bd_pins %s/aresetn]" % (rst_pin, sc)
            )

        if need_cdma:
            assert versal, "Only the Versal shell has an AXI CDMA for fixed-address memory banks"
            config.append(
                "add_mlo_cdma mlo_cdma %d %s %s %d %d"
                % (
                    pynq_native_port_width[self.platform],
                    clk_pin,
                    rst_pin,
                    aximm_idx,
                    axilite_idx,
                )
            )
            aximm_idx += 1
            axilite_idx += 1

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_pynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        fclk_mhz = int(1 / (self.period_ns * 0.001))

        # create a TCL recipe for the project
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        config = "\n".join(config) + "\n"
        num_workers = get_num_default_workers()
        assert num_workers >= 0, "Number of workers must be nonnegative."
        if num_workers == 0:
            num_workers = mp.cpu_count()
        template_repl = {
            "@FREQ_MHZ@": fclk_mhz,
            "@NUM_AXILITE@": axilite_idx,
            "@NUM_AXIMM@": aximm_idx,
            "@BOARD@": self.platform,
            "@FPGA_PART@": self.fpga_part,
            "@CONFIG@": config,
            "@ENABLE_DEBUG@": self.enable_debug,
            "@NUM_WORKERS@": num_workers,
            "@GOLDEN_DIR@": golden_dir,
        }
        for key, val in template_repl.items():
            if val is not None:
                template = template.replace(key, str(val))
        with open(ipcfg, "w") as f:
            f.write(template)

        # launch the synthesis; synth_project.sh repeats the run by hand
        run_tool(
            ["vivado", "-mode", "batch", "-source", ipcfg],
            cwd=vivado_pynq_proj_dir,
            replay=vivado_pynq_proj_dir + "/synth_project.sh",
        )
        deliverables = [
            vivado_pynq_proj_dir + "/finn_link/finn_link.runs/impl_1/finn_link_wrapper.bit",
            vivado_pynq_proj_dir + "/finn_link/finn_link.runs/impl_1/finn_link_wrapper_pld.pdi",
        ]
        bitfile_name = next((f for f in deliverables if os.path.isfile(f)), None)
        if bitfile_name is None:
            raise Exception(
                "Synthesis failed, no deliverable (bitfile/pdi) found. Check logs under %s"
                % vivado_pynq_proj_dir
            )
        deploy_bitfile_name = vivado_pynq_proj_dir + "/resizer" + os.path.splitext(bitfile_name)[1]
        copy(bitfile_name, deploy_bitfile_name)
        # set bitfile attribute
        model.set_metadata_prop("bitfile", deploy_bitfile_name)
        hwh_name_alts = [
            vivado_pynq_proj_dir
            + "/finn_link/finn_link.gen/sources_1/bd/finn_link/hw_handoff/finn_link.hwh",
            vivado_pynq_proj_dir
            + "/finn_link/finn_link.srcs/sources_1/bd/finn_link/hw_handoff/finn_link.hwh",
        ]
        hwh_name = None
        for hwh_name_cand in hwh_name_alts:
            if os.path.isfile(hwh_name_cand):
                hwh_name = hwh_name_cand
        if hwh_name is None or not os.path.isfile(hwh_name):
            raise Exception(
                "Synthesis failed, no hwh found. Check logs under %s" % vivado_pynq_proj_dir
            )
        deploy_hwh_name = vivado_pynq_proj_dir + "/resizer.hwh"
        copy(hwh_name, deploy_hwh_name)
        model.set_metadata_prop("hw_handoff", deploy_hwh_name)
        # filename for the synth utilization report
        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        # filename for the post-route timing summary report
        timing_report_filename = vivado_pynq_proj_dir + "/timing_summary_routed.rpt"
        model.set_metadata_prop("vivado_timing_rpt", timing_report_filename)
        return (model, False)


class PYNQBuild(Transformation):
    """Best-effort attempt at building the accelerator for Pynq.
    It assumes the model has only fpgadataflow nodes. Kernels with double-pumped
    nodes derive their ap_clk2x from the PL clock by an internal clock generator
    (CreateStitchedIP clk2x_gen); the Versal shell rejects them.
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
        self.axi_port_width = pynq_native_port_width[platform]
        self.period_ns = period_ns
        self.platform = platform
        self.enable_debug = enable_debug
        self.partition_model_dir = partition_model_dir

    def apply(self, model):
        if is_versal(self.fpga_part):
            check_versal_shell_clk2x(model, self.platform)
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
                CreateStitchedIP(
                    self.fpga_part, self.period_ns, sdp_node.onnx_node.name, clk2x_gen=True
                )
            )
            kernel_model.set_metadata_prop("platform", "pynq-iodma")
            kernel_model.save(dataflow_model_filename)
        # Assemble design from IPs
        model = model.transform(
            MakePYNQProject(self.platform, self.period_ns, enable_debug=self.enable_debug)
        )

        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "pynq-iodma")

        return (model, False)
