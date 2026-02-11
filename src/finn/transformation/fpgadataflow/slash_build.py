# Copyright (c) 2020, Xilinx, Inc.
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

import subprocess
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
)

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
from finn.util.basic import make_build_dir


class SlashLink(Transformation):
    def __init__(self):
        super().__init__()

    def apply(self, model):
        # create a temporary folder for the project and check out SLASH
        link_dir = Path(make_build_dir(prefix="slash_link_proj_"))
        model.set_metadata_prop("slash_link_proj", str(link_dir))
        checkout_cmd = [
            "git",
            "clone",
            "-b",
            "feature/finn_support",
            "--recurse-submodules",
            "https://github.com/JOOpdenhoevel/SLASH.git",
            str(link_dir),
        ]
        subprocess.run(checkout_cmd, check=True)

        # generate config file for SLASH
        config = ["[connectivity]\n"]
        component_xml_paths = []
        idma_idx = 0
        odma_idx = 0
        mem_idx = 0
        instance_names = {}
        for node in model.graph.node:
            assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)

            vivado_proj_dir = Path(kernel_model.get_metadata_prop("vivado_stitch_proj"))
            component_xml_path = vivado_proj_dir / "ip" / "component.xml"
            assert component_xml_path.is_file(), f"Missing component.xml for {node.name}"
            component_xml_paths.append(component_xml_path)

            # gather info on connectivity
            # assume each node connected to outputs/inputs is DMA:
            # has axis, aximm and axilite
            # everything else is axis-only
            # assume only one connection from each ip to the next
            if len(node.input) == 0:
                producer = None
            else:
                producer = model.find_producer(node.input[0])
            consumer = model.find_consumers(node.output[0])

            # define kernel instances
            # name kernels connected to graph inputs as idmaxx
            # name kernels connected to graph inputs as odmaxx
            # TODO not a good way of checking for external in/out
            # check top-level in/out list instead
            if producer is None:
                instance_name = "idma" + str(idma_idx)
                idma_idx += 1
            elif consumer == []:
                instance_name = "odma" + str(odma_idx)
                odma_idx += 1
            else:
                instance_name = node.name
            instance_names[node.name] = instance_name
            config.append("nk=%s:1:%s\n" % (node.name, instance_name))
            sdp_node.set_nodeattr("instance_name", instance_name)

            if producer is None or consumer is None or consumer == []:
                node_mem_port = sdp_node.get_nodeattr("mem_port")
                if node_mem_port == "":
                    mem_type = "HBM"
                    node_mem_port = "%s[%d]" % (mem_type, mem_idx)

            # connect streams
            if producer is not None:
                for i in range(len(node.input)):
                    producer = model.find_producer(node.input[i])
                    if producer is None:
                        continue

                    j = list(producer.output).index(node.input[i])
                    config.append(
                        "stream_connect=%s.m_axis_%d:%s.s_axis_%d\n"
                        % (
                            instance_names[producer.name],
                            j,
                            instance_names[node.name],
                            i,
                        )
                    )

        # Write the configuration
        config_path = link_dir / "config.cfg"
        with open(config_path, "w") as f:
            f.writelines(config)

        # Write build script
        build_script_path = link_dir / "build.sh"
        with open(build_script_path, "w") as f:
            kernels = " ".join(str(path) for path in component_xml_paths)
            f.write(
                f"""
                #!/bin/bash
                cd {link_dir}/linker/resources/base/iprepo/hbm_bandwidth
                make
                cd {link_dir}/linker/resources/base/iprepo/traffic_producer
                make
                cd {link_dir}/linker/src
                python3 main.py --cfg {config_path} -p finn --kernels {kernels}
            """
            )

        # Run the build script
        log_path = link_dir / "slash.log"
        with open(log_path, "w") as log_file:
            subprocess.run(
                ["bash", build_script_path], check=True, stdout=log_file, stderr=log_file
            )

        vbin_path = link_dir / "linker" / "results" / "finn" / "finn_hw.vbin"
        assert (
            vbin_path.is_file()
        ), f"SLASH linking failed, no bitfile generated. Check {log_path} for details."
        model.set_metadata_prop("bitfile", str(vbin_path))

        report_path = link_dir / "linker" / "results" / "finn" / "report_utilization_finn.xml"
        assert (
            report_path.is_file()
        ), f"SLASH linking failed, no report generated. Check {log_path} for details."
        model.set_metadata_prop("slash_report", str(report_path))

        return (model, False)


class SlashBuild(Transformation):
    """Best-effort attempt at building the accelerator with Slash.
    It assumes the model has only fpgadataflow nodes

    :parameter fpga_part: string identifying the target FPGA
    :parameter period_ns: target clock period
    """

    def __init__(
        self,
        fpga_part,
        period_ns,
        enable_link=True,
    ):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.enable_link = enable_link

    def apply(self, model):
        # prepare at global level, then break up into kernels
        prep_transforms = [InsertIODMA(512), InsertDWC(), SpecializeLayers(self.fpga_part)]
        for trn in prep_transforms:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())

        model = model.transform(Floorplan())

        model = model.transform(CreateDataflowPartition())
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
            kernel_model = kernel_model.transform(RemoveUnusedTensors())
            kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
            kernel_model.save(dataflow_model_filename)
            kernel_model = kernel_model.transform(PrepareIP(self.fpga_part, self.period_ns))
            kernel_model = kernel_model.transform(HLSSynthIP())
            kernel_model = kernel_model.transform(
                CreateStitchedIP(self.fpga_part, self.period_ns, sdp_node.onnx_node.name, True)
            )
            # set platform attribute for correct remote execution
            model.set_metadata_prop("platform", "alveo")
            kernel_model.save(dataflow_model_filename)
        # Assemble design from kernels
        if self.enable_link:
            model = model.transform(SlashLink())
        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "alveo")

        return (model, False)
