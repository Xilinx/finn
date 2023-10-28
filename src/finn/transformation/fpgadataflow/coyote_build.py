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
from pathlib import Path
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import Dict, Optional

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.util.basic import alveo_part_map, make_build_dir


class CreateStitchedIPForCoyote(Transformation):
    """Create a stitched IP configured for use with Coyote

    The stitched IP will be generated with vitis=True (i.e stitched_ip_gen_dcp=True
    from the config).
    Also, a TLastMarker is inserted as the last node.
    """

    def __init__(self, fpga_part, period_ns, signature):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.signature = signature

    def apply(self, model):
        # We want dynamic True so we leave default arguments
        # Also, we only want it at the output since the Coyote interface already provides tlast to
        # the input width converter
        print("Inserting tlast marker")
        model = model.transform(InsertTLastMarker())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(self.fpga_part, self.period_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(ReplaceVerilogRelPaths())
        # NOTE: Use CreateStitchedIP default IP name = "finn_design"
        print("Calling create stitched ip")
        model = model.transform(
            CreateStitchedIP(
                fpgapart=self.fpga_part,
                clk_ns=self.period_ns,
                vitis=True,
                signature=self.signature,
            )
        )

        return (model, False)


class GenerateCoyoteProject(Transformation):
    def __init__(self, fpga_part):
        super().__init__()
        self.fpga_part = fpga_part
        self.coyote_proj_dir = Path(make_build_dir(prefix="coyote_proj_"))
        self.coyote_repo_dir = self.coyote_proj_dir / "Coyote"
        self.coyote_hw_dir = self.coyote_repo_dir / "hw"
        self.coyote_hw_build_dir = self.coyote_hw_dir / "build"

    def generate_config(self, config: Dict[str, str]):
        config_tcl = []
        for key, value in config.items():
            config_tcl.append("CONFIG.%s {%s}" % (key, value))
        return " ".join(config_tcl)

    def create_ip(
        self,
        module_name: str,
        vlnv: Optional[str] = None,
        config: Optional[Dict[str, str]] = None,
        run_needed=False,
        name: Optional[str] = None,
        vendor: Optional[str] = None,
        version: Optional[str] = None,
        library: Optional[str] = None,
    ):
        assert vlnv is not None or (
            name is not None and vendor is not None and library is not None and version is not None
        )

        tcl = []
        if vlnv is not None:
            tcl.append("create_ip -vlnv %s -module_name %s" % (vlnv, module_name))
        else:
            tcl.append(
                "create_ip -name %s -vendor %s -library %s -version %s -module_name %s"
                % (name, vendor, library, version, module_name)
            )
        if config is not None:
            tcl.append(
                "set_property -dict [list %s] [get_ips %s]"
                % (self.generate_config(config), module_name)
            )

        tcl.append(
            "generate_target {instantiation_template} [get_files"
            " %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
            % (self.coyote_hw_build_dir, module_name, module_name)
        )

        tcl.append("update_compile_order -fileset sources_1")
        tcl.append(
            "generate_target all [get_files  %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
            % (self.coyote_hw_build_dir, module_name, module_name)
        )
        # NOTE: Only appears for width converter. Is this necessary?
        tcl.append("catch { config_ip_cache -export [get_ips -all %s] }" % module_name)

        tcl.append(
            "export_ip_user_files -of_objects [get_files"
            " %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
            " -no_script -sync -force -quiet" % (self.coyote_hw_build_dir, module_name, module_name)
        )

        if run_needed:
            tcl.append(
                "create_ip_run [get_files -of_objects [get_fileset sources_1]"
                " %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
                % (self.coyote_hw_build_dir, module_name, module_name)
            )
            tcl.append("launch_runs %s_synth_1 -jobs 4" % module_name)
        # NOTE: Seems involved. Is this required?
        tcl.append(
            "export_simulation -of_objects [get_files %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
            " -directory %s/lynx/lynx.ip_user_files/sim_scripts -ip_user_files_dir"
            " %s/lynx/lynx.ip_user_files -ipstatic_source_dir %s/lynx/lynx.ip_user_files/ipstatic"
            " -lib_map_path [list {modelsim=%s/lynx/lynx.cache/compile_simlib/modelsim}"
            " {questa=%s/lynx/lynx.cache/compile_simlib/questa}"
            " {xcelium=%s/lynx/lynx.cache/compile_simlib/xcelium}"
            " {vcs=%s/lynx/lynx.cache/compile_simlib/vcs}"
            " {riviera=%s/lynx/lynx.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force"
            " -quiet"
            % (
                self.coyote_hw_build_dir,
                module_name,
                module_name,
                self.coyote_hw_build_dir,
                self.coyote_hw_build_dir,
                self.coyote_hw_build_dir,
                self.coyote_hw_build_dir,
                self.coyote_hw_build_dir,
                self.coyote_hw_build_dir,
                self.coyote_hw_build_dir,
                self.coyote_hw_build_dir,
            )
        )

        return tcl

    def create_width_converter(self, module_name: str, config: Optional[Dict[str, str]]):
        return self.create_ip(
            module_name=module_name,
            vlnv=None,
            config=config,
            run_needed=True,
            name="axis_dwidth_converter",
            vendor="xilinx.com",
            version="1.1",
            library="ip",
        )

    def create_finn_kernel(self, vlnv: str):
        return self.create_ip(module_name="finn_design_0", vlnv=vlnv)

    def apply(self, model):
        # Clone Coyote git repo
        COYOTE_REPOSITORY = "https://github.com/fpgasystems/Coyote.git"

        git_clone_command = ["git", "clone", f"{COYOTE_REPOSITORY}", f"{self.coyote_repo_dir}"]
        process_compile = subprocess.Popen(git_clone_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        assert os.path.isdir(self.coyote_repo_dir)

        # Create build directory
        self.coyote_hw_build_dir.mkdir(parents=True)

        part_to_board = {v: k for k, v in alveo_part_map.items()}
        board = part_to_board[self.fpga_part]

        # Change working directory to Coyote hw build dir
        finn_cwd = os.getcwd()
        os.chdir(self.coyote_hw_build_dir)

        # CMake Coyote
        coyote_board = board.lower()
        cmake_command = ["/usr/bin/cmake", f"{self.coyote_hw_dir}", f"-DFDEV_NAME={coyote_board}"]
        process_compile = subprocess.Popen(cmake_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        make_shell_command = ["make", "shell"]
        process_compile = subprocess.Popen(make_shell_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        tcl = []
        tcl.append("open_project lynx/lynx.xpr")
        tcl.append("update_compile_order -fileset sources_1")

        # Add FINN kernel IP to Coyote
        tcl.append(
            "set_property  ip_repo_paths  {%s/ip} [current_project]"
            % model.get_metadata_prop("vivado_stitch_proj")
        )
        tcl.append("update_ip_catalog")

        # Generate IPs and their products
        vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
        tcl.extend(self.create_finn_kernel(vlnv))
        tcl.extend(
            self.create_width_converter(
                module_name="axis_dwidth_converter_host_to_finn",
                config={
                    "Component_Name": "axis_dwidth_converter_host_to_finn",
                    "HAS_TLAST": "1",
                    "M_TDATA_NUM_BYTES": "1",
                    "S_TDATA_NUM_BYTES": "64",
                },
            )
        )
        tcl.extend(
            self.create_width_converter(
                module_name="axis_dwidth_converter_finn_to_host",
                config={
                    "Component_Name": "axis_dwidth_converter_finn_to_host",
                    "HAS_TLAST": "1",
                    "M_TDATA_NUM_BYTES": "64",
                    "S_TDATA_NUM_BYTES": "1",
                },
            )
        )

        tcl_string = "\n".join(tcl) + "\n"
        with open(self.coyote_hw_build_dir / "automate.tcl", "w") as f:
            f.write(tcl_string)

        vivado_cmd = ["vivado", "-mode", "batch", "-source", "automate.tcl"]
        process_vivado = subprocess.Popen(vivado_cmd, stdout=subprocess.PIPE)
        process_vivado.communicate()

        # TODO: Write HDL to instantiate IPs

        os.chdir(finn_cwd)

        return (model, False)


class CoyoteUserLogic(Transformation):
    def __init__(self):
        super().__init__()

    def apply(self, model):
        # model.set_metadata_prop("vivado_stitch_ifnames", json.dumps(self.intf_names))
        return (model, False)


class CoyoteBuild(Transformation):
    def __init__(self, fpga_part, period_ns, signature):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.signature = signature

    def apply(self, model):
        # We want dynamic True so we leave default arguments
        # Also, we only want it at the output since the Coyote interface already provides tlast to
        # the input width converter
        model = model.transform(
            CreateStitchedIPForCoyote(
                fpga_part=self.fpga_part,
                period_ns=self.period_ns,
                signature=self.signature,
            )
        )

        model = model.transform(GenerateCoyoteProject(fpga_part=self.fpga_part))
        model = model.transform(CoyoteUserLogic())

        return (model, False)
