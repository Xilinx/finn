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


import json
import os
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import List, Optional, Tuple
from typing_extensions import TypeAlias

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.util.basic import alveo_part_map, make_build_dir

# IP definition

# Interfaces


class Interface(ABC):
    Signal: TypeAlias = str
    name: Signal
    _sub_signals: List[Tuple[Signal, int]]
    width: int
    owner: Optional["Instantiation"]

    def __init__(self, name: str, width: int):
        self.name = name
        self.width = width
        self._sub_signals = []
        self.connected = False

    @abstractmethod
    def connect(self, design: "Design", interface: "Interface", is_external=False):
        pass


class Delimiter(Enum):
    UNDERSCORE = 1
    POINT = 2


class AXIInterface(Interface):
    def __init__(self, name: str, width: int, delimiter: Delimiter):
        super().__init__(name=name, width=width)
        self.delimiter = delimiter

    def connect(self, design: "Design", interface: "AXIInterface", is_external=False):
        assert not interface.connected
        assert not self.connected
        if len(self._sub_signals) != len(interface._sub_signals):
            assert isinstance(self, "AXI4Stream") and isinstance(
                interface, "AXI4Stream"
            ), "This should not happen"

        # NOTE: This holds if the the potential additional signals are at the end of the list
        # NOTE: This is currently the case for AXI4Stream
        max_range = min(len(self._sub_signals), len(interface._sub_signals))
        for i in range(max_range):
            (sub_signal, width) = self._sub_signals[i]
            our_interface = "%s%s%s" % (self.name, self.delimiter, sub_signal)
            other_interface = "%s%s%s" % (interface.name, interface.delimiter, sub_signal)
            assert self.owner is not None
            if is_external:
                assert interface.owner is None
                self.owner.connections[our_interface] = other_interface
                self.connected = True
            else:
                assert interface.owner is not None
                wire = SimpleWire(
                    name="%s_wire_%s" % (our_interface, other_interface),
                    width=width,
                )
                wire.connected = True
                design.wires.append(wire)
                self.owner.connections[our_interface] = wire.name
                self.connected = True
                interface.owner.connections[other_interface] = wire.name
                interface.connected = True


class AXI4Stream(AXIInterface):
    def __init__(self, name: str, width: int, tlast: bool, delimiter: Delimiter):
        super().__init__(name, width, delimiter)
        self.tlast = tlast
        self._sub_signals = [("tdata", width), ("tvalid", 1), ("tready", 1)]
        if tlast:
            # TODO: Check if these can be larger than one depending on the number of bytes
            # TLAST is always one bit and TKEEP is WIDTH(in bits)/8
            self._sub_signals.append(("tlast", 1))
            self._sub_signals.append(("tkeep", width >> 3))


class AXI4Lite(AXIInterface):
    def __init__(self, name: str, width: int, delimiter: Delimiter):
        super().__init__(name, width, delimiter)
        # Infer address width from data width
        # bresp, rresp always two bits
        # One wstrb bit per byte on the data bus data width (in bits) / 2^3 (=8 bits)
        self._sub_signals = [
            ("awaddr", width.bit_length() - 1),
            ("awvalid", 1),
            ("awready", 1),
            ("wdata", width),
            ("wstrb", width >> 3),
            ("wvalid", 1),
            ("wready", 1),
            ("bresp", 2),
            ("bvalid", 1),
            ("bready", 1),
            ("araddr", width.bit_length() - 1),
            ("arvalid", 1),
            ("arready", 1),
            ("rdata", width),
            ("rresp", 2),
        ]


class SimpleWire(Interface):
    def __init__(self, name: str, width: int, generated=False):
        super().__init__(name, width)

    def connect(self, design: "Design", interface: "SimpleWire", is_external=False):
        assert not self.connected

        # NOTE: A wire is either generated and thus already connected, or belongs to the external
        # interface
        assert is_external
        assert self.owner is not None

        self.owner.connections[self.name] = interface.name


class IP:
    IPConfig: TypeAlias = "dict[str, str]"

    def __init__(
        self,
        vlnv: str,
        module_name: str,
        interfaces: List[Interface],
        ip_repo_path: Optional[str] = None,
        config: Optional[IPConfig] = None,
        run_needed=False,
    ):
        self.vlnv = vlnv
        self.ip_repo_path = ip_repo_path
        self.config = config
        self.run_needed = run_needed
        self.interfaces = {}
        self.module_name = module_name
        for interface in interfaces:
            self.interfaces[interface.name] = interface

    @staticmethod
    def build_vlnv(vendor: str, library: str, name: str, version: str):
        return "%s:%s:%s:%s" % (vendor, library, name, version)


class Instantiation:
    def __init__(self, instantiation_name: str, ip: IP):
        self.instantiation_name = instantiation_name
        self.ip = ip
        self.connections = {}
        for interface in ip.interfaces.values():
            interface.owner = self


class ExternalInterface:
    def __init__(self, interfaces: List[Interface]):
        self.interfaces = {}
        for interface in interfaces:
            self.interfaces[interface.name] = interface


class Design:
    def __init__(
        self,
        instantiations: List[Instantiation],
        external_interface: ExternalInterface,
    ):
        self.instantiations = instantiations
        self.external_interface = external_interface
        self.wires = []
        self.ips = set()
        self.ip_repo_paths = set()
        # Sets of unique IPs and unique repo paths
        for instantiation in instantiations:
            self.ips.add(instantiation.ip)
            if instantiation.ip.ip_repo_path is not None:
                self.ip_repo_paths.add(instantiation.ip.ip_repo_path)


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
        if not json.loads(model.get_metadata_prop("accl_mode")):
            model = model.transform(InsertTLastMarker())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(self.fpga_part, self.period_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(ReplaceVerilogRelPaths())
        # NOTE: Use CreateStitchedIP default IP name = "finn_design"
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
    def __init__(self, fpga_part, design: Design):
        super().__init__()
        self.fpga_part = fpga_part
        self.coyote_proj_dir = Path(make_build_dir(prefix="coyote_proj_"))
        self.coyote_repo_dir = self.coyote_proj_dir / "Coyote"
        self.coyote_hw_dir = self.coyote_repo_dir / "hw"
        self.coyote_hw_build_dir = self.coyote_hw_dir / "build"
        self.design = design

    def generate_config(self, config: IP.IPConfig):
        config_tcl = []
        for key, value in config.items():
            config_tcl.append("CONFIG.%s {%s}" % (key, value))
        return " ".join(config_tcl)

    @staticmethod
    def build_vlnv(vendor: str, library: str, name: str, version: str):
        return "%s:%s:%s:%s" % (vendor, library, name, version)

    def create_ip(
        self,
        ip: IP,
    ):
        tcl = []
        tcl.append("create_ip -vlnv %s -module_name %s" % (ip.vlnv, ip.module_name))
        if ip.config is not None:
            tcl.append(
                "set_property -dict [list %s] [get_ips %s]"
                % (self.generate_config(ip.config), ip.module_name)
            )

        tcl.append(
            "generate_target {instantiation_template} [get_files"
            " %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
            % (self.coyote_hw_build_dir, ip.module_name, ip.module_name)
        )

        tcl.append("update_compile_order -fileset sources_1")
        tcl.append(
            "generate_target all [get_files  %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
            % (self.coyote_hw_build_dir, ip.module_name, ip.module_name)
        )
        # NOTE: Only appears for width converter. Is this necessary?
        tcl.append("catch { config_ip_cache -export [get_ips -all %s] }" % ip.module_name)

        tcl.append(
            "export_ip_user_files -of_objects [get_files"
            " %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
            " -no_script -sync -force -quiet"
            % (self.coyote_hw_build_dir, ip.module_name, ip.module_name)
        )

        if ip.run_needed:
            tcl.append(
                "create_ip_run [get_files -of_objects [get_fileset sources_1]"
                " %s/lynx/lynx.srcs/sources_1/ip/%s/%s.xci]"
                % (self.coyote_hw_build_dir, ip.module_name, ip.module_name)
            )
            tcl.append("launch_runs %s_synth_1 -jobs 4" % ip.module_name)
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
                ip.module_name,
                ip.module_name,
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

        for ip_repo_path in self.design.ip_repo_paths:
            # Add IP paths to Coyote
            tcl.append("set_property  ip_repo_paths  {%s} [current_project]" % ip_repo_path)
        tcl.append("update_ip_catalog")

        for ip in self.design.ips:
            tcl.extend(self.create_ip(ip))

        # Generate IPs and their products
        # vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
        # tcl.extend(self.create_finn_kernel(vlnv))
        """
        if model.get_metadata_prop("accl_mode"):
            # TODO: Add ACCL ip repo
            # TODO: Instantiate ACCL IP
            raise NotImplementedError("ACCL mode not supported yet")
        else:
            tcl.extend(
                self.create_width_converter(
                    module_name="axis_dwidth_converter_host_to_finn",
                    config={
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
                        "HAS_TLAST": "1",
                        "M_TDATA_NUM_BYTES": "64",
                        "S_TDATA_NUM_BYTES": "1",
                    },
                )
            )
        """
        tcl_string = "\n".join(tcl) + "\n"
        with open(self.coyote_hw_build_dir / "automate.tcl", "w") as f:
            f.write(tcl_string)

        vivado_cmd = ["vivado", "-mode", "batch", "-source", "automate.tcl"]
        process_vivado = subprocess.Popen(vivado_cmd, stdout=subprocess.PIPE)
        process_vivado.communicate()

        os.chdir(finn_cwd)

        return (model, False)


class CoyoteUserLogic(Transformation):
    def __init__(self, intf_names):
        super().__init__()
        self.intf_names = intf_names

    def apply(self, model):
        return (model, False)


class CoyoteBuild(Transformation):
    def __init__(self, fpga_part, period_ns, signature):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.signature = signature

    @staticmethod
    def __is_accl_mode(model: ModelWrapper) -> bool:
        # TODO: Maybe find something a bit cleaner
        for node in model.graph.node:
            if "ACCL" in node.name or "cclo" in node.name:
                return True

        return False

    def apply(self, model):
        # We want dynamic True so we leave default arguments
        # Also, we only want it at the output since the Coyote interface already provides tlast to
        # the input width converter

        model.set_metadata_prop("accl_mode", json.dumps(CoyoteBuild.__is_accl_mode(model)))
        is_accl_mode = json.loads(model.get_metadata_prop("accl_mode"))

        model = ModelWrapper(
            "/home/antoine/Documents/coyote_finn/cybersecurity/outputs_u250_automate/"
            "intermediate_models/step_create_stitched_ip.onnx"
        )

        # model = model.transform(
        #     CreateStitchedIPForCoyote(
        #         fpga_part=self.fpga_part,
        #         period_ns=self.period_ns,
        #         signature=self.signature,
        #     )
        # )

        intf_names = json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))  # type: ignore

        # Only one main output and one main input
        # NOTE: This will probably stay like this
        assert len(intf_names["s_axis"]) == 1, "Only support one toplevel input"
        assert len(intf_names["m_axis"]) == 1, "Only support one toplevel output"

        # TODO: Support multiple AXI lites using interconnect
        assert len(intf_names["axilite"]) == 1, "Only support one AXI lite for now"

        # TODO: Create helpers to build IP
        finn_kernel_ip = IP(
            vlnv=model.get_metadata_prop("vivado_stitch_vlnv"),  # type: ignore
            module_name="finn_kernel_0",
            interfaces=[
                SimpleWire(intf_names["clk"][0], 1),
                SimpleWire(intf_names["rst"][0], 1),
                # NOTE: Input of FINN does not have TLast
                AXI4Stream(
                    intf_names["s_axis"][0][0],
                    intf_names["s_axis"][0][1],
                    False,
                    Delimiter.UNDERSCORE,
                ),
                AXI4Stream(
                    intf_names["m_axis"][0][0],
                    intf_names["m_axis"][0][1],
                    not is_accl_mode,
                    Delimiter.UNDERSCORE,
                ),
                AXI4Lite(
                    intf_names["axilite"][0],
                    # TODO: Figure out if this can be dynamic
                    32,
                    Delimiter.UNDERSCORE,
                ),
            ],
            ip_repo_path=model.get_metadata_prop("vivado_stitch_proj") + "/ip",  # type: ignore
        )

        axis_dwidth_convert_host_to_finn = IP(
            vlnv=IP.build_vlnv(
                vendor="xilinx.com", library="ip", name="axis_dwidth_converter", version="1.1"
            ),
            module_name="axis_dwidth_converter_host_to_finn",
            interfaces=[
                SimpleWire("aclk", 1),
                SimpleWire("aresetn", 1),
                AXI4Stream("s_axis", 512, not is_accl_mode, Delimiter.UNDERSCORE),
                AXI4Stream("m_axis", 8, not is_accl_mode, Delimiter.UNDERSCORE),
            ],
            ip_repo_path=None,
            config={
                "HAS_TLAST": "1",
                "M_TDATA_NUM_BYTES": "1",
                "S_TDATA_NUM_BYTES": "64",
            },
            run_needed=True,
        )

        axis_dwidth_convert_finn_to_host = IP(
            vlnv=IP.build_vlnv(
                vendor="xilinx.com", library="ip", name="axis_dwidth_converter", version="1.1"
            ),
            module_name="axis_dwidth_convert_finn_to_host",
            interfaces=[
                SimpleWire("aclk", 1),
                SimpleWire("aresetn", 1),
                AXI4Stream("s_axis", 8, not is_accl_mode, Delimiter.UNDERSCORE),
                AXI4Stream("m_axis", 512, not is_accl_mode, Delimiter.UNDERSCORE),
            ],
            ip_repo_path=None,
            config={
                "HAS_TLAST": "1",
                "M_TDATA_NUM_BYTES": "64",
                "S_TDATA_NUM_BYTES": "1",
            },
            run_needed=True,
        )

        coyote_interface = ExternalInterface(
            interfaces=[
                AXI4Lite(name="axi_ctrl", width=64, delimiter=Delimiter.POINT),
                SimpleWire(name="aclk", width=1),
                SimpleWire(name="aresetn", width=1),
                AXI4Stream(
                    name="axis_host_0_sink", width=512, tlast=True, delimiter=Delimiter.POINT
                ),
                AXI4Stream(
                    name="axis_host_0_src", width=512, tlast=True, delimiter=Delimiter.POINT
                ),
            ]
        )

        instantiations = []
        instantiations.append(
            Instantiation(instantiation_name="finn_kernel_inst", ip=finn_kernel_ip)
        )
        instantiations.append(
            Instantiation(
                instantiation_name="axis_dwidth_convert_host_to_finn_inst",
                ip=axis_dwidth_convert_host_to_finn,
            )
        )
        instantiations.append(
            Instantiation(
                instantiation_name="axis_dwidth_convert_finn_to_host_inst",
                ip=axis_dwidth_convert_finn_to_host,
            )
        )

        design = Design(instantiations, coyote_interface)
        # intf_names = json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))
        model = model.transform(GenerateCoyoteProject(fpga_part=self.fpga_part, design=design))
        # model = model.transform(CoyoteUserLogic())
        return (model, False)
