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
import re
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import TypeAlias

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.util.basic import alveo_part_map, make_build_dir


class Interface(ABC):
    """Interface used to represent an Interface, that is to say a simple pin (SimpleWire) or
    interface pins (AXI4Lite, AXI4Stream).
    """

    Width: TypeAlias = int
    Signal: TypeAlias = Tuple[str, Width]

    name: str
    _sub_signals: Dict[str, Width]
    owner: Optional["Instantiation"]
    external: bool

    def __init__(self, name: str, external: bool):
        self.name = name
        self._sub_signals = {}
        self.connected = False
        self.owner = None
        self.external = external

    @abstractmethod
    def connect(self, design: "Design", other: "Interface"):
        pass

    def __getitem__(self, key):
        return self._sub_signals[key]

    def __str__(self):
        desc: List[str] = []
        desc.append("Interface name is: %s" % self.name)
        desc.append("The sub signales are:")
        for name, width in self._sub_signals.items():
            desc.append("\t%s : %d (in bits)" % (name, width))
        if self.owner is not None:
            desc.append("Owner is:")
            desc.append(self.owner.__str__())
        desc.append("Interface is%s external" % ("" if self.external else " not"))
        return "\n".join(desc)


class AXIInterface(Interface):
    """Represents an AXIInterface. Could be AXI4Lite or AXI4Stream.

    Specifies how two AXIInterfaces should be connected.
    """

    class Delimiter(Enum):
        UNDERSCORE = "_"
        POINT = "."

        def __str__(self):
            return str(self.value)

    def __init__(self, name: str, delimiter: Delimiter, external: bool):
        super().__init__(name=name, external=external)
        self.delimiter = delimiter

    def connect(self, design: "Design", other: "AXIInterface"):
        assert not other.connected
        assert not self.connected
        assert not self.external
        # NOTE: It is possible that two axi interfaces do not have the same number of signals if
        # they are AXI4Streams. Some have tkeep and tlast some do not.
        if len(self._sub_signals) != len(other._sub_signals):
            assert isinstance(self, AXI4Stream) and isinstance(other, AXI4Stream), (
                "Different number of signals should only happen in the case of AXI4Stream"
                " interfaces. This should not happen.\nSelf is: %s\nOther is: %s"
                % (self.__str__(), other.__str__())
            )

        dict_iterated_over = (
            self._sub_signals
            if len(self._sub_signals) < len(other._sub_signals)
            else other._sub_signals
        )
        for sub_signal, width in dict_iterated_over.items():
            our_interface = "%s%s%s" % (self.name, self.delimiter, sub_signal)
            other_interface = "%s%s%s" % (other.name, other.delimiter, sub_signal)
            assert self.owner is not None
            if other.external:
                assert other.owner is None
                self.owner.connections[our_interface] = other_interface

            else:
                assert other.owner is not None
                wire = SimpleWire(
                    name="%s_%s_wire_%s_%s_%s"
                    % (
                        self.owner.ip.module_name,
                        self.name,
                        other.owner.ip.module_name,
                        other.name,
                        sub_signal,
                    ),
                    width=width,
                )
                wire.connected = True
                # NOTE: Wires are put into the design as they are global, that is to say, seen by
                # different IPs
                design.wires.append(wire)
                self.owner.connections[our_interface] = wire.name
                other.owner.connections[other_interface] = wire.name

        other.connected = True
        self.connected = True


class AXI4Stream(AXIInterface):
    """Represents an AXI4Stream interface.

    Its particularity is the fact that the number of signals depending on whether tlast is
    supported or not.
    """

    @staticmethod
    def next_multiple_of_8(width: int):
        return ((width + 7) >> 3) << 3

    def __init__(
        self,
        name: str,
        data_width: int,
        tlast: bool,
        delimiter: AXIInterface.Delimiter,
        external: bool,
    ):
        super().__init__(name, delimiter, external)
        self.tlast = tlast
        width_adjusted = AXI4Stream.next_multiple_of_8(data_width)
        self._sub_signals = {"tdata": width_adjusted, "tvalid": 1, "tready": 1}
        if tlast:
            # NOTE: TLAST is always one bit and TKEEP is WIDTH(in bits)/8
            self._sub_signals["tlast"] = 1
            self._sub_signals["tkeep"] = width_adjusted >> 3

    def __str__(self):
        return "Interface type: AXI4Stream\n%s" % super().__str__()


class AXI4Lite(AXIInterface):
    """Represents an AXI4Lite interface."""

    def __init__(
        self, name: str, width: int, delimiter: AXIInterface.Delimiter, external: bool, addr_width
    ):
        super().__init__(name, delimiter, external)
        # TODO: Add arprot and awprot, three bits (recommended value is 0b000 so
        # can safely be ignored)
        self._sub_signals = {
            "awaddr": addr_width,
            "awvalid": 1,
            "awready": 1,
            "wdata": width,
            "wstrb": width >> 3,
            "wvalid": 1,
            "wready": 1,
            "bresp": 2,
            "bvalid": 1,
            "bready": 1,
            "araddr": addr_width,
            "arvalid": 1,
            "arready": 1,
            "rdata": width,
            "rresp": 2,
            "rready": 1,
            "rvalid": 1,
        }

    def __str__(self):
        return "Interface type: AXI4Lite\n%s" % super().__str__()


class SimpleWire(Interface):
    """Represents a simple wire, that is to say, an interface with only one pin."""

    width: Interface.Width

    def __init__(self, name: str, width: int):
        super().__init__(name, True)
        self.width = width

    def connect(self, design: "Design", interface: "SimpleWire"):
        assert not self.connected

        # NOTE: A wire is either generated and thus already connected, or belongs to the external
        # interface
        assert self.owner is not None

        self.owner.connections[self.name] = interface.name

    def __str__(self):
        return "Interface type: SimpleWire\n%s" % super().__str__()


class IP:
    """Represents an IP and all that is required to instantiate it."""

    IPConfig: TypeAlias = "dict[str, str]"
    interfaces: Dict[str, Interface]

    def __init__(
        self,
        vlnv: str,
        module_name: str,  # NOTE: Name of the configured/instantiable version of the IP
        interfaces: List[Interface],
        ip_repo_path: Optional[str] = None,  # NOTE: No repo path for Vivado/Xilinx IPs
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

    def __getitem__(self, key):
        return self.interfaces[key]

    @staticmethod
    def build_vlnv(vendor: str, library: str, name: str, version: str):
        return "%s:%s:%s:%s" % (vendor, library, name, version)


class BD:
    """Represents a block design.

    `interfaces` represents the list of interfaces available to the outside not all the interfaces
    that are in the block design

    `ips` list of ips that constitute the block design. This is optional because it is only
    required for block designs we want to generate ourselves. Not for block design we only
    want to "instantiate".

    `make_external` list of bd-specific tcl commands that make the necessary pins available
    to the outside
    """

    module_name: str
    interfaces: Dict[str, Interface]
    ips: Optional[List[IP]]
    make_external: Optional[List[str]]

    def __init__(
        self,
        bd_name: str,
        interfaces: List[Interface],
        ips: Optional[List[IP]],
        make_external: Optional[List[str]],
    ):
        self.module_name = bd_name
        self.interfaces = {}
        for interface in interfaces:
            self.interfaces[interface.name] = interface
        self.ips = ips
        self.make_external = make_external

    def create_ips(self) -> List[str]:
        """Reponsible for acutally creating the block design"""
        tcl: List[str] = []
        if self.ips is None:
            return tcl
        for ip in self.ips:
            tcl.append("startgroup")
            tcl.append("create_bd_cell -type ip -vlnv %s %s" % (ip.vlnv, ip.module_name))
            tcl.append("endgroup")
            if ip.config is not None:
                config_keys: List[str] = []
                for key in ip.config.keys():
                    config_keys.append("CONFIG.%s.VALUE_SRC" % key)
                tcl.append(
                    "set_property -dict [list %s] [get_bd_cells %s]"
                    % (" ".join(config_keys), ip.module_name)
                )
                tcl.append(
                    "set_property -dict [list %s] [get_bd_cells %s]"
                    % (GenerateCoyoteProject.generate_config(ip.config), ip.module_name)
                )
        return tcl

    def __getitem__(self, key):
        return self.interfaces[key]


class Instantiation:
    ip: Union[IP, BD]

    def __init__(self, instantiation_name: str, ip: Union[IP, BD]):
        self.instantiation_name = instantiation_name
        self.ip = ip
        self.connections = {}
        for interface in ip.interfaces.values():
            interface.owner = self

    def __getitem__(self, key):
        return self.ip[key]


class ExternalInterface:
    def __init__(self, interfaces: List[Interface]):
        self.interfaces = {}
        for interface in interfaces:
            self.interfaces[interface.name] = interface

    def __getitem__(self, key):
        return self.interfaces[key]


class Design:
    def __init__(
        self,
        instantiations: Dict[str, Instantiation],
        external_interface: ExternalInterface,
    ):
        self.instantiations = instantiations
        self.external_interface = external_interface
        self.wires: List[SimpleWire] = []
        self.ips = set()
        self.ip_repo_paths = set()
        # Sets of unique IPs and unique repo paths
        for name, instantiation in instantiations.items():
            self.ips.add(instantiation.ip)
            if isinstance(instantiation.ip, IP) and instantiation.ip.ip_repo_path is not None:
                self.ip_repo_paths.add(instantiation.ip.ip_repo_path)


class CreateStitchedIPForCoyote(Transformation):
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

    @staticmethod
    def generate_config(config: IP.IPConfig):
        config_tcl = []
        for key, value in config.items():
            config_tcl.append("CONFIG.%s {%s}" % (key, value))
        return " ".join(config_tcl)

    def create_ip(
        self,
        ip: Union[IP, BD],
    ):
        tcl = []
        path_to_file: str = ""
        if isinstance(ip, IP):
            path_to_file = "ip/%s/%s.xci" % (ip.module_name, ip.module_name)
            tcl.append("create_ip -vlnv %s -module_name %s" % (ip.vlnv, ip.module_name))
            if ip.config is not None:
                tcl.append(
                    "set_property -dict [list %s] [get_ips %s]"
                    % (self.generate_config(ip.config), ip.module_name)
                )
            tcl.append(
                "generate_target {instantiation_template} [get_files"
                " %s/lynx/lynx.srcs/sources_1/%s]" % (self.coyote_hw_build_dir, path_to_file)
            )

        else:
            tcl.append('create_bd_design "%s"' % ip.module_name)
            tcl.append("update_compile_order -fileset sources_1")
            tcl.extend(ip.create_ips())
            if ip.make_external is not None:
                tcl.extend(ip.make_external)
            tcl.append("save_bd_design")
            path_to_file = "bd/%s/%s.bd" % (ip.module_name, ip.module_name)

        tcl.append("update_compile_order -fileset sources_1")
        tcl.append(
            "generate_target all [get_files  %s/lynx/lynx.srcs/sources_1/%s]"
            % (self.coyote_hw_build_dir, path_to_file)
        )

        tcl.append("set list_ips [get_ips -all %s*]" % ip.module_name)
        tcl.append(r"foreach ip $list_ips {catch { config_ip_cache -export [get_ips -all $ip] } }")

        tcl.append(
            "export_ip_user_files -of_objects [get_files"
            " %s/lynx/lynx.srcs/sources_1/%s]"
            " -no_script -sync -force -quiet" % (self.coyote_hw_build_dir, path_to_file)
        )

        # NOTE: Run always needed for BD
        if isinstance(ip, BD) or ip.run_needed:
            tcl.append(
                r"foreach ip ${list_ips} { create_ip_run [get_files -of_objects [get_fileset"
                r" sources_1]"
                r" %s/lynx/lynx.srcs/sources_1/%s] }" % (self.coyote_hw_build_dir, path_to_file)
            )
            tcl.append(r'foreach ip ${list_ips} {launch_runs "${ip}_synth_1"}')

        tcl.append(
            "export_simulation -of_objects [get_files %s/lynx/lynx.srcs/sources_1/%s]"
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
                path_to_file,
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

        if isinstance(ip, BD):
            tcl.append("update_compile_order -fileset sources_1")
            tcl.append(
                " make_wrapper -files [get_files %s/lynx/lynx.srcs/sources_1/%s] -top"
                % (self.coyote_hw_build_dir, path_to_file)
            )
            tcl.append(
                "add_files -norecurse %s/lynx/lynx.gen/sources_1/bd/%s/hdl/%s_wrapper.v"
                % (self.coyote_hw_build_dir, ip.module_name, ip.module_name)
            )
            tcl.append("update_compile_order -fileset sources_1")

        return tcl

    def apply(self, model):
        model.set_metadata_prop("coyote_hw_build", self.coyote_hw_build_dir.__str__())
        # Clone Coyote git repo
        COYOTE_REPOSITORY = "https://github.com/fpgasystems/Coyote.git"

        git_clone_command = [
            "git",
            "clone",
            f"{COYOTE_REPOSITORY}",
            f"{self.coyote_repo_dir}",
        ]
        process_compile = subprocess.Popen(git_clone_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        assert os.path.isdir(self.coyote_repo_dir)
        model.set_metadata_prop("coyote_dir", self.coyote_repo_dir.__str__())

        # Create build directory
        self.coyote_hw_build_dir.mkdir(parents=True)

        part_to_board = {v: k for k, v in alveo_part_map.items()}
        board = part_to_board[self.fpga_part]

        # Change working directory to Coyote hw build dir
        finn_cwd = os.getcwd()
        os.chdir(self.coyote_hw_build_dir)

        # CMake Coyote
        coyote_board = board.lower()
        cmake_command = [
            "/usr/bin/cmake",
            f"{self.coyote_hw_dir}",
            f"-DFDEV_NAME={coyote_board}",
        ]
        process_compile = subprocess.Popen(cmake_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        make_shell_command = ["make", "shell"]
        process_compile = subprocess.Popen(make_shell_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        tcl = []
        tcl.append("open_project lynx/lynx.xpr")
        tcl.append("update_compile_order -fileset sources_1")

        tcl.append("set paths [get_property ip_repo_paths [current_project]];")

        ip_repo_paths = []
        for ip_repo_path in self.design.ip_repo_paths:
            ip_repo_paths.append(ip_repo_path)

        tcl.append("set * [get_property ip_repo_paths [current_project]]")
        # Add IP paths to Coyote
        tcl.append(
            'set_property  ip_repo_paths  "${*} %s" [current_project]' % " ".join(ip_repo_paths)
        )

        tcl.append("update_ip_catalog")

        for ip in self.design.ips:
            tcl.extend(self.create_ip(ip))

        tcl_string = "\n".join(tcl) + "\n"
        with open(self.coyote_hw_build_dir / "automate.tcl", "w") as f:
            f.write(tcl_string)

        vivado_cmd = ["vivado", "-mode", "batch", "-source", "automate.tcl"]
        process_vivado = subprocess.Popen(vivado_cmd, stdout=subprocess.PIPE)
        process_vivado.communicate()

        os.chdir(finn_cwd)

        return (model, False)


class CoyoteUserLogic(Transformation):
    def __init__(self, design: Design):
        super().__init__()
        self.design = design

    def apply(self, model):
        coyote_hw_build_dir = Path(model.get_metadata_prop("coyote_hw_build"))
        finn_cwd = os.getcwd()
        os.chdir(coyote_hw_build_dir)

        user_logic_file = coyote_hw_build_dir / "lynx" / "hdl" / "config_0" / "user_logic_c0_0.sv"

        lines = None
        with open(user_logic_file, "r") as f:
            lines = f.read().splitlines()

        assert lines is not None

        user_logic_idx = None
        for i, line in enumerate(lines):
            if "USER LOGIC" in line:
                user_logic_idx = i

        assert user_logic_idx is not None

        user_logic = lines[: user_logic_idx + 1]
        after_user_logic = lines[user_logic_idx + 1 :]

        verilog = []
        for wire in self.design.wires:
            verilog.append("wire [%d:0] %s;" % (wire.width - 1, wire.name))

        verilog.append("")

        for instantiation_name, instantiation in self.design.instantiations.items():
            verilog.append(
                "%s %s ("
                % (
                    (
                        instantiation.ip.module_name
                        + ("_wrapper" if isinstance(instantiation.ip, BD) else "")
                    ),
                    instantiation_name,
                )
            )
            for port, external_wire in instantiation.connections.items():
                verilog.append("\t.%s(%s)," % (port, external_wire))

            verilog[len(verilog) - 1] = verilog[len(verilog) - 1][:-1]
            verilog.append(");\n")

        user_logic.extend(verilog)
        user_logic.extend(after_user_logic)

        with open(user_logic_file, "w") as f:
            f.write("\n".join(user_logic) + "\n")

        os.chdir(finn_cwd)
        return (model, False)


class CoyoteCompile(Transformation):
    def __init__(self):
        super().__init__()

    def apply(self, model):
        coyote_hw_build_dir = Path(model.get_metadata_prop("coyote_hw_build"))
        finn_cwd = os.getcwd()
        os.chdir(coyote_hw_build_dir)

        make_shell_command = ["make", "compile"]
        process_compile = subprocess.Popen(make_shell_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        bitfile_coyote_dir = coyote_hw_build_dir / "bitstreams"
        assert os.path.isdir(bitfile_coyote_dir.__str__())

        model.set_metadata_prop("bitfile", bitfile_coyote_dir.__str__())

        os.chdir(finn_cwd)
        return (model, False)


class CoyoteBuild(Transformation):
    def __init__(self, fpga_part, period_ns, signature):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.signature = signature

    @staticmethod
    def create_converter(
        suffix: str, tlast: bool, input_width_bits: int, output_width_bits: int
    ) -> IP:
        return IP(
            vlnv=IP.build_vlnv(
                vendor="xilinx.com",
                library="ip",
                name="axis_dwidth_converter",
                version="1.1",
            ),
            module_name="axis_dwidth_converter_%s" % suffix,
            interfaces=[
                SimpleWire("aclk", 1),
                SimpleWire("aresetn", 1),
                # NOTE: Coyote interface is 512 bits wide
                AXI4Stream(
                    "s_axis", input_width_bits, tlast, AXIInterface.Delimiter.UNDERSCORE, False
                ),
                AXI4Stream(
                    "m_axis", output_width_bits, tlast, AXIInterface.Delimiter.UNDERSCORE, False
                ),
            ],
            ip_repo_path=None,
            config={
                "HAS_TLAST": "1",
                "HAS_TKEEP": "1",
                "M_TDATA_NUM_BYTES": "%d" % (output_width_bits >> 3),
                "S_TDATA_NUM_BYTES": "%d" % (input_width_bits >> 3),
            },
            run_needed=True,
        )

    @staticmethod
    def create_interconnect(
        vivado_stitch_proj: str, axilites: List[str]
    ) -> Tuple[IP, List[Tuple[str, int]], List[Interface]]:
        design_file = Path(vivado_stitch_proj) / "finn_design.v"

        # TODO: Max amount of master out per interconnect is 16.
        # Will have to chain them if we have more axilite.

        contents = None
        with open(design_file, "r") as f:
            contents = f.read()

        assert contents is not None

        axilites_with_addr_width: List[Tuple[str, int]] = []

        for axilite in axilites:
            match = re.search(r"input \[(\d+)\:0\]%s_awaddr" % axilite, contents)
            assert match
            addr_width = int(match.group(1))
            assert addr_width <= 32
            axilites_with_addr_width.append((axilite, addr_width))

        axilites_with_addr_width = sorted(
            axilites_with_addr_width, key=lambda tup: tup[1], reverse=True
        )

        addr_width_and_base: Dict[str, Tuple[int, int]] = {}
        prev_width = 0
        # NOTE: AXI4L address sent through the Coyote interface start at 0x12_0000
        prev_base = 0x12_0000
        for signal_name, addr_width in axilites_with_addr_width:
            assert addr_width <= 32
            addr_width_and_base[signal_name] = (
                addr_width,
                prev_base + ((1 << prev_width) if prev_width > 0 else 0),
            )
            prev_width = addr_width_and_base[signal_name][0]
            prev_base = addr_width_and_base[signal_name][1]

            # NOTE: Max we can go before we start modifying bits above
            # the split done by Coyote (0x12_0000)
            assert (prev_base + (1 << prev_width) - 1) <= 0x13_FFFF

        config: Dict[str, str] = {}
        config["ADDR_WIDTH"] = "32"
        config["DATA_WIDTH"] = "32"
        config["NUM_MI"] = str(len(axilites))
        config["PROTOCOL"] = "AXI4LITE"
        for i, axilite in enumerate(axilites):
            config["M%0.2d_A00_ADDR_WIDTH" % i] = str(addr_width_and_base[axilite][0])
            if addr_width_and_base[axilite][1] != 0:
                config["M%0.2d_A00_BASE_ADDR" % i] = "0x%0.16x" % addr_width_and_base[axilite][1]

        interfaces: List[Interface] = [
            AXI4Lite(
                name="S00_AXI_0",
                width=32,
                delimiter=AXIInterface.Delimiter.UNDERSCORE,
                external=False,
                addr_width=32,
            )
        ]

        for i, axilite in enumerate(axilites):
            interfaces.append(
                AXI4Lite(
                    name="M%0.2d_AXI_0" % i,
                    width=32,
                    delimiter=AXIInterface.Delimiter.UNDERSCORE,
                    external=False,
                    addr_width=32,
                )
            )

        interfaces.extend([SimpleWire("aclk_0", 1), SimpleWire("aresetn_0", 1)])

        return (
            IP(
                vlnv=IP.build_vlnv(
                    vendor="xilinx.com",
                    library="ip",
                    name="axi_crossbar",
                    version="2.1",
                ),
                module_name="axi_crossbar_0",
                interfaces=interfaces,
                ip_repo_path=None,
                config=config,
                run_needed=True,
            ),
            axilites_with_addr_width,
            interfaces,
        )

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
        model = model.transform(
            CreateStitchedIPForCoyote(
                fpga_part=self.fpga_part,
                period_ns=self.period_ns,
                signature=self.signature,
            )
        )

        is_accl_mode = json.loads(model.get_metadata_prop("accl_mode"))  # type: ignore

        intf_names = json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))  # type: ignore

        # NOTE: s_axi_control_%d in intf_names["axilite"] does not match what is in the finn
        # design file. The index is 0 and not 3.
        # TODO: Be a bit more reliable, there, we might have several of them.
        # Keep a counter and update the name accordingly.
        for i, axilite in enumerate(intf_names["axilite"]):
            if "control" in axilite:
                intf_names["axilite"][i] = "s_axi_control_0"
                break

        # Only one main output and one main input
        # NOTE: This will probably stay like this
        assert len(intf_names["s_axis"]) == 1, "Only support one toplevel input"
        assert len(intf_names["m_axis"]) == 1, "Only support one toplevel output"

        (
            interconnect,
            axilites,
            interconnect_interfaces,
        ) = CoyoteBuild.create_interconnect(
            model.get_metadata_prop("vivado_stitch_proj"), intf_names["axilite"]
        )

        interconnect_bd = BD(
            bd_name="design_crossbar",
            interfaces=interconnect_interfaces,
            ips=[interconnect],
            make_external=[
                "startgroup",
                "make_bd_pins_external  [get_bd_cells %s]" % interconnect.module_name,
                "make_bd_intf_pins_external  [get_bd_cells %s]" % interconnect.module_name,
                "endgroup",
            ],
        )

        finn_interfaces: List[Interface] = []

        for axilite, width in axilites:
            finn_interfaces.append(
                AXI4Lite(
                    name=axilite,
                    width=32,
                    delimiter=AXIInterface.Delimiter.UNDERSCORE,
                    external=False,
                    addr_width=width,
                )
            )

        finn_interfaces.extend(
            [
                SimpleWire(intf_names["clk"][0], 1),
                SimpleWire(intf_names["rst"][0], 1),
                # NOTE: Input of FINN does not have TLast
                AXI4Stream(
                    intf_names["s_axis"][0][0],
                    intf_names["s_axis"][0][1],
                    False,
                    AXIInterface.Delimiter.UNDERSCORE,
                    False,
                ),
                AXI4Stream(
                    intf_names["m_axis"][0][0],
                    intf_names["m_axis"][0][1],
                    not is_accl_mode,
                    AXIInterface.Delimiter.UNDERSCORE,
                    False,
                ),
            ]
        )

        finn_kernel_ip = IP(
            vlnv=model.get_metadata_prop("vivado_stitch_vlnv"),  # type: ignore
            module_name="finn_kernel_0",
            interfaces=finn_interfaces,
            ip_repo_path=model.get_metadata_prop("vivado_stitch_proj") + "/ip",  # type: ignore
        )

        axis_dwidth_convert_host_to_finn = CoyoteBuild.create_converter(
            "host_to_finn",
            not is_accl_mode,
            512,
            finn_kernel_ip[intf_names["s_axis"][0][0]]["tdata"],
        )

        axis_dwidth_convert_finn_to_host = CoyoteBuild.create_converter(
            "finn_to_host",
            not is_accl_mode,
            finn_kernel_ip[intf_names["m_axis"][0][0]]["tdata"],
            512,
        )

        coyote_interface = ExternalInterface(
            interfaces=[
                AXI4Lite(
                    name="axi_ctrl",
                    width=64,
                    delimiter=AXIInterface.Delimiter.POINT,
                    external=True,
                    addr_width=64,
                ),
                SimpleWire(name="aclk", width=1),
                SimpleWire(name="aresetn", width=1),
                AXI4Stream(
                    name="axis_host_0_sink",
                    data_width=512,
                    tlast=True,
                    delimiter=AXIInterface.Delimiter.POINT,
                    external=True,
                ),
                AXI4Stream(
                    name="axis_host_0_src",
                    data_width=512,
                    tlast=True,
                    delimiter=AXIInterface.Delimiter.POINT,
                    external=True,
                ),
            ]
        )

        instantiations = {}
        instantiations["finn_kernel_inst"] = Instantiation(
            instantiation_name="finn_kernel_inst", ip=finn_kernel_ip
        )
        instantiations["axis_dwidth_convert_host_to_finn_inst"] = Instantiation(
            instantiation_name="axis_dwidth_convert_host_to_finn_inst",
            ip=axis_dwidth_convert_host_to_finn,
        )

        instantiations["axis_dwidth_convert_finn_to_host_inst"] = Instantiation(
            instantiation_name="axis_dwidth_convert_finn_to_host_inst",
            ip=axis_dwidth_convert_finn_to_host,
        )

        instantiations["axi_crossbar_0_inst"] = Instantiation(
            instantiation_name="axi_crossbar_0_inst", ip=interconnect_bd
        )

        design = Design(instantiations, coyote_interface)

        instantiations["finn_kernel_inst"][intf_names["m_axis"][0][0]].connect(
            design, instantiations["axis_dwidth_convert_finn_to_host_inst"]["s_axis"]
        )

        instantiations["finn_kernel_inst"][intf_names["s_axis"][0][0]].connect(
            design,
            instantiations["axis_dwidth_convert_host_to_finn_inst"]["m_axis"],
        )

        instantiations["finn_kernel_inst"][intf_names["clk"][0]].connect(
            design, coyote_interface["aclk"]
        )

        instantiations["finn_kernel_inst"][intf_names["rst"][0]].connect(
            design, coyote_interface["aresetn"]
        )

        instantiations["axis_dwidth_convert_finn_to_host_inst"]["aclk"].connect(
            design, coyote_interface["aclk"]
        )

        instantiations["axis_dwidth_convert_finn_to_host_inst"]["aresetn"].connect(
            design, coyote_interface["aresetn"]
        )

        instantiations["axis_dwidth_convert_finn_to_host_inst"]["m_axis"].connect(
            design, coyote_interface["axis_host_0_src"]
        )

        instantiations["axis_dwidth_convert_host_to_finn_inst"]["aclk"].connect(
            design, coyote_interface["aclk"]
        )

        instantiations["axis_dwidth_convert_host_to_finn_inst"]["aresetn"].connect(
            design, coyote_interface["aresetn"]
        )

        instantiations["axis_dwidth_convert_host_to_finn_inst"]["s_axis"].connect(
            design, coyote_interface["axis_host_0_sink"]
        )

        instantiations["axi_crossbar_0_inst"]["S00_AXI_0"].connect(
            design, coyote_interface["axi_ctrl"]
        )

        for i, (axilite, width) in enumerate(axilites):
            instantiations["axi_crossbar_0_inst"]["M%0.2d_AXI_0" % i].connect(
                design,
                instantiations["finn_kernel_inst"][axilite],
                # type: ignore
            )

        instantiations["axi_crossbar_0_inst"]["aclk_0"].connect(
            design,
            coyote_interface["aclk"],
        )

        instantiations["axi_crossbar_0_inst"]["aresetn_0"].connect(
            design,
            coyote_interface["aresetn"],
        )

        model = model.transform(GenerateCoyoteProject(fpga_part=self.fpga_part, design=design))
        model = model.transform(CoyoteUserLogic(design=design))
        model = model.transform(CoyoteCompile())

        return (model, False)
