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

import functools
import json
import operator
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
        desc.append("The sub signals are:")
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



    `ips` list of ips that constitute the block design. This is optional because it is only
    required for block designs we want to generate ourselves. Not for block design we only
    want to "instantiate".

    `intra_connections` is a list of tcl commands that establishes the intra block design
    connections

    `interfaces` represents the list of interfaces available to the outside not all the interfaces
    that are in the block design. Generated from the given list of ips.

    `make_external` list of bd-specific tcl commands that make the necessary pins available
    to the outside
    """

    module_name: str
    ips: Optional[List[IP]]
    intra_connections: List[str]
    interfaces: Dict[str, Interface]
    make_external: List[str]

    def __init__(
        self,
        bd_name: str,
        ips: Optional[List[IP]],
        intra_connections: List[str],
        extra_external_commands: Optional[List[str]],
    ):
        self.module_name = bd_name
        self.ips = ips

        self.intra_connections = intra_connections
        self.interfaces = {}
        if ips is not None:
            tcl: List[str] = []
            get_bd_cells: List[str] = []
            tcl.append("startgroup")
            for ip in ips:
                get_bd_cells.append("[get_bd_cells %s]" % ip.module_name)
                self.interfaces.update(ip.interfaces)

            tcl.append("make_bd_pins_external  %s" % " ".join(get_bd_cells))
            tcl.append("make_bd_intf_pins_external  %s" % " ".join(get_bd_cells))
            tcl.append("endgroup")
            if extra_external_commands is not None:
                tcl.extend(extra_external_commands)
            self.make_external = tcl

    def create_ips(self) -> List[str]:
        """Responsible for actually creating the block design"""
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
    """Represents the Verilog instantiation of an IP or a block design (in this case the wrapper of
    the block design)
    """

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
    """Represents the interface that is given to use in the verilog file we are trying to generate.
    In the Coyote case, it is the interface Coyote gives us inside the user logic file
    """

    def __init__(self, interfaces: List[Interface]):
        self.interfaces = {}
        for interface in interfaces:
            self.interfaces[interface.name] = interface

    def __getitem__(self, key):
        return self.interfaces[key]


class Design:
    """Represents the whole user logic file. Contains all the instantiations plus all the wires
    required to make them."""

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
        for name, instantiation in instantiations.items():
            self.ips.add(instantiation.ip)
            if isinstance(instantiation.ip, IP) and instantiation.ip.ip_repo_path is not None:
                self.ip_repo_paths.add(instantiation.ip.ip_repo_path)


class CreateStitchedIPForCoyote(Transformation):
    """
    Adds a TLastMarker node at the end of the graph and creates a stitched IP from the new graph.

    Outcome if successful: the vivado_stitch_proj attribute is set to the associated Vivado stitched
    IP project.
    """

    def __init__(self, fpga_part, period_ns, signature):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.signature = signature

    def apply(self, model):
        # NOTE: We want dynamic True so we leave default arguments
        # NOTE: We only want the TLastMarker node at the output since the Coyote interface already
        # provides tlast to the input width converter
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
    """Generate a Coyote Vivado project with the required configuration to make the instantiation
    of the FINN kernel possible. It will add the required IP repo paths, generate the required IPs
    and block designs.

    Outcome if successful: sets the coyote_hw_build attribute to the Coyote shell project path. The
    project will have the proper IP paths, IPs, output products and block designs setup.
    """

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
            tcl.extend(ip.intra_connections)
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
        # Clone Coyote git repo
        COYOTE_REPOSITORY = "https://github.com/fpgasystems/Coyote.git"

        git_clone_command = [
            "git",
            "clone",
            f"{COYOTE_REPOSITORY}",
            f"{self.coyote_repo_dir}",
        ]
        # TODO: Checkout a specific commit to ensure compatibility
        process_git_clone = subprocess.Popen(git_clone_command, stdout=subprocess.PIPE)
        process_git_clone.communicate()
        assert process_git_clone.returncode == 0, (
            "Failed to clone Coyote repo at address: %s" % COYOTE_REPOSITORY
        )
        assert os.path.isdir(self.coyote_repo_dir), (
            "Unable to find cloned directory: %s" % self.coyote_repo_dir
        )
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
        process_cmake = subprocess.Popen(cmake_command, stdout=subprocess.PIPE)
        process_cmake.communicate()
        assert (
            process_cmake.returncode == 0
        ), "Failed to generate CMake configuration for coyote,cmake command is: %s" % " ".join(
            cmake_command
        )

        make_shell_command = ["make", "shell"]
        process_shell = subprocess.Popen(make_shell_command, stdout=subprocess.PIPE)
        process_shell.communicate()
        assert (
            process_shell.returncode == 0
        ), "Failed to generate Coyote shell. Command is: %s" % " ".join(make_shell_command)
        assert Path("lynx/lynx.xpr").is_file()

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
        assert (
            process_vivado.returncode == 0
        ), "Failed to run project automation in Vivado,command is: %s" % " ".join(vivado_cmd)

        os.chdir(finn_cwd)

        model.set_metadata_prop("coyote_hw_build", self.coyote_hw_build_dir.__str__())

        return (model, False)


class CoyoteUserLogic(Transformation):
    """Generates the user_logic file for the Coyote shell containing the FINN kernel and all that
    is required to make it work.

    Outcome if successful: Modifies the user_logic verilog file provided by Coyote to contain the
    FINN kernel and all that is required to make it work inside with the shell.
    """

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
    """Asks Coyote to compile the project.

    Outcome if successful: A bitstream should be generated and the bitfile attribute should be set
    to point to the bitstream files.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        coyote_hw_build_dir = Path(model.get_metadata_prop("coyote_hw_build"))
        finn_cwd = os.getcwd()
        os.chdir(coyote_hw_build_dir)

        make_compile_command = ["make", "compile"]
        process_compile = subprocess.Popen(make_compile_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        assert process_compile.returncode == 0, "Failed to compile Coyote shell"

        bitfile_coyote_dir = coyote_hw_build_dir / "bitstreams"
        assert os.path.isdir(bitfile_coyote_dir.__str__())

        model.set_metadata_prop("bitfile", bitfile_coyote_dir.__str__())

        os.chdir(finn_cwd)
        return (model, False)


class CoyoteBuild(Transformation):
    """Generates a Coyote shell containing the accelerator.

    :parameter fpga_part: string identifying the target FPGA
    :parameter period_ns: target clock period

    """

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
    def next_power_of_2(x: int):
        return 1 << (x - 1).bit_length()

    @staticmethod
    def get_axilites_with_width(
        vivado_stitch_proj: str, axilites: List[str]
    ) -> List[Interface.Signal]:
        """The function reads the acutal axilites addresses width directly from the finn_design.v
        file generated by the CreateStitchedIPStep.

        Parameters:
        vivado_stitch_proj (str): Path to the Vivado stitched IP project
        axilites (List[str]): Names of the axilites interfaces

        Returns:
        List[Interface.Signal]: List of axilites interfaces names associated with their address
        width

        """
        design_file = Path(vivado_stitch_proj) / "finn_design.v"

        contents = None
        with open(design_file, "r") as f:
            contents = f.read()

        assert contents is not None

        axilites_with_addr_width: List[Tuple[str, int]] = []
        for axilite in axilites:
            match = re.search(r"input \[(\d+)\:0\]%s_awaddr" % axilite, contents)
            assert match
            addr_width = int(match.group(1)) + 1
            assert addr_width <= 32
            axilites_with_addr_width.append((axilite, addr_width))

        axilites_with_addr_width = sorted(
            axilites_with_addr_width, key=lambda tup: tup[1], reverse=True
        )

        return axilites_with_addr_width

    @staticmethod
    def create_interconnects(
        axilites_outer: List[Interface.Signal],
    ) -> Tuple[List[IP], Dict[str, str], List[str], List[str]]:
        """This function creates one or more interconnects to connect the Coyote AXI4Lite interface
        to the potentially multiple accelerator AX4List interfaces. In case the accelerator
        requires more than 16 AXI4Lite interfaces, multiple interconnects, chained together, are
        generated by this function.

        Parameters:
        axilites_outer (List[Interface.Signal]): List of axilite interfaces with their name and
        their address width

        Returns:
        Tuple[List[IP], Dict[str, str], List[str], List[str]]:
            - List[IP]: the interconnects IPs
            - Dict[str, str]: the interconnects connections to the axilites signals
            - List[str]: the connections between the interconnects, as TCL commands
            - List[str]: the specific TCL commands to make the appropriate pins external

        """

        def collapse_axilites_to_fit(
            axilites_inner: List[Interface.Signal],
        ) -> Tuple[List[Tuple[List[Interface.Signal], int]], Optional[List[Interface.Signal]]]:
            # NOTE: This represents the axilites connections as a list of list because everything
            # after and including the 15th elements of axilites_inner is collapsed into the 15th
            # element. This allows to treat the axilites that will be connected to the other
            # interconnects just like any other axi lite, except we will use the total address
            # width of the connected interfaces to generate the proper address map.
            axilites_list_list: List[Tuple[List[Interface.Signal], int]] = [
                ([(name, width)], width) for name, width in axilites_inner
            ]

            MAX_AMOUNT_OF_AXILITES = 16
            assert (
                MAX_AMOUNT_OF_AXILITES > 1
            ), "We need at least 2 outputs per interconnect for chaining"

            remaining_axilites = None
            if len(axilites_list_list) > MAX_AMOUNT_OF_AXILITES:
                # NOTE: This is what is responsible for collapsing everything after the 15th element
                axilites_list_list[MAX_AMOUNT_OF_AXILITES - 1] = (
                    [
                        item
                        for sublist in axilites_list_list[MAX_AMOUNT_OF_AXILITES - 1 :]
                        for item in sublist[0]
                    ],
                    # NOTE: This corresponds to how many address bits we will need for the
                    # remaining axilites
                    functools.reduce(
                        operator.add,
                        map(
                            lambda width: 1 << width,
                            [elem[1] for elem in axilites_list_list[MAX_AMOUNT_OF_AXILITES - 1 :]],
                        ),
                    ).bit_length(),
                )
                # NOTE: This is what is "left over" after we finish connecting the current
                # interconnect
                remaining_axilites = axilites_list_list[MAX_AMOUNT_OF_AXILITES - 1][0]
                # NOTE: Once everything is collapsed, we cut after the 15th element
                axilites_list_list = axilites_list_list[:MAX_AMOUNT_OF_AXILITES]

            axilites_list_list = sorted(axilites_list_list, key=lambda tup: tup[1], reverse=True)
            # NOTE: After the cut, if one was needed, axilites_list_list should not contain more
            # than 16 elements
            assert len(axilites_list_list) <= MAX_AMOUNT_OF_AXILITES

            return (axilites_list_list, remaining_axilites)

        def configure_and_connect(
            axilites_list_list: List[Tuple[List[Interface.Signal], int]],
            current_idx: int,
            indices_count: Dict[int, int],
        ) -> Tuple[Dict[str, str], int, Dict[str, str]]:
            config: Dict[str, str] = {}
            config["ADDR_WIDTH"] = "32"
            config["DATA_WIDTH"] = "32"
            config["NUM_MI"] = str(len(axilites_list_list))
            config["PROTOCOL"] = "AXI4LITE"
            prev_width = 0
            # NOTE: AXI4L address sent through the Coyote interface start at 0x12_0000
            prev_base = 0x12_0000 if current_idx == 0 else 0

            chain_interconnect_idx = 0
            axilites_connections: Dict[str, str] = {}
            for i, (axilites_with_width, total_width) in enumerate(axilites_list_list):
                assert total_width <= 32

                (signal_name, addr_width) = (
                    axilites_with_width[0]
                    if len(axilites_with_width) == 1
                    else ("interconnect_chain", total_width)
                )
                config["M%0.2d_A00_ADDR_WIDTH" % i] = str(addr_width)
                if prev_width == 0:
                    config["M%0.2d_A00_BASE_ADDR" % i] = "0x%0.16x" % prev_base
                else:
                    prev_base = prev_base + (1 << prev_width)
                    config["M%0.2d_A00_BASE_ADDR" % i] = "0x%0.16x" % prev_base

                if len(axilites_with_width) == 1:
                    axilites_connections[
                        "M%0.2d_AXI_%d" % (i, indices_count.get(i, 0))
                    ] = signal_name
                    indices_count[i] = indices_count.get(i, 0) + 1
                else:
                    chain_interconnect_idx = i
                prev_width = addr_width

                # NOTE: Max we can go before we start modifying bits above
                # the split done by Coyote (0x12_0000)
                assert (
                    prev_base + (1 << prev_width) - 1
                ) <= 0x13_FFFF, "The axi lites interfaces cover too big of a range."

            return (axilites_connections, chain_interconnect_idx, config)

        def create_interconnects_inner(
            axilites_inner: List[Interface.Signal], current_idx: int, indices_count: Dict[int, int]
        ) -> Tuple[List[IP], Dict[str, str], List[str]]:
            """This function is responsible for actually creating the interconnects. It should be
            called recurisevly with a smaller list axilites signals everytime.

            Parameters:
            axilites_inner (List[Interface.Signal]): List of remaining axilite interfaces with
            their name and their address width.
            current_idx (int): index of the currently generated interconnect. Useful for generating
            interfaces and IPs names.
            indices_count (Dict[int, int]): When generating multiple interconnects, it is possible
            that several master connections with the same index gets exposed to the outside. For
            instance, it is possible that the M01_AXI interface of interconnect 0 and the M01_AXI
            of interconnect 1 get exposed to the outside. In that case, the exposed interface for
            interconnect 0, will be named M01_AXI_0 and the exposed interface of interconnect 1
            will be named M01_AXI_1. `indices_count` allows to keep track of how many master
            interfaces that are going to get exposed have the same inner index.

            Returns:
            Tuple[List[IP], Dict[str, str], List[str], List[str]]:
                - List[IP]: the interconnects IPs
                - Dict[str, str]: the interconnects connections to the axilites signals
                - List[str]: the connections between the interconnects, as TCL commands

            """

            (axilites_list_list, remaining_axilites) = collapse_axilites_to_fit(axilites_inner)
            old_indices_count = indices_count.copy()

            ips: List[IP] = []
            intra_connections_tcl: List[str] = []

            (axilites_connections, chain_interconnect_idx, config) = configure_and_connect(
                axilites_list_list, current_idx, indices_count
            )

            if remaining_axilites is not None:
                assert remaining_axilites is not None
                (
                    intra_ips,
                    intra_axilites_connections,
                    intra_intra_connections_tcl,
                ) = create_interconnects_inner(
                    remaining_axilites, current_idx=current_idx + 1, indices_count=indices_count
                )
                ips.extend(intra_ips)
                axilites_connections.update(intra_axilites_connections)
                intra_connections_tcl.extend(intra_intra_connections_tcl)
                ip_name = "axi_crossbar_%d" % current_idx
                intra_connections_tcl.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/M%0.2d_AXI]"
                    " [get_bd_intf_pins axi_crossbar_%d/S00_AXI]"
                    % (ip_name, chain_interconnect_idx, current_idx + 1)
                )

            interfaces: List[Interface] = []
            if current_idx == 0:
                interfaces.append(
                    AXI4Lite(
                        name="S00_AXI_0",
                        width=32,
                        delimiter=AXIInterface.Delimiter.UNDERSCORE,
                        external=False,
                        addr_width=32,
                    )
                )
                interfaces.extend([SimpleWire("aclk_0", 1), SimpleWire("aresetn_0", 1)])

            for i in range(len(axilites_list_list)):
                if remaining_axilites is None or i != chain_interconnect_idx:
                    interfaces.append(
                        AXI4Lite(
                            name="M%0.2d_AXI_%d" % (i, old_indices_count.get(i, 0)),
                            width=32,
                            delimiter=AXIInterface.Delimiter.UNDERSCORE,
                            external=False,
                            addr_width=32,
                        )
                    )
            # Prepend this IP
            ips[:0] = [
                IP(
                    vlnv=IP.build_vlnv(
                        vendor="xilinx.com",
                        library="ip",
                        name="axi_crossbar",
                        version="2.1",
                    ),
                    module_name="axi_crossbar_%d" % current_idx,
                    interfaces=interfaces,
                    ip_repo_path=None,
                    config=config,
                    run_needed=True,
                )
            ]
            return (
                ips,
                axilites_connections,
                intra_connections_tcl,
            )

        (
            interconnects,
            interconnects_axilites_connections,
            intra_connections,
        ) = create_interconnects_inner(axilites_outer, 0, {})

        extra_external_commands: List[str] = []
        if len(interconnects) > 1:
            extra_external_commands.append("startgroup")
            get_bd_nets: List[str] = []
            get_bd_ports: List[str] = []
            connections_to_external: List[str] = []
            for i in range(1, len(interconnects)):
                get_bd_nets.append("[get_bd_nets aclk_%d_1]" % i)
                get_bd_nets.append("[get_bd_nets aresetn_%d_1]" % i)
                get_bd_ports.append("[get_bd_ports aclk_%d]" % i)
                get_bd_ports.append("[get_bd_ports aresetn_%d]" % i)
                connections_to_external.append(
                    "connect_bd_net [get_bd_ports aclk_0] [get_bd_pins axi_crossbar_%d/aclk]" % i
                )
                connections_to_external.append(
                    "connect_bd_net [get_bd_ports aresetn_0] [get_bd_pins axi_crossbar_%d/aresetn]"
                    % i
                )

            # NOTE: Necessary for clock and reset inputs. Did not find a cleaner way to do this
            extra_external_commands.append("delete_bd_objs %s" % " ".join(get_bd_nets))
            extra_external_commands.append("delete_bd_objs %s" % " ".join(get_bd_ports))
            extra_external_commands.append("endgroup")
            extra_external_commands.append("startgroup")
            extra_external_commands.extend(connections_to_external)
            extra_external_commands.append("endgroup")

        return (
            interconnects,
            interconnects_axilites_connections,
            intra_connections,
            extra_external_commands,
        )

    @staticmethod
    def __is_accl_mode(model: ModelWrapper) -> bool:
        # TODO: Maybe find something a bit cleaner
        for node in model.graph.node:
            if "ACCL" in node.name or "cclo" in node.name:
                return True

        return False

    @staticmethod
    def __update_intf_axilite_control(intf_names_axilite: List[str]):
        s_axi_control_counter = 0
        for i, axilite in enumerate(intf_names_axilite):
            if "control" in axilite:
                intf_names_axilite[i] = "s_axi_control_%d" % s_axi_control_counter
                s_axi_control_counter = s_axi_control_counter + 1

    def apply(self, model):
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

        CoyoteBuild.__update_intf_axilite_control(intf_names["axilite"])

        # Only one main output and one main input
        # NOTE: This will change for ACCL mode where we have several inputs and outputs
        assert len(intf_names["s_axis"]) == 1, "Only support one toplevel input"
        assert len(intf_names["m_axis"]) == 1, "Only support one toplevel output"

        axilites_with_addr_width = CoyoteBuild.get_axilites_with_width(
            model.get_metadata_prop("vivado_stitch_proj"), intf_names["axilite"]
        )

        (
            interconnects,
            interconnects_connections,
            intra_connections,
            extra_external_commands,
        ) = CoyoteBuild.create_interconnects(axilites_with_addr_width)

        interconnect_bd = BD(
            bd_name="design_crossbar",
            ips=interconnects,
            intra_connections=intra_connections,
            extra_external_commands=extra_external_commands,
        )

        finn_interfaces: List[Interface] = []

        for axilite, width in axilites_with_addr_width:
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

        for interconnect_master, finn_signal in interconnects_connections.items():
            instantiations["axi_crossbar_0_inst"][interconnect_master].connect(
                design, instantiations["finn_kernel_inst"][finn_signal]
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
