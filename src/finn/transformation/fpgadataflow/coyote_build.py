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
from typing import Dict, List, Optional, Tuple
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
    Signal: TypeAlias = str
    name: Signal
    _sub_signals: List[Tuple[Signal, int]]
    width: int
    owner: Optional["Instantiation"]
    external: bool

    def __init__(self, name: str, width: int, external: bool):
        self.name = name
        self.width = width
        self._sub_signals = []
        self.connected = False
        self.owner = None
        self.external = external

    @abstractmethod
    def connect(self, design: "Design", other: "Interface"):
        pass

    @abstractmethod
    def connect_multiple(self, design: "Design", others: List["AXIInterface"]):
        pass


class Delimiter(Enum):
    UNDERSCORE = "_"
    POINT = "."

    def __str__(self):
        return str(self.value)


class AXIInterface(Interface):
    def __init__(self, name: str, width: int, delimiter: Delimiter, external: bool):
        super().__init__(name=name, width=width, external=external)
        self.delimiter = delimiter

    def connect(self, design: "Design", other: "AXIInterface"):
        assert not other.connected
        assert not self.connected
        assert not self.external
        if len(self._sub_signals) != len(other._sub_signals):
            assert isinstance(self, AXI4Stream) and isinstance(
                other, AXI4Stream
            ), "This should not happen"

        # NOTE: This holds only if the potential additional signals are at the end of the list
        # NOTE: This is currently the case for AXI4Stream
        max_range = min(len(self._sub_signals), len(other._sub_signals))
        for i in range(max_range):
            (sub_signal, width) = self._sub_signals[i]
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
                design.wires.append(wire)
                self.owner.connections[our_interface] = wire.name
                other.owner.connections[other_interface] = wire.name

        print("Connected %s" % other.name)
        other.connected = True
        self.connected = True
        print("Connected %s" % self.name)

    def connect_multiple(self, design: "Design", others: List["AXIInterface"]):
        assert isinstance(self, AXI4Lite)
        for other in others:
            if other.connected:
                print("Failed is %s" % other.name)
            assert not other.connected
            assert isinstance(
                other, AXI4Stream
            ), "AXI4Stream not supported for multiple connections"
        assert not self.connected
        assert not self.external
        assert self.owner is not None

        wires: List[SimpleWire] = []
        for sub_signal, width in self._sub_signals:
            wire = SimpleWire(
                name="%s_%s_wire_%s"
                % (
                    self.owner.ip.module_name,
                    self.name,
                    sub_signal,
                ),
                width=width,
            )
            design.wires.append(wire)
            wires.append(wire)
            our_interface = "%s%s%s" % (self.name, self.delimiter, sub_signal)
            self.owner.connections[our_interface] = wire.name

        for i, other in enumerate(others):
            assert not other.external
            assert other.owner is not None
            for j, (other_sub_signal, other_sub_signal_width) in enumerate(other._sub_signals):
                (sub_signal, width) = self._sub_signals[j]

                # NOTE: Guaranteed to give an integer by construction, see scale in AXI4Lite
                width_to_use = width / len(others)

                other_interface = "%s%s%s" % (other.name, other.delimiter, other_sub_signal)

                wire = wires[j]
                wire.connected = True
                print("Connected multiple %s" % self.name)
                other.owner.connections[other_interface] = "%s[%d:%d]" % (
                    wire.name,
                    i * width_to_use + other_sub_signal_width - 1,
                    i * width_to_use,
                )
            other.connected = True
            print("Connected multiple  %s" % other.name)

        self.connected = True
        print("Connected multiple %s" % self.name)


class AXI4Stream(AXIInterface):
    @staticmethod
    def next_multiple_of_8(width: int):
        return ((width + 7) >> 3) << 3

    def __init__(self, name: str, width: int, tlast: bool, delimiter: Delimiter, external: bool):
        super().__init__(name, AXI4Stream.next_multiple_of_8(width), delimiter, external)
        self.tlast = tlast
        self._sub_signals = [("tdata", self.width), ("tvalid", 1), ("tready", 1)]
        if tlast:
            # TLAST is always one bit and TKEEP is WIDTH(in bits)/8
            self._sub_signals.append(("tlast", 1))
            self._sub_signals.append(("tkeep", self.width >> 3))


class AXI4Lite(AXIInterface):
    def __init__(
        self, name: str, width: int, delimiter: Delimiter, external: bool, addr_width=0, scale=1
    ):
        super().__init__(name, width, delimiter, external)
        # Infer address width from data width
        # bresp, rresp always two bits
        # One wstrb bit per byte on the data bus data width (in bits) / 2^3 (=8 bits)
        self._sub_signals = [
            ("awaddr", (width.bit_length() - 1 if addr_width == 0 else addr_width) * scale),
            ("awvalid", 1 * scale),
            ("awready", 1 * scale),
            ("wdata", width * scale),
            ("wstrb", (width >> 3) * scale),
            ("wvalid", 1 * scale),
            ("wready", 1 * scale),
            ("bresp", 2 * scale),
            ("bvalid", 1 * scale),
            ("bready", 1 * scale),
            ("araddr", (width.bit_length() - 1 if addr_width == 0 else addr_width) * scale),
            ("arvalid", 1 * scale),
            ("arready", 1 * scale),
            ("rdata", width * scale),
            ("rresp", 2 * scale),
            ("rready", 1 * scale),
            ("rvalid", 1 * scale),
        ]


class SimpleWire(Interface):
    def __init__(self, name: str, width: int):
        super().__init__(name, width, True)

    def connect(self, design: "Design", interface: "SimpleWire"):
        assert not self.connected

        # NOTE: A wire is either generated and thus already connected, or belongs to the external
        # interface
        assert self.owner is not None

        self.owner.connections[self.name] = interface.name

    def connect_multiple(self, design: "Design", others: List["AXIInterface"]):
        assert False, "Operation not supported for SimpleWire"


class IP:
    IPConfig: TypeAlias = "dict[str, str]"
    interfaces: Dict[str, Interface]

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

    def __getitem__(self, key):
        return self.interfaces[key]

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
        model.set_metadata_prop("coyote_hw_build", self.coyote_hw_build_dir.__str__())
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

        tcl.append("set paths [get_property ip_repo_paths [current_project]];")

        ip_repo_paths = []
        for ip_repo_path in self.design.ip_repo_paths:
            ip_repo_paths.append(ip_repo_path)

        # Add IP paths to Coyote
        tcl.append("set_property  ip_repo_paths  {%s} [current_project]" % " ".join(ip_repo_paths))

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
        with open(user_logic_file) as f:
            lines = f.read().splitlines()

        assert lines is not None

        user_logic_idx = None
        for i, line in enumerate(lines):
            if "USER LOGIC" in line:
                user_logic_idx = i

        assert user_logic_idx is not None

        before_user_logic = lines[: user_logic_idx + 1]
        after_user_logic = lines[user_logic_idx + 1 :]

        verilog = []
        for wire in self.design.wires:
            verilog.append("wire [%d:0] %s;" % (wire.width - 1, wire.name))

        verilog.append("")

        for instantiation_name, instantiation in self.design.instantiations.items():
            verilog.append("%s %s (" % (instantiation.ip.module_name, instantiation_name))
            for port, external_wire in instantiation.connections.items():
                verilog.append("\t.%s(%s)," % (port, external_wire))

            verilog[len(verilog) - 1] = verilog[len(verilog) - 1][:-1]
            verilog.append(");\n")

        before_user_logic.extend(verilog)
        before_user_logic.extend(after_user_logic)

        with open(user_logic_file, "w") as f:
            f.write("\n".join(before_user_logic) + "\n")

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
                vendor="xilinx.com", library="ip", name="axis_dwidth_converter", version="1.1"
            ),
            module_name="axis_dwidth_converter_%s" % suffix,
            interfaces=[
                SimpleWire("aclk", 1),
                SimpleWire("aresetn", 1),
                # NOTE: Coyote interface is 512 bits wide
                AXI4Stream("s_axis", input_width_bits, tlast, Delimiter.UNDERSCORE, False),
                AXI4Stream("m_axis", output_width_bits, tlast, Delimiter.UNDERSCORE, False),
            ],
            ip_repo_path=None,
            config={
                "HAS_TLAST": "1",
                "HAS_TKEEP": "1",
                "M_TDATA_NUM_BYTES": "%d" % (input_width_bits >> 3),
                "S_TDATA_NUM_BYTES": "%d" % (output_width_bits >> 3),
            },
            run_needed=True,
        )

    """
    create_ip -name axi_crossbar -vendor xilinx.com -library ip -version 2.1
    -module_name axi_crossbar_1
    set_property -dict [list \
    CONFIG.ADDR_WIDTH {24} \
    CONFIG.M00_A00_ADDR_WIDTH {8} \
    CONFIG.M01_A00_ADDR_WIDTH {8} \
    CONFIG.M01_A00_BASE_ADDR {0x0000000000000100} \
    CONFIG.M02_A00_ADDR_WIDTH {8} \
    CONFIG.M02_A00_BASE_ADDR {0x0000000000000200} \
    CONFIG.M03_A00_ADDR_WIDTH {8} \
    CONFIG.M03_A00_BASE_ADDR {0x0000000000000300} \
    CONFIG.NUM_MI {4} \
    CONFIG.PROTOCOL {AXI4LITE} \
    ] [get_ips axi_crossbar_1]

    """

    @staticmethod
    def create_interconnect(
        vivado_stitch_proj: str, axilites: List[str]
    ) -> Tuple[IP, List[Tuple[str, int]]]:
        design_file = Path(vivado_stitch_proj) / "finn_design.v"

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
        prev_base = 0
        for signal_name, addr_width in axilites_with_addr_width:
            assert addr_width <= 32
            addr_width_and_base[signal_name] = (
                addr_width,
                prev_base + (1 << prev_width) if prev_width > 0 else 0,
            )
            prev_width = addr_width_and_base[signal_name][0]
            prev_base = addr_width_and_base[signal_name][1]

            assert (prev_base + (1 << prev_width)) <= (1 << 17)

        config: Dict[str, str] = {}
        config["ADDR_WIDTH"] = "32"
        config["NUM_MI"] = str(len(axilites))
        config["PROTOCOL"] = "AXI4LITE"
        for i, axilite in enumerate(axilites):
            config["M%0.2d_A00_ADDR_WIDTH" % i] = str(addr_width_and_base[axilite][0])
            if addr_width_and_base[axilite][1] != 0:
                config["M%0.2d_A00_BASE_ADDR" % i] = "0x%0.16x" % addr_width_and_base[axilite][1]

        print(config)

        return (
            IP(
                vlnv=IP.build_vlnv(
                    vendor="xilinx.com", library="ip", name="axi_crossbar", version="2.1"
                ),
                module_name="axi_crossbar_0",
                interfaces=[
                    AXI4Lite(
                        name="s_axi_crossbar",
                        width=32,
                        delimiter=Delimiter.UNDERSCORE,
                        external=False,
                        addr_width=32,
                    ),
                    AXI4Lite(
                        name="m_axi_crossbar",
                        width=32,
                        delimiter=Delimiter.UNDERSCORE,
                        external=False,
                        addr_width=32,
                        scale=len(axilites),
                    ),
                ],
            ),
            axilites_with_addr_width,
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

        # model = ModelWrapper(
        #     "/home/antoine/Documents/coyote_finn/cybersecurity/outputs_u250_automate/"
        #     "intermediate_models/step_create_stitched_ip.onnx"
        # )
        model.set_metadata_prop("accl_mode", json.dumps(CoyoteBuild.__is_accl_mode(model)))
        is_accl_mode = json.loads(model.get_metadata_prop("accl_mode"))  # type: ignore

        model = model.transform(
            CreateStitchedIPForCoyote(
                fpga_part=self.fpga_part,
                period_ns=self.period_ns,
                signature=self.signature,
            )
        )

        intf_names = json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))  # type: ignore

        # NOTE: s_axi_control_%d in intf_names["axilite"] does not match what is in the finn
        # design file. The index is 0 and not 3.
        for i, axilite in enumerate(intf_names["axilite"]):
            if "control" in axilite:
                intf_names["axilite"][i] = "s_axi_control_0"
                break

        # Only one main output and one main input
        # NOTE: This will probably stay like this
        assert len(intf_names["s_axis"]) == 1, "Only support one toplevel input"
        assert len(intf_names["m_axis"]) == 1, "Only support one toplevel output"

        (interconnect, axilites) = CoyoteBuild.create_interconnect(
            model.get_metadata_prop("vivado_stitch_proj"), intf_names["axilite"]
        )

        interfaces: List[Interface] = []

        for axilite, width in axilites:
            interfaces.append(
                AXI4Lite(
                    name=axilite,
                    width=32,
                    delimiter=Delimiter.UNDERSCORE,
                    external=False,
                    addr_width=width,
                )
            )

        interfaces.extend(
            [
                SimpleWire(intf_names["clk"][0], 1),
                SimpleWire(intf_names["rst"][0], 1),
                # NOTE: Input of FINN does not have TLast
                AXI4Stream(
                    intf_names["s_axis"][0][0],
                    intf_names["s_axis"][0][1],
                    False,
                    Delimiter.UNDERSCORE,
                    False,
                ),
                AXI4Stream(
                    intf_names["m_axis"][0][0],
                    intf_names["m_axis"][0][1],
                    not is_accl_mode,
                    Delimiter.UNDERSCORE,
                    False,
                ),
            ]
        )

        finn_kernel_ip = IP(
            vlnv=model.get_metadata_prop("vivado_stitch_vlnv"),  # type: ignore
            module_name="finn_kernel_0",
            interfaces=interfaces,
            ip_repo_path=model.get_metadata_prop("vivado_stitch_proj") + "/ip",  # type: ignore
        )

        axis_dwidth_convert_host_to_finn = CoyoteBuild.create_converter(
            "host_to_finn",
            not is_accl_mode,
            512,
            finn_kernel_ip[intf_names["s_axis"][0][0]].width,
        )

        axis_dwidth_convert_finn_to_host = CoyoteBuild.create_converter(
            "finn_to_host",
            not is_accl_mode,
            finn_kernel_ip[intf_names["m_axis"][0][0]].width,
            512,
        )

        coyote_interface = ExternalInterface(
            interfaces=[
                AXI4Lite(name="axi_ctrl", width=64, delimiter=Delimiter.POINT, external=True),
                SimpleWire(name="aclk", width=1),
                SimpleWire(name="aresetn", width=1),
                AXI4Stream(
                    name="axis_host_0_sink",
                    width=512,
                    tlast=True,
                    delimiter=Delimiter.POINT,
                    external=True,
                ),
                AXI4Stream(
                    name="axis_host_0_src",
                    width=512,
                    tlast=True,
                    delimiter=Delimiter.POINT,
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
            instantiation_name="axi_crossbar_0_inst", ip=interconnect
        )

        design = Design(instantiations, coyote_interface)

        instantiations["finn_kernel_inst"][intf_names["m_axis"][0][0]].connect(
            design, instantiations["axis_dwidth_convert_finn_to_host_inst"]["s_axis"]
        )

        instantiations["finn_kernel_inst"][intf_names["s_axis"][0][0]].connect(
            design,
            instantiations["axis_dwidth_convert_host_to_finn_inst"]["m_axis"],
        )

        # instantiations["finn_kernel_inst"][intf_names["axilite"][0]].connect(
        #     design, coyote_interface["axi_ctrl"]
        # )

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

        instantiations["axi_crossbar_0_inst"]["m_axi_crossbar"].connect_multiple(
            design,
            [
                instantiations["finn_kernel_inst"][key] for key in intf_names["axilite"]
            ],  # type: ignore
        )

        model = model.transform(GenerateCoyoteProject(fpga_part=self.fpga_part, design=design))
        model = model.transform(CoyoteUserLogic(design=design))
        # model = model.transform(CoyoteCompile())

        return (model, False)