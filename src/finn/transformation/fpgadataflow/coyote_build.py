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
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper

# from qonnx.custom_op.base import CustomOp
# from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from shutil import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import TypeAlias

import finn.custom_op.fpgadataflow.templates_coyote as templates
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP

# from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.util.basic import alveo_part_map, make_build_dir


class Interface(ABC):
    """Interface used to represent an Interface, that is to say a simple pin (SimpleWire)
    or interface pins (AXI4Lite, AXI4Stream).
    """

    Width: TypeAlias = int
    Signal: TypeAlias = Tuple[str, Width]

    name: str
    sub_signals: Dict[str, Width]
    owner: Optional["Instantiation"]
    external: bool

    def __init__(self, name: str, external: bool):
        self.name = name
        self.sub_signals = {}
        self.connected = False
        self.owner = None
        self.external = external

    @abstractmethod
    def connect(self, design: "Design", other: "Interface", is_self_input: bool = True) -> None:
        pass

    def __getitem__(self, key) -> Width:
        return self.sub_signals[key]

    def __str__(self) -> str:
        desc: List[str] = []
        desc.append("Interface name is: %s" % self.name)
        desc.append("The sub signals are:")
        for name, width in self.sub_signals.items():
            desc.append("\t%s : %d (in bits)" % (name, width))
        if self.owner is not None:
            desc.append("Owner is:")
            desc.append(self.owner.__str__())
        desc.append("Interface is%s external" % ("" if self.external else " not"))
        return "\n".join(desc)

    def get(self, sub_signal: str) -> Union[Tuple[str, Width], None]:
        width = self.sub_signals.get(sub_signal)
        if width is None:
            return None
        return (sub_signal, width)


class AXIInterface(Interface):
    """Represents an AXIInterface. Could be AXI4Lite or AXI4Stream.

    Specifies how two AXIInterfaces should be connected.
    """

    delimiter: "Delimiter"
    shift_left: Optional[int]

    class Delimiter(Enum):
        UNDERSCORE = "_"
        POINT = "."

        def __str__(self):
            return str(self.value)

    def __init__(
        self,
        name: str,
        delimiter: Delimiter,
        external: bool,
        shift_left: Optional[int] = None,
    ):
        super().__init__(name=name, external=external)
        self.delimiter = delimiter
        self.shift_left = shift_left

    @staticmethod
    def get_sub_signal(interface: "AXIInterface", sub_signal: str) -> str:
        our_sub_signal: str = ""
        if (altered_signal_and_width := interface.get(sub_signal)) is not None:
            our_sub_signal = altered_signal_and_width[0]
        elif (altered_signal_and_width := interface.get(sub_signal.swapcase())) is not None:
            our_sub_signal = altered_signal_and_width[0]
        else:
            warnings.warn(f"Signal {sub_signal} was not found in {interface}. It will be ignored.")
            return ""
        return our_sub_signal

    @staticmethod
    def get_sub_range(
        interface_1: "AXIInterface",
        interface_2: "AXIInterface",
        proper_sub_signal_1: str,
        proper_sub_signal_2: str,
    ) -> str:
        return (
            ""
            if interface_1.sub_signals[proper_sub_signal_1]
            >= interface_2.sub_signals[proper_sub_signal_2]
            or not interface_1.shift_left
            or interface_1.external
            else f"[{interface_1.sub_signals[proper_sub_signal_1] - 1}:{interface_1.shift_left}]"
        )

    @staticmethod
    def get_full_signal_name(name: str, delimiter: Delimiter, sub_signal: str):
        return "%s%s%s" % (name, delimiter, sub_signal)

    def connect(self, design: "Design", other: "AXIInterface", is_self_input: bool = True) -> None:
        assert not other.connected
        assert not self.connected
        assert not self.external

        # NOTE: It is possible that two axi interfaces do not have the same number of signals if
        # they are AXI4Streams. Some have tkeep and tlast, some do not.
        if len(self.sub_signals) != len(other.sub_signals):
            assert isinstance(self, AXI4Stream) and isinstance(other, AXI4Stream), (
                "Different number of signals should only happen in the case of AXI4Stream"
                " interfaces. This should not happen.\nSelf is: %s\nOther is: %s"
                % (self.__str__(), other.__str__())
            )

        dict_iterated_over = (
            self.sub_signals
            if len(self.sub_signals) <= len(other.sub_signals)
            else other.sub_signals
        )
        for sub_signal, width in dict_iterated_over.items():
            our_sub_signal: str = AXIInterface.get_sub_signal(self, sub_signal)
            if not our_sub_signal:
                continue

            other_sub_signal: str = AXIInterface.get_sub_signal(other, sub_signal)
            if not other_sub_signal:
                continue

            our_interface = AXIInterface.get_full_signal_name(
                self.name, self.delimiter, our_sub_signal
            )
            other_interface = AXIInterface.get_full_signal_name(
                other.name, other.delimiter, other_sub_signal
            )
            assert self.owner is not None

            our_sub_range = AXIInterface.get_sub_range(
                self, other, our_sub_signal, other_sub_signal
            )

            other_sub_range = AXIInterface.get_sub_range(
                other, self, other_sub_signal, our_sub_signal
            )

            if other.external:
                assert other.owner is None
                self.owner.connections[our_interface] = f"{other_interface}{our_sub_range}"
            else:
                assert other.owner is not None
                wire = SimpleWire(
                    name="%s_%s_wire_%s_%s_%s"
                    % (
                        self.owner.ip.module_name,
                        self.name,
                        other.owner.ip.module_name,
                        other.name,
                        sub_signal.lower(),
                    ),
                    width=max(
                        self.sub_signals[our_sub_signal], other.sub_signals[other_sub_signal]
                    ),
                )
                wire.connected = True
                # NOTE: Wires are put into the design as they are global, that is to say, seen by
                # different IPs
                design.wires.append(wire)
                self.owner.connections[our_interface] = f"{wire.name}{our_sub_range}"
                other.owner.connections[other_interface] = f"{wire.name}{other_sub_range}"

        other.connected = True
        self.connected = True


class AXI4Stream(AXIInterface):
    """Represents an AXI4Stream interface.

    Its particularity is the fact that the number of signals can vary depending on which
    it supports.
    """

    tlast: bool
    connect_tid_to_tdest: bool
    is_meta_intf: bool

    @staticmethod
    def next_multiple_of_8(width: int):
        return ((width + 7) >> 3) << 3

    def __init__(
        self,
        name: str,
        data_width: Interface.Width,
        tlast: bool,
        delimiter: AXIInterface.Delimiter,
        external: bool,
        tdest: Optional[int] = None,
        tstrb: bool = False,
        tid: Optional[int] = None,
        connect_tid_to_tdest: bool = False,
        is_meta_intf: bool = False,
    ):
        super().__init__(name, delimiter, external)
        self.tlast = tlast
        width_adjusted = AXI4Stream.next_multiple_of_8(data_width)
        if is_meta_intf:
            # NOTE: We were required to add this because metaIntf signals from the Coyote interface
            # are connected to stream signals from ACCL
            self.sub_signals = {"data": width_adjusted, "valid": 1, "ready": 1}
        else:
            self.sub_signals = {"tdata": width_adjusted, "tvalid": 1, "tready": 1}
        if tlast:
            assert not is_meta_intf
            # NOTE: TLAST is always one bit and TKEEP is WIDTH(in bits)/8
            self.sub_signals["tlast"] = 1
            self.sub_signals["tkeep"] = width_adjusted >> 3
        if tdest:
            assert not is_meta_intf
            self.sub_signals["tdest"] = tdest
        if tstrb:
            assert not is_meta_intf
            self.sub_signals["tstrb"] = width_adjusted >> 3
        if tid:
            assert not is_meta_intf
            self.sub_signals["tid"] = tid

        if connect_tid_to_tdest:
            assert not is_meta_intf
            # NOTE: We need to add this because the ACCL BD s_axis_eth_rx_data and
            # m_axis_eth_tx_data's tdest(s) are connected to axis_rdma_0_sink and axis_rdma_0_src's
            # tid respectively
            # We consider that in this case, the signals must have one of tid or tdest but not both
            # at the same time
            assert (tid is None and tdest is not None) or (tid is not None and tdest is None)
        self.connect_tid_to_tdest = connect_tid_to_tdest
        self.is_meta_intf = is_meta_intf
        assert (is_meta_intf and external) or not is_meta_intf

    def connect(self, design: "Design", other: "AXI4Stream", is_self_input: bool = True) -> None:
        assert not other.connected
        assert not self.connected
        assert not self.external

        # NOTE: We restricted meta_intf to only be possible for external interfaces
        # So we only need to check it for other

        if not other.is_meta_intf and self["tdata"] != other["tdata"]:
            input: AXI4Stream = self if is_self_input else other
            output: AXI4Stream = other if is_self_input else self
            # We need to instantiate a converter here

            # NOTE: TLast is based on the input stream.
            # We consider it ok to leave it unconnected if the input does not have TLast.
            input_to_output_converter = CoyoteBuild.create_converter(
                "%s_to_%s" % (input.name, output.name),
                input.tlast,
                input["tdata"],
                output["tdata"],
            )

            instantiation_name = "%s_inst" % input_to_output_converter.module_name
            design.instantiations[instantiation_name] = Instantiation(
                instantiation_name=instantiation_name,
                ip=input_to_output_converter,
            )

            design.ips.add(design.instantiations[instantiation_name].ip)

            assert isinstance(design.instantiations[instantiation_name]["s_axis"], AXI4Stream)
            assert isinstance(design.instantiations[instantiation_name]["m_axis"], AXI4Stream)

            s_axis_signal: AXIInterface = design.instantiations[instantiation_name]["s_axis"]
            m_axis_signal: AXIInterface = design.instantiations[instantiation_name]["m_axis"]

            if input.external:
                AXIInterface.connect(s_axis_signal, design, input)
            else:
                AXIInterface.connect(input, design, s_axis_signal)

            AXIInterface.connect(m_axis_signal, design, output)
        elif other.is_meta_intf and self["tdata"] != other["data"]:
            raise Exception(
                "Meta intf interfaces cannot connect to interfaces of different widths.\nFirst "
                f"interface is:\n{self}\n Second interface is:\n{other}"
            )
        else:
            AXIInterface.connect(self, design, other)

        if self.connect_tid_to_tdest and other.connect_tid_to_tdest:
            tid_owner: AXI4Stream = self
            tdest_owner: AXI4Stream = other
            if (
                self.sub_signals.get("tid") is not None and other.sub_signals.get("tid") is None
            ) or (
                self.sub_signals.get("tdest") is not None and other.sub_signals.get("tdest") is None
            ):
                if (
                    self.sub_signals.get("tdest") is not None
                    and other.sub_signals.get("tdest") is None
                ):
                    tdest_owner = self
                    tid_owner = other

            else:
                raise Exception(
                    "Trying to connect tid from one interface to the tdest of another but both "
                    "interfaces provide the same signal!\n"
                    f"First interface is:\n{self}\nOther is:\n{other}"
                )

            assert tid_owner.sub_signals["tid"] == tdest_owner.sub_signals["tdest"]
            tid_simple_wire: SimpleWire = SimpleWire(
                name=AXIInterface.get_full_signal_name(tid_owner.name, tid_owner.delimiter, "tid"),
                width=tid_owner.sub_signals["tid"],
            )
            tid_simple_wire.owner = tid_owner.owner
            tdest_simple_wire: SimpleWire = SimpleWire(
                name=AXIInterface.get_full_signal_name(
                    tdest_owner.name, tdest_owner.delimiter, "tdest"
                ),
                width=tdest_owner.sub_signals["tdest"],
            )
            tdest_simple_wire.owner = tdest_owner.owner

            tid_simple_wire.external = tid_simple_wire.external
            tdest_simple_wire.external = tdest_owner.external

            # NOTE: We are guaranteed that exactly one of the two interfaces is external
            assert self.external ^ other.external
            if not tid_owner.external:
                tid_simple_wire.connect(design, tdest_simple_wire)
            else:
                tdest_simple_wire.connect(design, tid_simple_wire)

    def __str__(self):
        return "Interface type: AXI4Stream\n%s" % super().__str__()

    def get(self, sub_signal: str) -> Union[Tuple[str, Interface.Width], None]:
        altered_sub_signal: str = sub_signal
        if self.is_meta_intf and self.sub_signals.get(sub_signal) is None:
            # NOTE: We remove the t for meta intf
            altered_sub_signal = sub_signal[1:]
        elif not self.is_meta_intf and self.sub_signals.get(sub_signal) is None:
            altered_sub_signal = "t" + sub_signal
        width = self.sub_signals.get(altered_sub_signal)
        if width is None:
            return None
        else:
            return (altered_sub_signal, width)


class AXI4Lite(AXIInterface):
    """Represents an AXI4Lite interface."""

    name: str
    wdith: Interface.Width
    delimiter: AXIInterface.Delimiter
    external: bool
    addr_width: int
    upper_case: bool

    def __init__(
        self,
        name: str,
        width: Interface.Width,
        delimiter: AXIInterface.Delimiter,
        external: bool,
        addr_width: int,
        upper_case: bool = False,
        shift_left: Optional[int] = None,
    ):
        super().__init__(name, delimiter, external, shift_left)
        # NOTE: arprot and awprot can safely be ignored (recommended value is 0b000)
        self.sub_signals = {
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

        if upper_case:
            self.sub_signals = {k.upper(): v for k, v in self.sub_signals.items()}

    def __str__(self):
        return "Interface type: AXI4Lite\n%s" % super().__str__()


class SimpleWire(Interface):
    """Represents a simple wire, that is to say, an interface with only one pin."""

    width: Interface.Width

    def __init__(self, name: str, width: Interface.Width):
        super().__init__(name, True)
        self.width = width

    def connect(self, design: "Design", interface: "SimpleWire"):
        assert not self.connected

        # NOTE: A wire is either generated and thus already connected, or belongs to the external
        # interface, or is connected to an external interface signal
        assert self.owner is not None

        self.owner.connections[self.name] = interface.name

    def __str__(self):
        return "Interface type: SimpleWire\n%s" % super().__str__()


class Clk(SimpleWire):
    def __init__(self, name: str, width: Interface.Width):
        super().__init__(name, width)


class Reset(SimpleWire):
    def __init__(self, name: str, width: Interface.Width):
        super().__init__(name, width)


class IP:
    """Represents an IP and all that is required to instantiate it."""

    IPConfig: TypeAlias = "dict[str, str]"

    vlnv: str
    module_name: str
    interfaces: Dict[str, Interface]
    ip_repo_path: Optional[str]
    config: Optional[IPConfig]

    def __init__(
        self,
        vlnv: str,
        module_name: str,  # NOTE: Name of the configured/instantiable version of the IP
        interfaces: List[Interface],
        ip_repo_path: Optional[str] = None,  # NOTE: No repo path for Vivado/Xilinx IPs
        config: Optional[IPConfig] = None,
    ):
        self.vlnv = vlnv
        self.ip_repo_path = ip_repo_path
        self.config = config
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

    bd_name: str
    ips: Optional[List[IP]]
    interfaces: Dict[str, Interface]
    intra_connections: Sequence[str]
    extra_external_commands: Optional[List[str]]

    def __init__(
        self,
        bd_name: str,
        ips: Optional[List[IP]],
        interfaces: Optional[List[Interface]],
        intra_connections: Sequence[str],
        extra_external_commands: Optional[List[str]],
    ):
        self.module_name = bd_name
        self.ips = ips

        self.intra_connections = intra_connections
        self.interfaces = {}
        if interfaces:
            self.interfaces = {interface.name: interface for interface in interfaces}
        if ips is not None:
            tcl: List[str] = []
            get_bd_cells: List[str] = []
            for ip in ips:
                if ip.interfaces:
                    get_bd_cells.append("[get_bd_cells %s]" % ip.module_name)
                    self.interfaces.update(ip.interfaces)

            if get_bd_cells:
                tcl.append("make_bd_pins_external  %s" % " ".join(get_bd_cells))
                tcl.append("make_bd_intf_pins_external  %s" % " ".join(get_bd_cells))
            if extra_external_commands is not None:
                tcl.extend(extra_external_commands)
            self.make_external = tcl

    def create_ips(self) -> List[str]:
        """Responsible for actually creating the block design"""
        tcl: List[str] = []
        if self.ips is None:
            return tcl
        for ip in self.ips:
            tcl.append("create_bd_cell -type ip -vlnv %s %s" % (ip.vlnv, ip.module_name))
            if ip.config is not None:
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
    In the Coyote case, it is the interface Coyote gives us inside the user logic file.
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
        clk_name: str,
        rst_name: str,
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
            elif isinstance(instantiation.ip, BD):
                assert instantiation.ip.ips is not None
                for ip in instantiation.ip.ips:
                    if ip is not None and ip.ip_repo_path is not None:
                        self.ip_repo_paths.add(ip.ip_repo_path)
        self.clk_name = clk_name
        self.rst_name = rst_name


class CreateHLSBridge(Transformation):
    """
    Creates an HLS bridge to allow writing the weights with a unlimited address space.
    """

    fpga_part: str

    def __init__(self, fpga_part: str):
        super().__init__()
        self.fpga_part = fpga_part

    def apply(self, model):
        hls_bridge_folder: Path = Path(make_build_dir("hls_bridge_"))
        model.set_metadata_prop("hls_bridge_dir", hls_bridge_folder.__str__())
        finn_cwd = os.getcwd()
        os.chdir(hls_bridge_folder)

        c_file = hls_bridge_folder / "bridge.c"
        with open(c_file, "w") as text_file:
            text_file.write(templates.hls_bridge)

        hls_script = templates.hls_bridge_script.replace("$PART$", self.fpga_part)

        vitis_hls_cmd = ["vitis_hls", "-eval", hls_script]
        process_vitis_hls = subprocess.Popen(vitis_hls_cmd, stdout=subprocess.PIPE)
        process_vitis_hls.communicate()
        assert (
            process_vitis_hls.returncode == 0
        ), "Failed to run hls bridge generation in Vitis_HLS, command is: %s" % " ".join(
            vitis_hls_cmd
        )

        os.chdir(finn_cwd)

        model.set_metadata_prop(
            "hls_bridge_ip_path",
            (hls_bridge_folder / "hls_bridge" / "hls_bridge_sol" / "impl" / "ip").__str__(),
        )

        return (model, False)


class CreateStitchedIPForCoyote(Transformation):
    """
    Adds a TLastMarker node at the end of the graph and creates a stitched IP from the new graph.

    Outcome if successful: the vivado_stitch_proj attribute is set to the associated Vivado stitched
    IP project.
    """

    def __init__(self, fpga_part, period_ns, signature, is_accl_mode: bool):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.signature = signature
        self.is_accl_mode = is_accl_mode

    def apply(self, model):
        # NOTE: We want dynamic True so we leave default arguments
        # NOTE: We only want the TLastMarker node at the output since the Coyote interface already
        # provides tlast to the input width converter
        # model = model.transform(InsertTLastMarker())
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


class PrepareCoyoteProject(Transformation):
    fpga_part: str
    is_accl_mode: bool
    coyote_proj_dir: Path
    coyote_repo_dir: Path
    coyote_hw_dir: Path
    coyote_hw_build_dir: Path
    m_frequency: int

    def __init__(self, fpga_part: str, is_accl_mode: bool, period_ns: int):
        super().__init__()
        self.fpga_part = fpga_part
        self.is_accl_mode = is_accl_mode
        self.coyote_proj_dir = Path(make_build_dir(prefix="coyote_proj_"))
        self.coyote_repo_dir = self.coyote_proj_dir / "Coyote"
        self.coyote_hw_dir = self.coyote_repo_dir / "hw"
        self.coyote_hw_build_dir = self.coyote_hw_dir / "build"
        self.m_frequency = int((1.0 / float(period_ns)) * 10**3)

    def apply(self, model):
        self.accl_repo_dir = model.get_metadata_prop("accl_repo_dir")

        model.set_metadata_prop("coyote_hw_build", self.coyote_hw_build_dir.__str__())

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

        # Prepare Coyote Shell
        if self.is_accl_mode:
            git_checkout_cmd = ["git", "checkout", "accl_integration"]
            process_checkout = subprocess.Popen(git_checkout_cmd, stdout=subprocess.PIPE)
            process_checkout.communicate()
            assert (
                process_checkout.returncode == 0
            ), "Unable to checkout branch with git, command is: %s" % " ".join(git_checkout_cmd)

        # CMake Coyote
        coyote_board = board.lower()
        cmake_command = ["/usr/bin/cmake"]
        if self.is_accl_mode:
            cmake_command.extend(
                [
                    f"{self.coyote_hw_dir}",
                    f"-DFDEV_NAME={coyote_board}",
                    "-DEN_MEM=1",
                    "-DEN_STRM=1",
                    "-DEN_BPSS=1",
                    "-DEN_RDMA_0=1",
                    "-DEN_RPC=1",
                    "-DN_STRM_AXI=4",
                    "-DN_CARD_AXI=3",
                    "-DEN_HLS=0",
                    f"-DACLK_F={self.m_frequency}",
                    "-DTLBL_A=12",
                ]
            )
        else:
            cmake_command.extend([f"{self.coyote_hw_dir}", f"-DFDEV_NAME={coyote_board}"])

        process_cmake = subprocess.Popen(cmake_command, stdout=subprocess.PIPE)
        process_cmake.communicate()
        assert (
            process_cmake.returncode == 0
        ), "Failed to generate CMake configuration for coyote, cmake command is: %s" % " ".join(
            cmake_command
        )

        make_shell_command = ["make", "shell"]
        process_shell = subprocess.Popen(make_shell_command, stdout=subprocess.PIPE)
        process_shell.communicate()
        assert (
            process_shell.returncode == 0
        ), "Failed to generate Coyote shell. Command is: %s" % " ".join(make_shell_command)
        assert Path("lynx/lynx.xpr").is_file()

        os.chdir(finn_cwd)
        return (model, False)


class GenerateCoyoteProject(Transformation):
    """Generate a Coyote Vivado project with the required configuration to make the instantiation
    of the FINN kernel possible. It will add the required IP repo paths, generate the required IPs
    and block designs.

    Outcome if successful: sets the coyote_hw_build attribute to the Coyote shell project path. The
    project will have the proper IP paths, IPs, output products and block designs setup.
    """

    coyote_hw_build_dir: Path

    def __init__(
        self, fpga_part: str, design: Design, is_accl_mode: bool, coyote_hw_build_dir: Path
    ):
        super().__init__()
        self.fpga_part = fpga_part
        self.design = design
        self.is_accl_mode = is_accl_mode
        self.coyote_hw_build_dir = coyote_hw_build_dir

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
        finn_cwd = os.getcwd()
        os.chdir(self.coyote_hw_build_dir)

        tcl = []
        tcl.append("open_project lynx/lynx.xpr")
        tcl.append("update_compile_order -fileset sources_1")

        tcl.append("set paths [get_property ip_repo_paths [current_project]];")

        tcl.append("set * [get_property ip_repo_paths [current_project]]")
        # Add IP paths to Coyote
        tcl.append(
            'set_property  ip_repo_paths  "${*} %s" [current_project]'
            % " ".join(self.design.ip_repo_paths)
        )

        tcl.append("update_ip_catalog")

        for ip in self.design.ips:
            tcl.extend(self.create_ip(ip))

        tcl_string = "\n".join(tcl) + "\n"
        with open(self.coyote_hw_build_dir / Path("automate.tcl"), "w") as f:
            f.write(tcl_string)

        vivado_cmd = ["vivado", "-mode", "batch", "-source", "automate.tcl"]
        process_vivado = subprocess.Popen(vivado_cmd, stdout=subprocess.PIPE)
        process_vivado.communicate()
        assert (
            process_vivado.returncode == 0
        ), "Failed to run project automation in Vivado,command is: %s" % " ".join(vivado_cmd)

        os.chdir(finn_cwd)

        return (model, False)


class CoyoteUserLogic(Transformation):
    """Generates the user_logic file for the Coyote shell containing the FINN kernel and all that
    is required to make it work.

    Outcome if successful: Modifies the user_logic verilog file provided by Coyote to contain the
    FINN kernel and all that is required to make it work inside with the shell.
    """

    def __init__(self, design: Design, user_logic_config: str):
        super().__init__()
        self.design = design
        self.user_logic_config = user_logic_config

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
        verilog.extend(self.user_logic_config.splitlines())
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
            for interface_name, interface in instantiation.ip.interfaces.items():
                if isinstance(interface, Clk) and not interface.connected:
                    interface.connect(
                        self.design, self.design.external_interface[self.design.clk_name]
                    )
                elif isinstance(interface, Reset) and not interface.connected:
                    interface.connect(
                        self.design, self.design.external_interface[self.design.rst_name]
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
                Clk("aclk", 1),
                Reset("aresetn", 1),
                AXI4Stream(
                    "s_axis",
                    input_width_bits,
                    tlast,
                    AXIInterface.Delimiter.UNDERSCORE,
                    False,
                ),
                AXI4Stream(
                    "m_axis",
                    output_width_bits,
                    tlast,
                    AXIInterface.Delimiter.UNDERSCORE,
                    False,
                ),
            ],
            ip_repo_path=None,
            config={
                "HAS_TLAST": "1",
                "HAS_TKEEP": "1",
                "M_TDATA_NUM_BYTES": "%d" % (output_width_bits >> 3),
                "S_TDATA_NUM_BYTES": "%d" % (input_width_bits >> 3),
            },
        )

    @staticmethod
    def next_power_of_2(x: int):
        return 1 << (x - 1).bit_length()

    @staticmethod
    def get_axilites_with_width(wrapper_file: str, axilites: List[str]) -> List[Interface.Signal]:
        """The function reads the actual axilites addresses width directly from the finn_design.v
        file generated by the CreateStitchedIPStep.

        Parameters:
        wrapper_file (str): Path to the Vivado stitched IP wrapper file
        axilites (List[str]): Names of the axilites interfaces

        Returns:
        List[Interface.Signal]: List of axilites interfaces names associated with their address
        width

        """

        contents = None
        with open(wrapper_file, "r") as f:
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
        axilites_outer: List[Tuple[str, Interface.Signal]], global_base_addr: int, limit: int
    ) -> Tuple[List[IP], Dict[str, Dict[str, str]], List[str], List[str], List[str]]:
        """This function creates one or more interconnects to connect the Coyote AXI4Lite interface
        to the potentially multiple accelerator AX4List interfaces. In case the accelerator
        requires more than 16 AXI4Lite interfaces, multiple interconnects, chained together, are
        generated by this function.

        Parameters:
        axilites_outer (List[Tuple[str, Interface.Signal]]): List of axilite interfaces with their
        name and their address width. The first element of the tuple corresponds to the
        instantiation name associated with the interface.
        global_base_addr (int): the base address the crossbar address mappings should start from
        limit (int): the upper limit on the available address range

        Returns:
        Tuple[List[IP], Dict[str, str], List[str], List[str]]:
            - List[IP]: the interconnects IPs
            - Dict[str, Dict[str, str]]: Instantiation -> (the interconnects connections to the
            axilites signals)
            - List[str]: the connections between the interconnects, as TCL commands
            - List[str]: the specific TCL commands to make the appropriate pins external
            - List[str]: the address map

        """

        def collapse_axilites_to_fit(
            axilites_inner: List[Tuple[str, Interface.Signal]],
        ) -> Tuple[
            List[Tuple[List[Tuple[str, Interface.Signal]], int]],
            Optional[List[Tuple[str, Interface.Signal]]],
            Optional[int],
        ]:
            # NOTE: This represents the axilites connections as a list of list because everything
            # after and including the 15th elements of axilites_inner is collapsed into the 15th
            # element. This allows to treat the axilites that will be connected to the other
            # interconnects just like any other axi lite, except we will use the total address
            # width of the connected interfaces to generate the proper address map.
            axilites_list_list: List[Tuple[List[Tuple[str, Interface.Signal]], int]] = [
                ([(instantiation_name, (name, width))], width)
                for instantiation_name, (name, width) in axilites_inner
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

            axilites_list_list_with_indices = [
                (i, elem) for i, elem in enumerate(axilites_list_list)
            ]
            axilites_list_list_with_indices = sorted(
                axilites_list_list_with_indices, key=lambda tup: tup[1][1], reverse=True
            )

            chain_interconnect_idx = None
            if remaining_axilites:
                for i, (k, elem) in enumerate(axilites_list_list_with_indices):
                    if k == MAX_AMOUNT_OF_AXILITES - 1:
                        chain_interconnect_idx = i

            axilites_list_list = [elem for (i, elem) in axilites_list_list_with_indices]
            # NOTE: After the cut, if one was needed, axilites_list_list should not contain more
            # than 16 elements
            assert len(axilites_list_list) <= MAX_AMOUNT_OF_AXILITES

            return (axilites_list_list, remaining_axilites, chain_interconnect_idx)

        def configure_and_connect(
            axilites_list_list: List[Tuple[List[Tuple[str, Interface.Signal]], int]],
            indices_count: Dict[int, int],
            base_addr: int,
            limit: int,
            chain_interconnect_idx: Optional[int],
        ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], List[str]]:
            config: Dict[str, str] = {}
            config["ADDR_WIDTH"] = "32"
            config["DATA_WIDTH"] = "32"
            config["NUM_MI"] = str(len(axilites_list_list))
            config["PROTOCOL"] = "AXI4LITE"
            prev_width = 0
            prev_base = base_addr

            axilites_connections: Dict[str, Dict[str, str]] = {}
            address_map: List[str] = []
            for i, (axilites_with_width, total_width) in enumerate(axilites_list_list):
                assert total_width <= 32

                (component_name, (signal_name, addr_width)) = (
                    axilites_with_width[0]
                    if i != chain_interconnect_idx
                    else ("interconnect_component", ("interconnect_chain", total_width))
                )
                config["M%0.2d_A00_ADDR_WIDTH" % i] = str(addr_width)
                if prev_width == 0:
                    config["M%0.2d_A00_BASE_ADDR" % i] = "0x%0.16x" % prev_base
                else:
                    prev_base = prev_base + (1 << prev_width)
                    config["M%0.2d_A00_BASE_ADDR" % i] = "0x%0.16x" % prev_base

                if i != chain_interconnect_idx:
                    if component_name not in axilites_connections:
                        axilites_connections[component_name] = {}
                    axilites_connections[component_name][
                        "M%0.2d_AXI_%d" % (i, indices_count.get(i, 0))
                    ] = signal_name
                    indices_count[i] = indices_count.get(i, 0) + 1
                    address_map.append(
                        "%s\t: %s" % (signal_name, config["M%0.2d_A00_BASE_ADDR" % i])
                    )
                prev_width = addr_width

                assert (
                    prev_base + (1 << prev_width) - 1
                ) <= limit, "The axi lites interfaces cover too big of a range."

            return (axilites_connections, config, address_map)

        def create_interconnects_inner(
            axilites_inner: List[Tuple[str, Interface.Signal]],
            current_idx: int,
            indices_count: Dict[int, int],
            base_addr: int,
        ) -> Tuple[List[IP], Dict[str, Dict[str, str]], List[str], List[str]]:
            """This function is responsible for actually creating the interconnects. It should be
            called recurisevly with a smaller list of axilites signals everytime.

            Parameters:
            axilites_inner (List[Tuple[str, Interface.Signal]]): List of remaining axilite
            interfaces with their name and their address width. The first element of
            the tuple is the instantiation name associated with the signal.
            current_idx (int): index of the currently generated interconnect. Useful for generating
            interfaces and IPs names.
            indices_count (Dict[int, int]): When generating multiple interconnects, it is possible
            that several master connections with the same index gets exposed to the outside. For
            instance, it is possible that the M01_AXI interface of interconnect 0 and the M01_AXI
            of interconnect 1 get exposed to the outside. In that case, the exposed interface for
            interconnect 0, will be named M01_AXI_0 and the exposed interface of interconnect 1
            will be named M01_AXI_1. `indices_count` allows to keep track of how many master
            interfaces that are going to get exposed have the same inner index.
            base_addr (int): start base address for this particular interconnect

            Returns:
            Tuple[List[IP], Dict[str, Dict[str, str]], List[str], List[str]]:
                - List[IP]: the interconnects IPs
                - Dict[str, Dict[str, str]]: the interconnects connections to the axilites signals
                - List[str]: the connections between the interconnects, as TCL commands
                - List[str]: the address map

            """

            (
                axilites_list_list,
                remaining_axilites,
                chain_interconnect_idx,
            ) = collapse_axilites_to_fit(axilites_inner)
            old_indices_count = indices_count.copy()

            ips: List[IP] = []
            intra_connections_tcl: List[str] = []

            (
                axilites_connections,
                config,
                address_map_outer,
            ) = configure_and_connect(
                axilites_list_list, indices_count, base_addr, limit, chain_interconnect_idx
            )

            if remaining_axilites is not None:
                assert chain_interconnect_idx is not None
                (
                    intra_ips,
                    intra_axilites_connections,
                    intra_intra_connections_tcl,
                    address_map_inner,
                ) = create_interconnects_inner(
                    remaining_axilites,
                    current_idx=current_idx + 1,
                    indices_count=indices_count,
                    base_addr=int(config["M%0.2d_A00_BASE_ADDR" % chain_interconnect_idx], 16),
                )

                address_map_outer.extend(address_map_inner)
                ips.extend(intra_ips)
                for component, connections in intra_axilites_connections.items():
                    if axilites_connections.get(component) is not None:
                        axilites_connections[component].update(connections)
                    else:
                        axilites_connections[component] = connections
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
                interfaces.extend([Clk("aclk_0", 1), Reset("aresetn_0", 1)])

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
                )
            ]
            return (ips, axilites_connections, intra_connections_tcl, address_map_outer)

        (
            interconnects,
            interconnects_axilites_connections,
            intra_connections,
            address_map,
        ) = create_interconnects_inner(axilites_outer, 0, {}, global_base_addr)

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
            address_map,
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

    @staticmethod
    def __update_intf_axilite_weight(intf_names_axilite: List[str]) -> Optional[int]:
        s_axi_weight_counter: int = 0
        weight_start_idx: Optional[int] = None
        for i, axilite in enumerate(intf_names_axilite):
            if "axilite" in axilite:
                if weight_start_idx is None:
                    match = re.search(r"s_axilite_(\d+)", axilite)
                    assert match
                    weight_start_idx = int(match.group(1))
                intf_names_axilite[i] = "s_axilite_%d" % s_axi_weight_counter
                s_axi_weight_counter = s_axi_weight_counter + 1
        return weight_start_idx

    def get_hls_address_map(self, model: ModelWrapper):
        # tlast_node = model.get_nodes_by_op_type("TLastMarker")
        # assert len(tlast_node) == 1
        # tlast_op: CustomOp = getCustomOp(tlast_node[0])
        # header_file_path = (
        #     Path(tlast_op.get_nodeattr("ip_path"))
        #     / "drivers"
        #     / "TLastMarker_0_v1_0"
        #     / "src"
        #     / "xtlastmarker_0_hw.h"
        # )
        # copy(header_file_path, model.get_metadata_prop("address_map"))

        bridge_header_file_path = (
            Path(model.get_metadata_prop("hls_bridge_dir"))
            / "hls_bridge"
            / "hls_bridge_sol"
            / "impl"
            / "ip"
            / "drivers"
            / "write_intf_bridge_v1_0"
            / "src"
            / "xwrite_intf_bridge_hw.h"
        )
        if bridge_header_file_path.exists():
            copy(bridge_header_file_path, model.get_metadata_prop("address_map"))

    def write_address_map_to_file(self, model: ModelWrapper, address_map: List[str]) -> None:
        address_map.append("")
        with open(
            model.get_metadata_prop("address_map") + "/address_map.txt", "w"
        ) as address_map_file:
            address_map_file.write("\n".join(address_map))

    def apply(self, model: ModelWrapper):
        is_accl_mode: bool = CoyoteBuild.__is_accl_mode(model)
        model.set_metadata_prop("address_map", make_build_dir("address_map_"))

        # Prepare everything
        model = model.transform(
            CreateStitchedIPForCoyote(
                fpga_part=self.fpga_part,
                period_ns=self.period_ns,
                signature=self.signature,
                is_accl_mode=is_accl_mode,
            )
        )

        model = model.transform(
            PrepareCoyoteProject(
                fpga_part=self.fpga_part, is_accl_mode=is_accl_mode, period_ns=self.period_ns
            )
        )

        # Handle FINN interface
        intf_names = json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))  # type: ignore

        print(f"Axilites before update:\n{intf_names['axilite']}")

        CoyoteBuild.__update_intf_axilite_control(intf_names["axilite"])
        weight_start_idx = CoyoteBuild.__update_intf_axilite_weight(intf_names["axilite"])
        print(f"Weight idx is: {weight_start_idx}")

        print(f"Axilites after update:\n{intf_names['axilite']}")

        if not is_accl_mode:
            assert len(intf_names["s_axis"]) == 1, "Only support one toplevel input"
            assert len(intf_names["m_axis"]) == 1, "Only support one toplevel output"

        axilites_with_addr_width = CoyoteBuild.get_axilites_with_width(
            model.get_metadata_prop("wrapper_filename"), intf_names["axilite"]
        )

        if len(axilites_with_addr_width) > 0:
            model = model.transform(CreateHLSBridge(self.fpga_part))

        self.get_hls_address_map(model=model)

        finn_kernel_ip: IP = templates.get_finn_interface(
            is_accl_mode=is_accl_mode,
            axilites=axilites_with_addr_width,
            intf_names=intf_names,
            model=model,
        )

        # Instnatiate IPS
        instantiations = {}
        instantiations["finn_kernel_inst"] = Instantiation(
            instantiation_name="finn_kernel_inst", ip=finn_kernel_ip
        )

        if len(axilites_with_addr_width) > 0:
            path_to_hls_bridge_ip: str = model.get_metadata_prop("hls_bridge_ip_path")
            hls_bridge_ip: IP = templates.get_hls_bridge_ip(
                path_to_hls_bridge_ip=path_to_hls_bridge_ip
            )
            instantiations["hls_bridge_inst"] = Instantiation(
                instantiation_name="hls_bridge_inst", ip=hls_bridge_ip
            )

        if is_accl_mode:
            instantiations["accl_bd_inst"] = Instantiation(
                instantiation_name="accl_bd_inst",
                ip=templates.generate_accl_bd(
                    (Path(model.get_metadata_prop("accl_repo_dir")) / "kernels").__str__()
                ),
            )

        coyote_interconnects_connections = None
        finn_interconnects_connections = None
        if len(axilites_with_addr_width) > 0:
            COYOTE_BASE_ADDR = 0x12_0000
            COYOTE_LIMIT = 0x13_FFFF
            # Top level axilite signals
            (
                coyote_interconnects,
                coyote_interconnects_connections,
                coyote_intra_connections,
                coyote_extra_external_commands,
                address_map_coyote,
                # NOTE: ACCL driver expects the appropriate axi interfaces to be mapped at the
                # start of the address space
            ) = CoyoteBuild.create_interconnects(
                [("accl_bd_inst", ("S00_AXI_0", 15)), ("hls_bridge_inst", ("s_axi_control", 5))],
                global_base_addr=COYOTE_BASE_ADDR,
                limit=COYOTE_LIMIT,
            )

            coyote_interconnect_bd = BD(
                bd_name="design_crossbar_coyote",
                ips=coyote_interconnects,
                interfaces=None,
                intra_connections=coyote_intra_connections,
                extra_external_commands=coyote_extra_external_commands,
            )

            instantiations["axi_crossbar_coyote_0_inst"] = Instantiation(
                instantiation_name="axi_crossbar_coyote_0_inst", ip=coyote_interconnect_bd
            )

            if len(axilites_with_addr_width) > 1:
                # FINN axilite signals that will be connected to the bridge
                (
                    finn_interconnects,
                    finn_interconnects_connections,
                    finn_intra_connections,
                    finn_extra_external_commands,
                    address_map_finn,
                ) = CoyoteBuild.create_interconnects(
                    [
                        ("finn_kernel_inst", axilite_with_addr_width)
                        for axilite_with_addr_width in axilites_with_addr_width
                    ],
                    global_base_addr=0x0,
                    limit=(1 << 32) - 1,
                )

                print(f"Address map before update:\n{address_map_finn}")
                assert weight_start_idx is not None
                for i, address_map_element in enumerate(address_map_finn):
                    if "axilite" in address_map_finn:
                        match = re.search(r"s_axilite_(\d+)", address_map_element)
                        assert match
                        current_idx = int(match.group(1))
                        new_idx = current_idx + weight_start_idx
                        address_map_finn[i] = address_map_finn[i].replace(
                            f"s_axilite_{current_idx}",
                            f"s_axilite_{new_idx}",
                        )
                print(f"Address map after update:\n{address_map_finn}")

                self.write_address_map_to_file(
                    model=model,
                    address_map=["Address map for first interconnect: Coyote -> (ACCL + BRIDGE)"]
                    + address_map_coyote
                    + ["Address for FINN specific interconnect: BRIDGE -> (FINN axilites)"]
                    + address_map_finn,
                )

                finn_interconnect_bd = BD(
                    bd_name="design_crossbar_finn",
                    ips=finn_interconnects,
                    interfaces=None,
                    intra_connections=finn_intra_connections,
                    extra_external_commands=finn_extra_external_commands,
                )

                instantiations["axi_crossbar_finn_0_inst"] = Instantiation(
                    instantiation_name="axi_crossbar_finn_0_inst", ip=finn_interconnect_bd
                )
            else:
                self.write_address_map_to_file(
                    model=model,
                    address_map=["Address map for first interconnect: Coyote -> (ACCL + BRIDGE)"]
                    + address_map_coyote,
                )

        coyote_interface: ExternalInterface = (
            templates.get_coyote_interface()
            if not is_accl_mode
            else templates.get_coyote_interface_accl()
        )

        design = Design(instantiations, coyote_interface, "aclk", "aresetn")

        if len(axilites_with_addr_width) > 0:
            assert coyote_interconnects_connections
            for component_name, connection_dict in coyote_interconnects_connections.items():
                for interconnect_master, signal in connection_dict.items():
                    instantiations["axi_crossbar_coyote_0_inst"][interconnect_master].connect(
                        design, instantiations[component_name][signal]
                    )

            if len(axilites_with_addr_width) > 1:
                assert finn_interconnects_connections
                for component_name, connection_dict in finn_interconnects_connections.items():
                    for interconnect_master, finn_signal in connection_dict.items():
                        instantiations["axi_crossbar_finn_0_inst"][interconnect_master].connect(
                            design, instantiations[component_name][finn_signal]
                        )

                instantiations["hls_bridge_inst"]["m_axi_gmem"].connect(
                    design, instantiations["axi_crossbar_finn_0_inst"]["S00_AXI_0"]
                )
            else:
                instantiations["hls_bridge_inst"]["m_axi_gmem"].connect(
                    design, instantiations["finn_kernel_inst"][intf_names["axilite"][0]]
                )

            instantiations["axi_crossbar_coyote_0_inst"]["S00_AXI_0"].connect(
                design, coyote_interface["axi_ctrl"]
            )

        if is_accl_mode:
            # ACCL BD connections except for AXI4Lite

            instantiations["accl_bd_inst"]["cyt_byp_rd_cmd_0"].connect(
                design, coyote_interface["bpss_rd_req"]
            )
            instantiations["accl_bd_inst"]["cyt_byp_rd_sts_0"].connect(
                design, coyote_interface["bpss_rd_done"]
            )
            instantiations["accl_bd_inst"]["cyt_byp_wr_cmd_0"].connect(
                design, coyote_interface["bpss_wr_req"]
            )
            instantiations["accl_bd_inst"]["cyt_byp_wr_sts_0"].connect(
                design, coyote_interface["bpss_wr_done"]
            )

            instantiations["accl_bd_inst"]["m_axis_host_0"].connect(
                design, coyote_interface["axis_host_0_src_s"]
            )
            instantiations["accl_bd_inst"]["m_axis_host_1"].connect(
                design, coyote_interface["axis_host_1_src_s"]
            )
            instantiations["accl_bd_inst"]["m_axis_host_2"].connect(
                design, coyote_interface["axis_host_2_src_s"]
            )

            instantiations["accl_bd_inst"]["m_axis_card_0"].connect(
                design, coyote_interface["axis_card_0_src_s"]
            )
            instantiations["accl_bd_inst"]["m_axis_card_1"].connect(
                design, coyote_interface["axis_card_1_src_s"]
            )
            instantiations["accl_bd_inst"]["m_axis_card_2"].connect(
                design, coyote_interface["axis_card_2_src_s"]
            )

            instantiations["accl_bd_inst"]["s_axis_host_0"].connect(
                design, coyote_interface["axis_host_0_sink_s"]
            )
            instantiations["accl_bd_inst"]["s_axis_host_1"].connect(
                design, coyote_interface["axis_host_1_sink_s"]
            )
            instantiations["accl_bd_inst"]["s_axis_host_2"].connect(
                design, coyote_interface["axis_host_2_sink_s"]
            )

            instantiations["accl_bd_inst"]["s_axis_card_0"].connect(
                design, coyote_interface["axis_card_0_sink_s"]
            )
            instantiations["accl_bd_inst"]["s_axis_card_1"].connect(
                design, coyote_interface["axis_card_1_sink_s"]
            )
            instantiations["accl_bd_inst"]["s_axis_card_2"].connect(
                design, coyote_interface["axis_card_2_sink_s"]
            )

            instantiations["accl_bd_inst"]["s_axis_eth_rx_data"].connect(
                design, coyote_interface["axis_rdma_0_sink"]
            )
            # NOTE: Apparently tid is not driven here, is that also the case for the s signal?
            instantiations["accl_bd_inst"]["m_axis_eth_tx_data"].connect(
                design, coyote_interface["axis_rdma_0_src"]
            )

            instantiations["accl_bd_inst"]["s_axis_rdma_wr_req"].connect(
                design, coyote_interface["rdma_0_wr_req"]
            )
            instantiations["accl_bd_inst"]["s_axis_rdma_rd_req"].connect(
                design, coyote_interface["rdma_0_rd_req"]
            )
            instantiations["accl_bd_inst"]["m_axis_rdma_sq"].connect(
                design, coyote_interface["rdma_0_sq"]
            )

            if len(axilites_with_addr_width) == 0:
                instantiations["accl_bd_inst"]["S00_AXI_0"].connect(
                    design, coyote_interface["axi_ctrl"]
                )

            # FINN connections

            instantiations["finn_kernel_inst"][intf_names["s_axis"][0][0]].connect(
                design, coyote_interface["axis_host_3_sink"], is_self_input=False
            )

            instantiations["finn_kernel_inst"][intf_names["m_axis"][0][0]].connect(
                design, coyote_interface["axis_host_3_src"]
            )

            instantiations["finn_kernel_inst"][intf_names["s_axis"][1][0]].connect(
                design, instantiations["accl_bd_inst"]["ack_clients_1_0"]
            )

            instantiations["finn_kernel_inst"][intf_names["s_axis"][2][0]].connect(
                design, instantiations["accl_bd_inst"]["m_axis_krnl_0"]
            )

            instantiations["finn_kernel_inst"][intf_names["m_axis"][1][0]].connect(
                design, instantiations["accl_bd_inst"]["cmd_clients_1_0"]
            )

            instantiations["finn_kernel_inst"][intf_names["m_axis"][2][0]].connect(
                design, instantiations["accl_bd_inst"]["s_axis_krnl_0"]
            )

        else:
            instantiations["finn_kernel_inst"][intf_names["m_axis"][0][0]].connect(
                design, coyote_interface["axis_host_0_src"]
            )

            instantiations["finn_kernel_inst"][intf_names["s_axis"][0][0]].connect(
                design, coyote_interface["axis_host_0_sink"], is_self_input=False
            )

        model = model.transform(
            GenerateCoyoteProject(
                fpga_part=self.fpga_part,
                design=design,
                is_accl_mode=is_accl_mode,
                coyote_hw_build_dir=model.get_metadata_prop("coyote_hw_build"),
            )
        )
        model = model.transform(
            CoyoteUserLogic(design=design, user_logic_config=templates.user_logic_config)
        )
        model = model.transform(CoyoteCompile())

        return (model, False)
