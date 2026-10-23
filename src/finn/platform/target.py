# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one resolution of a build's target: from what a build states (a board, a
part, a shell flow, a Vitis platform) to a ``Target``. What cannot be resolved is
refused with a ``PlatformError`` whose ``code`` names the rule; FINN's documentation
lists the codes (Targets, parts, and platform data).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from finn.platform.boards import get_board, golden_dir, missing_golden_files
from finn.platform.catalog import get_part_architecture, get_part_device
from finn.platform.registry import (
    SHELL_ARCHITECTURES,
    SHELL_DEVICES,
    Architecture,
    Board,
    Device,
    PlatformError,
    platform_data,
)


class ShellFlowType(str, Enum):
    """For builds that produce a bitfile, select the shell flow that will integrate
    the FINN-generated accelerator."""

    VIVADO_ZYNQ = "vivado_zynq"
    VIVADO_VERSAL = "vivado_versal"
    VITIS_ALVEO = "vitis_alveo"
    SLASH_ALVEO = "slash_alveo"


SHELL_VIVADO_MIN = {ShellFlowType.SLASH_ALVEO: (2025, 1)}
#: the width in bits of the memory port of the shells that state it for every board
SHELL_AXI_PORT_WIDTH = {ShellFlowType.VITIS_ALVEO: 512, ShellFlowType.SLASH_ALVEO: 512}
#: the shell flows that build on a board's Vivado preset
BOARD_SHELLS = (ShellFlowType.VIVADO_ZYNQ, ShellFlowType.VIVADO_VERSAL)
DRIVER_PLATFORMS = {
    ShellFlowType.VIVADO_ZYNQ: "pynq-iodma",
    ShellFlowType.VIVADO_VERSAL: "pynq-iodma",
    ShellFlowType.VITIS_ALVEO: "vitis-xrt",
    ShellFlowType.SLASH_ALVEO: "slash-vrt",
}
ZYNQ_TYPES = {"zynq7000": "zynq_7000", "zynquplus": "zynq_us+"}

_DOCS = "See FINN's documentation: Targets, parts, and platform data."


@dataclass(frozen=True)
class Target:
    """A build's resolved target. ``part`` is as stated, or the board's; ``board`` is
    set only if a board is named; ``shell_flow_type`` ``None`` means no shell;
    ``mem_type`` is the board's preferred memory and ``mem_port`` its default Vitis
    memory port."""

    part: str
    device: Device
    architecture: Architecture
    board: Optional[Board]
    shell_flow_type: Optional[ShellFlowType]
    vitis_platform: Optional[str]
    mem_type: Optional[str]
    mem_port: Optional[str]
    axi_port_width: Optional[int]
    driver_platform: Optional[str]

    @property
    def zynq_type(self) -> Optional[str]:
        """``zynq_us+`` or ``zynq_7000`` for a Zynq part, else ``None``."""
        return ZYNQ_TYPES.get(self.architecture.class_)


def _release(version: Tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def _board(name: str) -> Board:
    board = get_board(name)
    if board.retired:
        raise PlatformError(
            "retired-board",
            f"Board {name!r} is retired: {board.retired}. The build flow is unchanged, so "
            "to build for it anyway, add a row for it under another name, without "
            '"retired", to a boards.json of your own, in a directory listed in '
            "FINN_PLATFORM_PATH. " + _DOCS,
        )
    return board


def _shell(board: Optional[Board], shell_flow_type) -> Optional[ShellFlowType]:
    stated = None if shell_flow_type is None else ShellFlowType(shell_flow_type)
    boards = None if board is None or board.shell is None else ShellFlowType[board.shell]
    if board is not None and stated is not None and stated != boards:
        raise PlatformError(
            "board-shell-mismatch",
            f"Board {board.name!r} is built with the "
            + (f"{boards.name} shell flow" if boards else "no shell flow")
            + f", but shell_flow_type is {stated.name}. Leave shell_flow_type unset to use "
            "the board's, or name another board.",
        )
    return stated if stated is not None else boards


def _require_shell(board, shell, need_shell, need_bitfile):
    if (need_shell or need_bitfile) and shell is None:
        what = f"board {board.name!r} has no shell flow" if board else "no shell flow is stated"
        raise PlatformError(
            "no-shell",
            f"A bitfile or driver needs a shell flow, but {what}. Remove BITFILE and the "
            "drivers from generate_outputs, or name a board or shell_flow_type with one.",
        )
    if not need_bitfile:
        return
    if shell in BOARD_SHELLS + (ShellFlowType.SLASH_ALVEO,) and board is None:
        raise PlatformError(
            "board-required",
            f"A bitfile with the {shell.name} shell flow needs a board: set board to a "
            f"board of the {shell.name} shell flow (finn-platform show <board>).",
        )
    if shell in BOARD_SHELLS and board.board_part is None:
        raise PlatformError(
            "no-board-preset",
            f"Board {board.name!r} states no Vivado board preset (board_part), which the "
            f"{shell.name} shell flow needs. Add it to the board's row. " + _DOCS,
        )


def _part(board: Optional[Board], fpga_part: Optional[str]) -> str:
    if board is not None and fpga_part is not None and fpga_part.lower() != board.part.lower():
        raise PlatformError(
            "board-part-mismatch",
            f"Board {board.name!r} carries part {board.part}, but fpga_part is "
            f"{fpga_part}. Set only one of them, or name the board of that part.",
        )
    part = fpga_part if fpga_part is not None else (board.part if board else None)
    if part is None:
        raise PlatformError(
            "target-unstated",
            "The build states no target: set fpga_part, or board for a board FINN knows.",
        )
    return part


def _vitis_platform(board, shell, vitis_platform, need_bitfile) -> Optional[str]:
    if vitis_platform is not None and shell is None:
        raise PlatformError(
            "vitis-platform-unused",
            f"vitis_platform is {vitis_platform}, but the build has no shell flow to use "
            "it. Set shell_flow_type=VITIS_ALVEO, or board to an Alveo board, or remove "
            "vitis_platform.",
        )
    if vitis_platform is not None and shell != ShellFlowType.VITIS_ALVEO:
        raise PlatformError(
            "vitis-platform-unused",
            f"vitis_platform is {vitis_platform}, but the {shell.name} shell flow uses "
            "none. Remove vitis_platform, or build with VITIS_ALVEO.",
        )
    if shell != ShellFlowType.VITIS_ALVEO:
        return None
    found = vitis_platform or (board.vitis_platform if board else None)
    if found is None and need_bitfile:
        raise PlatformError(
            "vitis-platform-required",
            "A bitfile with the VITIS_ALVEO shell flow needs a Vitis platform: set "
            "vitis_platform, or board to an Alveo board with a default platform.",
        )
    return found


def _vivado(board, shell, stated_shell, need_tools, need_bitfile, vivado_release):
    """For a build that runs AMD tools, refuse a Vivado release below the board's or the
    shell's minimum, or an unknown release where a minimum applies."""
    if not (need_tools or need_bitfile):
        return
    minimums = []
    if board is not None and board.vivado_min is not None:
        minimums.append((board.vivado_min, f"board {board.name!r}"))
    if shell in SHELL_VIVADO_MIN and (stated_shell is not None or need_bitfile):
        minimums.append((SHELL_VIVADO_MIN[shell], f"the {shell.name} shell flow"))
    for minimum, what in minimums:
        if vivado_release is None:
            raise PlatformError(
                "vivado-unknown",
                f"{what[0].upper() + what[1:]} needs Vivado {_release(minimum)} or later, "
                "but the Vivado release is unknown: set XILINX_VIVADO to the Vivado "
                "installation (source its settings64.sh).",
            )
        if tuple(vivado_release) < minimum:
            raise PlatformError(
                "vivado-too-old",
                f"{what[0].upper() + what[1:]} needs Vivado {_release(minimum)} or later, "
                f"found {_release(vivado_release)}. Upgrade Vivado.",
            )


def _golden_missing(board: str, missing) -> str:
    found = golden_dir(board)
    if board == "VCK190":
        return (
            f"{board} golden reference files missing from {found}: {', '.join(missing)}. "
            "Launch the container with FINN_VCK190_GOLDEN=1 ./run-docker.sh to build them "
            "(needs Vivado, takes a while on first launch)."
        )
    return (
        f"Board {board!r} builds on a golden reference design, whose files are missing: "
        f"{', '.join(missing)}. Put them in {found}: the design's block design Tcl "
        "(golden_ref.tcl), NoC solution (golden_noc.ncr) and routed checkpoint "
        "(golden_routed.dcp)."
    )


_resolved = {}


def _resolve(
    board, fpga_part, shell_flow_type, vitis_platform, need_shell, need_tools, need_bitfile, vivado
):
    row = None if board is None else _board(board)
    shell = _shell(row, shell_flow_type)
    _require_shell(row, shell, need_shell, need_bitfile)
    part = _part(row, fpga_part)
    device = get_part_device(part)
    architecture = get_part_architecture(part)
    builds = SHELL_ARCHITECTURES[shell.name] if shell is not None else None
    if builds is not None and architecture.name not in builds:
        raise PlatformError(
            "shell-architecture-mismatch",
            f"The {shell.name} shell flow builds for parts of the {', '.join(builds)} "
            f"architectures, but {part} is of architecture {architecture.name}.",
        )
    devices = SHELL_DEVICES.get(shell.name) if shell is not None else None
    if devices is not None and device.name.lower() not in devices:
        raise PlatformError(
            "shell-device-mismatch",
            f"The {shell.name} shell flow builds for the device {', '.join(devices)} "
            f"only, but {part} is on device {device.name}.",
        )
    platform = _vitis_platform(row, shell, vitis_platform, need_bitfile)
    _vivado(row, shell, shell_flow_type, need_tools, need_bitfile, vivado)
    width = row.axi_port_width if row is not None else None
    return Target(
        part=part,
        device=device,
        architecture=architecture,
        board=row,
        shell_flow_type=shell,
        vitis_platform=platform,
        mem_type=row.memory[0] if row is not None and row.memory else None,
        mem_port=row.mem_port if row is not None else None,
        axi_port_width=width if width is not None else SHELL_AXI_PORT_WIDTH.get(shell),
        driver_platform=DRIVER_PLATFORMS.get(shell),
    )


def resolve_target(
    board: Optional[str] = None,
    fpga_part: Optional[str] = None,
    shell_flow_type=None,
    vitis_platform: Optional[str] = None,
    need_shell: bool = False,
    need_bitfile: bool = False,
    vivado_release: Optional[Tuple[int, int]] = None,
    need_tools: bool = False,
) -> Target:
    """Resolve a build's target from its ``board``, ``fpga_part``,
    ``shell_flow_type`` and ``vitis_platform``. ``need_shell`` asks for a shell flow
    (a driver), ``need_bitfile`` for everything a bitfile needs. ``need_tools`` says
    the build runs AMD tools (HLS, Vivado or Vitis; a bitfile always does): then
    ``vivado_release`` (year, minor) is checked against the board's and the shell's
    minimums (``None``: unknown, refused where a minimum applies). A build that runs
    no tools, such as estimate reports alone, needs no Vivado. Raises
    ``PlatformError`` with the refusal's ``code``. Results are cached per loaded
    platform data; the golden reference files are checked on every call."""
    shell = None if shell_flow_type is None else ShellFlowType(shell_flow_type)
    vivado = None if vivado_release is None else tuple(vivado_release)
    key = (id(platform_data()), board, fpga_part, shell, vitis_platform, need_shell)
    key += (need_tools, need_bitfile, vivado)
    if key not in _resolved:
        _resolved[key] = _resolve(
            board, fpga_part, shell, vitis_platform, need_shell, need_tools, need_bitfile, vivado
        )
    target = _resolved[key]
    if need_bitfile and target.board is not None and target.board.golden:
        missing = missing_golden_files(target.board.name)
        if missing:
            raise PlatformError("golden-missing", _golden_missing(target.board.name, missing))
    return target
