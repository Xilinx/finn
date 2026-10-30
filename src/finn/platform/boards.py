# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The boards FINN builds for: their rows in the platform data (``boards.json``,
see ``finn.platform.registry``), the golden reference files of the Versal shell, and a
comparison of each row with the board's own definition (``board.xml``).
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from finn.platform.registry import Board, PlatformError, platform_data

#: the files of a golden reference design the Versal shell builds on
GOLDEN_FILES = ("golden_ref.tcl", "golden_noc.ncr", "golden_routed.dcp")


def get_boards() -> Dict[str, Board]:
    """Every board of the loaded platform data, by name."""
    return platform_data().boards


def get_board(name: str) -> Board:
    """The row of the board ``name``. Raises ``PlatformError`` (``unknown-board``)
    for a board no platform data holds, listing the boards; a retired board's row is
    returned (``finn.platform.resolve_target`` refuses it for a build)."""
    boards = get_boards()
    if name not in boards:
        names = ", ".join(sorted(each for each in boards if not boards[each].retired))
        raise PlatformError(
            "unknown-board",
            f"Board {name!r} is not in FINN's platform data (from: "
            f"{platform_data().describe()}). Boards: {names}. To add a board, add its row "
            "to a boards.json of your own, in a directory listed in FINN_PLATFORM_PATH "
            "(see FINN's documentation: "
            "Targets, parts, and platform data).",
        )
    return boards[name]


def golden_dir(board: str) -> str:
    """The directory of the golden reference design the Versal shell builds on for
    ``board``: ``$FINN_ROOT/deps/pynq_golden/<board>``."""
    root = os.environ.get("FINN_ROOT")
    if root is None:
        raise PlatformError("golden-missing", "FINN_ROOT is not set, so no golden files")
    return os.path.join(root, "deps", "pynq_golden", board.lower())


def missing_golden_files(board: str) -> List[str]:
    """The golden reference files the Versal shell needs for ``board`` that are
    missing from ``golden_dir(board)``."""
    found = golden_dir(board)
    return [each for each in GOLDEN_FILES if not os.path.isfile(os.path.join(found, each))]


def board_file_roots() -> List[Path]:
    """The directories Vivado board files are read from, where present: FINN's
    fetched ones (``$FINN_ROOT/deps/board_files``) and Vivado's bundled store
    (``$XILINX_VIVADO/data/xhub/boards/XilinxBoardStore/boards``)."""
    roots = []
    if os.environ.get("FINN_ROOT"):
        roots.append(Path(os.environ["FINN_ROOT"]) / "deps" / "board_files")
    if os.environ.get("XILINX_VIVADO"):
        vivado = Path(os.environ["XILINX_VIVADO"]) / "data"
        roots.append(vivado / "xhub" / "boards" / "XilinxBoardStore" / "boards")
        roots.append(vivado / "boards" / "board_files")
    return [each for each in roots if each.is_dir()]


def read_board_xml(path) -> Optional[Tuple[str, str]]:
    """The preset id (``vendor:name:part0:file_version``) and part a ``board.xml``
    states, or ``None`` if it states neither."""
    try:
        board = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    version = board.findtext("file_version")
    fpga = [each for each in board.iter("component") if each.get("name") == "part0"]
    if board.tag != "board" or version is None or not fpga:
        return None
    preset = f"{board.get('vendor')}:{board.get('name')}:part0:{version.strip()}"
    return preset, fpga[0].get("part_name")


def find_board_xml(board_part: str, roots) -> Optional[Tuple[Path, str]]:
    """The ``board.xml`` stating the preset ``board_part`` under ``roots``, and the
    part it states, or ``None``. Vivado compares preset ids without case."""
    name = board_part.split(":")[1]
    for root in roots:
        for path in sorted(root.rglob(f"{name}/**/board.xml", case_sensitive=False)):
            found = read_board_xml(path)
            if found is not None and found[0].lower() == board_part.lower():
                return path, found[1]
    return None


def compare_board_files(boards, roots=None) -> List[Tuple[str, str, str]]:
    """Compare each board row with a preset (``board_part``) against its
    ``board.xml`` under ``roots`` (by default ``board_file_roots()``): a list of
    ``(board, status, detail)``, the status ``match``, ``mismatch`` (the part
    differs) or ``absent`` (no board file states the preset)."""
    roots = board_file_roots() if roots is None else roots
    found = []
    for name, each in sorted(boards.items()):
        if each.board_part is None:
            continue
        located = find_board_xml(each.board_part, roots)
        if located is None:
            found.append((name, "absent", f"no board.xml states {each.board_part}"))
        elif located[1] is None or located[1].lower() != each.part.lower():
            found.append(
                (
                    name,
                    "mismatch",
                    f"its row states part {each.part}, its board file {located[0]} "
                    f"states {located[1]}",
                )
            )
        else:
            found.append((name, "match", str(located[0])))
    return found
