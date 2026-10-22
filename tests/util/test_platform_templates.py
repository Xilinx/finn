# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import re

from finn.platform import PlatformError, get_board, get_part_architecture, registry
from finn.platform.registry import DATA, read_source
from finn.platform.target import ZYNQ_TYPES
from finn.transformation.fpgadataflow import templates
from finn.transformation.fpgadataflow.make_pynq_proj import (
    PYNQBuild,
    board_template_values,
)

# the built-in boards of the Zynq shell, retired ones included
ZYNQ_BOARDS = sorted(
    name for name, row in read_source(DATA).boards.items() if row.shell == "VIVADO_ZYNQ"
)
# what MakePYNQProject substitutes itself, not from the board
OWN = {"@FREQ_MHZ@", "@NUM_AXILITE@", "@NUM_AXIMM@", "@CONFIG@", "@ENABLE_DEBUG@"}
OWN |= {"@NUM_WORKERS@", "@GOLDEN_DIR@"}


def fill(template, values):
    for key, value in values.items():
        template = template.replace(key, str(value))
    return template


@pytest.mark.util
@pytest.mark.parametrize("board", ZYNQ_BOARDS)
def test_the_zynq_template_takes_the_boards_preset_zynq_type_and_ps_settings(board):
    row = get_board(board)
    template = templates.custom_zynq_shell_template
    tcl = fill(template, board_template_values(board))
    assert set(re.findall(r"@[A-Z_]+@", tcl)) == OWN - {"@GOLDEN_DIR@"}
    assert f"set BOARD_PART {row.board_part}\n" in tcl
    assert "set_property board_part $BOARD_PART [current_project]" in tcl
    zynq_type = ZYNQ_TYPES[get_part_architecture(row.part).class_]
    assert f'set ZYNQ_TYPE "{zynq_type}"\n' in tcl
    assert "Unrecognized board" not in tcl
    # the PS settings the board's row adds, and no others
    added = [line for line in tcl.splitlines() if "CONFIG." in line]
    added = [line.strip() for line in added if line not in template.splitlines()]
    assert added == [
        "set_property -dict [list CONFIG.%s {%s}] [get_bd_cells zynq_ps]" % each
        for each in row.ps_config.items()
    ]


@pytest.mark.util
def test_the_versal_template_takes_the_boards_preset():
    tcl = fill(templates.custom_versal_shell_template, board_template_values("VCK190"))
    assert set(re.findall(r"@[A-Z_]+@", tcl)) == OWN
    assert "set_property board_part xilinx.com:vck190:part0:3.4 [current_project]" in tcl


@pytest.mark.util
@pytest.mark.parametrize(
    "board, code",
    [
        ("U55C", "board-shell-mismatch"),
        ("VEK280", "board-shell-mismatch"),
        ("ZCU999", "unknown-board"),
    ],
)
def test_a_board_the_pynq_shells_cannot_build_is_refused(board, code):
    with pytest.raises(PlatformError) as refused:
        PYNQBuild(board, 10)
    assert refused.value.code == code


@pytest.mark.util
def test_a_board_without_a_preset_is_refused_before_vivado(tmp_path, monkeypatch):
    row = {"part": "xczu7ev-ffvc1156-2-e", "shell": "VIVADO_ZYNQ", "axi_port_width": 128}
    (tmp_path / "boards.json").write_text(json.dumps({"MyBoard": row}))
    registry._loaded.cache_clear()
    monkeypatch.setenv("FINN_PLATFORM_PATH", str(tmp_path))
    try:
        with pytest.raises(PlatformError) as refused:
            PYNQBuild("MyBoard", 10)
        assert refused.value.code == "no-board-preset"
    finally:
        registry._loaded.cache_clear()
