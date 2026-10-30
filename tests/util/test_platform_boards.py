# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import os

from finn.platform import (
    PlatformError,
    ShellFlowType,
    get_board,
    get_boards,
    registry,
    resolve_target,
)
from finn.platform.boards import board_file_roots, compare_board_files, read_board_xml
from finn.platform.cli import main
from finn.platform.registry import DATA, load, read_source
from finn.platform.target import DRIVER_PLATFORMS, SHELL_AXI_PORT_WIDTH

ZYNQ, VERSAL = ShellFlowType.VIVADO_ZYNQ, ShellFlowType.VIVADO_VERSAL
VITIS, SLASH = ShellFlowType.VITIS_ALVEO, ShellFlowType.SLASH_ALVEO

# each shipped board of the built-in platform data
SHIPPED = sorted(name for name, row in read_source(DATA).boards.items() if not row.retired)


@pytest.fixture
def reloaded():
    registry._loaded.cache_clear()
    yield
    registry._loaded.cache_clear()


@pytest.fixture
def golden(tmp_path, monkeypatch):
    """A FINN_ROOT holding VCK190's golden reference files."""
    found = tmp_path / "deps" / "pynq_golden" / "vck190"
    found.mkdir(parents=True)
    for each in ("golden_ref.tcl", "golden_noc.ncr", "golden_routed.dcp"):
        (found / each).write_text("")
    monkeypatch.setenv("FINN_ROOT", str(tmp_path))
    return found


@pytest.mark.util
@pytest.mark.parametrize("name", SHIPPED)
def test_every_board_resolves(name, golden):
    row = get_board(name)
    shell = ShellFlowType[row.shell] if row.shell else None
    target = resolve_target(board=name, need_bitfile=shell is not None, vivado_release=(2025, 2))
    assert target.part == row.part and target.board.name == name
    assert row.part.lower().startswith(target.device.name.lower())
    assert target.shell_flow_type == shell
    assert target.axi_port_width == (row.axi_port_width or SHELL_AXI_PORT_WIDTH.get(shell))
    assert target.mem_type == (row.memory[0] if row.memory else None)
    assert target.vitis_platform == row.vitis_platform
    assert target.driver_platform == (DRIVER_PLATFORMS[shell] if shell else None)
    if shell == ZYNQ:
        assert target.zynq_type is not None


@pytest.mark.util
def test_a_part_alone_resolves_without_a_board():
    target = resolve_target(fpga_part="xc7z020clg400-1")
    assert target.board is None and target.shell_flow_type is None
    assert target.zynq_type == "zynq_7000" and target.mem_type is None
    target = resolve_target(fpga_part="xcu55c-fsvh2892-2L-e", shell_flow_type=VITIS)
    assert target.axi_port_width == 512 and target.vitis_platform is None
    assert target.mem_port is None  # no board: a Vitis DMA's memory port is not known


def refused(code, **stated):
    with pytest.raises(PlatformError) as found:
        resolve_target(**stated)
    assert found.value.code == code, str(found.value)
    return str(found.value)


@pytest.mark.util
def test_every_refusal(golden):
    message = refused("unknown-board", board="ZCU999")
    assert "Board 'ZCU999' is not in FINN's platform data" in message and "ZCU104" in message
    assert "Pynq-Z1" not in message.split("Boards:")[1]
    message = refused("retired-board", board="Pynq-Z2")
    assert "retired from official FINN support" in message and "another name" in message
    message = refused("board-part-mismatch", board="AUP-ZU3_8GB", fpga_part="xczu7ev-ffvc1156-2-e")
    assert "carries part xczu3eg-sfvc784-2-e, but fpga_part is xczu7ev-ffvc1156-2-e" in message
    refused("target-unstated")
    refused("target-unstated", shell_flow_type=ZYNQ)
    refused("unknown-part", fpga_part="xcmy100tffg1761-2")
    refused("board-shell-mismatch", board="U55C", shell_flow_type=ZYNQ)
    refused("board-shell-mismatch", board="AUP-ZU3_8GB", shell_flow_type=VITIS)
    refused("board-shell-mismatch", board="V80", shell_flow_type=VITIS)
    refused("board-shell-mismatch", board="VEK280", shell_flow_type=VERSAL)
    refused("shell-architecture-mismatch", fpga_part="xcu55c-fsvh2892-2L-e", shell_flow_type=ZYNQ)
    refused("shell-architecture-mismatch", fpga_part="xczu7ev-ffvc1156-2-e", shell_flow_type=VERSAL)
    # a part-only build of a shell flow that does not build for the part's architecture
    message = refused(
        "shell-architecture-mismatch", fpga_part="xcv80-lsva4737-2MHP-e-s", shell_flow_type=VITIS
    )
    assert "virtexuplus, virtexuplusHBM, virtexuplus58g" in message and "versal" in message
    refused(
        "shell-architecture-mismatch",
        fpga_part="xcv80-lsva4737-2MHP-e-s",
        shell_flow_type=VITIS,
        vitis_platform="xilinx_u55c_gen3x16_xdma_3_202210_1",
        need_bitfile=True,
    )
    refused("shell-architecture-mismatch", fpga_part="xcu55c-fsvh2892-2L-e", shell_flow_type=SLASH)
    refused("shell-architecture-mismatch", fpga_part="xczu7ev-ffvc1156-2-e", shell_flow_type=VITIS)
    # SLASH links against the V80's platform: another Versal part is refused
    message = refused(
        "shell-device-mismatch", fpga_part="xcvc1902-vsva2197-2MP-e-S", shell_flow_type=SLASH
    )
    assert "xcv80" in message and "xcvc1902" in message
    refused("no-shell", board="VEK280", need_bitfile=True)
    refused("no-shell", board="VEK280", need_shell=True)
    refused("no-shell", fpga_part="xczu7ev-ffvc1156-2-e", need_bitfile=True)
    refused(
        "board-required", fpga_part="xczu7ev-ffvc1156-2-e", shell_flow_type=ZYNQ, need_bitfile=True
    )
    refused(
        "board-required",
        fpga_part="xcv80-lsva4737-2MHP-e-s",
        shell_flow_type=SLASH,
        need_bitfile=True,
    )
    refused(
        "vitis-platform-required",
        fpga_part="xcu55c-fsvh2892-2L-e",
        shell_flow_type=VITIS,
        need_bitfile=True,
    )
    refused(
        "vitis-platform-unused",
        board="AUP-ZU3_8GB",
        vitis_platform="xilinx_u55c_gen3x16_xdma_3_202210_1",
    )
    # a Vitis platform with no shell flow at all is not dropped
    message = refused(
        "vitis-platform-unused",
        fpga_part="xczu3eg-sbva484-1-e",
        vitis_platform="my_platform",
    )
    assert "no shell flow" in message and "shell_flow_type=VITIS_ALVEO" in message
    message = refused(
        "vivado-too-old", board="AUP-ZU3_8GB", need_tools=True, vivado_release=(2023, 2)
    )
    assert "Board 'AUP-ZU3_8GB' needs Vivado 2024.1 or later, found 2023.2" in message
    refused("vivado-too-old", board="V80", need_tools=True, vivado_release=(2024, 1))
    refused("vivado-too-old", board="V80", need_bitfile=True, vivado_release=(2024, 2))
    refused(
        "vivado-too-old",
        fpga_part="xcv80-lsva4737-2MHP-e-s",
        shell_flow_type=SLASH,
        need_tools=True,
        vivado_release=(2024, 2),
    )
    # a stated minimum and an unknown release: refused, not passed
    message = refused("vivado-unknown", board="AUP-ZU3_8GB", need_tools=True)
    assert "needs Vivado 2024.1 or later" in message and "XILINX_VIVADO" in message
    refused("vivado-unknown", board="V80", need_tools=True)
    refused("vivado-unknown", board="AUP-ZU3_8GB", need_bitfile=True)
    refused("vivado-unknown", board="V80", need_bitfile=True)
    refused(
        "vivado-unknown",
        fpga_part="xcv80-lsva4737-2MHP-e-s",
        shell_flow_type=SLASH,
        need_tools=True,
    )
    for each in os.listdir(golden):
        os.remove(golden / each)
    message = refused("golden-missing", board="VCK190", need_bitfile=True)
    assert "golden_ref.tcl" in message and "FINN_VCK190_GOLDEN=1" in message


@pytest.mark.util
def test_golden_files_missing_for_a_board_of_ones_own(tmp_path, monkeypatch, reloaded):
    row = {
        "part": "xcvc1902-vsva2197-2MP-e-S",
        "shell": "VIVADO_VERSAL",
        "board_part": "vendor.com:myvck:part0:1.0",
        "axi_port_width": 512,
        "golden": True,
    }
    monkeypatch.setenv(
        "FINN_PLATFORM_PATH", str(user_platform_data(tmp_path / "mine", {"MyVCK": row}))
    )
    monkeypatch.setenv("FINN_ROOT", str(tmp_path / "root"))
    message = refused("golden-missing", board="MyVCK", need_bitfile=True)
    assert "FINN_VCK190_GOLDEN" not in message
    found = str(tmp_path / "root" / "deps" / "pynq_golden" / "myvck")
    assert f"golden_ref.tcl, golden_noc.ncr, golden_routed.dcp. Put them in {found}" in message


@pytest.mark.util
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB", "V80"])
def test_a_build_that_runs_no_tools_needs_no_vivado(board):
    assert resolve_target(board=board).board.name == board
    assert resolve_target(board=board, vivado_release=(2022, 2)).board.name == board


@pytest.mark.util
def test_what_is_not_needed_is_not_refused():
    # no bitfile: neither a preset, a Vitis platform, nor the shell's Vivado minimum
    resolve_target(board="V80", vivado_release=(2024, 2))
    resolve_target(fpga_part="xcu55c-fsvh2892-2L-e", shell_flow_type=VITIS)
    resolve_target(board="VCK190")  # no golden files, and no bitfile
    # no Vivado minimum stated: an unknown release is not checked
    resolve_target(board="ZCU104", need_bitfile=True)
    resolve_target(fpga_part="xcu55c-fsvh2892-2L-e", shell_flow_type=VITIS)
    # each shell flow builds for the architectures of its shipped boards
    resolve_target(fpga_part="xck26-sfvc784-2LV-c", shell_flow_type=ZYNQ)
    resolve_target(fpga_part="xczu28dr-ffvg1517-2-e", shell_flow_type=ZYNQ)
    resolve_target(fpga_part="xc7z020clg400-1", shell_flow_type=ZYNQ)
    resolve_target(fpga_part="xcu250-figd2104-2L-e", shell_flow_type=VITIS)
    # the same part spelled in another case is the board's part
    target = resolve_target(
        board="V80", fpga_part="xcv80-lsva4737-2MHP-e-S", vivado_release=(2025, 2)
    )
    assert target.part.endswith("-S")


def user_platform_data(directory, boards, name="mine"):
    directory.mkdir()
    manifest = {"name": name, "source": {"kind": "hand-written"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "boards.json").write_text(json.dumps(boards, indent=1))
    return directory


@pytest.mark.util
def test_platform_data_of_ones_own_adds_a_board(tmp_path, monkeypatch, reloaded):
    row = {
        "part": "xczu7ev-ffvc1156-2-e",
        "shell": "VIVADO_ZYNQ",
        "board_part": "vendor.com:myboard:part0:1.0",
        "axi_port_width": 128,
        "memory": ["DDR"],
    }
    monkeypatch.setenv(
        "FINN_PLATFORM_PATH", str(user_platform_data(tmp_path / "mine", {"MyBoard": row}))
    )
    target = resolve_target(board="MyBoard", need_bitfile=True)
    assert target.part == "xczu7ev-ffvc1156-2-e" and target.board.board_part == row["board_part"]


@pytest.mark.util
@pytest.mark.parametrize(
    "row, rule",
    [
        ({"part": "xczu7ev-ffvc1156-2-e", "shell": "ZYNQ"}, "shell is 'ZYNQ'"),
        ({"part": "xczu7ev-ffvc1156-2-e", "shell": "VIVADO_ZYNQ"}, "states its axi_port_width"),
        (
            {"part": "xcu55c-fsvh2892-2L-e", "shell": "VITIS_ALVEO", "axi_port_width": 512},
            "the VITIS_ALVEO shell states",
        ),
        (
            {"part": "xcu55c-fsvh2892-2L-e", "mem_port": "HBM[0]"},
            "mem_port applies to boards of the VITIS_ALVEO",
        ),
        (
            {"part": "xcu55c-fsvh2892-2L-e", "shell": "VITIS_ALVEO", "mem_port": "HBM0"},
            "not like HBM[0]",
        ),
        ({"part": "xczu7ev-ffvc1156-2-e", "vivado_min": "2024"}, "not a release"),
        ({"part": "xczu7ev-ffvc1156-2-e", "memory": ["SRAM"]}, "memory is"),
        ({"part": "xczu7ev-ffvc1156-2-e", "memory": [{}]}, "memory is [{}], not a list of"),
        ({"part": "xczu7ev-ffvc1156-2-e", "preset": "x"}, "unknown keys ['preset']"),
        (
            {
                "part": "xczu7ev-ffvc1156-2-e",
                "shell": "VIVADO_ZYNQ",
                "axi_port_width": 128,
                "board_part": "oops",
            },
            "board_part is 'oops', not a Vivado board preset like vendor:name:part0:version",
        ),
        (
            {
                "part": "xczu7ev-ffvc1156-2-e",
                "shell": "VIVADO_ZYNQ",
                "axi_port_width": 128,
                "board_part": "xilinx.com::part0:1.1",
            },
            "not a Vivado board preset",
        ),
        (
            {
                "part": "xcu55c-fsvh2892-2L-e",
                "shell": "VITIS_ALVEO",
                "memory": ["DDR"],
                "mem_port": "HBM[0]",
            },
            "mem_port HBM[0] is on HBM, which is not one of its memory ['DDR']",
        ),
    ],
)
def test_a_board_row_is_validated(tmp_path, capsys, row, rule):
    mine = user_platform_data(tmp_path / "mine", {"B": row})
    with pytest.raises(PlatformError) as found:
        read_source(mine)
    assert "boards.json:2: schema: board 'B': " in str(found.value)
    assert rule in str(found.value)
    assert main(["check", str(mine)]) == 1
    err = capsys.readouterr().err
    assert "boards.json:2: schema: board 'B': " in err and rule in err


@pytest.mark.util
@pytest.mark.parametrize(
    "text, where",
    [
        (
            '{\n "B": {"part": "xczu3eg-sbva484-1-e"},\n'
            ' "B": {"part": "xczu7ev-ffvc1156-2-e"}\n}\n',
            "boards.json:3: duplicate: key 'B' is stated twice in the top-level object",
        ),
        (
            '{\n "B": {\n  "part": "xczu3eg-sbva484-1-e",\n'
            '  "part": "xczu7ev-ffvc1156-2-e"\n }\n}\n',
            "boards.json:4: duplicate: key 'part' is stated twice in 'B'",
        ),
    ],
)
def test_a_board_or_field_stated_twice_is_refused(tmp_path, capsys, text, where):
    mine = user_platform_data(tmp_path / "mine", {})
    (mine / "boards.json").write_text(text)
    with pytest.raises(PlatformError, match=where):
        read_source(mine)
    assert main(["check", str(mine)]) == 1
    assert where in capsys.readouterr().err


@pytest.mark.util
@pytest.mark.parametrize(
    "row, rule",
    [
        ({"part": "xcnosuch1-ffg900-2"}, "part-undefined"),
        (
            {"part": "xcu55c-fsvh2892-2L-e", "shell": "VIVADO_ZYNQ", "axi_port_width": 128},
            "shell-architecture",
        ),
        (
            {"part": "xcv80-lsva4737-2MHP-e-s", "shell": "VITIS_ALVEO"},
            "shell-architecture: board B's shell VITIS_ALVEO does not build for its part's "
            "architecture versal",
        ),
        (
            {
                "part": "xc7z020clg400-1",
                "shell": "VIVADO_ZYNQ",
                "axi_port_width": 64,
                "ps_config": {"PSU__USE__M_AXI_GP0": "1"},
            },
            "ps-config",
        ),
        (
            {"part": "xcvc1902-vsva2197-2MP-e-S", "shell": "SLASH_ALVEO", "memory": ["DDR"]},
            "shell-device: board B's shell SLASH_ALVEO builds for the device xcv80 only",
        ),
    ],
)
def test_a_board_row_is_checked_against_the_union(tmp_path, row, rule):
    with pytest.raises(PlatformError, match=rule):
        load([DATA, user_platform_data(tmp_path / "mine", {"B": row})])


@pytest.mark.util
def test_a_board_stated_differently_in_ones_own_platform_data_is_refused(tmp_path):
    row = {"part": "xczu7ev-ffvc1156-2-e", "memory": ["DDR"]}
    with pytest.raises(PlatformError, match="conflict: board ZCU104 is defined differently"):
        load([DATA, user_platform_data(tmp_path / "mine", {"ZCU104": row})])


def write_board_xml(directory, vendor, name, version, part):
    directory.mkdir(parents=True)
    (directory / "board.xml").write_text(
        f'<board schema_version="2.2" vendor="{vendor}" name="{name}">'
        f"<file_version>{version}</file_version><components>"
        f'<component name="part0" type="fpga" part_name="{part}"/>'
        "</components></board>"
    )


@pytest.mark.util
def test_rows_are_compared_with_their_board_files(tmp_path):
    root = tmp_path / "board_files"
    write_board_xml(root / "zcu104" / "1.1", "xilinx.com", "zcu104", "1.1", "xczu7ev-ffvc1156-2-e")
    write_board_xml(root / "zcu102" / "3.3", "xilinx.com", "zcu102", "3.3", "xczu9eg-ffvb1156-1-e")
    write_board_xml(
        root / "Ultra96v2" / "1.2", "avnet.com", "Ultra96v2", "1.2", "xczu3eg-sbva484-1-i"
    )
    assert read_board_xml(root / "zcu104" / "1.1" / "board.xml") == (
        "xilinx.com:zcu104:part0:1.1",
        "xczu7ev-ffvc1156-2-e",
    )
    boards = {
        name: get_board(name) for name in ("ZCU104", "ZCU102", "Ultra96-V2", "ZCU111", "U55C")
    }
    found = {board: status for board, status, _ in compare_board_files(boards, [root])}
    assert found == {
        "ZCU104": "match",
        "ZCU102": "mismatch",
        "Ultra96-V2": "match",
        "ZCU111": "absent",
    }


@pytest.mark.util
def test_every_board_row_states_what_its_board_file_states():
    roots = board_file_roots()
    if not roots:
        pytest.skip("no board files: neither $FINN_ROOT/deps/board_files nor $XILINX_VIVADO")
    found = compare_board_files(get_boards(), roots)
    assert not [each for each in found if each[1] == "mismatch"], found
    if not [each for each in found if each[1] == "match"]:
        pytest.skip(f"no board file found for any board in {roots}")


@pytest.mark.util
def test_finn_platform_shows_and_checks_boards(tmp_path, capsys, monkeypatch):
    assert main(["show", "ZCU111"]) == 0
    out = capsys.readouterr().out
    assert "board ZCU111 (from built-in)" in out
    assert "board_part      xilinx.com:zcu111:part0:1.4" in out
    assert "device        xczu28dr (from built-in)" in out
    board_files = tmp_path / "root" / "deps" / "board_files"
    write_board_xml(
        board_files / "zcu111" / "1.4", "xilinx.com", "zcu111", "1.4", "xczu28dr-ffvg1517-2-e"
    )
    monkeypatch.setenv("FINN_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("XILINX_VIVADO", raising=False)
    assert main(["check", str(DATA)]) == 0
    out = capsys.readouterr().out
    assert "board ZCU111: match" in out and "board ZCU104: absent" in out
    row = {
        "part": "xczu28dr-ffvg1517-1-e",
        "shell": "VIVADO_ZYNQ",
        "board_part": "xilinx.com:zcu111:part0:1.4",
        "axi_port_width": 128,
    }
    assert main(["check", str(user_platform_data(tmp_path / "mine", {"MyZCU111": row}))]) == 1
    out = capsys.readouterr().out
    assert "board MyZCU111: mismatch: its row states part xczu28dr-ffvg1517-1-e" in out
