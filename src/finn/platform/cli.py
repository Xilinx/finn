# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``finn-platform``: inspect FINN's registered target platforms and add your own
(FINN's documentation: Targets, parts, and platform data).
"""

import argparse
import sys
from pathlib import Path

from finn.platform import extract as extraction
from finn.platform.boards import board_file_roots, compare_board_files
from finn.platform.catalog import get_part_device
from finn.platform.registry import (
    DATA,
    PlatformError,
    platform_data,
    read_source,
    union,
)


def _same(first, second) -> bool:
    return Path(first).resolve() == Path(second).resolve()


def _size(source) -> str:
    return (
        f"{len(source.architectures)} architectures, {len(source.devices)} devices, "
        f"{len(source.boards)} boards"
    )


def _resources(each) -> str:
    return f"LUT {each.lut}, FF {each.ff}, BRAM18 {each.bram18}, URAM {each.uram}, DSP {each.dsp}"


def _defined_in(data, kind, key) -> str:
    return ", ".join(data.defined_in.get((kind, key), ()))


def show_part(data, part):
    device = get_part_device(part)
    architecture = data.architectures[device.architecture]
    print(f"part {part}")
    print(
        f"  device        {device.name} (from {_defined_in(data, 'device', device.name.lower())})"
    )
    print(
        f"  architecture  {architecture.name}: {architecture.dsp}, class "
        f"{architecture.class_ or 'none'}, UltraRAM {'yes' if architecture.uram else 'no'} "
        f"(from {_defined_in(data, 'architecture', architecture.name)})"
    )
    print(f"  family        {device.family}")
    print(f"  resources     {_resources(device.resources)}")
    if len(device.slrs) > 1:
        cap = " (each SLR's site capacity; the totals cap their sum)" if device.capped else ""
        print(f"  SLRs          {len(device.slrs)}{cap}")
        for index, slr in enumerate(device.slrs):
            print(f"    SLR{index}        {_resources(slr)}")


def show_board(data, board):
    print(f"board {board.name} (from {_defined_in(data, 'board', board.name)})")
    for key, value in vars(board).items():
        if key == "name" or value in (None, False, (), {}):
            continue
        if isinstance(value, dict):
            value = ", ".join(f"{name}={each}" for name, each in value.items())
        elif isinstance(value, tuple):
            value = ".".join(map(str, value)) if key == "vivado_min" else ", ".join(value)
        print(f"  {key:<15} {value}")


def show(name) -> int:
    data = platform_data()
    if name in data.boards:
        show_board(data, data.boards[name])
        name = data.boards[name].part
    show_part(data, name)
    return 0


def check(directories) -> int:
    """Validate each platform data directory alone, and compare its board rows with their
    board files where present; then validate all of them together with FINN's built-in
    platform data, in the given order, as a build loads them."""
    builtin = read_source(DATA)
    roots = board_file_roots()
    failed = False
    together = []
    for directory in directories:
        source = read_source(directory)
        if not _same(directory, DATA):
            together.append(source)
        print(f"OK: {source.describe()} in {directory}: {_size(source)}")
        if not source.boards:
            continue
        if not roots:
            print("  board files: none found ($FINN_ROOT/deps/board_files, $XILINX_VIVADO)")
        for board, status, detail in compare_board_files(source.boards, roots) if roots else ():
            failed |= status == "mismatch"
            print(f"  board {board}: {status}: {detail}")
    union([builtin] + together)
    if together:
        names = ", ".join(each.name for each in together)
        print(f"OK: together with the built-in platform data, in this order: {names}")
    if failed:
        print("finn-platform: board-file-mismatch: a board row and its board.xml differ")
        return 1
    return 0


def sources() -> int:
    for source in platform_data().sources:
        print(f"{source.describe():40} {source.path}  ({_size(source)})")
    return 0


def extract(options) -> int:
    data = platform_data()
    architectures = dict(data.architectures)
    if (Path(options.out) / "architectures.json").is_file():
        architectures.update(read_source(options.out).architectures)
    known = [
        device.name
        for source in data.sources
        if not _same(source.path, options.out)
        for device in source.devices.values()
    ]
    counts, left_out = extraction.extract(
        options.parts,
        options.out,
        architectures,
        known,
        work=options.work,
        from_logs=options.from_logs,
        jobs=options.jobs,
        name=options.name,
    )
    written = read_source(options.out)
    union([each for each in data.sources if not _same(each.path, options.out)] + [written])
    print(f"wrote {written.describe()} to {options.out}: {counts}")
    if left_out:
        print(
            f"left out the parts of architectures no platform data defines: "
            f"{', '.join(left_out)} (add them to an architectures.json in a directory of "
            "FINN_PLATFORM_PATH to include them)"
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="finn-platform",
        description="Inspect FINN's registered target platforms and add your own.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    extracting = commands.add_parser("extract", help="extract platform data from Vivado's parts")
    extracting.add_argument("--parts", required=True, help="a Vivado part pattern, e.g. 'xc7v*'")
    extracting.add_argument("--out", required=True, type=Path, help="a platform data directory")
    extracting.add_argument("--name", help="the data's name (default: the directory's)")
    extracting.add_argument("--jobs", type=int, default=1, help="Vivado processes at once")
    extracting.add_argument("--work", type=Path, help="where to keep Vivado's logs")
    extracting.add_argument("--from-logs", type=Path, help="derive from these logs, no Vivado")
    checking = commands.add_parser(
        "check",
        help="validate platform data directories, each and together, as a build loads them",
    )
    checking.add_argument("directories", nargs="+", type=Path)
    showing = commands.add_parser("show", help="what FINN resolves for a part or a board")
    showing.add_argument("name")
    commands.add_parser("sources", help="where the loaded platform data comes from")
    options = parser.parse_args(argv)
    try:
        if options.command == "extract":
            return extract(options)
        if options.command == "check":
            return check(options.directories)
        if options.command == "show":
            return show(options.name)
        return sources()
    except PlatformError as error:
        print(f"finn-platform: {error.code}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
