# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The registry of FINN's target platforms: loads and validates the platform data
directories (FINN's built-in ``finn/platform/data``, then those listed in
``FINN_PLATFORM_PATH``) and takes their union. The files are described in FINN's
documentation: Targets, parts, and platform data.
"""

import functools
import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA = Path(__file__).resolve().parent / "data"
PATH_VARIABLE = "FINN_PLATFORM_PATH"

DSP_BLOCKS = ("DSP48E1", "DSP48E2", "DSP58")
ARCHITECTURE_CLASSES = ("versal", "zynq7000", "zynquplus")
RESOURCE_NAMES = ("lut", "ff", "bram18", "uram", "dsp")
SOURCE_KINDS = ("vivado", "hand-written")
SHELLS = ("VIVADO_ZYNQ", "VIVADO_VERSAL", "VITIS_ALVEO", "SLASH_ALVEO")
MEMORIES = ("DDR", "HBM")
SHELL_ARCHITECTURES = {
    "VIVADO_ZYNQ": ("zynq", "zynquplus", "zynquplusRFSOC"),
    "VIVADO_VERSAL": ("versal",),
    "VITIS_ALVEO": ("virtexuplus", "virtexuplusHBM", "virtexuplus58g"),
    "SLASH_ALVEO": ("versal",),
}
#: the devices a shell flow builds for, where it builds for some devices of its
#: architectures only: SLASH links against the V80's platform
SHELL_DEVICES = {"SLASH_ALVEO": ("xcv80",)}


class PlatformError(ValueError):
    """A platform fact FINN cannot resolve, or platform data it cannot load. ``code``
    names the rule that refused, for example ``unknown-part`` or
    ``invalid-platform-data``."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Resources:
    """Counts of a device's or an SLR's resources: CLB LUTs, CLB registers, block RAM
    in RAMB18 halves, URAM288 blocks and DSP slices."""

    lut: int
    ff: int
    bram18: int
    uram: int
    dsp: int


@dataclass(frozen=True)
class Architecture:
    """One of Vivado's ARCHITECTURE values. ``class_`` is the processing system class
    (``versal``, ``zynq7000``, ``zynquplus`` or ``None``)."""

    name: str
    dsp: str
    class_: Optional[str]
    uram: bool


@dataclass(frozen=True)
class Device:
    """A device: its ``name`` as Vivado spells it, Vivado's ``architecture`` and
    ``family``, its totals (``resources``) and each SLR's resources (``slrs``). On a
    ``capped`` device (a reduced die of several SLRs) each SLR states its site
    capacity and the totals cap their sum."""

    name: str
    architecture: str
    family: str
    resources: Resources
    slrs: Tuple[Resources, ...]
    capped: bool


@dataclass(frozen=True)
class Board:
    """A board's row in ``boards.json``. ``vivado_min`` is a (year, minor) tuple."""

    name: str
    part: str
    shell: Optional[str] = None
    board_part: Optional[str] = None
    vitis_platform: Optional[str] = None
    axi_port_width: Optional[int] = None
    memory: Tuple[str, ...] = ()
    mem_port: Optional[str] = None
    vivado_min: Optional[Tuple[int, int]] = None
    golden: bool = False
    retired: Optional[str] = None
    ps_config: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PlatformSource:
    """One loaded directory of platform data: its name, path, origin (the manifest's
    ``source``) and facts. ``devices`` are keyed by their lower-cased names."""

    name: str
    path: Path
    origin: dict
    counts: dict
    architectures: Dict[str, Architecture]
    devices: Dict[str, Device]
    boards: Dict[str, Board] = field(default_factory=dict)

    def describe(self) -> str:
        """The name and origin, for example ``built-in [Vivado 2025.2]``."""
        if self.origin.get("kind") == "vivado":
            return f"{self.name} [Vivado {self.origin['release']}]"
        return f"{self.name} [hand-written]"


@dataclass(frozen=True)
class PlatformData:
    """The union of the loaded platform data, in load order. ``defined_in`` names the
    sources that define each fact, keyed by ``(kind, key)``, for example
    ``("device", "xczu3eg")``."""

    sources: Tuple[PlatformSource, ...]
    architectures: Dict[str, Architecture]
    devices: Dict[str, Device]
    defined_in: Dict[Tuple[str, str], Tuple[str, ...]]
    boards: Dict[str, Board] = field(default_factory=dict)
    _found: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def find_device(self, part: str) -> Optional[Device]:
        """The device of ``part``: the longest device name that prefixes the part's
        name, compared without case (``xczu3eg-sbva484-1-e`` is on ``xczu3eg``, and
        ``xcku085_CIV-flva1517-1-i`` on ``xcku085_CIV``, not ``xcku085``), or
        ``None``."""
        lowered = part.lower()
        if lowered not in self._found:
            name = max((n for n in self.devices if lowered.startswith(n)), key=len, default=None)
            self._found[lowered] = None if name is None else self.devices[name]
        return self._found[lowered]

    def describe(self) -> str:
        """Where the loaded platform data comes from, for example ``built-in
        [Vivado 2025.2], my-devices [hand-written]``."""
        return ", ".join(each.describe() for each in self.sources)


class _Problems:
    def __init__(self, name: str, path: Path):
        self.name, self.path, self.found = name, path, []

    def add(self, file: str, line: Optional[int], rule: str, detail: str):
        where = file if line is None else f"{file}:{line}"
        self.found.append(f"{self.name!r} ({self.path}), {where}: {rule}: {detail}")

    def raise_any(self):
        if self.found:
            raise PlatformError("invalid-platform-data", "\n".join(self.found))


def _key_line(text: str, key: str) -> Optional[int]:
    for number, line in enumerate(text.splitlines(), start=1):
        if f'"{key}"' in line:
            return number
    return None


class _Object(dict):
    """A JSON object, with the keys it states more than once."""

    twice: Tuple[str, ...] = ()


def _object(pairs) -> _Object:
    found = _Object(pairs)
    keys = [key for key, _ in pairs]
    found.twice = tuple(dict.fromkeys(key for key in keys if keys.count(key) > 1))
    return found


def _stated_twice(value, path=()):
    """Each (path, key) of a key stated twice in one object of ``value``."""
    if isinstance(value, _Object):
        for key in value.twice:
            yield path, key
        for key, each in value.items():
            yield from _stated_twice(each, path + (key,))
    elif isinstance(value, list):
        for each in value:
            yield from _stated_twice(each, path)


def _key_positions(text: str, key: str, start: int = 0):
    """The positions after each ``"key":`` in ``text`` from ``start``."""
    quoted = json.dumps(key)
    at = text.find(quoted, start)
    while at >= 0:
        after = at + len(quoted)
        rest = text[after:].lstrip()
        if rest.startswith(":"):
            yield at, len(text) - len(rest) + 1
        at = text.find(quoted, after)


def _second_line(text: str, path, key: str) -> Optional[int]:
    """The line of the second statement of ``key`` in the object at ``path``, found by
    following the path's keys through ``text``."""
    at = 0
    for each in path:
        found = next(_key_positions(text, each, at), None)
        if found is None:
            return None
        at = found[1]
    found = [start for start, _ in _key_positions(text, key, at)]
    return text.count("\n", 0, found[1]) + 1 if len(found) > 1 else None


def _parse(text: str, file: str, line: Optional[int], problems: _Problems):
    """``text`` parsed, or ``None`` with its problems added: not JSON, or a key stated
    twice in one object (JSON would keep only the last)."""
    try:
        value = json.loads(text, object_pairs_hook=_object)
    except json.JSONDecodeError as error:
        problems.add(
            file, error.lineno if line is None else line, "schema", f"not JSON: {error.msg}"
        )
        return None
    twice = list(_stated_twice(value))
    for path, key in twice:
        where = "the top-level object" if not path else repr("/".join(map(str, path)))
        found = line if line is not None else _second_line(text, path, key)
        problems.add(file, found, "duplicate", f"key {key!r} is stated twice in {where}")
    return None if twice else value


def _read_json(path: Path, problems: _Problems):
    text = path.read_text()
    return _parse(text, path.name, None, problems), text


def _read_jsonl(path: Path, problems: _Problems) -> List[Tuple[int, object]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = _parse(line, path.name, number, problems)
        if row is not None:
            rows.append((number, row))
    return rows


def _keys(value, required, optional=()) -> Optional[str]:
    """Why ``value`` is not an object of the ``required`` and ``optional`` keys, or
    ``None``."""
    if not isinstance(value, dict):
        return f"expected an object, found {value!r}"
    missing = [key for key in required if key not in value]
    unknown = [key for key in value if key not in required and key not in optional]
    if missing or unknown:
        return f"missing {missing}, unknown {unknown}" if missing else f"unknown keys {unknown}"
    return None


def _counts(value) -> Optional[Resources]:
    if _keys(value, RESOURCE_NAMES) is not None:
        return None
    if not all(type(value[name]) is int and value[name] >= 0 for name in RESOURCE_NAMES):
        return None
    return Resources(**value)


def _sum(slrs) -> Resources:
    return Resources(**{name: sum(getattr(slr, name) for slr in slrs) for name in RESOURCE_NAMES})


def _read_manifest(directory: Path, problems: _Problems) -> Tuple[str, dict, dict]:
    path = directory / "manifest.json"
    if not path.is_file():
        return directory.name, {"kind": "hand-written"}, {}
    manifest, text = _read_json(path, problems)
    if manifest is None:
        return directory.name, {"kind": "hand-written"}, {}
    why = _keys(manifest, ("name", "source"), ("counts",))
    source = manifest.get("source") if isinstance(manifest, dict) else None
    if why is None and not isinstance(manifest["name"], str):
        why = f"name is {manifest['name']!r}, not a string"
    if why is None:
        kind = source.get("kind") if isinstance(source, dict) else None
        if kind == "vivado":
            why = _keys(source, ("kind", "release", "build"))
        elif kind == "hand-written":
            why = _keys(source, ("kind",), ("note",))
        else:
            why = f"source is {source!r}: its kind is one of {list(SOURCE_KINDS)}"
    if why is not None:
        problems.add(path.name, 1, "schema", why)
        return directory.name, {"kind": "hand-written"}, {}
    problems.name = manifest["name"]
    return manifest["name"], source, manifest.get("counts", {})


def _read_architectures(directory: Path, problems: _Problems) -> Dict[str, Architecture]:
    path = directory / "architectures.json"
    if not path.is_file():
        return {}
    rows, text = _read_json(path, problems)
    if not isinstance(rows, dict):
        if rows is not None:
            problems.add(path.name, 1, "schema", "expected an object keyed by architecture")
        return {}
    found = {}
    for name, row in rows.items():
        line = _key_line(text, name)
        why = _keys(row, ("dsp", "class", "uram"))
        if why is None and row["dsp"] not in DSP_BLOCKS:
            why = f"dsp is {row['dsp']!r}, not a DSP block FINN knows ({', '.join(DSP_BLOCKS)})"
        if why is None and row["class"] is not None and row["class"] not in ARCHITECTURE_CLASSES:
            why = f"class is {row['class']!r}, not one of {list(ARCHITECTURE_CLASSES)} or null"
        if why is None and not isinstance(row["uram"], bool):
            why = f"uram is {row['uram']!r}, not true or false"
        if why is not None:
            problems.add(path.name, line, "schema", f"architecture {name!r}: {why}")
            continue
        found[name] = Architecture(name, row["dsp"], row["class"], row["uram"])
    return found


def _read_resources(directory: Path, problems: _Problems) -> Dict[str, tuple]:
    """Resource records by digest: (totals, slrs, capped)."""
    path = directory / "resources.jsonl"
    if not path.is_file():
        return {}
    records = {}
    for line, row in _read_jsonl(path, problems):
        why = _keys(row, ("digest", "slr_count", "totals", "slrs"), ("capped",))
        if why is not None:
            problems.add(path.name, line, "schema", why)
            continue
        totals = _counts(row["totals"])
        slrs = row["slrs"] if isinstance(row["slrs"], list) else []
        slrs = [_counts(each) for each in slrs]
        capped = row.get("capped", False)
        if totals is None or not slrs or None in slrs:
            problems.add(
                path.name,
                line,
                "schema",
                "totals and slrs are counts of " + ", ".join(RESOURCE_NAMES),
            )
            continue
        if not isinstance(row["digest"], str) or capped not in (True, False):
            problems.add(path.name, line, "schema", "digest is a string and capped true or false")
            continue
        if row["digest"] in records:
            problems.add(path.name, line, "duplicate", f"digest {row['digest']} is stated twice")
            continue
        if row["slr_count"] != len(slrs):
            problems.add(
                path.name,
                line,
                "slr-count",
                f"slr_count is {row['slr_count']!r} but {len(slrs)} SLRs are stated",
            )
            continue
        summed = _sum(slrs)
        over = all(getattr(summed, name) >= getattr(totals, name) for name in RESOURCE_NAMES)
        if not capped and summed != totals:
            problems.add(
                path.name, line, "slr-sum", f"the SLRs sum to {summed}, not the totals {totals}"
            )
            continue
        if capped and (len(slrs) < 2 or summed == totals or not over):
            problems.add(
                path.name,
                line,
                "slr-sum",
                "a capped record has several SLRs whose sum exceeds its totals",
            )
            continue
        records[row["digest"]] = (totals, tuple(slrs), capped)
    return records


def _read_devices(directory: Path, problems: _Problems, records: dict) -> Dict[str, Device]:
    path = directory / "devices.jsonl"
    if not path.is_file():
        return {}
    devices = {}
    for line, row in _read_jsonl(path, problems):
        why = _keys(row, ("name", "architecture", "family", "resources"))
        if why is None and not all(isinstance(row[key], str) and row[key] for key in row):
            why = "name, architecture, family and resources are strings"
        if why is not None:
            problems.add(path.name, line, "schema", why)
            continue
        if row["resources"] not in records:
            problems.add(
                path.name,
                line,
                "resources-undefined",
                f"device {row['name']}: no record {row['resources']} in resources.jsonl",
            )
            continue
        if row["name"].lower() in devices:
            problems.add(path.name, line, "duplicate", f"device {row['name']} is stated twice")
            continue
        totals, slrs, capped = records[row["resources"]]
        devices[row["name"].lower()] = Device(
            row["name"], row["architecture"], row["family"], totals, slrs, capped
        )
    return devices


_BOARD_KEYS = (
    "shell",
    "board_part",
    "vitis_platform",
    "axi_port_width",
    "memory",
    "mem_port",
    "vivado_min",
    "golden",
    "retired",
    "ps_config",
    "note",
)
_RELEASE = re.compile(r"^(20\d\d)\.(\d)$")
_MEM_PORT = re.compile(r"^(DDR|HBM)\[(\d+|\{slr\})\]$")
_BOARD_PART = re.compile(r"^[^:\s]+:[^:\s]+:[^:\s]+:[^:\s]+$")


def _board_problem(row) -> Optional[str]:
    """Why a board row breaks the schema or its own rules, or ``None``."""
    why = _keys(row, ("part",), _BOARD_KEYS)
    if why is not None:
        return why
    strings = ("part", "board_part", "vitis_platform", "mem_port", "retired", "note")
    for key in strings:
        if key in row and not (isinstance(row[key], str) and row[key]):
            return f"{key} is {row[key]!r}, not a string"
    shell = row.get("shell")
    if shell is not None and shell not in SHELLS:
        return f"shell is {shell!r}, not one of {list(SHELLS)} or null"
    width = row.get("axi_port_width")
    if width is not None and not (type(width) is int and width > 0 and width % 8 == 0):
        return f"axi_port_width is {width!r}, not a width in bits"
    memory = row.get("memory", [])
    if not isinstance(memory, list) or not all(each in MEMORIES for each in memory):
        return f"memory is {memory!r}, not a list of {list(MEMORIES)}"
    port = _MEM_PORT.match(row.get("mem_port", ""))
    if "mem_port" in row and not port:
        return f"mem_port is {row['mem_port']!r}, not like HBM[0] or DDR[{{slr}}]"
    if "board_part" in row and not _BOARD_PART.match(row["board_part"]):
        return (
            f"board_part is {row['board_part']!r}, not a Vivado board preset like "
            "vendor:name:part0:version"
        )
    if "vivado_min" in row and not (
        isinstance(row["vivado_min"], str) and _RELEASE.match(row["vivado_min"])
    ):
        return f"vivado_min is {row['vivado_min']!r}, not a release like \"2024.1\""
    if not isinstance(row.get("golden", False), bool):
        return f"golden is {row['golden']!r}, not true or false"
    config = row.get("ps_config", {})
    if not isinstance(config, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in config.items()
    ):
        return f"ps_config is {config!r}, not an object of strings"
    if shell in ("VIVADO_ZYNQ", "VIVADO_VERSAL") and width is None:
        return f"a board of the {shell} shell states its axi_port_width"
    if shell in ("VITIS_ALVEO", "SLASH_ALVEO") and width is not None:
        return f"the {shell} shell states the axi_port_width, not its boards"
    only = {
        "vitis_platform": ("VITIS_ALVEO",),
        "mem_port": ("VITIS_ALVEO",),
        "golden": ("VIVADO_VERSAL",),
        "board_part": ("VIVADO_ZYNQ", "VIVADO_VERSAL"),
        "ps_config": ("VIVADO_ZYNQ",),
    }
    for key, shells in only.items():
        if row.get(key) and shell not in shells:
            return f"{key} applies to boards of the {' or '.join(shells)} shell only"
    if port and port[1] not in memory:
        return f"mem_port {port[0]} is on {port[1]}, which is not one of its memory {memory!r}"
    return None


def _read_boards(directory: Path, problems: _Problems) -> Dict[str, Board]:
    path = directory / "boards.json"
    if not path.is_file():
        return {}
    rows, text = _read_json(path, problems)
    if not isinstance(rows, dict):
        if rows is not None:
            problems.add(path.name, 1, "schema", "expected an object keyed by board name")
        return {}
    found = {}
    for name, row in rows.items():
        why = _board_problem(row)
        if why is not None:
            problems.add(path.name, _key_line(text, name), "schema", f"board {name!r}: {why}")
            continue
        release = _RELEASE.match(row.get("vivado_min", ""))
        found[name] = Board(
            name=name,
            part=row["part"],
            shell=row.get("shell"),
            board_part=row.get("board_part"),
            vitis_platform=row.get("vitis_platform"),
            axi_port_width=row.get("axi_port_width"),
            memory=tuple(row.get("memory", ())),
            mem_port=row.get("mem_port"),
            vivado_min=(int(release[1]), int(release[2])) if release else None,
            golden=row.get("golden", False),
            retired=row.get("retired"),
            ps_config=dict(row.get("ps_config", {})),
        )
    return found


def read_source(directory) -> PlatformSource:
    """The platform data in ``directory``, validated on its own. Raises
    ``PlatformError`` (``invalid-platform-data``) naming each problem's file, line and
    rule."""
    directory = Path(directory)
    problems = _Problems(directory.name, directory)
    if not directory.is_dir():
        problems.add(".", None, "not-a-directory", "platform data is a directory")
        problems.raise_any()
    name, origin, counts = _read_manifest(directory, problems)
    architectures = _read_architectures(directory, problems)
    records = _read_resources(directory, problems)
    devices = _read_devices(directory, problems, records)
    boards = _read_boards(directory, problems)
    problems.raise_any()
    return PlatformSource(name, directory, origin, counts, architectures, devices, boards)


def _differing(first, second) -> str:
    return next(f.name for f in fields(first) if getattr(first, f.name) != getattr(second, f.name))


def union(sources) -> PlatformData:
    """The union of ``sources``, in order. Refuses (``PlatformError``,
    ``invalid-platform-data``) a fact defined differently in two sources, a device
    whose architecture no source defines, a device with UltraRAM on an architecture
    without it, a board whose part no source holds, and a board whose shell does not
    build on its part's architecture or device."""
    sources = tuple(sources)
    architectures, devices, boards, defined_in, first = {}, {}, {}, {}, {}
    problems = []

    def add(kind, key, fact, source, into, file):
        if key in into and into[key] != fact:
            other = first[(kind, key)]
            problems.append(
                f"{source.name!r} ({source.path}), {file}: conflict: {kind} {key} is "
                f"defined differently in {other.name!r} ({other.path}) and "
                f"{source.name!r} ({source.path}) (field "
                f"{_differing(into[key], fact)!r}); a fact may be stated in several "
                "platform data directories only if it is identical"
            )
            return
        into.setdefault(key, fact)
        first.setdefault((kind, key), source)
        defined_in[(kind, key)] = defined_in.get((kind, key), ()) + (source.name,)

    for source in sources:
        for name, each in source.architectures.items():
            add("architecture", name, each, source, architectures, "architectures.json")
        for name, each in source.devices.items():
            add("device", name, each, source, devices, "devices.jsonl")
        for name, each in source.boards.items():
            add("board", name, each, source, boards, "boards.json")
    for source in sources:
        for name, each in source.devices.items():
            found = architectures.get(each.architecture)
            if found is None:
                problems.append(
                    f"{source.name!r} ({source.path}), devices.jsonl: "
                    f"architecture-undefined: device {each.name} is of architecture "
                    f"{each.architecture!r}, which no platform data defines; add it to "
                    "an architectures.json in a directory of FINN_PLATFORM_PATH"
                )
            elif each.resources.uram and not found.uram:
                problems.append(
                    f"{source.name!r} ({source.path}), devices.jsonl: uram: device "
                    f"{each.name} has UltraRAM, but architecture {found.name!r} states none"
                )
    data = PlatformData(sources, architectures, devices, defined_in, boards)
    for source in sources:
        for name, each in source.boards.items():
            where = f"{source.name!r} ({source.path}), boards.json"
            device = data.find_device(each.part)
            if device is None:
                problems.append(
                    f"{where}: part-undefined: board {name}'s part {each.part} is of no "
                    "device any platform data holds"
                )
                continue
            found = architectures.get(device.architecture)
            builds = SHELL_ARCHITECTURES.get(each.shell)
            if found is not None and builds is not None and found.name not in builds:
                problems.append(
                    f"{where}: shell-architecture: board {name}'s shell {each.shell} does "
                    f"not build for its part's architecture {found.name}"
                )
            devices = SHELL_DEVICES.get(each.shell)
            if devices is not None and device.name.lower() not in devices:
                problems.append(
                    f"{where}: shell-device: board {name}'s shell {each.shell} builds for "
                    f"the device {', '.join(devices)} only, but its part is on device "
                    f"{device.name}"
                )
            if found is not None and each.ps_config and found.class_ != "zynquplus":
                problems.append(
                    f"{where}: ps-config: board {name}'s ps_config sets Zynq UltraScale+ "
                    f"PS properties, but its part is of architecture {found.name}"
                )
    if problems:
        raise PlatformError("invalid-platform-data", "\n".join(problems))
    return data


def source_paths() -> List[Path]:
    """The platform data directories to load: FINN's built-in one, then those listed
    in ``FINN_PLATFORM_PATH``, in order."""
    listed = os.environ.get(PATH_VARIABLE, "")
    return [DATA] + [Path(each).expanduser() for each in listed.split(os.pathsep) if each]


def load(paths=None) -> PlatformData:
    """Load and validate the platform data in the directories ``paths`` (by default
    ``source_paths()``) and return their union."""
    return union(read_source(each) for each in (source_paths() if paths is None else paths))


@functools.lru_cache(maxsize=None)
def _loaded(listed: str) -> PlatformData:
    return load()


def platform_data() -> PlatformData:
    """The union of FINN's built-in platform data and the user's, loaded once per
    process (and again if ``FINN_PLATFORM_PATH`` changes)."""
    return _loaded(os.environ.get(PATH_VARIABLE, ""))


def platform_sources() -> Tuple[PlatformSource, ...]:
    """Where the loaded platform data comes from, in load order: each directory's
    ``describe()`` and ``path``."""
    return platform_data().sources
