# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Extract platform data (``finn.platform.registry``) from an installed Vivado's part
database: ``finn-platform extract``.

Two passes of ``vivado -mode batch`` over ``extract.tcl`` beside this module:

1. the *registry pass* lists every installed part matching a pattern, with its
   device, Vivado's ARCHITECTURE and FAMILY and the device's totals
   (``LUT_ELEMENTS``, ``FLIPFLOPS``, ``BLOCK_RAMS``, ``ULTRA_RAMS``, ``DSP``,
   ``SLRS``);
2. the *link pass* links an empty design (no synthesis) on one part of each device
   of several SLRs, and counts each SLR's sites. A device of one SLR needs no link:
   its SLR is its totals.

From their logs (``parts-*.log``, ``devices-*.log``) the platform data files are derived:

- each SLR's resources: ``lut`` and ``ff`` the slices (SLICEL and SLICEM sites)
  times a SLICEM's LUT6 and flip-flop BELs, ``bram18`` the RAMB18 sites, ``uram`` the
  URAM288 sites, ``dsp`` the DSP sites. Their number must be the device's SLRS and
  their sum at least its totals (BLOCK_RAMS counts RAMB36, two RAMB18 each). A device
  sold on a larger die (an XCVU5P on the XCVU9P's) has more sites than its totals:
  each SLR then states its sites, an upper bound for that SLR alone, and the record
  is ``capped``: the totals cap their sum;
- the devices, each with its architecture and family.

The DSP block is a fact of the architecture (``architectures.json``), not of the
device, so a part whose architecture no loaded platform data defines is left out and
reported. Where a device was linked, its DSP sites must be of its architecture's
block. Every part of a device must state the same architecture, family and totals,
and the longest device name prefixing each part's name must be its device, which is
how FINN finds a part's device without a list of parts.

The parsers also run on saved logs (``--from-logs``), without Vivado.
"""

import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from finn.platform.registry import RESOURCE_NAMES, PlatformError

TCL = Path(__file__).resolve().parent / "extract.tcl"

SLICE_SITES = frozenset({"SLICEL", "SLICEM"})
BRAM18_SITES = frozenset({"RAMB18E1", "FIFO18E1", "RAMB181", "RAMBFIFO18", "RAMB18_L", "RAMB18_U"})
URAM_SITES = frozenset({"URAM288"})
DSP_SITES = {"DSP48E1": "DSP48E1", "DSP48E2": "DSP48E2", "DSP58_PRIMARY": "DSP58"}
NOT_COUNTED = frozenset(
    {"RAMBFIFO36E1", "RAMBFIFO36", "RAMB36", "DSP58_CPLX", "URAM_CAS_DLY", "SLICE_FE"}
)
"""Memory and DSP sites that are another view of counted ones (a RAMB36 is two RAMB18
sites; a complex DSP58 pair is two DSP58_PRIMARY sites) or hold no resource."""
COUNTED = SLICE_SITES | BRAM18_SITES | URAM_SITES | frozenset(DSP_SITES)
_RESOURCE_SITE = re.compile(r"^(SLICE|RAMB|FIFO|URAM|DSP)")
_LUT6 = re.compile(r"^[A-H]6LUT$")
_FF = re.compile(r"^[A-H](5FF|FF|FF2)$")


class ExtractError(PlatformError):
    """An extraction platform data cannot be derived from."""

    def __init__(self, message: str):
        super().__init__("extract-failed", message)


def _lines(text, tag):
    prefix = f"CATALOG {tag} "
    return [line[len(prefix) :].split("|") for line in text.splitlines() if line.startswith(prefix)]


def read_logs(work, kind):
    return [path.read_text() for path in sorted(Path(work).glob(f"{kind}-*.log"))]


def parse_tool(texts):
    """The Vivado release and build, which every log states alike."""
    found = {tuple(row) for text in texts for row in _lines(text, "TOOL")}
    if len(found) != 1:
        raise ExtractError(f"the logs name {len(found)} Vivado builds: {sorted(found)}")
    ((release, build),) = found
    return {"release": release, "build": build}


def parse_parts(texts):
    """Every part of the registry pass's logs, each exactly once, sorted by name: a
    dict of name, device, architecture, family and the device's totals as stated."""
    counts = set()
    rows = {}
    for text in texts:
        if "CATALOG COMPLETE" not in text:
            raise ExtractError("a registry pass did not complete")
        counts.update(int(count) for (count,) in _lines(text, "COUNT"))
        for name, device, _pkg, _speed, _temp, arch, family, *totals in _lines(text, "PART"):
            if name in rows:
                raise ExtractError(f"{name} was extracted twice")
            rows[name] = {
                "name": name,
                "device": device,
                "architecture": arch,
                "family": family,
                "totals": tuple(totals),
            }
    if len(counts) != 1 or counts.pop() != len(rows):
        raise ExtractError(f"{len(rows)} parts extracted, not every matching part")
    return [rows[name] for name in sorted(rows)]


def parse_devices(texts):
    """Every linked device of the link pass's logs, by device name: its SLRs' sites by
    type, in Vivado's SLR order, and one SLICEM's BELs."""
    probes = {}
    for text in texts:
        failed = _lines(text, "LINK_FAIL")
        if failed:
            raise ExtractError(f"link_design failed: {failed}")
        slrs = defaultdict(list)
        for part, _slr, _fabric, counts in _lines(text, "SLR"):
            sites = (each.split("=") for each in counts.split(",") if each)
            slrs[part].append({site: int(count) for site, count in sites})
        bels = {part: set(filter(None, found.split(","))) for part, found in _lines(text, "SLICEM")}
        for part, device in _lines(text, "DEVICE"):
            if part not in bels:
                raise ExtractError(f"{part}: the link did not complete")
            probes[device] = {"slrs": slrs[part], "bels": bels[part]}
    return probes


def _slr_count(row):
    return int(row["totals"][-1] or 0)


def link_list(parts, index=0, count=1):
    """One part (its first) of each device of several SLRs, the largest devices first,
    dealt round over ``count`` runs: the share of the run numbered ``index``."""
    first = {}
    for row in parts:
        if _slr_count(row) > 1:
            first.setdefault(row["device"], row)
    by_size = sorted(first.values(), key=lambda row: -int(row["totals"][0] or 0))
    return [row["name"] for row in by_size[index::count]]


def _counts(**counts):
    return {name: counts[name] for name in RESOURCE_NAMES}


def slr_resources(probe):
    """Each SLR's resources from its sites and the device's SLICEM BELs."""
    luts = sum(1 for bel in probe["bels"] if _LUT6.match(bel))
    ffs = sum(1 for bel in probe["bels"] if _FF.match(bel))
    found = []
    for sites in probe["slrs"]:
        slices = sum(sites.get(each, 0) for each in SLICE_SITES)
        found.append(
            _counts(
                lut=slices * luts,
                ff=slices * ffs,
                bram18=sum(sites.get(each, 0) for each in BRAM18_SITES),
                uram=sum(sites.get(each, 0) for each in URAM_SITES),
                dsp=sum(sites.get(each, 0) for each in DSP_SITES),
            )
        )
    return found


def device_totals(row):
    """The device-level totals a part states, in the platform data's units, and its
    SLRs."""
    lut, ff, bram, uram, dsp, slrs = (value or None for value in row["totals"])
    if None in (lut, ff, bram, dsp, slrs):
        raise ExtractError(f"{row['name']} states no totals: {row['totals']}")
    totals = _counts(
        lut=int(lut), ff=int(ff), bram18=2 * int(bram), uram=int(uram or 0), dsp=int(dsp)
    )
    return totals, int(slrs)


def linked_dsp_block(device, probe):
    """The DSP block a linked device's sites show; any resource site the extraction
    does not know is refused rather than miscounted."""
    types = {each for slr in probe["slrs"] for each in slr}
    unknown = {each for each in types if _RESOURCE_SITE.match(each)} - COUNTED - NOT_COUNTED
    if unknown:
        raise ExtractError(f"{device}: sites {sorted(unknown)} are not recognised")
    blocks = {DSP_SITES[each] for each in types if each in DSP_SITES}
    if len(blocks) != 1:
        raise ExtractError(f"{device}: DSP sites {sorted(blocks)}, not one DSP block")
    return blocks.pop()


def record(totals, slrs, capped):
    """A resource record as stored, keyed by the digest of its facts."""
    facts = {"slr_count": len(slrs), "totals": totals, "slrs": slrs}
    if capped:
        facts["capped"] = True
    encoded = json.dumps(facts, separators=(",", ":")).encode()
    return {"digest": "sha256:" + hashlib.sha256(encoded).hexdigest()[:16], **facts}


def _add(slrs):
    return _counts(**{name: sum(slr[name] for slr in slrs) for name in RESOURCE_NAMES})


def _slrs(device, totals, slr_count, probe):
    if slr_count == 1:
        return [totals], False
    if probe is None:
        raise ExtractError(f"{device} has {slr_count} SLRs but was not linked")
    slrs = slr_resources(probe)
    summed = _add(slrs)
    if len(slrs) != slr_count or any(summed[n] < totals[n] for n in RESOURCE_NAMES):
        raise ExtractError(
            f"{device}: {len(slrs)} SLRs summing to {summed}, the device states "
            f"{slr_count} and {totals}"
        )
    # a reduced die: each SLR's sites are its capacity, and the totals cap their sum
    return slrs, summed != totals


def derive(tool, parts, probes, architectures, known=(), name="extracted"):
    """Platform data files from an extraction: ``resources.jsonl``, ``devices.jsonl``
    and ``manifest.json``, and the architectures left out. ``architectures`` holds the
    defined architectures by name; ``known`` the names of devices already in other
    platform data, which the prefix check must not confuse."""
    left_out = sorted({row["architecture"] for row in parts} - set(architectures))
    parts = [row for row in parts if row["architecture"] in architectures]
    by_device = defaultdict(list)
    for row in parts:
        by_device[row["device"]].append(row)
    folded = {}
    for each in [row["name"] for row in parts] + list(by_device):
        if folded.setdefault(each.lower(), each) != each:
            raise ExtractError(f"{each} and {folded[each.lower()]} differ only in case")
    records, devices, capped = {}, [], []
    for device in sorted(by_device):
        rows = by_device[device]
        stated = {(row["architecture"], row["family"], row["totals"]) for row in rows}
        if len(stated) != 1:
            raise ExtractError(f"{device}'s parts differ: {sorted(stated)}")
        architecture = rows[0]["architecture"]
        totals, slr_count = device_totals(rows[0])
        slrs, is_capped = _slrs(device, totals, slr_count, probes.get(device))
        if device in probes:
            block = linked_dsp_block(device, probes[device])
            if block != architectures[architecture].dsp:
                raise ExtractError(
                    f"{device}: its DSP sites are {block}, but architecture {architecture} "
                    f"states {architectures[architecture].dsp}"
                )
        if is_capped:
            capped.append(device)
        found = record(totals, slrs, is_capped)
        records[found["digest"]] = found
        devices.append(
            {
                "name": device,
                "architecture": architecture,
                "family": rows[0]["family"],
                "resources": found["digest"],
            }
        )
    names = {each.lower() for each in known} | {each["name"].lower() for each in devices}
    names = sorted(names, key=len, reverse=True)
    for row in parts:
        found = next((n for n in names if row["name"].lower().startswith(n)), None)
        if found != row["device"].lower():
            raise ExtractError(
                f"{row['name']}: the longest device name prefixing it is {found}, not its "
                f"device {row['device']}; FINN would need a list of parts"
            )
    manifest = {
        "name": name,
        "source": {"kind": "vivado", **tool},
        "counts": {
            "parts": len(parts),
            "devices": len(devices),
            "resources": len(records),
            "capped": len(capped),
        },
    }
    files = {
        "resources.jsonl": [records[digest] for digest in sorted(records)],
        "devices.jsonl": devices,
        "manifest.json": manifest,
    }
    return files, left_out


def write(files, directory):
    """Write derived platform data files into ``directory``; other files there (such as a
    hand-written ``architectures.json``) are left as they are."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if name.endswith(".jsonl"):
            text = "".join(json.dumps(each, separators=(", ", ": ")) + "\n" for each in content)
        else:
            text = json.dumps(content, indent=2) + "\n"
        (directory / name).write_text(text)


def _run_vivado(runs, work, kind):
    """Run Vivado once per argument list in ``runs``, in parallel, each in its own
    directory under ``work``, logging to ``{kind}-{index}.log``."""
    vivado = shutil.which("vivado")
    if vivado is None:
        raise ExtractError("vivado is not on the PATH: source Vivado's settings64.sh first")
    started = []
    for index, args in enumerate(runs):
        cwd = Path(tempfile.mkdtemp(prefix=f"{kind}-{index}-", dir=work))
        log = open(Path(work) / f"{kind}-{index}.log", "w")
        command = [vivado, "-mode", "batch", "-nojournal", "-nolog", "-source", str(TCL)]
        process = subprocess.Popen(
            command + ["-tclargs", kind] + args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT
        )
        started.append((process, log, cwd))
    for process, log, cwd in started:
        process.wait()
        log.close()
        shutil.rmtree(cwd, ignore_errors=True)
    failed = [index for index, (process, _, _) in enumerate(started) if process.returncode]
    if failed:
        raise ExtractError(f"Vivado failed in the {kind} runs {failed}; see their logs in {work}")


def extract(pattern, out, architectures, known=(), work=None, from_logs=None, jobs=1, name=None):
    """Extract the parts matching ``pattern`` into platform data in ``out``. With
    ``from_logs``, read a previous extraction's logs from that directory instead of
    running Vivado (the parts outside ``pattern`` are left out). Returns the counts
    written and the architectures left out."""
    if from_logs is None:
        work = Path(tempfile.mkdtemp(prefix="finn-platform-extract-") if work is None else work)
        work.mkdir(parents=True, exist_ok=True)
        _run_vivado([[pattern, str(index), str(jobs)] for index in range(jobs)], work, "parts")
        parts = parse_parts(read_logs(work, "parts"))
        if not parts:
            raise ExtractError(f"no installed part matches {pattern!r}")
        linked = [link_list(parts, index, jobs) for index in range(jobs)]
        linked = [each for each in linked if each]
        if linked:
            _run_vivado(linked, work, "devices")
    else:
        work = Path(from_logs)
        parts = parse_parts(read_logs(work, "parts"))
        parts = [row for row in parts if fnmatch.fnmatchcase(row["name"].lower(), pattern.lower())]
        if not parts:
            raise ExtractError(f"no part in the logs in {work} matches {pattern!r}")
    texts = read_logs(work, "parts") + read_logs(work, "devices")
    probes = parse_devices(read_logs(work, "devices"))
    name = Path(out).name if name is None else name
    files, left_out = derive(parse_tool(texts), parts, probes, architectures, known, name)
    write(files, out)
    print(f"logs in {work}", file=sys.stderr)
    return files["manifest.json"]["counts"], left_out
