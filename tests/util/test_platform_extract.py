# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""On saved, synthetic Vivado output: no Vivado needed."""

import pytest

import json

from finn.platform import extract
from finn.platform.cli import main
from finn.platform.registry import Architecture, PlatformError, read_source

ARCHITECTURES = {
    "zynquplus": Architecture("zynquplus", "DSP48E2", "zynquplus", True),
    "virtexuplus": Architecture("virtexuplus", "DSP48E2", None, True),
}
VU5P_SLR = "SLICEL=30000,SLICEM=19260,RAMB18_L=720,RAMB18_U=720,DSP48E2=2280,URAM288=320"
VU9P_SLR = VU5P_SLR
SLICEM = ",".join(
    [f"{x}5LUT,{x}6LUT,{x}FF,{x}FF2" for x in "ABCDEFGH"] + ["CARRY8", "F7MUX_AB", "F9MUX"]
)

PARTS = [
    # name|device|package|speed|temperature|architecture|family|totals (LUT_ELEMENTS,
    # FLIPFLOPS, BLOCK_RAMS, ULTRA_RAMS, DSP, SLRS)
    "xczu2eg-sbva484-1-e|xczu2eg|sbva484|-1|E|zynquplus|zynquplus|47232|94464|150||240|1",
    "xczu3eg-sbva484-1-e|xczu3eg|sbva484|-1|E|zynquplus|zynquplus|70560|141120|216||360|1",
    "xczu3eg-sbva484-1-i|xczu3eg|sbva484|-1|I|zynquplus|zynquplus|70560|141120|216||360|1",
    "xcvu5p-flva2104-1-e|xcvu5p|flva2104|-1|E|virtexuplus|virtexuplus|"
    "600577|1201154|1024|470|3474|2",
    "xcvu9p-flga2104-2L-e|xcvu9p|flga2104|-2L|E|virtexuplus|virtexuplus|"
    "1182240|2364480|2160|960|6840|3",
    "xc7a35t-cpg236-1|xc7a35t|cpg236|-1|C|artix7|artix7|20800|41600|50||90|1",
]


def write_logs(work, parts=PARTS, vu9p_slr=VU9P_SLR, extra_probes=()):
    tool = "CATALOG TOOL 2025.2|SW Build 1\n"
    lines = [tool, f"CATALOG COUNT {len(parts)}\n"]
    lines += [f"CATALOG PART {each}\n" for each in parts]
    (work / "parts-0.log").write_text("".join(lines) + "CATALOG COMPLETE\n")
    probes = [
        ("xcvu5p-flva2104-1-e", "xcvu5p", [VU5P_SLR, VU5P_SLR]),  # on the XCVU9P's die
        ("xcvu9p-flga2104-2L-e", "xcvu9p", [vu9p_slr] * 3),
    ] + list(extra_probes)
    lines = [tool]
    for part, device, slrs in probes:
        lines.append(f"CATALOG DEVICE {part}|{device}\n")
        lines += [f"CATALOG SLR {part}|SLR{i}|1|{sites}\n" for i, sites in enumerate(slrs)]
        lines.append(f"CATALOG SLICEM {part}|{SLICEM}\n")
    (work / "devices-0.log").write_text("".join(lines) + "CATALOG COMPLETE\n")
    return work


def derive(work, known=()):
    texts = extract.read_logs(work, "parts") + extract.read_logs(work, "devices")
    return extract.derive(
        extract.parse_tool(texts),
        extract.parse_parts(extract.read_logs(work, "parts")),
        extract.parse_devices(extract.read_logs(work, "devices")),
        ARCHITECTURES,
        known,
        "test",
    )


def counts(lut, ff, bram18, uram, dsp):
    return {"lut": lut, "ff": ff, "bram18": bram18, "uram": uram, "dsp": dsp}


@pytest.mark.util
def test_extraction_derives_each_devices_slrs(tmp_path):
    files, left_out = derive(write_logs(tmp_path))
    assert left_out == ["artix7"]  # no architecture row: left out, and reported
    records = {each["digest"]: each for each in files["resources.jsonl"]}
    devices = {each["name"]: each for each in files["devices.jsonl"]}
    assert sorted(devices) == ["xcvu5p", "xcvu9p", "xczu2eg", "xczu3eg"]
    assert devices["xczu3eg"] == {
        "name": "xczu3eg",
        "architecture": "zynquplus",
        "family": "zynquplus",
        "resources": devices["xczu3eg"]["resources"],
    }
    # one SLR: the SLR is the device's totals, with no link
    zu3 = records[devices["xczu3eg"]["resources"]]
    assert zu3["slrs"] == [zu3["totals"]] == [counts(70560, 141120, 432, 0, 360)]
    assert "capped" not in zu3
    # several SLRs: each SLR's sites from the link
    vu9p = records[devices["xcvu9p"]["resources"]]
    assert vu9p["slrs"] == [counts(394080, 788160, 1440, 320, 2280)] * 3
    assert "capped" not in vu9p
    # a reduced die of two SLRs: each SLR's sites, capped by the device's totals
    vu5p = records[devices["xcvu5p"]["resources"]]
    assert vu5p["capped"] is True and vu5p["totals"] == counts(600577, 1201154, 2048, 470, 3474)
    assert vu5p["slrs"] == [counts(394080, 788160, 1440, 320, 2280)] * 2
    assert files["manifest.json"] == {
        "name": "test",
        "source": {"kind": "vivado", "release": "2025.2", "build": "SW Build 1"},
        "counts": {"parts": 5, "devices": 4, "resources": 4, "capped": 1},
    }


@pytest.mark.util
def test_only_devices_of_several_slrs_are_linked_largest_first(tmp_path):
    parts = extract.parse_parts(extract.read_logs(write_logs(tmp_path), "parts"))
    assert extract.link_list(parts) == ["xcvu9p-flga2104-2L-e", "xcvu5p-flva2104-1-e"]
    assert extract.link_list(parts, 1, 2) == ["xcvu5p-flva2104-1-e"]


@pytest.mark.util
def test_a_part_its_device_name_does_not_prefix_fails_extraction(tmp_path):
    stray = "xc9eg3-sbva484-1-e|xczu3eg|sbva484|-1|E|zynquplus|zynquplus|70560|141120|216||360|1"
    write_logs(tmp_path, parts=PARTS + [stray])
    with pytest.raises(PlatformError, match="FINN would need a list of parts"):
        derive(tmp_path)


@pytest.mark.util
def test_the_prefix_check_counts_the_devices_of_other_platform_data(tmp_path):
    # a device already in other platform data whose name extends xczu3eg would take its parts
    write_logs(tmp_path)
    with pytest.raises(PlatformError, match="the longest device name prefixing it is xczu3eg-s"):
        derive(tmp_path, known=["xczu3eg-s"])


@pytest.mark.util
@pytest.mark.parametrize(
    "vu9p_slr, extra, reason",
    [
        (VU9P_SLR + ",DSP99=4", (), r"sites \['DSP99'\] are not recognised"),
        (VU9P_SLR.replace("RAMB18_U=720", "RAMB18_U=100"), (), "SLRs summing to"),
        (VU9P_SLR.replace("DSP48E2", "DSP58_PRIMARY"), (), "architecture virtexuplus states"),
        (
            VU9P_SLR,
            [("xczu3eg-sbva484-1-e", "xczu3eg", [VU5P_SLR.replace("DSP48E2", "DSP48E1")])],
            "its DSP sites are DSP48E1",
        ),
    ],
)
def test_an_extraction_the_platform_data_cannot_account_for_fails(
    tmp_path, vu9p_slr, extra, reason
):
    write_logs(tmp_path, vu9p_slr=vu9p_slr, extra_probes=extra)
    with pytest.raises(PlatformError, match=reason) as refused:
        derive(tmp_path)
    assert refused.value.code == "extract-failed"


@pytest.mark.util
def test_a_device_of_several_slrs_must_be_linked(tmp_path):
    write_logs(tmp_path)
    (tmp_path / "devices-0.log").write_text("CATALOG TOOL 2025.2|SW Build 1\nCATALOG COMPLETE\n")
    with pytest.raises(PlatformError, match="xcvu5p has 2 SLRs but was not linked"):
        derive(tmp_path)


@pytest.mark.util
def test_an_incomplete_registry_pass_fails(tmp_path):
    write_logs(tmp_path)
    log = tmp_path / "parts-0.log"
    log.write_text(log.read_text().replace("CATALOG COUNT 6", "CATALOG COUNT 7"))
    with pytest.raises(PlatformError, match="not every matching part"):
        derive(tmp_path)


@pytest.mark.util
def test_finn_platform_extract_writes_platform_data_from_saved_logs(tmp_path, capsys):
    (tmp_path / "work").mkdir()
    work = write_logs(tmp_path / "work")
    out = tmp_path / "mine"
    assert main(["extract", "--parts", "xcvu*", "--out", str(out), "--from-logs", str(work)]) == 0
    source = read_source(out)
    assert source.name == "mine" and source.origin["release"] == "2025.2"
    assert sorted(source.devices) == ["xcvu5p", "xcvu9p"]
    assert "wrote mine [Vivado 2025.2]" in capsys.readouterr().out
    # the extraction leaves the hand-written files of a platform data directory alone
    (out / "architectures.json").write_text(json.dumps({}))
    assert main(["extract", "--parts", "xczu*", "--out", str(out), "--from-logs", str(work)]) == 0
    assert (out / "architectures.json").read_text() == "{}"
    assert sorted(read_source(out).devices) == ["xczu2eg", "xczu3eg"]
    assert main(["extract", "--parts", "xcku*", "--out", str(out), "--from-logs", str(work)]) == 1
    assert "no part in the logs" in capsys.readouterr().err
