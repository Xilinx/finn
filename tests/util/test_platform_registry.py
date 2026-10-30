# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
from collections import defaultdict
from dataclasses import astuple

from finn.platform import catalog, registry
from finn.platform.registry import (
    DATA,
    PlatformError,
    Resources,
    load,
    read_source,
    union,
)

ZU3 = {"lut": 70560, "ff": 141120, "bram18": 432, "uram": 0, "dsp": 360}
ZU7 = {"lut": 230400, "ff": 460800, "bram18": 624, "uram": 96, "dsp": 1728}
ZYNQUPLUS = {"dsp": "DSP48E2", "class": "zynquplus", "uram": True}
VIRTEX7 = {"dsp": "DSP48E1", "class": None, "uram": False}


def record(digest, slrs, capped=False, totals=None):
    totals = totals or {name: sum(slr[name] for slr in slrs) for name in ZU3}
    found = {"digest": digest, "slr_count": len(slrs), "totals": totals, "slrs": slrs}
    if capped:
        found["capped"] = True
    return found


def device(name, resources, architecture="zynquplus", family="zynquplus"):
    return {"name": name, "architecture": architecture, "family": family, "resources": resources}


def write_platform_data(directory, name=None, architectures=None, resources=(), devices=()):
    """Write a platform data directory; ``None`` leaves a file out."""
    directory.mkdir(parents=True, exist_ok=True)
    if name is not None:
        manifest = {"name": name, "source": {"kind": "hand-written", "note": "a test"}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
    if architectures is not None:
        (directory / "architectures.json").write_text(json.dumps(architectures, indent=1))
    for file, rows in (("resources.jsonl", resources), ("devices.jsonl", devices)):
        if rows:
            (directory / file).write_text("".join(json.dumps(each) + "\n" for each in rows))
    return directory


@pytest.fixture
def base(tmp_path):
    """Platform data of four devices on two records: ZU3CG and ZU3EG share one; xczu7 and
    xczu7ev stand for two devices of which one's name prefixes the other's."""
    return write_platform_data(
        tmp_path / "base",
        "base",
        {"zynquplus": ZYNQUPLUS},
        [record("zu3", [ZU3]), record("zu7", [ZU7])],
        [
            device("xczu3cg", "zu3"),
            device("xczu3eg", "zu3"),
            device("xczu7", "zu7"),
            device("xczu7ev", "zu7"),
        ],
    )


@pytest.mark.util
def test_a_part_is_on_the_longest_device_name_that_prefixes_it(base):
    data = load([base])
    found = data.find_device("XCZU3EG-SBVA484-1-E")
    assert found.name == "xczu3eg" and found.architecture == "zynquplus"
    assert found.resources == Resources(**ZU3) and found.slrs == (Resources(**ZU3),)
    assert not found.capped
    assert data.find_device("xczu7ev-ffvc1156-2-e").name == "xczu7ev"
    assert data.find_device("xczu7-ffvc1156-2-e").name == "xczu7"
    assert data.find_device("xczu3cg").name == "xczu3cg"
    assert data.find_device("xczu9eg-ffvb1156-2-e") is None


@pytest.mark.util
def test_platform_data_directories_add_devices_and_architectures(base, tmp_path):
    virtex7 = write_platform_data(
        tmp_path / "virtex7",
        "virtex7",
        {"virtex7": VIRTEX7},
        [record("my100t", [dict(ZU3, uram=0)])],
        [device("xcmy100t", "my100t", "virtex7", "virtex7")],
    )
    data = load([base, virtex7])
    assert [each.name for each in data.sources] == ["base", "virtex7"]
    assert data.find_device("xcmy100tffg1761-2").architecture == "virtex7"
    assert data.architectures["virtex7"].dsp == "DSP48E1"
    assert data.defined_in[("device", "xcmy100t")] == ("virtex7",)
    assert data.describe() == "base [hand-written], virtex7 [hand-written]"


@pytest.mark.util
def test_a_fact_stated_identically_in_two_directories_is_accepted(base, tmp_path):
    # e.g. a user who extracts a whole family the built-in data already holds members of
    again = write_platform_data(
        tmp_path / "again",
        "again",
        {"zynquplus": ZYNQUPLUS},
        [record("other-key", [ZU3])],  # the same record under another key
        [device("XCZU3EG", "other-key")],  # spelled otherwise: a different fact
    )
    with pytest.raises(PlatformError, match="field 'name'"):
        load([base, again])
    write_platform_data(again, devices=[device("xczu3eg", "other-key")])
    data = load([base, again])
    assert data.defined_in[("device", "xczu3eg")] == ("base", "again")
    assert data.defined_in[("architecture", "zynquplus")] == ("base", "again")


@pytest.mark.util
def test_a_fact_stated_differently_in_two_directories_is_refused(base, tmp_path):
    other = write_platform_data(
        tmp_path / "other",
        "other",
        resources=[record("zu3x", [dict(ZU3, bram18=412)])],
        devices=[device("xczu3eg", "zu3x")],
    )
    with pytest.raises(PlatformError) as refused:
        load([base, other])
    assert refused.value.code == "invalid-platform-data"
    message = str(refused.value)
    assert (
        f"conflict: device xczu3eg is defined differently in 'base' ({base}) and "
        f"'other' ({other})"
    ) in message
    assert "(field 'resources')" in message
    dsp58 = write_platform_data(
        tmp_path / "dsp58", "dsp58", {"zynquplus": dict(ZYNQUPLUS, dsp="DSP58")}
    )
    with pytest.raises(PlatformError, match="architecture zynquplus .*field 'dsp'"):
        load([base, dsp58])


@pytest.mark.util
def test_any_platform_data_file_may_be_absent(base, tmp_path):
    empty = write_platform_data(tmp_path / "empty")
    source = read_source(empty)
    assert source.name == "empty" and source.origin == {"kind": "hand-written"}
    assert len(load([base, empty]).devices) == 4


@pytest.mark.util
@pytest.mark.parametrize(
    "architectures, resources, devices, rule",
    [
        ({"zynquplus": dict(ZYNQUPLUS, dsp="DSP99")}, [], [], "architectures.json:2: schema"),
        ({"zynquplus": dict(ZYNQUPLUS, extra=1)}, [], [], "unknown keys"),
        ({"zynquplus": dict(ZYNQUPLUS, **{"class": "zynq"})}, [], [], "schema"),
        (None, [dict(record("zu3", [ZU3]), slr_count=2)], [], "resources.jsonl:1: slr-count"),
        (None, [record("zu3", [ZU3], totals=dict(ZU3, lut=1))], [], "slr-sum"),
        (None, [record("zu3", [ZU3], capped=True)], [], "slr-sum"),
        (None, [record("zu3", [ZU3]), record("zu3", [ZU3])], [], "resources.jsonl:2: duplicate"),
        (None, [], [device("xczu3eg", "zu3")], "devices.jsonl:1: resources-undefined"),
        (
            None,
            [record("zu3", [ZU3])],
            [device("xczu3eg", "zu3"), device("XCZU3EG", "zu3")],
            "devices.jsonl:2: duplicate",
        ),
        (None, [record("zu3", [dict(ZU3, lut=-1)])], [], "schema"),
    ],
)
def test_platform_data_is_validated_on_load(tmp_path, architectures, resources, devices, rule):
    broken = write_platform_data(tmp_path / "broken", "broken", architectures, resources, devices)
    with pytest.raises(PlatformError) as refused:
        read_source(broken)
    assert refused.value.code == "invalid-platform-data"
    assert f"'broken' ({broken})" in str(refused.value)
    assert rule in str(refused.value)


@pytest.mark.util
def test_the_union_refuses_a_device_of_an_undefined_architecture(base, tmp_path):
    stray = write_platform_data(
        tmp_path / "stray",
        "stray",
        resources=[record("vx", [ZU3])],
        devices=[device("xcmy100t", "vx", "virtex7", "virtex7")],
    )
    with pytest.raises(PlatformError, match="architecture-undefined: device xcmy100t"):
        load([base, stray])
    uram = write_platform_data(
        tmp_path / "uram",
        "uram",
        {"virtex7": VIRTEX7},
        [record("vx", [ZU7])],
        [device("xcmy100t", "vx", "virtex7", "virtex7")],
    )
    with pytest.raises(PlatformError, match="uram: device xcmy100t has UltraRAM"):
        load([base, uram])


@pytest.mark.util
def test_a_manifest_names_its_source(tmp_path):
    source = write_platform_data(tmp_path / "named", "named")
    assert read_source(source).describe() == "named [hand-written]"
    manifest = {"name": "x", "source": {"kind": "vivado", "release": "2025.2", "build": "b"}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    assert read_source(source).describe() == "x [Vivado 2025.2]"
    manifest["source"] = {"kind": "vivado"}
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PlatformError, match="manifest.json:1: schema"):
        read_source(source)


@pytest.fixture
def reloaded():
    """Clear the per-process cache of the loaded platform data, before and after."""
    registry._loaded.cache_clear()
    yield
    registry._loaded.cache_clear()


@pytest.mark.util
def test_platform_data_is_loaded_from_finn_platform_path(base, tmp_path, monkeypatch, reloaded):
    monkeypatch.setattr(registry, "DATA", base)
    other = write_platform_data(
        tmp_path / "other",
        "other",
        resources=[record("zu9", [ZU7])],
        devices=[device("xczu9eg", "zu9")],
    )
    monkeypatch.setenv("FINN_PLATFORM_PATH", str(other))
    assert [each.name for each in registry.platform_sources()] == ["base", "other"]
    assert catalog.get_part_device("xczu9eg-ffvb1156-2-e").name == "xczu9eg"
    monkeypatch.setenv("FINN_PLATFORM_PATH", "")
    with pytest.raises(PlatformError) as refused:
        catalog.get_part_device("xczu9eg-ffvb1156-2-e")
    assert refused.value.code == "unknown-part"


@pytest.mark.util
def test_an_unknown_part_is_refused_with_how_to_add_it():
    with pytest.raises(PlatformError) as refused:
        catalog.get_part_device("xczu3ex-sbva484-1-e")
    assert refused.value.code == "unknown-part"
    message = str(refused.value)
    release = read_source(DATA).origin["release"]
    assert "FPGA part 'xczu3ex-sbva484-1-e' is not in FINN's platform data" in message
    assert f"(from: built-in [Vivado {release}])" in message
    assert "Closest devices: xczu3eg" in message
    assert "finn-platform extract --parts 'xczu3ex*' --out <DIR>" in message
    assert "export FINN_PLATFORM_PATH=<DIR>" in message
    assert "finn-platform check <DIR>" in message
    assert "Targets, parts, and platform data" in message
    for name in ("", None):
        with pytest.raises(PlatformError, match="is not an FPGA part name"):
            catalog.get_part_device(name)


# -- FINN's built-in platform data ----------------------------------------------------


@pytest.mark.util
def test_the_built_in_data_is_valid_with_sorted_records_keyed_by_their_facts():
    builtin = read_source(DATA)
    assert builtin.name == "built-in" and builtin.origin["kind"] == "vivado"
    assert union([builtin]).devices == builtin.devices
    for name, key in (("resources.jsonl", "digest"), ("devices.jsonl", "name")):
        keys = [json.loads(line)[key] for line in (DATA / name).read_text().splitlines()]
        assert keys == sorted(keys) and len(set(keys)) == len(keys), name
    assert builtin.counts["devices"] == len(builtin.devices) > 0
    assert builtin.counts["capped"] == sum(each.capped for each in builtin.devices.values())


@pytest.mark.util
def test_the_built_in_data_defines_every_architecture_finn_can_use():
    architectures = read_source(DATA).architectures
    seven_series = {"artix7", "kintex7", "virtex7", "spartan7", "zynq"}
    assert seven_series <= set(architectures)
    assert {architectures[each].dsp for each in seven_series} == {"DSP48E1"}
    by_dsp = defaultdict(set)
    for each in architectures.values():
        by_dsp[each.dsp].add(each.class_)
    assert by_dsp["DSP58"] == {"versal"}
    assert {each.name for each in architectures.values() if each.class_ == "zynquplus"} == {
        "zynquplus",
        "zynquplusRFSOC",
    }


@pytest.mark.util
def test_a_built_in_device_of_one_slr_is_its_totals_and_one_of_several_states_each():
    for each in read_source(DATA).devices.values():
        if len(each.slrs) == 1:
            assert each.slrs == (each.resources,) and not each.capped, each.name
            continue
        totals = astuple(each.resources)
        summed = tuple(sum(counts) for counts in zip(*map(astuple, each.slrs)))
        if each.capped:
            assert all(s >= t for s, t in zip(summed, totals)), each.name
        else:
            assert summed == totals, each.name


@pytest.mark.util
def test_a_device_whose_name_extends_another_finds_its_own_parts():
    names = [each.name for each in read_source(DATA).devices.values()]
    extending = [n for n in names if any(n != m and n.lower().startswith(m.lower()) for m in names)]
    assert extending  # the _CIV variants, among others
    for name in extending:
        assert catalog.get_part_device(name + "-pkg-1-e").name == name


@pytest.mark.util
@pytest.mark.parametrize(
    "file, text, where",
    [
        # two definitions of one architecture: JSON keeps the last
        (
            "architectures.json",
            '{\n "virtex7": {"dsp": "DSP48E1", "class": null, "uram": false},\n'
            ' "virtex7": {"dsp": "DSP58", "class": null, "uram": false}\n}\n',
            "architectures.json:3: duplicate: key 'virtex7' is stated twice in the "
            "top-level object",
        ),
        # one field stated twice in one architecture
        (
            "architectures.json",
            '{\n "virtex7": {\n  "dsp": "DSP48E1",\n  "class": null,\n  "uram": false,\n'
            '  "dsp": "DSP58"\n }\n}\n',
            "architectures.json:6: duplicate: key 'dsp' is stated twice in 'virtex7'",
        ),
        (
            "devices.jsonl",
            '{"name": "xczu3eg", "name": "xczu7ev", "architecture": "zynquplus", '
            '"family": "zynquplus", "resources": "zu3"}\n',
            "devices.jsonl:1: duplicate: key 'name' is stated twice in the top-level object",
        ),
        (
            "manifest.json",
            '{"name": "a", "name": "b", "source": {"kind": "hand-written"}}',
            "manifest.json:1: duplicate: key 'name' is stated twice",
        ),
    ],
)
def test_a_key_stated_twice_in_one_object_is_refused(base, file, text, where):
    (base / file).write_text(text)
    with pytest.raises(PlatformError) as found:
        read_source(base)
    assert where in str(found.value)
