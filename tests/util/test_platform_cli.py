# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json

from finn.platform import registry
from finn.platform.cli import main
from finn.platform.registry import DATA

MY100T = {"lut": 100000, "ff": 200000, "bram18": 250, "uram": 0, "dsp": 350}


def write_my_devices(directory, totals=MY100T):
    """Hand-written platform data of one made-up Virtex-7 device, as the docs describe it."""
    directory.mkdir()
    manifest = {"name": "my-devices", "source": {"kind": "hand-written", "note": "made up"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    record = {"digest": "my100t", "slr_count": 1, "totals": totals, "slrs": [totals]}
    (directory / "resources.jsonl").write_text(json.dumps(record) + "\n")
    device = {
        "name": "xcmy100t",
        "architecture": "virtex7",
        "family": "virtex7",
        "resources": "my100t",
    }
    (directory / "devices.jsonl").write_text(json.dumps(device) + "\n")
    return directory


@pytest.fixture
def reloaded():
    registry._loaded.cache_clear()
    yield
    registry._loaded.cache_clear()


@pytest.mark.util
def test_check_accepts_the_built_in_and_hand_written_platform_data(tmp_path, capsys):
    assert main(["check", str(DATA)]) == 0
    assert main(["check", str(write_my_devices(tmp_path / "my-devices"))]) == 0
    out = capsys.readouterr().out
    assert "OK: built-in [Vivado " in out
    assert "OK: my-devices [hand-written]" in out and "0 architectures, 1 devices" in out


@pytest.mark.util
def test_check_refuses_broken_platform_data_naming_file_line_and_rule(tmp_path, capsys):
    broken = write_my_devices(tmp_path / "broken", dict(MY100T, lut=-1))
    assert main(["check", str(broken)]) == 1
    err = capsys.readouterr().err
    assert "finn-platform: invalid-platform-data: 'my-devices'" in err
    assert "resources.jsonl:1: schema" in err


@pytest.mark.util
def test_check_refuses_platform_data_that_contradicts_the_built_in(tmp_path, capsys):
    other = tmp_path / "other"
    other.mkdir()
    zu3 = {"lut": 1, "ff": 1, "bram18": 1, "uram": 0, "dsp": 1}
    record = {"digest": "zu3", "slr_count": 1, "totals": zu3, "slrs": [zu3]}
    (other / "resources.jsonl").write_text(json.dumps(record) + "\n")
    device = {"name": "xczu3eg", "architecture": "zynquplus", "family": "zynquplus"}
    (other / "devices.jsonl").write_text(json.dumps(dict(device, resources="zu3")) + "\n")
    assert main(["check", str(other)]) == 1
    err = capsys.readouterr().err
    assert "conflict: device xczu3eg is defined differently in 'built-in' (" in err
    assert f"and 'other' ({other})" in err


@pytest.mark.util
def test_show_names_what_a_part_resolves_to_and_from_where(tmp_path, capsys, monkeypatch, reloaded):
    assert main(["show", "xcvu5p-flva2104-1-e"]) == 0
    out = capsys.readouterr().out
    assert "device        xcvu5p (from built-in)" in out
    assert "architecture  virtexuplus: DSP48E2, class none, UltraRAM yes" in out
    assert "SLRs          2 (each SLR's site capacity; the totals cap their sum)" in out
    assert main(["show", "xcmy100tffg1761-2"]) == 1
    assert "finn-platform: unknown-part: FPGA part 'xcmy100tffg1761-2'" in capsys.readouterr().err
    monkeypatch.setenv("FINN_PLATFORM_PATH", str(write_my_devices(tmp_path / "my-devices")))
    assert main(["show", "xcmy100tffg1761-2"]) == 0
    out = capsys.readouterr().out
    assert "device        xcmy100t (from my-devices)" in out
    assert "architecture  virtex7: DSP48E1, class none, UltraRAM no (from built-in)" in out
    assert main(["sources"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("built-in [Vivado ") and out[1].startswith("my-devices [hand-written]")


@pytest.mark.util
def test_check_validates_the_directories_together_as_a_build_loads_them(tmp_path, capsys):
    first = write_my_devices(tmp_path / "first")
    same = write_my_devices(tmp_path / "same")
    assert main(["check", str(first), str(same)]) == 0
    assert "OK: together with the built-in platform data, in this order" in capsys.readouterr().out
    # each is valid with the built-in data, but they contradict each other
    other = write_my_devices(tmp_path / "other", dict(MY100T, dsp=351))
    assert main(["check", str(first), str(other)]) == 1
    err = capsys.readouterr().err
    assert "conflict: device xcmy100t is defined differently in 'my-devices'" in err


@pytest.mark.util
def test_check_accepts_a_directory_that_builds_on_another(tmp_path, capsys):
    arch = tmp_path / "arch"
    arch.mkdir()
    (arch / "architectures.json").write_text(
        json.dumps({"myarch": {"dsp": "DSP48E2", "class": None, "uram": False}})
    )
    devices = write_my_devices(tmp_path / "devices")
    device = json.loads((devices / "devices.jsonl").read_text())
    device.update(name="xcmy100", architecture="myarch", family="myarch")
    (devices / "devices.jsonl").write_text(json.dumps(device) + "\n")
    assert main(["check", str(arch), str(devices)]) == 0
    assert main(["check", str(devices)]) == 1
    assert "architecture-undefined: device xcmy100" in capsys.readouterr().err
