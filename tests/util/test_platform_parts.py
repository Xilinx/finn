# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import re
from pathlib import Path

from finn.platform import (
    PlatformError,
    dsp_block,
    dsp_datapath_limits,
    get_part_device,
    is_versal,
    registry,
)
from finn.platform.registry import DATA, read_source

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.util
def test_the_helpers_answer_the_architecture_of_every_built_in_device():
    builtin = read_source(DATA)
    for device in builtin.devices.values():
        part = device.name + "-pkg-1-e"
        architecture = builtin.architectures[device.architecture]
        assert is_versal(part) == (device.architecture == "versal"), device.name
        assert dsp_block(part) == architecture.dsp, device.name


@pytest.mark.util
def test_the_helpers_read_the_platform_data_not_the_name(tmp_path, monkeypatch):
    totals = {"lut": 1, "ff": 2, "bram18": 0, "uram": 0, "dsp": 1}
    record = {"digest": "r", "slr_count": 1, "totals": totals, "slrs": [totals]}
    (tmp_path / "resources.jsonl").write_text(json.dumps(record) + "\n")
    devices = [
        {"name": "xcnew1", "architecture": "versal", "family": "versalnew", "resources": "r"},
        {"name": "xcvc9", "architecture": "zynquplus", "family": "zynquplus", "resources": "r"},
    ]
    (tmp_path / "devices.jsonl").write_text("".join(json.dumps(each) + "\n" for each in devices))
    registry._loaded.cache_clear()
    monkeypatch.setenv("FINN_PLATFORM_PATH", str(tmp_path))
    try:
        assert is_versal("xcnew1-pkg1-1-e") and dsp_block("xcnew1-pkg1-1-e") == "DSP58"
        assert not is_versal("xcvc9-pkg1-1-e") and dsp_block("xcvc9-pkg1-1-e") == "DSP48E2"
    finally:
        registry._loaded.cache_clear()


@pytest.mark.util
@pytest.mark.parametrize("helper", [is_versal, dsp_block])
def test_a_part_the_platform_data_lacks_is_refused(helper):
    with pytest.raises(PlatformError) as refused:
        helper("xcnosuch1-ffg900-2")
    assert refused.value.code == "unknown-part"
    assert "FPGA part 'xcnosuch1-ffg900-2' is not in FINN's platform data" in str(refused.value)


@pytest.mark.util
def test_an_unknown_dsp_block_is_refused():
    with pytest.raises(ValueError, match="'DSP99' is not a DSP block FINN knows"):
        dsp_datapath_limits("DSP99")


# -- coverage: every part FINN names resolves ---------------------------------------------

# a part name: xc, xq or xa, then the device's letters and digits, then the package,
# speed and temperature, if any
PART = re.compile(r"""["'\s=(\[]((?:xc|xq|xa)[a-z0-9]{3,}(?:[-_][a-z0-9]+)*)""", re.I)
NOT_PARTS = ("xcelium", "xclbin", "xcvr", "xcl")
SCANNED = ("src", "tests", "notebooks", "docs")
SUFFIXES = {".py", ".ipynb", ".rst", ".md", ".sh", ".tcl", ".cfg", ".txt", ".json"}
# the platform package's own data and tests, which name made-up parts
SKIPPED = ("src/finn/platform/", "tests/util/test_platform_")
# the made-up part the docs and the config check tests name because no platform data holds it
UNKNOWN_ON_PURPOSE = {"xcmy100t", "xcmy100tffg1761-2"}


def named_parts():
    """Every part-like name in FINN's sources, tests, notebooks and docs, with a file
    that names it."""
    found = {}
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*")):
            relative = path.relative_to(ROOT).as_posix()
            if path.suffix not in SUFFIXES or not path.is_file() or relative.startswith(SKIPPED):
                continue
            for match in PART.finditer(" " + path.read_text(errors="ignore")):
                name = match.group(1)
                if re.search(r"\d", name) and not name.lower().startswith(NOT_PARTS):
                    found.setdefault(name, relative)
    return found


@pytest.mark.util
def test_every_part_finn_names_is_in_the_built_in_platform_data():
    found = named_parts()
    assert "xc7z020clg400-1" in found and "xcvc1902-vsva2197-2MP-e-S" in found
    missing = {}
    for name, where in found.items():
        if name in UNKNOWN_ON_PURPOSE:
            continue
        try:
            get_part_device(name)
        except PlatformError:
            missing[name] = where
    assert not missing, f"parts FINN names that no platform data holds: {missing}"
