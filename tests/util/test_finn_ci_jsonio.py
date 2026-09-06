# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn_ci import jsonio

pytestmark = pytest.mark.util


def test_read_json_warns_on_corrupt_file(tmp_path, capsys):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{")

    assert jsonio.read_json(str(corrupt), default={"fallback": True}) == {"fallback": True}

    captured = capsys.readouterr()
    # FileNotFoundError stays silent (idle state), corrupt files warn loudly
    # so a broken master state doesn't silently degrade sharding to round-robin.
    assert "finn_ci jsonio read_json" in captured.err
    assert str(corrupt) in captured.err


def test_read_json_missing_file_is_silent(tmp_path, capsys):
    missing = tmp_path / "absent.json"

    assert jsonio.read_json(str(missing), default=None) is None

    captured = capsys.readouterr()
    assert captured.err == ""
