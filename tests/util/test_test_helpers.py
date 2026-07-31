# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.util.test import make_runtime_weight_stream

pytestmark = pytest.mark.util


class RuntimeWeightOp:
    def __init__(self, content):
        self.content = content

    def make_weight_file(self, weights, mode, path):
        assert weights == "weights"
        assert mode == "decoupled_runtime"
        with open(path, "w") as f:
            f.write(self.content)


def test_make_runtime_weight_stream_removes_clean_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path))

    stream = make_runtime_weight_stream(RuntimeWeightOp("1\na\nff\n"), "weights")

    assert stream == [1, 10, 255]
    assert list(tmp_path.iterdir()) == []


def test_make_runtime_weight_stream_retains_failed_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        make_runtime_weight_stream(RuntimeWeightOp("not-hex\n"), "weights")

    retained = list(tmp_path.iterdir())
    assert len(retained) == 1
    assert (retained[0] / "weights.dat").read_text() == "not-hex\n"
