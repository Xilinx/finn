# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os

from finn.util.basic import CppBuilder


def _builder(tmp_path, flags=("-O1",)):
    builder = CppBuilder()
    builder.append_sources(tmp_path / "*.cpp")
    builder.append_includes(["-I", str(tmp_path / "inc"), *flags])
    builder.set_executable_path(tmp_path / "node_model")
    return builder


def _mtime(tmp_path):
    return os.stat(tmp_path / "node_model").st_mtime_ns


@pytest.mark.util
def test_cppbuilder_rebuilds_only_changed_executables(tmp_path):
    (tmp_path / "inc").mkdir()
    (tmp_path / "inc" / "value.h").write_text("#define VALUE 1\n")
    (tmp_path / "main.cpp").write_text('#include "value.h"\nint main() { return VALUE; }\n')

    assert _builder(tmp_path).build_if_changed(tmp_path)
    built = _mtime(tmp_path)

    # unchanged, also if the sources are regenerated with the same contents
    assert not _builder(tmp_path).build_if_changed(tmp_path)
    (tmp_path / "main.cpp").write_text('#include "value.h"\nint main() { return VALUE; }\n')
    assert not _builder(tmp_path).build_if_changed(tmp_path)
    assert _mtime(tmp_path) == built

    # a changed source, included header or command line triggers a rebuild
    (tmp_path / "main.cpp").write_text('#include "value.h"\nint main() { return VALUE + 1; }\n')
    assert _builder(tmp_path).build_if_changed(tmp_path)
    (tmp_path / "inc" / "value.h").write_text("#define VALUE 2\n")
    assert _builder(tmp_path).build_if_changed(tmp_path)
    assert _builder(tmp_path, flags=("-O2",)).build_if_changed(tmp_path)
    assert not _builder(tmp_path, flags=("-O2",)).build_if_changed(tmp_path)

    # a missing executable is rebuilt
    os.remove(tmp_path / "node_model")
    assert _builder(tmp_path, flags=("-O2",)).build_if_changed(tmp_path)
    assert os.path.isfile(tmp_path / "node_model")
