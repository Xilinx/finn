############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import errno
import os
import shutil
import time
from pathlib import Path

from finn.util.basic import make_build_dir, robust_rmtree


@pytest.mark.util
def test_robust_rmtree_succeeds_first_attempt():
    target = make_build_dir("test_rmtree_")
    sub = os.path.join(target, "sub")
    os.makedirs(sub)
    Path(os.path.join(sub, "f")).write_text("x")
    robust_rmtree(target)
    assert not os.path.exists(target)


@pytest.mark.util
def test_robust_rmtree_missing_path_is_noop():
    target = make_build_dir("test_rmtree_")
    robust_rmtree(os.path.join(target, "does_not_exist"))
    robust_rmtree("")
    robust_rmtree(None)
    robust_rmtree(target)


@pytest.mark.util
@pytest.mark.parametrize("transient_errno", [errno.ENOTEMPTY, errno.EBUSY])
def test_robust_rmtree_retries_on_transient_then_succeeds(monkeypatch, transient_errno):
    target = make_build_dir("test_rmtree_")
    Path(os.path.join(target, "f")).write_text("x")

    state = {"calls": 0}
    real_rmtree = shutil.rmtree

    def flaky(path, *a, **kw):
        state["calls"] += 1
        if state["calls"] < 3:
            raise OSError(transient_errno, "fake", str(path))
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(shutil, "rmtree", flaky)
    # skip the delay after the stubbed failures only: the real removal (NFS in
    # CI) may need the backoff that robust_rmtree provides
    real_sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: real_sleep(s) if state["calls"] >= 3 else None)

    robust_rmtree(target, retries=5)

    assert state["calls"] >= 3
    assert not os.path.exists(target)


@pytest.mark.util
def test_robust_rmtree_propagates_non_transient_oserror(monkeypatch):
    target = make_build_dir("test_rmtree_")

    calls = []

    def always_eacces(path, *a, **kw):
        calls.append(path)
        raise OSError(errno.EACCES, "fake", str(path))

    monkeypatch.setattr(shutil, "rmtree", always_eacces)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(OSError) as excinfo:
        robust_rmtree(target, retries=5)

    assert excinfo.value.errno == errno.EACCES
    # No retry: an errno outside the transient ones propagates on the first attempt.
    assert len(calls) == 1


@pytest.mark.util
def test_robust_rmtree_raises_after_retries_exhausted(monkeypatch):
    target = make_build_dir("test_rmtree_")

    calls = []

    def always_enotempty(path, *a, **kw):
        calls.append(path)
        raise OSError(errno.ENOTEMPTY, "fake", str(path))

    monkeypatch.setattr(shutil, "rmtree", always_enotempty)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(OSError) as excinfo:
        robust_rmtree(target, retries=4)

    assert excinfo.value.errno == errno.ENOTEMPTY
    assert len(calls) == 4


@pytest.mark.util
def test_robust_rmtree_tolerates_concurrent_removal(monkeypatch):
    target = make_build_dir("test_rmtree_")
    real_rmtree = shutil.rmtree

    def removed_meanwhile(path, *a, **kw):
        real_rmtree(path)
        raise FileNotFoundError(errno.ENOENT, "fake", str(path))

    monkeypatch.setattr(shutil, "rmtree", removed_meanwhile)
    robust_rmtree(target)
    assert not os.path.exists(target)


@pytest.mark.util
def test_robust_rmtree_retries_when_an_entry_vanishes(monkeypatch):
    # shutil.rmtree re-raises the FileNotFoundError of an entry that vanished
    # after the directory was listed (e.g. a stale NFS listing); the tree itself
    # is still there and must not be reported as removed
    target = make_build_dir("test_rmtree_")
    Path(os.path.join(target, "f")).write_text("x")
    state = {"calls": 0}
    real_rmtree = shutil.rmtree

    def entry_vanishes(path, *a, **kw):
        state["calls"] += 1
        if state["calls"] == 1:
            raise FileNotFoundError(errno.ENOENT, "fake", os.path.join(str(path), "g"))
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(shutil, "rmtree", entry_vanishes)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    robust_rmtree(target)

    assert state["calls"] == 2
    assert not os.path.exists(target)
