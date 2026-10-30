# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Assert and record the Brevitas overlay in pytest and each xdist worker."""
import pytest

import brevitas
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

# Run in a new interpreter too; sys.executable and PATH must retain the overlay.
PROBE = """import importlib.metadata as m, json
import brevitas
x = m.distribution("brevitas")
print(json.dumps({
    "path": brevitas.__file__,
    "distribution_path": str(x.locate_file("brevitas/__init__.py")),
    "direct_url": json.loads(x.read_text("direct_url.json") or "{}"),
}))
"""


def pytest_sessionstart(session):
    """Fail before collection if any execution context imports another revision."""
    expected = os.environ.get("FINN_BREVITAS_EXPECTED_COMMIT")
    if not expected:
        raise pytest.UsageError("Brevitas candidate plugin requires FINN_BREVITAS_EXPECTED_COMMIT")
    dist = importlib.metadata.distribution("brevitas")
    current = {
        "path": brevitas.__file__,
        "distribution_path": str(dist.locate_file("brevitas/__init__.py")),
        "direct_url": json.loads(dist.read_text("direct_url.json") or "{}"),
    }
    children = {}
    # Both launch styles occur in FINN helpers. They must select the overlay.
    for name, executable in (("sys_executable", sys.executable), ("path_python", "python")):
        children[name] = json.loads(
            subprocess.check_output([executable, "-c", PROBE], text=True, timeout=600)
        )
    for name, context in {"pytest": current, **children}.items():
        actual = context["direct_url"].get("vcs_info", {}).get("commit_id")
        if (
            actual != expected
            or Path(context["path"]).resolve() != Path(context["distribution_path"]).resolve()
        ):
            raise pytest.UsageError(
                f"{name} imported Brevitas {actual} from {context['path']}; expected {expected}"
            )
    # Overlay and base site-packages can both expose a distribution. Keep the
    # first metadata record on Python's lookup path, matching version(name).
    effective = {}
    for dist in importlib.metadata.distributions():
        key = dist.metadata["Name"].lower().replace("_", "-")
        effective.setdefault(key, dist)
    distributions = [
        {
            "name": d.metadata["Name"],
            "version": d.version,
            "direct_url": json.loads(d.read_text("direct_url.json") or "{}"),
        }
        for _, d in sorted(effective.items())
    ]
    worker = getattr(session.config, "workerinput", {}).get("workerid", "parent")
    output = Path(os.environ.get("FINN_BREVITAS_PROVENANCE", "brevitas-provenance"))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{worker}.json").write_text(
        json.dumps(
            {
                "requested_commit": expected,
                "finn": {
                    key: os.environ.get(key)
                    for key in (
                        "FINN_SOURCE_REVISION",
                        "FINN_SOURCE_DIRTY",
                        "FINN_IMAGE_ID",
                        "FINN_IMAGE_REVISION",
                    )
                },
                "pytest": current,
                "subprocesses": children,
                "distributions": distributions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
