# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Build unchanged upstream sources; its find: metadata omits the namespace."""
from setuptools import find_namespace_packages, setup

import json
import shutil
import sys
from pathlib import Path

# Builds run before FINN is installed; the resource implementation is stdlib only.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from finn import resources

source = Path(resources.path("finn-experimental"))
# A private build copy keeps setuptools' paths relative and the resource immutable.
shutil.rmtree("_source", ignore_errors=True)
shutil.copytree(source, "_source", ignore=shutil.ignore_patterns("*.egg-info", "build"))
origin = resources.declarations()["finn-experimental"]
(Path("_source/src/finnexperimental") / "_source.json").write_text(
    json.dumps(
        {
            "git": origin.git,
            "declared_commit": origin.commit,
            "declared_digest": origin.digest,
            "actual_digest": resources.tree_digest(source),
            "state": resources.status("finn-experimental").state,
        },
        sort_keys=True,
    )
    + "\n"
)
setup(
    packages=find_namespace_packages("_source/src"),
    package_dir={"": "_source/src"},
    license_files=["_source/LICENSE.txt"],
    package_data={"finnexperimental": ["_source.json"]},
)
