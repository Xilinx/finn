# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JSON read/write helpers shared across the finn_ci package."""

import json
import os
import sys
import tempfile


def read_json(path, default=None):
    if not path:
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        # File present but unreadable or malformed: warn so a corrupt timing
        # file does not silently degrade sharding to round-robin.
        print(
            "finn_ci jsonio read_json: %s: %s: %s" % (path, exc.__class__.__name__, exc),
            file=sys.stderr,
        )
        return default


def write_json_atomic(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    # exist_ok=True so two concurrent first-time callers on a shared NFS root
    # cannot race on mkdir.
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.rename(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
