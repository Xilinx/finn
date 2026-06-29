# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JSON read helper shared across the finn_ci package."""

import json
import sys


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
