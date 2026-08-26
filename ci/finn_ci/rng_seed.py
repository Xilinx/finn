# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
from finn_ci import sharding


def seed_from_nodeid(nodeid):
    group_suffix = sharding.GROUP_SUFFIX_RE.search(nodeid)
    if group_suffix is not None:
        nodeid = nodeid[: group_suffix.start()]
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
