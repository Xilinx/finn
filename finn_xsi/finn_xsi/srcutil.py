# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os.path

# these helpers live in a separate module from adapter.py so they can be
# imported without the xsi C-extension (pulled in via sim_engine), keeping them
# unit-testable in a checkout where finn_xsi has not been built yet


def is_pkg_src(path: str) -> bool:
    """Return True for ``*_pkg.{sv,v}`` files (SystemVerilog/Verilog packages).

    ``compile_sim_obj`` uses this to move packages ahead of the modules that
    import them, which xelab requires. Matching the ``_pkg`` basename suffix is
    just a useful heuristic that may catch new finn-rtllib packages without
    manually editing a whitelist.
    """
    base = os.path.basename(path)
    return base.endswith("_pkg.sv") or base.endswith("_pkg.v")


def order_pkg_first(source_list):
    """Return ``source_list`` reordered so ``*_pkg.{sv,v}`` files come first.

    The partition is stable, so files keep their original relative order within
    each group. This only guarantees packages before modules. A package that
    imports another package still depends on ``source_list`` already listing
    them in dependency order.
    """
    pkg_srcs = []
    other_srcs = []
    for src in source_list:
        (pkg_srcs if is_pkg_src(src) else other_srcs).append(src)
    return pkg_srcs + other_srcs
