# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Code generation for the finn-rtllib pad core, shared by the RTL padding ops."""

import os


def pad_hdl_dir() -> str:
    """Directory of the pad core and its template."""
    return os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/pad/hdl")


def pad_hdl(
    top: str,
    c: int,
    pe: int,
    w: int,
    ih: int,
    iw: int,
    pads: tuple[int, int, int, int],
    eager: bool,
    tokens: tuple[str, str] | None = None,
) -> str:
    """Verilog top module instantiating the pad core.

    ``pads`` is (N, W, S, E), the order of FINN's ``Padding`` attribute
    (top, left, bottom, right). ``tokens`` are sized Verilog literals of the west
    and east token tables, ``max(1, PAD_W) * c * w`` and ``max(1, PAD_E) * c * w``
    bits, leftmost token in the least significant bits; without them the
    padding is zeros.
    """
    pad_n, pad_w, pad_s, pad_e = (int(p) for p in pads)
    values = {
        "TOP_MODULE_NAME": top,
        "C": int(c),
        "PE": int(pe),
        "W": int(w),
        "IH": int(ih),
        "IW": int(iw),
        "PAD_N": pad_n,
        "PAD_S": pad_s,
        "PAD_W": pad_w,
        "PAD_E": pad_e,
        "EAGER": int(eager),
        "TOKENS": int(tokens is not None),
        "TOKENS_W_BITS": max(1, pad_w) * c * w,
        "TOKENS_E_BITS": max(1, pad_e) * c * w,
        "TOKENS_W": tokens[0] if tokens is not None else "0",
        "TOKENS_E": tokens[1] if tokens is not None else "0",
    }
    with open(os.path.join(pad_hdl_dir(), "pad_template.v"), "r") as f:
        template = f.read()
    for key, value in values.items():
        template = template.replace("$%s$" % key, str(value))
    return template
