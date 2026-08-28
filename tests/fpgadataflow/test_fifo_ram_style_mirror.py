# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Guards finn.custom_op.fpgadataflow.streamingfifo._resolve() against drift from the
# RAM_STYLE_EFF selection in finn-rtllib/fifo/hdl/fifo.sv. _resolve() is a hand-written
# Python mirror of that selection, used for resource estimation and the build report; if
# someone edits fifo.sv without updating _resolve(), this test fails.
#
# It does not restate the selection logic: it parses fifo.sv, turns each condition into a
# callable, and uses that as the oracle. So a threshold change in fifo.sv moves the oracle
# too, and the mismatch surfaces against the unchanged _resolve().

import pytest

import os
import re

from finn.custom_op.fpgadataflow.streamingfifo import _resolve

# fifo.sv spells the SRL backing "shift"; FINN's vocabulary calls it "srl" (translated
# back at the codegen boundary in streamingfifo_rtl.py).
RTL_TO_FINN = {"shift": "srl"}


def _read_ram_style_eff():
    """Return the ordered (condition_source, result_token) branches of fifo.sv's
    RAM_STYLE_EFF selection, or fail loudly if it cannot be located/parsed."""
    sv_path = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "fifo", "hdl", "fifo.sv")
    with open(sv_path, "r") as f:
        sv_text = f.read()
    m = re.search(r"localparam\s+RAM_STYLE_EFF\s*=(.*?);", sv_text, re.S)
    if m is None:
        pytest.fail(
            "Could not find the RAM_STYLE_EFF selection in %s. If fifo.sv's style "
            "selection was restructured, update streamingfifo._resolve() and this test." % sv_path
        )
    body = m.group(1)
    # drop /* ... */ comments (e.g. the trailing "/* else */") so only code remains
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    # the expression is  <cond>? <result> : <cond>? <result> : ... : <result>
    segments = [seg.strip() for seg in body.split(":") if seg.strip()]
    branches = []
    for seg in segments:
        if "?" in seg:
            cond, result = seg.split("?", 1)
            branches.append((cond.strip(), result.strip()))
        else:
            # the final fall-through branch has no condition
            branches.append((None, seg.strip()))
    if len(branches) < 2 or branches[-1][0] is not None:
        pytest.fail(
            "Parsed an unexpected RAM_STYLE_EFF shape from fifo.sv: %r. Update "
            "streamingfifo._resolve() and this test to match." % branches
        )
    return branches


def _eval_condition(cond_src, depth, width, requested):
    """Evaluate one SystemVerilog branch condition for the given inputs."""
    expr = cond_src
    expr = expr.replace("DATA_WIDTH", "width")
    expr = expr.replace("DEPTH", "depth")
    expr = expr.replace("RAM_STYLE", "requested")
    expr = expr.replace("&&", " and ").replace("||", " or ")
    try:
        return bool(
            eval(  # noqa: S307 - expression comes from the trusted in-tree fifo.sv
                expr,
                {"__builtins__": {}},
                {"depth": depth, "width": width, "requested": requested},
            )
        )
    except Exception as e:  # pragma: no cover - only hit if fifo.sv adds new syntax
        pytest.fail(
            "Could not evaluate RAM_STYLE_EFF condition %r from fifo.sv (%s). Extend "
            "this test's parser and check streamingfifo._resolve()." % (cond_src, e)
        )


def _oracle(branches, depth, width, requested):
    """The style fifo.sv elaborates, expressed in FINN's vocabulary."""
    for cond_src, result in branches:
        if cond_src is None or _eval_condition(cond_src, depth, width, requested):
            if result == "RAM_STYLE":  # the passthrough branch honors the request as-is
                token = requested
            else:
                token = result.strip('"')
            return RTL_TO_FINN.get(token, token)
    pytest.fail("RAM_STYLE_EFF produced no result for (%d, %d, %s)" % (depth, width, requested))


# boundaries of every threshold in the selection, plus one on each side
DEPTHS = [2, 32, 33, 34, 63, 64, 65, 256, 257, 258, 2027, 2028, 2029, 65536]
WIDTHS = [1, 4, 5, 6, 11, 12, 13, 64]
REQUESTS = ["auto", "srl", "distributed", "block", "ultra"]


def test_resolve_mirrors_fifo_sv():
    branches = _read_ram_style_eff()
    for depth in DEPTHS:
        for width in WIDTHS:
            for requested in REQUESTS:
                expected = _oracle(branches, depth, width, requested)
                actual = _resolve(depth, width, requested)
                assert actual == expected, (
                    "_resolve(%d, %d, %s) = %s but fifo.sv elaborates %s. "
                    "streamingfifo._resolve() has drifted from fifo.sv's RAM_STYLE_EFF."
                    % (depth, width, requested, actual, expected)
                )
