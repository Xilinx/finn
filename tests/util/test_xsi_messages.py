# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Framing and classification of the simulator messages that decide every rtlsim's
pass/fail (finn_xsi/finn_xsi/messages.py; C++ twin: drain_messages() in
finn_xsi/rtlsim_xsi.cpp). The frames below are what xsi_get_error_info() returns
under Vivado 2026.1 for the named SystemVerilog statements (paths shortened)."""

import pytest

import os
import sys

# As in test_xsi_pkg_ordering.py: finn_xsi is not pip-installed, and messages has
# no xsi C-extension import, so this runs without a built finn_xsi.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_XSI_ROOT = os.path.join(os.environ.get("FINN_ROOT", _REPO_ROOT), "finn_xsi")
if _XSI_ROOT not in sys.path:
    sys.path.insert(0, _XSI_ROOT)

from finn_xsi.messages import (  # noqa: E402
    SimulationError,
    is_error_message,
    split_messages,
)

LOC = "Time: 20 ns  Iteration: 1  Process: /top/Always12_4  Scope: top  File: top.sv Line: 14\n"
INFO = "Info: an info\n" + LOC  # $info("an info")
WARNING = "Warning: a warning\n" + LOC  # $warning("a warning")
ERROR = "Error: an error\n" + LOC  # $error("an error")
FATAL = "Fatal: a fatal\n" + LOC  # $fatal(1, "a fatal")
ASSERTION = "Error: Assertion violation\nTime: 50 ns  Iteration: 0  Process: /top  File: top.sv\n"
# from the simulator kernel, framed without the leading @
KERNEL_WARNING = "WARNING: File data.dat referenced on top.sv at line 7 cannot be opened\n"
KERNEL_ERROR = "ERROR: File data.dat referenced on top.sv at line 7 cannot be opened\n"


def design(text):
    "Frame of a message from the design ($display, $error, ...)."
    return '@"' + text + '"\n'


def kernel(text):
    "Frame of a message from the simulator kernel."
    return '"' + text + '"\n'


@pytest.mark.util
@pytest.mark.parametrize("info", [None, ""])
def test_split_messages_empty(info):
    assert split_messages(info) == []


@pytest.mark.util
@pytest.mark.parametrize(
    "msgs",
    [
        pytest.param(["plain display\n"], id="display"),
        pytest.param(["\n"], id="empty-display"),
        pytest.param([""], id="empty-frame"),
        pytest.param(["two\nlines\n"], id="multi-line"),
        pytest.param([INFO, WARNING, ERROR, FATAL], id="severities"),
        pytest.param([ASSERTION], id="assertion"),
        # $write without a newline: the frame closes all the same
        pytest.param(["no newline"], id="write"),
        # $display("say \"hi\""), $display("ends with quote\""): no escaping
        pytest.param(['say "hi"\n', 'ends with quote"\n'], id="quotes"),
        # $display("@\"fake frame\""): a frame start inside a message
        pytest.param(['@"fake frame"\n', "next\n"], id="fake-frame"),
        pytest.param(['"\n', 'x"\n'], id="quote-newline-text"),
    ],
)
def test_split_messages_design_frames(msgs):
    assert split_messages("".join(design(m) for m in msgs)) == msgs


@pytest.mark.util
def test_split_messages_kernel_frames():
    info = kernel(KERNEL_WARNING) + design(ERROR) + kernel(KERNEL_ERROR) + design("done\n")
    assert split_messages(info) == [KERNEL_WARNING, ERROR, KERNEL_ERROR, "done\n"]


@pytest.mark.util
@pytest.mark.parametrize(
    "info, msgs",
    [
        # an incomplete frame is kept verbatim, as are the frames after it
        pytest.param(design("a\n") + '@"partial', ["a\n", '@"partial'], id="open-frame"),
        pytest.param(design("a\n") + '@"part"', ["a\n", '@"part"'], id="no-newline"),
        # a lone @ is no frame start, so the frame before it does not close
        pytest.param(design("a\n") + "@", [design("a\n") + "@"], id="lone-at"),
        pytest.param(design("a\n") + '"', ["a\n", '"'], id="lone-quote"),
        # a frame that does not close absorbs the next one
        pytest.param('@"x\n' + design("b\n"), ['x\n@"b\n'], id="open-frame-absorbs-next"),
        # text that is no frame at all
        pytest.param("stray text", ["stray text"], id="unframed"),
        pytest.param(design("a\n") + "stray\n", [design("a\n") + "stray\n"], id="trailing-text"),
        pytest.param(design("a\n") + "\n", [design("a\n") + "\n"], id="trailing-newline"),
    ],
)
def test_split_messages_unframed_rest(info, msgs):
    """A frame closes with "\\n followed by the end or the next frame. Anything from
    the first frame that does not close on is kept verbatim, so no text is lost."""
    assert split_messages(info) == msgs


@pytest.mark.util
@pytest.mark.parametrize(
    "cut, first, second, detected",
    [
        # after the @: the rest reads as a kernel frame
        pytest.param(1, ["@"], [ERROR], True, id="after-at"),
        # within the frame: both parts verbatim, the second is classified by its
        # start, so the error is only seen if the cut falls just before it
        pytest.param(2, ['@"'], [ERROR + '"\n'], True, id="before-text"),
        pytest.param(7, ['@"Error'], [ERROR[5:] + '"\n'], False, id="in-severity"),
        pytest.param(20, ['@"' + ERROR[:18]], [ERROR[18:] + '"\n'], False, id="in-text"),
        pytest.param(len(ERROR) + 3, ['@"' + ERROR + '"'], ["\n"], False, id="before-newline"),
    ],
)
def test_split_messages_split_across_reads(cut, first, second, detected):
    """Each read is framed on its own. XSI hands out whole frames, but a frame cut
    between two reads would lose no text: its parts are kept verbatim. Its
    classification, though, mostly is lost."""
    info = design(ERROR)
    assert split_messages(info[:cut]) == first
    assert split_messages(info[cut:]) == second
    assert any(is_error_message(m) for m in first + second) == detected


@pytest.mark.util
@pytest.mark.parametrize(
    "msg, error",
    [
        (ERROR, True),
        (FATAL, True),
        (ASSERTION, True),
        (KERNEL_ERROR, True),
        ("FATAL_ERROR: Vivado Simulator kernel has discovered an exceptional condition\n", True),
        (INFO, False),
        (WARNING, False),
        (KERNEL_WARNING, False),
        ("plain display\n", False),
        ("", False),
        # only a leading severity counts, not one quoted in the text
        ("got Error: 3\n", False),
        (" Error: indented\n", False),
        ("error: lower case\n", False),
    ],
)
def test_is_error_message(msg, error):
    assert is_error_message(msg) == error


@pytest.mark.util
def test_errors_among_split_messages():
    info = design("start\n") + design(INFO) + design(ERROR) + design(WARNING) + design(FATAL)
    errors = [m for m in split_messages(info) if is_error_message(m)]
    assert errors == [ERROR, FATAL]


@pytest.mark.util
def test_simulation_error_is_runtime_error():
    assert issubclass(SimulationError, RuntimeError)
