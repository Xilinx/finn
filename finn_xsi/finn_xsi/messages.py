#############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	Framing and classification of the XSI simulator messages.
# @author	Thomas B. Preußer <thomas.preusser@amd.com>
#############################################################################

# This lives apart from sim_engine.py, which imports the xsi C-extension, so
# that the logic deciding every rtlsim's pass/fail can be imported and unit-
# tested without a built finn_xsi. The C++ twin is drain_messages() in
# rtlsim_xsi.cpp; keep both in step.

import re


class SimulationError(RuntimeError):
    "The simulated design reported an $error or $fatal."


# Message frames: @"<text>"\n from the design ($display, $error, ...), "<text>"\n
# from the simulator kernel (e.g. "WARNING: File ... cannot be opened"). The
# closing "\n must be followed by the end or the next frame. As kernel messages
# never start with a newline, a following "\n is not taken as a frame start:
# this keeps text ending in a quote, e.g. $display("say \"hi\""), in one piece.
_FRAME = re.compile(r'@?"(.*?)"\n(?=@"|"(?!\n)|\Z)', re.DOTALL)


def split_messages(info):
    """Split the output of xsi_get_error_info() into individual messages.

    The frames are not escaped, so splitting is best effort: a message that
    itself contains a frame boundary is split in two. Anything from the first
    incomplete frame on is kept verbatim.
    """
    msgs, pos = [], 0
    while info and pos < len(info):
        m = _FRAME.match(info, pos)
        if m is None:
            msgs.append(info[pos:])
            break
        msgs.append(m.group(1))
        pos = m.end()
    return msgs


# Design $error/$fatal (incl. failed assertions) and simulator kernel errors
_ERROR_PREFIXES = ("Error:", "Fatal:", "ERROR:", "FATAL_ERROR:")


def is_error_message(msg):
    "Whether a message reports an error."
    return msg.startswith(_ERROR_PREFIXES)
