# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What FINN knows of a part: its device and the device's architecture, from the
loaded platform data (``finn.platform.registry``).
"""

import difflib
import re
from typing import Tuple

from finn.platform.registry import Architecture, Device, PlatformError, platform_data

#: (activation, weight, accumulator) operand widths in bits that fit the datapath of
#: each DSP block: its B, A and P ports
DSP_DATAPATH_LIMITS = {
    "DSP48E1": (18, 25, 48),
    "DSP48E2": (18, 27, 48),
    "DSP58": (24, 27, 58),
}


def _unknown_part(data, part: str) -> str:
    stem = re.split(r"[-_]", part.lower())[0]
    # a part name continues its device's name, so compare each device with as much
    scored = [
        (difflib.SequenceMatcher(None, stem[: len(each.name)], each.name.lower()).ratio(), each)
        for each in data.devices.values()
    ]
    close = sorted((each for each in scored if each[0] >= 0.6), key=lambda e: -e[0])[:5]
    return (
        f"FPGA part {part!r} is not in FINN's platform data (from: {data.describe()}). "
        "Closest devices: "
        + (", ".join(each.name for _, each in close) if close else "none")
        + ".\nIf the part name is right, add its device to platform data of your own (a "
        "directory listed in FINN_PLATFORM_PATH):\n"
        f"  finn-platform extract --parts '{stem}*' --out <DIR>"
        "    (needs a Vivado that installs the part)\n"
        f"  export FINN_PLATFORM_PATH=<DIR>\n"
        "or write the files by hand and check them with `finn-platform check <DIR>`.\n"
        "See FINN's documentation: Targets, parts, and platform data."
    )


def get_part_device(part: str) -> Device:
    """The device of the FPGA part ``part`` (a device's own name gives that device).
    Raises ``PlatformError`` (``unknown-part``) for a part of no device in the loaded
    platform data, naming where it comes from, the closest devices and how to add the
    part."""
    if not isinstance(part, str) or not part:
        raise PlatformError("unknown-part", f"{part!r} is not an FPGA part name")
    data = platform_data()
    found = data.find_device(part)
    if found is None:
        raise PlatformError("unknown-part", _unknown_part(data, part))
    return found


def get_part_architecture(part: str) -> Architecture:
    """The architecture of the part's device, as the loaded platform data states it."""
    return platform_data().architectures[get_part_device(part).architecture]


def is_versal(part: str) -> bool:
    """Whether the part is of the Versal architecture class."""
    return get_part_architecture(part).class_ == "versal"


def dsp_block(part: str) -> str:
    """The DSP block of the part: ``DSP48E1``, ``DSP48E2`` or ``DSP58``."""
    return get_part_architecture(part).dsp


def dsp_datapath_limits(block: str) -> Tuple[int, int, int]:
    """The maximum (activation, weight, accumulator) operand widths in bits that fit
    the datapath of the DSP block ``block``. These correspond to the DSP B, A and P
    ports respectively. Widths exceeding these limits would be silently truncated (or
    fail synthesis) if mapped onto the DSP-based RTL MVU. Raises ``ValueError`` for
    a DSP block other than DSP48E1, DSP48E2 and DSP58."""
    if block not in DSP_DATAPATH_LIMITS:
        raise ValueError(
            f"{block!r} is not a DSP block FINN knows: {', '.join(DSP_DATAPATH_LIMITS)}"
        )
    return DSP_DATAPATH_LIMITS[block]
