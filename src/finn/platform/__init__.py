# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's platform facts: FPGA devices and their architectures, read from platform
data (``finn.platform.registry``) and queried by part (``finn.platform.catalog``).
"""

from finn.platform.catalog import (
    DSP_DATAPATH_LIMITS,
    dsp_block,
    dsp_datapath_limits,
    get_part_architecture,
    get_part_device,
    is_versal,
)
from finn.platform.registry import PlatformError, load, platform_data, platform_sources

__all__ = [
    "DSP_DATAPATH_LIMITS",
    "PlatformError",
    "dsp_block",
    "dsp_datapath_limits",
    "get_part_architecture",
    "get_part_device",
    "is_versal",
    "load",
    "platform_sources",
    "platform_data",
]
