# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's platform facts: FPGA devices and their architectures, and the boards FINN
builds for, read from platform data (``finn.platform.registry``), queried by part
(``finn.platform.catalog``) and by board (``finn.platform.boards``), and resolved
into a build's target (``finn.platform.target``).
"""

from finn.platform.boards import (
    GOLDEN_FILES,
    get_board,
    get_boards,
    golden_dir,
    missing_golden_files,
)
from finn.platform.catalog import (
    DSP_DATAPATH_LIMITS,
    dsp_block,
    dsp_datapath_limits,
    get_part_architecture,
    get_part_device,
    is_versal,
)
from finn.platform.registry import (
    Board,
    PlatformError,
    load,
    platform_data,
    platform_sources,
)
from finn.platform.target import ShellFlowType, Target, resolve_target

__all__ = [
    "DSP_DATAPATH_LIMITS",
    "GOLDEN_FILES",
    "Board",
    "PlatformError",
    "ShellFlowType",
    "Target",
    "dsp_block",
    "dsp_datapath_limits",
    "get_board",
    "get_boards",
    "get_part_architecture",
    "get_part_device",
    "golden_dir",
    "is_versal",
    "load",
    "platform_sources",
    "missing_golden_files",
    "platform_data",
    "resolve_target",
]
