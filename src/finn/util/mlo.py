# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware constants shared by the compiler and the external-memory RTL.

The compiler is the single source of truth for these values. The memory
layouts it computes (DDR weight images, intermediate-frame slots, stream
widths) depend on them, and it passes them into the RTL through the wrapper
templates, so the RTL modules never hard-code them.
"""

from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

# AXI-MM data bus width in bits of the DMA engines: fetch_weights (MVAU and
# Elementwise weights from external memory) and the FINNLoop intermediate frames.
DMA_DATA_BITS = 256

# Number of intermediate feature-map slots in external memory that the FINNLoop
# controller (intermediate_frames.sv) keeps in flight.
N_OUTSTANDING_DMAS = 128


def mlo_set_select_bits(n_iterations: int) -> int:
    """Byte-padded width of the MLO set-select (iteration index) stream.

    The FINNLoop stream taps emit the index of the current iteration, and every
    consumer of a per-iteration parameter set (its ``in1_V`` pin) and the
    parallel MLO simulation must agree on this width.
    """
    bits = DataType.get_smallest_possible(n_iterations).bitwidth()
    return roundup_to_integer_multiple(bits, 8)
