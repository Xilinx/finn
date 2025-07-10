from finn.kernels.kernel_registry import gkr
from .fmpadding_rtl import FMPaddingRTL

gkr.register("FMPadding", FMPaddingRTL, -1)
