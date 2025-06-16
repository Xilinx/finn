from finn.kernels.kernel_registry import gkr
from .fmpadding_hls_square import FMPaddingHLS_Square 

gkr.register("FMPadding", FMPaddingHLS_Square, -1)