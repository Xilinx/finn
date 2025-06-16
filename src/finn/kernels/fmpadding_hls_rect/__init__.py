from finn.kernels.kernel_registry import gkr
from .fmpadding_hls_rect import FMPaddingHLS_Rect

gkr.register("FMPadding", FMPaddingHLS_Rect, 0)
