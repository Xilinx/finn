from finn.kernels.kernel_registry import gkr
from .streamingfifo_rtl import StreamingFIFORTL

gkr.register("StreamingFIFO", StreamingFIFORTL, 0)
