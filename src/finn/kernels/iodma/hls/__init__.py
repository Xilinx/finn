from finn.kernels.kernel_registry import gkr
from .iodma_hls import IODMAHLS

gkr.register("IODMA", IODMAHLS, 0)
