from finn.kernels.kernel_registry import gkr
from .thresholding_rtl import ThresholdingRTL

gkr.register("Thresholding", ThresholdingRTL, 0)
