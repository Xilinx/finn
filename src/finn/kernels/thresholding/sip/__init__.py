from finn.kernels.kernel_registry import gkr
from .thresholding_sip import ThresholdingSIP

gkr.register("Thresholding", ThresholdingSIP, 0)
