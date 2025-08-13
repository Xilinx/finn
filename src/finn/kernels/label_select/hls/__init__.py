from finn.kernels.kernel_registry import gkr
from .labelselect_hls import LabelSelectHLS

gkr.register("LabelSelect", LabelSelectHLS, 0)
