from finn.kernels.kernel_registry import gkr
from .matrixvectoractivation_sip import MVAUSIP

gkr.register("MVAU", MVAUSIP, 0)
