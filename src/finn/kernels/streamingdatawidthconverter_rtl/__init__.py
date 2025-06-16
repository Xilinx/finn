from finn.kernels.kernel_registry import gkr
from .streamingdatawidthconverter_rtl import StreamingDataWidthConverterRTL

gkr.register("StreamingDataWidthConverter", StreamingDataWidthConverterRTL, 0)
