from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl

class DynMVAU_rtl(MVAU_rtl):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)