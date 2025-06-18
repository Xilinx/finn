from finn.util.context import Context
from finn.util.kernel_util import get_node_attr
from finn.kernels import Kernel
from finn.kernels import gkr

from .rtl_code_builder import gen_rtl_node
from .hls_code_builder import gen_hls_node
from .sip_code_builder import gen_sip_node

from qonnx.transformation.base import NodeLocalTransformation


class CodeBuilder(NodeLocalTransformation):

    """ Takes an ONNX graph and constructs the codegen for it """

    def __init__(self, ctx: Context, num_workers=None):
        super().__init__(num_workers)
        self.ctx = ctx

    def applyNodeLocal(self, node):

        # Extract node attributes
        attributes = get_node_attr(node, self.ref_input_model)

        # Fetch kernel, skip this node if kernel is not RTL
        kernel: Kernel = gkr.kernel(node.op_type, attributes)

        if kernel.impl_style == 'rtl':
            gen_rtl_node(kernel, self.ctx)
        elif kernel.impl_style == 'hls':
            gen_hls_node(kernel, self.ctx)
        elif kernel.impl_style == 'sip':
            gen_sip_node(kernel, self.ctx)
        else:
            raise RuntimeError(f"Kernel.impl_style of {kernel.name} is {kernel.impl_style}, not recognised by CodeBuilder.")

        return (node, False)
