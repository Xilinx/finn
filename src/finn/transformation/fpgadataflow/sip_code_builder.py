from finn.util.context import Context
from finn.kernels import Kernel

from .rtl_code_builder import gen_rtl_node
from .hls_code_builder import gen_hls_node

from pathlib import Path


def gen_sip_node(kernel: Kernel, ctx: Context):

    # Make subcontext
    node_ctx = ctx.get_subcontext(Path(kernel.name))

    # Generate instance HLS files in "output/node"
    kernel.generate_instance_files(node_ctx)

    for subkernel in kernel.subkernels:

        if subkernel.impl_style == 'rtl':
            gen_rtl_node(subkernel, node_ctx)
        elif subkernel.impl_style == 'hls':
            gen_hls_node(subkernel, node_ctx)
        elif subkernel.impl_style == 'sip':
            gen_sip_node(subkernel, node_ctx)
        else:
            raise RuntimeError(f"Kernel.impl_style of {subkernel.name} is {subkernel.impl_style}, not recognised by CodeBuilder.")
