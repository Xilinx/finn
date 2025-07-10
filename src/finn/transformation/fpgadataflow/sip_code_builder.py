from finn.util.context import Context
from finn.kernels import Kernel
from finn.kernels.cache_manager import cache_manager
from finn.kernels.cache_config_extractor import extract_kernel_config_for_cache

from .rtl_code_builder import gen_rtl_node
from .hls_code_builder import gen_hls_node


def gen_sip_node(kernel: Kernel, node_ctx: Context):

    # Extract kernel configuration for caching using standardized logic
    kernel_config = extract_kernel_config_for_cache(kernel)

    # Check if cached files exist and are valid
    cache_hash = cache_manager._compute_hash(kernel, kernel_config)
    sip_cache_hit = False
    if cache_manager.has_cached_files(kernel, kernel_config):
        print(f"Using cached files for SIP kernel {kernel.name} (hash: {cache_hash[:12]}...)")
        if cache_manager.get_cached_files(kernel, kernel_config, node_ctx.directory):
            print(f"Successfully restored cached SIP files for {kernel.name} (hash: {cache_hash[:12]}...)")
            sip_cache_hit = True
        else:
            print(f"Cached files incomplete for SIP kernel {kernel.name} (hash: {cache_hash[:12]}...), regenerating...")

    if not sip_cache_hit:
        # Generate instance files in "output/node"
        kernel.generate_instance_files(node_ctx)

    # Always generate subkernels (they have their own caching logic)
    for subkernel in kernel.subkernels:

        if subkernel.impl_style == 'rtl':
            gen_rtl_node(subkernel, node_ctx.get_subcontext(subkernel.name))
        elif subkernel.impl_style == 'hls':
            gen_hls_node(subkernel, node_ctx.get_subcontext(subkernel.name))
        elif subkernel.impl_style == 'sip':
            gen_sip_node(subkernel, node_ctx.get_subcontext(subkernel.name))
        else:
            raise RuntimeError(f"Kernel.impl_style of {subkernel.name} is {subkernel.impl_style}, not recognised by CodeBuilder.")

    # Store generated files in cache for future use (only if we didn't hit cache)
    if not sip_cache_hit:
        if cache_manager.store_generated_files(kernel, kernel_config, node_ctx.directory):
            print(f"Cached generated SIP files for kernel {kernel.name} (hash: {cache_hash[:12]}...)")
        else:
            print(f"Warning: Failed to cache generated SIP files for kernel {kernel.name} (hash: {cache_hash[:12]}...)")
