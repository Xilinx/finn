from finn.util.context import Context
from finn.kernels import Kernel
from finn.kernels.cache_manager import cache_manager
from finn.kernels.cache_config_extractor import extract_kernel_config_for_cache

import importlib
import shutil


def gen_rtl_node(kernel: Kernel, node_ctx: Context):

    # Extract kernel configuration for caching using standardized logic
    kernel_config = extract_kernel_config_for_cache(kernel)

    # Check if cached files exist and are valid
    cache_hash = cache_manager._compute_hash(kernel, kernel_config)
    if cache_manager.has_cached_files(kernel, kernel_config):
        print(f"Using cached files for RTL kernel {kernel.name} (hash: {cache_hash[:12]}...)")
        if cache_manager.get_cached_files(kernel, kernel_config, node_ctx.directory):
            print(f"Successfully restored cached RTL files for {kernel.name} (hash: {cache_hash[:12]}...)")
            return
        else:
            print(f"Cached files incomplete for RTL kernel {kernel.name} (hash: {cache_hash[:12]}...), regenerating...")

    # Generate instance files in "output/node"
    kernel.generate_instance_files(node_ctx)

    # Record shared files with context
    for sharedFile in kernel.sharedFiles:
        node_ctx.add_shared(sharedFile)

    # Record kernel files with context
    for path in kernel.kernelFiles:
        node_ctx.add_kernel_file(type(kernel).__name__, path)

    # Kernel files go to "output/kernel_name"
    for kernel_name, paths in node_ctx.kernel_files.items():
        if len(paths) != 0:
            dst_path = node_ctx.get_kernel_dir(kernel_name)

            for path in paths:
                full_path = importlib.resources.files("finn") / path
                if full_path.is_file():
                    shutil.copy(full_path, dst_path)
                else:
                    shutil.copytree(full_path, dst_path, dirs_exist_ok=True)

    # Shared files go to "output/shared"
    if len(node_ctx.shared) != 0:
        dst_path = node_ctx.shared_dir

        for shared_path in node_ctx.shared:
            if shared_path.is_file():
                shutil.copy(shared_path, dst_path)
            else:
                shutil.copytree(shared_path, dst_path, dirs_exist_ok=True)

    # Store generated files in cache for future use
    if cache_manager.store_generated_files(kernel, kernel_config, node_ctx.directory):
        print(f"Cached generated RTL files for kernel {kernel.name} (hash: {cache_hash[:12]}...)")
    else:
        print(f"Warning: Failed to cache generated RTL files for kernel {kernel.name} (hash: {cache_hash[:12]}...)")
