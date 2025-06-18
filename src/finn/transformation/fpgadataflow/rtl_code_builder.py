from finn.util.context import Context
from finn.kernels import Kernel

from pathlib import Path
import importlib
import shutil


def gen_rtl_node(kernel: Kernel, ctx: Context):

    # Make subcontext
    node_ctx = ctx.get_subcontext(Path(kernel.name))

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
            dst_path = ctx.get_kernel_dir(kernel_name)

            for path in paths:
                full_path = importlib.resources.files("finn") / path
                if full_path.is_file():
                    shutil.copy(full_path, dst_path)
                else:
                    shutil.copytree(full_path, dst_path, dirs_exist_ok=True)

    # Shared files go to "output/shared"
    if len(node_ctx.shared) != 0:
        dst_path = ctx.shared_dir

        for shared_path in node_ctx.shared:
            if shared_path.is_file():
                shutil.copy(shared_path, dst_path)
            else:
                shutil.copytree(shared_path, dst_path, dirs_exist_ok=True)
