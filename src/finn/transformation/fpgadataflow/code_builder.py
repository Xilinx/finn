from finn.util.context import Context
from finn.util.kernel_util import get_node_attr
from finn.kernels import Kernel
from finn.kernels import gkr

from .rtl_code_builder import gen_rtl_node
from .hls_code_builder import gen_hls_node
from .sip_code_builder import gen_sip_node

from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
import copy
import multiprocessing as mp
from qonnx.util.basic import get_num_default_workers
import importlib
import shutil


class CodeBuilder(Transformation):

    """ Takes an ONNX graph and constructs the codegen for it.
        Needed something similar to qonnx.transformation.base.NodeLocalTransformation
        but with the ability to pass a Context and return it from processes. """

    def __init__(self, ctx: Context, num_workers=None):
        super().__init__()

        # Set number of workers, from qonnx.transformation.base.NodeLocalTransformation
        if num_workers is None:
            self._num_workers = get_num_default_workers()
        else:
            self._num_workers = num_workers
        assert self._num_workers >= 0, "Number of workers must be nonnegative."
        if self._num_workers == 0:
            self._num_workers = mp.cpu_count()
        
        # Context that was passed in to CodeBuilder
        self.ctx = ctx

    # Modified version from qonnx.transformation.base.NodeLocalTransformation
    def apply(self, model: ModelWrapper):
        # make a detached copy of the input model that applyNodeLocal
        # can use for read-only access
        self.ref_input_model = copy.deepcopy(model)
        # Remove old nodes from the current model
        old_nodes = []
        for i in range(len(model.graph.node)):
            node = model.graph.node.pop()
            old_nodes.append((node, self.ctx.get_subcontext(node.name)))

        # Execute transformation in parallel
        if self._num_workers > 1:
            with mp.Pool(self._num_workers) as p:
                new_nodes_and_bool = p.starmap(self.applyNodeLocal, old_nodes, chunksize=1)
        # execute without mp.Pool in case of 1 worker to simplify debugging
        else:
            new_nodes_and_bool = [self.applyNodeLocal(*node) for node in old_nodes]

        # extract nodes and check if the transformation needs to run again
        # Note: .pop() had initially reversed the node order
        run_again = False
        for node, node_ctx, run in reversed(new_nodes_and_bool):
            # Reattach new nodes to old model
            model.graph.node.append(node)
            self.ctx.update_subcontext(node_ctx)
            if run is True:
                run_again = True

        # Kernel files go to "output/kernel_name"
        for kernel_name, paths in self.ctx.kernel_files.items():
            if len(paths) != 0:
                dst_path = self.ctx.get_kernel_dir(kernel_name)

                for path in paths:
                    full_path = importlib.resources.files("finn") / path
                    if full_path.is_file():
                        shutil.copy(full_path, dst_path)
                    else:
                        shutil.copytree(full_path, dst_path, dirs_exist_ok=True)

        # Shared files go to "output/shared"
        if len(self.ctx.shared) != 0:
            dst_path = self.ctx.shared_dir

            for shared_path in self.ctx.shared:
                if shared_path.is_file():
                    shutil.copy(shared_path, dst_path)
                else:
                    shutil.copytree(shared_path, dst_path, dirs_exist_ok=True)

        return (model, run_again)

    def applyNodeLocal(self, node, node_ctx: Context):

        # Extract node attributes
        attributes = get_node_attr(node, self.ref_input_model)

        # Fetch kernel, skip this node if kernel is not RTL
        kernel: Kernel = gkr.kernel(node.op_type, attributes)

        if kernel.impl_style == 'rtl':
            gen_rtl_node(kernel, node_ctx)
        elif kernel.impl_style == 'hls':
            gen_hls_node(kernel, node_ctx)
        elif kernel.impl_style == 'sip':
            gen_sip_node(kernel, node_ctx)
        else:
            raise RuntimeError(f"Kernel.impl_style of {node.name} is {kernel.impl_style}, not recognised by CodeBuilder.")

        return (node, node_ctx, False)
