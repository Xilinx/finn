from finn.util.context import Context
from finn.util.kernel_util import get_node_attr
from finn.kernels import Kernel
from finn.kernels import gkr
import finn_xsi.adapter as finnxsi

from .code_builder import CodeBuilder
from .stitched_ip_builder import StitchedIPBuilder

from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
import copy
import multiprocessing as mp
from qonnx.util.basic import get_num_default_workers
from pathlib import Path


class RTLSimBuilder(Transformation):

    """ Takes an ONNX graph and creates a xsi emulation library for it,
        sets the rtlsim_so attribute to its path. """

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
        self.rtlsim_dir = self.ctx.directory.parent / Path(f"{self.ctx.directory.name}_rtlsim")
        self.rtlsim_dir.mkdir(exist_ok=True)
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

        return (model, run_again)

    def applyNodeLocal(self, node, node_ctx: Context):

        # Extract node attributes
        attributes = get_node_attr(node, self.ref_input_model)

        # Fetch kernel, skip this node if kernel is not RTL
        kernel: Kernel = gkr.kernel(node.op_type, attributes)

        kernel.build_rtlsim(node_ctx, self.rtlsim_dir, self.ref_input_model.get_metadata_prop("rtlsim_trace"))

        return (node, node_ctx, False)
