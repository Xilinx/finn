import os
import subprocess
import psutil

from concurrent.futures import ThreadPoolExecutor, as_completed

import qonnx.analysis.topology as ta
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import (
    get_sanitize_quant_tensors,
    sanitize_quant_values,
)

class DistributedDataflow(CustomOp):
    def get_nodeattr_types(self):
        return {
            "model": ("s", True, ""),
            "instance_name": ("s", False, ""),
            "return_full_exec_context": ("i", False, 0),
            "world_size": ("i", True, -1),
        }

    def make_shape_compatible_op(self, model):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        return []

    def execute_node(self, context, graph):
        model = ModelWrapper(self.get_nodeattr("model"))
        return_full_exec_context = self.get_nodeattr(
            "return_full_exec_context") == 1
        node = self.onnx_node

        inp_ctx = dict(filter(lambda x: x[0] in node.input, context.items()))
        for i, old_iname in enumerate(node.input):
            new_iname = model.graph.input[i].name
            if old_iname != new_iname:
                inp_ctx[new_iname] = inp_ctx[old_iname]
                del inp_ctx[old_iname]

        ret = execute_distributed_onnx(model, inp_ctx, return_full_exec_context)

        for i, node_oname in enumerate(node.output):
            model_oname = model.graph.output[i].name
            context[node_oname] = ret[model_oname]

        if return_full_exec_context:
            for tname in ret.keys():
                if tname not in [x.name for x in model.graph.output]:
                    context[node.name + "_" + tname] = ret[tname]

