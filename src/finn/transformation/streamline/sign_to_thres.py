import numpy as np
from onnx import helper as oh

from finn.core.datatype import DataType
from finn.transformation import Transformation


class ConvertSignToThres(Transformation):
    """Convert Sign node instances to MultiThreshold with threshold at 0."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Sign":
                sign_in_name = n.input[0]
                sign_out_name = n.output[0]
                # find consumer
                consumer = model.find_consumer(sign_out_name)
                assert consumer is not None, """There is no consumer of the 
                sign_out tensor."""
                # create thresholds
                thres_param_name = model.make_new_valueinfo_name()
                thres_param = np.asarray([[0]], dtype=np.float32)
                model.set_initializer(thres_param_name, thres_param)
                # create a new node
                mt_node = oh.make_node(
                    "MultiThreshold",
                    [sign_in_name, thres_param_name],
                    [sign_out_name],
                    domain="finn",
                    out_scale=2.0,
                    out_bias=-1.0,
                    out_dtype="BIPOLAR",
                )
                # remove old node, add new node to graph at correct position
                graph.node.insert(node_ind, mt_node)
                graph.node.remove(n)
                # add quantization annotations
                model.set_tensor_datatype(sign_out_name, DataType.BIPOLAR)
                graph_modified = True
        return (model, graph_modified)
