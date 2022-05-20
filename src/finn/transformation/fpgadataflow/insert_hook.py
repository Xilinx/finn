import numpy as np
from onnx import TensorProto
from onnx import helper as oh

from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.util.fpgadataflow import is_fpgadataflow_node


def _is_hook_node(node):
    if node.op_type in ["checksum"]:
        return True
    else:
        return False


def _suitable_node(node):
    if node is not None:
        if is_fpgadataflow_node(node) is True:
            if _is_hook_node(node) is False:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


class InsertHook(Transformation):
    """Inserting hook layer after each layer that has the node attribute
    'output_hook' specified"""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        list_supported_hooks = ["checksum"]
        graph = model.graph
        node_ind = -1
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if _suitable_node(n):
                for output_name in n.output:
                    consumers = model.find_consumers(output_name)
                    assert len(consumers) <= 1, (
                        n.name
                        + ": HLS node with fan-out higher than 1 cannot be stitched"
                    )
                    n0 = getCustomOp(n)
                    n0_hook = n0.get_nodeattr("output_hook")
                    if n0_hook in list_supported_hooks:
                        if n0_hook == "checksum":
                            if len(consumers) == 1:
                                if consumers[0].op_type == "checksum":
                                    continue
                            n0_normal_oshape = n0.get_normal_output_shape()
                            n0_folded_oshape = n0.get_folded_output_shape()
                            n0_odt = n0.get_output_datatype()
                            items_per_word = n0.get_nodeattr("PE")
                            words_per_frame = np.prod(n0_folded_oshape[:-1])
                            chk_otensor = oh.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                n0_normal_oshape,
                            )
                            chk_result = oh.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                [1],
                            )
                            chk_node = oh.make_node(
                                "checksum",
                                [output_name],
                                outputs=[chk_otensor.name, chk_result.name],
                                domain="finn.custom_op.fpgadataflow",
                                backend="fpgadataflow",
                                words_per_frame=words_per_frame,
                                items_per_word=items_per_word,
                                inputDataType=str(n0_odt.name),
                                folded_shape=n0_folded_oshape,
                            )
                            # insert checksum node
                            graph.node.insert(node_ind + 1, chk_node)

                            # set chk output tensor as new input tensor of second node
                            if len(consumers) == 1:
                                consumers[0].input[0] = chk_otensor.name
                            else:
                                model.graph.output.pop()
                                model.graph.output.append(chk_otensor)
                                model = model.transform(GiveUniqueNodeNames())
                                model = model.transform(GiveReadableTensorNames())
                            graph_modified = True
                            return (model, graph_modified)

        return (model, graph_modified)
