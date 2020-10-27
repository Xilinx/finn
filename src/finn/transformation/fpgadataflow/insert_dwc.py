from onnx import TensorProto
from onnx import helper as oh

from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.util.fpgadataflow import is_fpgadataflow_node
import warnings


def _is_dwc_node(node):
    if node.op_type == "StreamingDataWidthConverter_Batch":
        return True
    else:
        return False


def _suitable_node(node):
    if node is not None:
        if is_fpgadataflow_node(node) is True:
            if _is_dwc_node(node) is False:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


class InsertDWC(Transformation):
    """Ensure that the graph is terminated with a TLastMarker node, inserting
    one if necessary."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = -1
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if _suitable_node(n):
                for n_output in n.output:
                    consumers = model.find_consumers(n_output)
                    if consumers is None:
                        continue
                    if len(consumers) > 1:
                        warnings.warn(
                            n.name
                            + ": HLS node with fan-out higher than 1 cannot be stitched"
                        )

                    consumer = consumers[0]
                    if _suitable_node(consumer) is True:
                        n0 = getCustomOp(n)
                        n1 = getCustomOp(consumer)
                        n0_out_shape = n0.get_folded_output_shape()
                        n1_in_shape = n1.get_folded_input_shape()
                        if n0_out_shape[-1] != n1_in_shape[-1]:
                            graph_modified = True
                            # determine dwc inwidth
                            dwc_in_width = n0.get_outstream_width()
                            # determine dwc outwidth
                            dwc_out_width = n1.get_instream_width()

                            # determine shape for dwc
                            dwc_shape = n0.get_normal_output_shape()

                            # determine dtype for dwc
                            dtype = n0.get_output_datatype()

                            dwc_output_tensor = oh.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                dwc_shape,
                            )
                            graph.value_info.append(dwc_output_tensor)

                            dwc_node = oh.make_node(
                                "StreamingDataWidthConverter_Batch",
                                [n_output],
                                [dwc_output_tensor.name],
                                domain="finn",
                                backend="fpgadataflow",
                                shape=dwc_shape,
                                inWidth=dwc_in_width,
                                outWidth=dwc_out_width,
                                dataType=str(dtype.name),
                            )
                            # insert dwc
                            graph.node.insert(node_ind + 1, dwc_node)

                            # set dwc output tensor as new input tensor of second node
                            for idx, inp in enumerate(consumer.input):
                                if inp == n_output:
                                    consumer.input[idx] = dwc_output_tensor.name

        return (model, graph_modified)
