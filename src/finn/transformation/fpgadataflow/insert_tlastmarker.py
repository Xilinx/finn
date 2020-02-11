from onnx import TensorProto
from onnx import helper as oh

from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation


class InsertTLastMarker(Transformation):
    """Ensure that the graph is terminated with a TLastMarker node, inserting
    one if necessary."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        # TODO only makes sense for a pure fpgadataflow graph -- check!
        graph_out_name = model.graph.output[0].name
        final_node = model.find_producer(graph_out_name)
        if final_node.op_type == "TLastMarker":
            # TODO maybe check the correctness of properties
            return (model, False)
        else:
            custom_op = getCustomOp(final_node)
            num_iters = int(custom_op.get_number_output_values())
            stream_width = int(custom_op.get_outstream_width())
            out_shape = model.get_tensor_shape(graph_out_name)
            out_dtype = model.get_tensor_datatype(graph_out_name)
            # make new buffer
            final_node_out = oh.make_tensor_value_info(
                model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
            )
            model.graph.value_info.append(final_node_out)
            model.set_tensor_datatype(final_node_out.name, out_dtype)
            # reroute final node output to final_node_out_name
            final_node.output[0] = final_node_out.name
            tlast_node = oh.make_node(
                "TLastMarker",
                [final_node_out.name],
                [graph_out_name],
                NumIters=num_iters,
                StreamWidth=stream_width,
            )
            model.graph.node.append(tlast_node)
            return (model, True)
