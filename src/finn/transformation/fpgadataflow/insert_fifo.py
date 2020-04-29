from onnx import TensorProto
from onnx import helper as oh

from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation
from finn.util.fpgadataflow import is_fpgadataflow_node


def _is_fifo_node(node):
    if node.op_type == "StreamingFIFO":
        return True
    else:
        return False


def _suitable_node(node):
    if node is not None:
        if is_fpgadataflow_node(node) is True:
            if _is_fifo_node(node) is False:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


class InsertFIFO(Transformation):
    """Ensure that the graph is terminated with a TLastMarker node, inserting
    one if necessary."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        # default depth for FIFOs
        graph = model.graph
        node_ind = -1
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if _suitable_node(n):
                n_output = n.output[0]
                consumer = model.find_consumer(n_output)
                if _suitable_node(consumer) is True:
                    graph_modified = True
                    n0 = getCustomOp(n)
                    # determine fifo node attributes
                    fld_shape = n0.get_folded_output_shape()
                    dtype = n0.get_output_datatype()

                    # check if outFIFOdepth attribute of first node
                    # and inFIFOdepth attribute of consumer node is equal
                    n0_depth = n0.get_nodeattr("outFIFODepth")
                    n1 = getCustomOp(consumer)
                    n1_depth = n1.get_nodeattr("inFIFODepth")
                    if n0_depth == n1_depth:
                        fifo_depth = n0_depth
                    elif n0_depth != n1_depth:
                        fifo_depth = max(n0_depth, n1_depth)
                        n0.set_nodeattr("outFIFODepth", fifo_depth)
                        n1.set_nodeattr("inFIFODepth", fifo_depth)

                    # create fifo node
                    fifo_output_tensor = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        n0.get_normal_output_shape(),
                    )
                    graph.value_info.append(fifo_output_tensor)

                    fifo_node = oh.make_node(
                        "StreamingFIFO",
                        [n_output],
                        [fifo_output_tensor.name],
                        domain="finn",
                        backend="fpgadataflow",
                        depth=fifo_depth,
                        folded_shape=fld_shape,
                        dataType=str(dtype.name),
                    )
                    # insert fifo
                    graph.node.insert(node_ind + 1, fifo_node)

                    # set fifo output tensor as new input tensor of second node
                    consumer.input[0] = fifo_output_tensor.name

        if graph_modified is False:
            # insert FIFO as first node
            if graph.node[0].op_type != "StreamingFIFO":
                n = graph.node[0]
                n_input = n.input[0]
                n0 = getCustomOp(n)
                # determine fifo node attributes
                fld_shape = n0.get_folded_input_shape()
                dtype = n0.get_input_datatype()
                fifo_depth = n0.get_nodeattr("inFIFODepth")

                # create fifo node
                fifo_output_tensor = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    n0.get_normal_input_shape(),
                )
                graph.value_info.append(fifo_output_tensor)

                fifo_node = oh.make_node(
                    "StreamingFIFO",
                    [n_input],
                    [fifo_output_tensor.name],
                    domain="finn",
                    backend="fpgadataflow",
                    depth=fifo_depth,
                    folded_shape=fld_shape,
                    dataType=str(dtype.name),
                )
                # insert fifo
                graph.node.insert(0, fifo_node)

                # set fifo output tensor as new input tensor of second node
                n.input[0] = fifo_output_tensor.name

            # insert FIFO as last node
            if graph.node[-1].op_type != "StreamingFIFO":
                n = graph.node[-1]
                assert (
                    n.op_type != "TLastMarker"
                ), """Insert tlast marker should be done
                    after inserting the FIFOs"""
                graph_out_name = graph.output[0].name
                n0 = getCustomOp(n)
                # determine fifo node attributes
                fld_shape = n0.get_folded_output_shape()
                dtype = n0.get_output_datatype()
                fifo_depth = n0.get_nodeattr("inFIFODepth")

                # create fifo node
                fifo_input_tensor = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    n0.get_normal_output_shape(),
                )
                graph.value_info.append(fifo_input_tensor)

                fifo_node = oh.make_node(
                    "StreamingFIFO",
                    [fifo_input_tensor.name],
                    [graph_out_name],
                    domain="finn",
                    backend="fpgadataflow",
                    depth=fifo_depth,
                    folded_shape=fld_shape,
                    dataType=str(dtype.name),
                )
                # insert fifo
                graph.node.append(fifo_node)

                # set fifo output tensor as new input tensor of second node
                n.output[0] = fifo_input_tensor.name

        return (model, graph_modified)
