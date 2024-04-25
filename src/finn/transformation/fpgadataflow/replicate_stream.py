# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Utility for handling ONNX nodes and tensors
from onnx import TensorProto
from onnx import helper as oh

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# QONNX graph transformation base class
from qonnx.transformation.base import Transformation

# Transformations running qonnx datatype inference
from qonnx.transformation.infer_datatypes import InferDataTypes

# Transformation running onnx shape inference
from qonnx.transformation.infer_shapes import InferShapes


# Inserts the ReplicateStream hardware operator on tensors with multiple
# consumers
class InferReplicateStream(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Check each output of the node, as there might be multiple distinct
            # outputs, each feeding multiple consumers
            for out in node.output:
                # Get the list of all consumers of this output tensor
                consumers = model.find_consumers(out)
                # No need to replicate if there is just one or no consumer
                if consumers is None or len(consumers) <= 1:
                    # Check next output tensor
                    continue
                # Ok, now we have multiple consumers of a single output tensor
                # which requires streams to be replicated for HLS synthesis
                # Get the shape of the original output tensor
                out_shape = model.get_tensor_shape(out)
                # Generate a list of unique replicas of the output tensor, one
                # for each consumer
                replicas = [model.make_new_valueinfo_name() for _ in consumers]
                # Create an instance of the ReplicateStream operator for this
                # output
                replicate_stream = oh.make_node(
                    # Name of the operator class as it can be found within FINN
                    "ReplicateStream",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    backend="fpgadataflow",
                    # Connect to the original output tensor
                    inputs=[out],
                    # Connect to a unique output tensor for each consumer
                    outputs=replicas,
                    # The operator needs to now the number of replicas as an
                    # attribute
                    num=len(replicas),
                    # Number of input elements in the last dimension
                    num_elems=out_shape[-1],
                    # Number of elements to process in parallel: default fully
                    # sequential
                    PE=1,
                    # Number of inputs to be processed sequentially
                    num_inputs=out_shape[:-1],
                    # Infer the datatype from the original output
                    dtype=model.get_tensor_datatype(out).name,
                    # Derive a node name based on the original node name
                    name=f"ReplicateStream_{node.name}"
                )
                # Insert the replicate operator into the graph right behind the
                # current node
                graph.node.insert(index + 1, replicate_stream)
                # Need to modify each consumer to have the replica as input
                for replica, consumer in zip(replicas, consumers):
                    # Properly construct a value info object for the  new tensor
                    # replica
                    model.graph.value_info.append(oh.make_tensor_value_info(
                        replica, TensorProto.FLOAT, out_shape
                    ))
                    # Find the first input of the consumer corresponding to the
                    # original output tensor
                    for i, inp in enumerate(consumer.input):
                        # Check whether this input is the original output
                        if inp == out:
                            # Connect this input to the replica of the output
                            consumer.input[i] = replica
                            # Break here as multiple inputs to the node might
                            # connect to the original output, but each gets its
                            # own replica.
                            break
                # The graph has been modified, needs to be reported back to the
                # caller
                graph_modified = True
        # After rewiring need to re-do the shape annotations
        model = model.transform(InferShapes())  # noqa: Shadows model
        # As new tensor value infos have been inserted, it is necessary to re-do
        # the datatype annotations
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified
