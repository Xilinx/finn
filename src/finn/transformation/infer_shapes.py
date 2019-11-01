import onnx.shape_inference as si
import onnx.helper as helper



def infer_shapes(model):
    """Ensure every tensor in the model has a specified shape (ValueInfo)."""
    for node in model.graph.node:
        if node.domain == 'finn':
            print("finn shape inference was called for node with op_type: " + str(node.op_type))
        else:
            # onnx shape inference unfortunately does not implement shape inference on a single node,
            # it can only check the shapes of the node in an entire models 
            # -- create a model which solely consists of current node (see also onnx_exec.py).
            node_inputs = list(filter(lambda x: x.name in node.input, model.graph.input))
            node_inputs += list(filter(lambda x: x.name in node.input, model.graph.value_info))
            node_outputs = list(filter(lambda x: x.name in node.output, model.graph.output))
            node_outputs += list(filter(lambda x: x.name in node.output, model.graph.value_info))
            node_graph = helper.make_graph(
                nodes=[node],
                name="single-node-exec",
                inputs=node_inputs,
                outputs=node_outputs,
            )
            node_model = helper.make_model(node_graph)

            node_model = si.infer_shapes(node_model)
            print("onnx shape inference was called for node with op_type: " + str(node.op_type))
    # single-step operation, no need to call multiple times so return
    # model_was_changed = false
    return (model, False)
