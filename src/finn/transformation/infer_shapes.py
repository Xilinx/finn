import finn.core.onnx_exec as oxe


def infer_shapes(model):
    """Ensure every tensor in the model has a specified shape (ValueInfo)."""
    for node in model.graph.node:

        # create an empty execution context
        execution_context = model.make_empty_exec_context()

        # execute node with empty context
        oxe.execute_node(node, execution_context, model.graph)

        # set the tensor shape for all outputs of the node
        for output in node.output:
            model.set_tensor_shape(output, execution_context[output].shape)

    # single-step operation, no need to call multiple times so return
    # model_was_changed = false
    return (model, False)
