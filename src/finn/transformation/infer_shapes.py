import onnx.helper as helper
import onnx.shape_inference as si

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper


def infer_shapes(model):
    """Ensure every tensor in the model has a specified shape (ValueInfo)."""
    for node in model.graph.node:
        if node.domain == "finn":

            # create an empty execution context
            execution_context = model.make_empty_exec_context()

            # execute node with empty context
            oxe.execute_node(node, execution_context, model.graph)

            # set the tensor shape for all outputs of the node
            for output in node.output:
                model.set_tensor_shape(output, execution_context[output].shape)

        else:
            # onnx shape inference unfortunately does not take single node,
            # it can only analyze entire models -- so we create a model which solely
            # consists of our current node.
            node_inputs = list(
                filter(lambda x: x.name in node.input, model.graph.input)
            )
            node_inputs += list(
                filter(lambda x: x.name in node.input, model.graph.value_info)
            )
            node_outputs = list(
                filter(lambda x: x.name in node.output, model.graph.output)
            )
            node_outputs += list(
                filter(lambda x: x.name in node.output, model.graph.value_info)
            )
            node_graph = helper.make_graph(
                nodes=[node],
                name="single-node-exec",
                inputs=node_inputs,
                outputs=node_outputs,
            )
            node_model = helper.make_model(node_graph)
            node_model = si.infer_shapes(node_model)

            node_model = ModelWrapper(node_model)

            # set the corresponding tensors in the whole model
            for output in node.output:
                model.set_tensor_shape(output, node_model.get_tensor_shape(output))

    # single-step operation, no need to call multiple times so return
    # model_was_changed = false
    return (model, False)
