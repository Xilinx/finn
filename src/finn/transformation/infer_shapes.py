import onnx.helper as helper
import onnx.shape_inference as si

import finn.core.execute_custom_node as ex_cu_node
from finn.core.modelwrapper import ModelWrapper


def infer_shapes(model):
    """Ensure every tensor in the model has a specified shape (ValueInfo)."""
    for node in model.graph.node:
        if node.domain == "finn":
            execution_context = model.make_empty_exec_context()
            ex_cu_node.execute_custom_node(node, execution_context, model.graph)
            for output in node.output:
                model = ModelWrapper(model)
                model.set_tensor_shape(output, execution_context[output].shape)

        else:
            """onnx shape inference unfortunately does not implement shape
            inference on a single node, it can only check the shapes of the nodes
            in an entire model -- create a model which solely consists of
            current node (see also onnx_exec.py). """

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

    # single-step operation, no need to call multiple times so return
    # model_was_changed = false
    return (model, False)
