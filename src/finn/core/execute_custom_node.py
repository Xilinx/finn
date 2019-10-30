#import onnx.helper as helper

#import finn.core.MultiThreshold

def execute_custom_node(node, context, graph)
    """Call custom implementation to execute a single custom node. Input/output provided via context."""
    node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
    print(node_inputs)
