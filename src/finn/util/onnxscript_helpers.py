import ast
from collections.abc import Iterable
from onnx_ir import _enums
from onnxscript import ir
from onnxscript.rewriter._pattern_ir import (
    GraphPattern,
    NodeOutputPattern,
    ValuePattern,
)
from onnxscript.rewriter._rewrite_rule import (
    ReplacementPatternFunction,
    ReplacementSubgraph,
)
from onnxscript.rewriter.pattern import (
    MatchResult,
    OpsetPatternBuilder,
    RewriterContext,
    pattern_builder,
)
from qonnx.custom_op.registry import is_custom_op
from typing import List, Optional


class SubGraphView(ir.GraphView):
    """Create a read-only view of a subgraph defined by a set of nodes.

    Args:
        graph (ir.Graph): The parent graph containing the nodes.
        name (str): Name of the subgraph.
        nodes (List[ir.Node]): List of nodes that make up the subgraph.
        include_initializers (bool): Whether to include initializers connected to the
            subgraph nodes as part of the subgraph.
    """

    def __init__(self, graph, name, nodes, include_initializers=False):
        self._assert_graph_subset(graph, nodes)
        self.include_initializers = include_initializers
        super().__init__(
            name=name,
            inputs=self._identify_inputs(nodes),
            initializers=self._identify_initializers(nodes),
            outputs=self._identify_outputs(nodes),
            nodes=nodes,
        )

    def _assert_graph_subset(self, graph, nodes):
        for node in nodes:
            if node.graph != graph:
                raise ValueError("All nodes must belong to the same graph")

    def _identify_inputs(self, nodes):
        inputs = set()
        for node in nodes:
            for input in node.inputs:
                if input.is_graph_input():
                    inputs.add(input)
                elif input.producer() not in nodes:
                    inputs.add(input)
        return list(inputs)

    def _identify_initializers(self, nodes):
        initializers = set()
        if self.include_initializers:
            for node in nodes:
                for input in node.inputs:
                    if input.is_initializer():
                        initializers.add(input)
        return list(initializers)

    def _identify_outputs(self, nodes):
        outputs = set()
        for node in nodes:
            for output in node.outputs:
                if output.is_graph_output():
                    outputs.add(output)
                else:
                    for consumer in output.consumers():
                        if consumer not in nodes:
                            outputs.add(output)
        return list(outputs)


class PytorchMetadataNode:
    """Wrap an ONNX IR node and expose PyTorch exporter metadata.

    The Torch ONNX exporter stores per-node metadata describing the originating
    module instance hierarchy and class names. This helper parses the serialized
    metadata strings into Python objects and provides convenience accessors for
    querying instance/class names at different nesting depths.
    """

    def __init__(self, node):
        self._node = node

        if self.check_node_metadata_exists():
            self.instance_metadata = ast.literal_eval(
                self._node.metadata_props["pkg.torch.onnx.name_scopes"]
            )
            self.class_metadata = ast.literal_eval(
                self._node.metadata_props["pkg.torch.onnx.class_hierarchy"]
            )

    def check_node_metadata_exists(self):
        if (
            "pkg.torch.onnx.name_scopes" in self._node.metadata_props
            and "pkg.torch.onnx.class_hierarchy" in self._node.metadata_props
        ):
            return True
        else:
            return False

    def is_last_level(self, level):
        if len(self.instance_metadata) - 1 == level:
            return True
        else:
            return False

    def get_instance_name(self, depth=0):
        if depth >= len(self.instance_metadata):
            return None
        else:
            return self.instance_metadata[depth]

    def get_class_name(self, depth=0):
        if depth >= len(self.instance_metadata):
            return None
        else:
            return self.class_metadata[depth]


class PytorchHierarchyNode:
    """Represent a node in the hierarchy reconstructed from PyTorch metadata.

    Each instance mirrors a PyTorch module captured by the exporter. It stores
    child modules plus the wrapped ONNX nodes and exposes helpers that let
    callers traverse or query the reconstructed module tree.

    Example::

        root = PytorchHierarchyNode()
        for ir_node in graph._nodes:
            root.add_node(ir_node)

        root.print_hierarchy()
        target_path = ["top_module", "encoder", "layer_0"]
        ir_nodes = root.get_nodes(target_path)

    Notes
    -----
    ``add_node`` can be called in any order because the structure is built
    incrementally. ``get_nodes`` performs prefix matching so supplying
    ``["top_module", "encoder"]`` returns every descendant of that subtree.
    Nodes that are missing exporter metadata are ignored, and the maximum
    depth matches the length of the serialized ``name_scopes`` list.
    """

    def __init__(self):
        self.instance_name = None
        self.module_type = None
        self.children = []
        self.nodes = []

    def print_hierarchy(self, instance_hierarchy: Optional[List[str]] = None):
        if instance_hierarchy is None:
            instance_hierarchy = []
        if self.instance_name is not None:
            instance_hierarchy.append(self.instance_name)

        for child in self.children:
            child.print_hierarchy(list(instance_hierarchy))

        for node in self.nodes:
            print(
                f"Node: {node._node.name}, Instance: {'/'.join(instance_hierarchy)},"
                f" Module: {self.module_type}"
            )

    def get_unwrapped_nodes(self):
        # Return _node from self._nodes
        return [node._node for node in self.nodes]

    # Checks if the search hierarchy matches the instance hierarchy
    def hierarchy_matches(
        self, search_hierarchy: List[str], instance_hierarchy: Optional[List[str]] = None
    ):
        if instance_hierarchy is None:
            instance_hierarchy = []
        search_length = min(len(search_hierarchy), len(instance_hierarchy))
        for i in range(search_length):
            if search_hierarchy[i] != instance_hierarchy[i]:
                return False
        return True

    # Return all nodes from the given name hierarchy on down
    def get_nodes(
        self, search_hierarchy: List[str], instance_hierarchy: Optional[List[str]] = None
    ):
        if instance_hierarchy is None:
            instance_hierarchy = []

        nodes_to_return = []
        # base case for recursion
        # 1 - search_hierarchy does not match instance_hierarchy
        if self.instance_name is not None:
            instance_hierarchy.append(self.instance_name)

        if not self.hierarchy_matches(search_hierarchy, instance_hierarchy):
            return []

        for child in self.children:
            child_nodes = child.get_nodes(search_hierarchy, list(instance_hierarchy))
            nodes_to_return.extend(child_nodes)

        if len(instance_hierarchy) >= len(search_hierarchy):
            nodes_to_return.extend(self.get_unwrapped_nodes())

        return nodes_to_return

    def add_node(self, node, level=0):
        if not isinstance(node, PytorchMetadataNode):
            node = PytorchMetadataNode(node)
            if node.check_node_metadata_exists() is False:
                return False

        if self.instance_name is None:
            self.instance_name = node.get_instance_name(level)
        if self.module_type is None:
            self.module_type = node.get_class_name(level)

        # check that instance name and module type match
        if self.instance_name != node.get_instance_name(level):
            return False
        if self.module_type != node.get_class_name(level):
            return False
        # if this is the last level of the hierarchy, add the node to this node
        # otherwise find the child node that matches the next level of the hierarchy
        # and add the node to that child
        if node.is_last_level(level):
            self.nodes.append(node)
            return True
        else:
            for child in self.children:
                if child.instance_name == node.get_instance_name(level + 1):
                    return child.add_node(node, level + 1)

            # if no child matches the next level of the hierarchy, create a new child node
            new_child = PytorchHierarchyNode()
            new_child.instance_name = node.get_instance_name(level + 1)
            new_child.module_type = node.get_class_name(level + 1)
            self.children.append(new_child)
            return new_child.add_node(node, level + 1)


def direct_convert_ir_graph_to_pattern(graph):
    """Convert an IR graph into an ONNX Script ``GraphPattern``.

    The conversion walks nodes in order, mapping each IR ``Value`` to the
    corresponding ``ValuePattern``/``NodeOutputPattern`` produced by the
    pattern builder. The resulting pattern preserves input/output ordering and
    captures every constructed operator so it can later drive rewrite rules.
    """
    # Transform IR values to ValuePatterns

    vmap = {}
    for input in graph.inputs:
        vmap[input] = ValuePattern(input.name)

    for init in graph.initializers:
        vmap[init] = ValuePattern(init)

    for node in graph._nodes:
        if node.op_type == "Constant":
            vmap[node.outputs[0]] = ValuePattern(node.outputs[0].name)

    builder = OpsetPatternBuilder("", record=True)

    with pattern_builder(builder):
        for node in graph._nodes:
            ninputs = []
            for ninput in node.inputs:
                ninputs.append(vmap[ninput])

            vp_outputs = builder.__getattr__(node.op_type)(
                *ninputs, _domain=node.domain, _outputs=len(node.outputs)
            )

            if isinstance(vp_outputs, NodeOutputPattern):
                vp_outputs = [vp_outputs]

            for vp_output in iter(vp_outputs):
                vmap[node.outputs[vp_output.output_index]] = vp_output

    pinputs = []
    for input in graph.inputs:
        pinputs.append(vmap[input])

    # build graph outputs
    poutputs = []
    for output in graph.outputs:
        poutputs.append(vmap[output])

    return GraphPattern(inputs=pinputs, outputs=poutputs, nodes=builder.nodes())


def remove_input_from_node(node, inp):
    node._inputs = [x for x in node._inputs if x is not inp]
    inp._remove_usage(node)


def same(input_list):
    return len(set(input_list)) == 1


def vdisconnect(value):
    value._uses = {}
    value._producer = None
    value._index = None
    value._graph = None
    return value


def is_fpgadataflow_onnxir_node(node):
    """Returns True if given node is fpgadataflow node. Otherwise False."""
    is_node = False
    if node is not None:
        if is_custom_op(node.domain):
            if "backend" in node.attributes:
                backend_value = node.attributes["backend"].as_string()
                if backend_value in ["fpgadataflow", "hls", "rtl"]:
                    is_node = True

    return is_node


class ReplacementPatternGraph(ReplacementPatternFunction):
    """Instantiate a replacement pattern graph from an ONNX Script IR graph.

    The class adapts an ``ir.Graph`` into the replacement side of a rewrite
    rule: when the pattern matches, ``get_replacement`` materialises the stored
    graph inside the active rewrite context while remapping bound values to the
    match result.
    """

    def __init__(self, ir_graph):
        self._graph = ir_graph

    def get_replacement(self, match: MatchResult) -> ReplacementSubgraph | None:
        context = RewriterContext()
        # ``match.bindings`` maps ``value_name`` (str) from the replacement
        # subgraph pattern to actual IR values.
        vvmap = {}  # Maps pattern values to the values that will populate the replacement

        for value in self._graph.inputs:
            if value.name in match.bindings:
                vvmap[value] = match.bindings[value.name]
            else:
                vvmap[value] = value

        for node in self._graph._nodes:
            ninputs = []
            for ninput in node.inputs:
                ninputs.append(vvmap[ninput])

            coutput = context.__getattr__(node.op_type)(
                *ninputs,
                **node.attributes,
                _outputs=len(node.outputs),
                _domain=node.domain,
                _version=node.version,
            )
            if not isinstance(coutput, Iterable):
                coutput = [coutput]

            for i, cout in enumerate(coutput):
                cout._type = node.outputs[i].type
                cout._shape = node.outputs[i].shape
                for key in node.outputs[i].meta:
                    cout.meta[key] = node.outputs[i].meta[key]
                vvmap[node.outputs[cout.index()]] = cout

        new_outputs = [vvmap[x] for x in self._graph.outputs]
        return ReplacementSubgraph(
            match, new_outputs, context.nodes, context.initializers, context.used_opsets
        )


def find_nodes_of_optype(graph, layername):
    nodes = []
    for node in ir.traversal.RecursiveGraphIterator(graph):
        if node.op_type == layername:
            nodes.append(node)
    return nodes


def build_constant_from_tensor(name, tensor):
    value_attribute = ir.Attr(name="value", type=ir.AttributeType.TENSOR, value=tensor)
    ir_value_out = ir.Value(name=name + "_out", type=ir.TensorType(tensor.dtype))
    return ir.Node(
        "", "Constant", name=name, inputs=[], outputs=[ir_value_out], attributes=[value_attribute]
    )


def build_concat_node_from_inputs(inputs):
    axis = ir.Attr(name="axis", type=ir.AttributeType.INT, value=0)

    ndim = len(inputs) * inputs[0].shape.dims[0]
    output_shape = ir.Shape([ndim, *inputs[0].shape.dims[1:]])
    output = ir.Value(name=f"{inputs[0].name}_concat", shape=output_shape, type=inputs[0].type)
    return ir.Node("", "Concat", inputs=inputs, attributes=[axis], outputs=[output])


def build_reshape_node(inp, reshape_shape):
    reshape_out = ir.Value(name=f"{inp.name}_reshape", type=inp.type)
    return ir.Node("", "Reshape", inputs=[inp, reshape_shape], outputs=[reshape_out])


def tensor_type_to_finn_datatype_string(tensor_type):
    if tensor_type == ir.TensorType(_enums.DataType.FLOAT):
        return "FLOAT32"
    elif tensor_type == ir.TensorType(_enums.DataType.INT8):
        return "INT8"
    elif tensor_type == ir.TensorType(_enums.DataType.INT16):
        return "INT16"
    elif tensor_type == ir.TensorType(_enums.DataType.INT32):
        return "INT32"
    elif tensor_type == ir.TensorType(_enums.DataType.INT64):
        return "INT64"
    elif tensor_type == ir.TensorType(_enums.DataType.UINT8):
        return "UINT8"
    elif tensor_type == ir.TensorType(_enums.DataType.UINT16):
        return "UINT16"
    elif tensor_type == ir.TensorType(_enums.DataType.UINT32):
        return "UINT32"
    elif tensor_type == ir.TensorType(_enums.DataType.UINT64):
        return "UINT64"
    elif tensor_type == ir.TensorType(_enums.DataType.BOOL):
        return "BOOL"
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")
