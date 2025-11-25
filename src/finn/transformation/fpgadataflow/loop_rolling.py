import copy
import numpy as np
import onnx
import onnxscript
from enum import Enum
from onnxscript import ir
from onnxscript.rewriter import pattern, rewrite
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.transformation.fold_constants import FoldConstants
from typing import List, Tuple

from finn.util import onnxscript_helpers as osh


def get_constant_from_value(value):
    """
    Get the constant value of a tensor.
    """
    # Handle input and/or inititalizer values
    if value.producer() is None:
        return value.const_value.numpy()
    elif value.producer().op_type == "Constant":
        return value.producer().attributes["value"].value.numpy()


def same_values(inputs):
    """
    Check if all inputs have the same constant value.
    """
    if not inputs:
        return False

    first_value = get_constant_from_value(inputs[0])

    for inp in inputs[1:]:
        if not np.array_equal(first_value, get_constant_from_value(inp)):
            return False

    return True


def build_loop_replace_pattern(graph, LoopBody):
    nodes = osh.find_nodes_of_optype(graph, LoopBody.function.name)
    iterations = len(nodes)

    graph_nodes = []
    loop_inputs = []

    graph_inputs = []
    const_indexes = []
    for i, LoopInputType in enumerate(LoopBody.signature):
        if LoopInputType == LoopBodyInputType.PARAMETER:
            # validate parameter shapes
            g_shape = nodes[0].inputs[i].shape
            for node in nodes:
                if node.inputs[i].shape != g_shape:
                    print(
                        (
                            f"LoopRolling: Index {i} expected shape {g_shape}, "
                            f"got {node.inputs[i].shape}."
                        )
                    )
                    raise Exception(
                        (
                            "LoopRolling: all loop-body initializers of the same index "
                            "must have the same shape."
                        )
                    )

            # Build Concat Node
            concat_inputs = []
            for node in nodes:
                nvalue = osh.vdisconnect(copy.copy(node.inputs[i]))
                graph_inputs.append(nvalue)
                concat_inputs.append(nvalue)

            # if inputs are scalars then we need to manually perform a concat
            if len(concat_inputs[0].shape.dims) == 0:
                const_values_as_numpy = np.array([x.const_value.numpy() for x in concat_inputs])
                const_values_as_tensor = ir.Tensor(const_values_as_numpy)
                const_values_as_const_node = osh.build_constant_from_tensor(
                    f"concat_{i}", const_values_as_tensor
                )
                graph_nodes.append(const_values_as_const_node)

                reshape_shape_const = osh.build_constant_from_tensor(
                    f"reshape_shape_const_{i}", ir.Tensor(np.array([len(concat_inputs), 1]))
                )
                reshape_node = osh.build_reshape_node(
                    const_values_as_const_node.outputs[0], reshape_shape_const.outputs[0]
                )
            else:
                concat_node = osh.build_concat_node_from_inputs(concat_inputs)
                graph_nodes.append(concat_node)
                # Build Reshape Node
                reshape_shape_const = osh.build_constant_from_tensor(
                    f"reshape_shape_const_{i}",
                    ir.Tensor(np.array([len(nodes), *concat_inputs[0].shape.dims])),
                )

                reshape_node = osh.build_reshape_node(
                    concat_node.outputs[0], reshape_shape_const.outputs[0]
                )
            graph_nodes.append(reshape_shape_const)
            graph_nodes.append(reshape_node)
            loop_inputs.append(reshape_node.outputs[0])

            # Add mlo_max_iter attribute to loop input consumers
            # assuming that each input only has a single consumer
            inp = LoopBody.function.inputs[i]
            assert len(inp.consumers()) == 1
            consumer = inp.consumers()[0]
            if osh.is_fpgadataflow_onnxir_node(consumer):
                consumer.attributes["mlo_max_iter"] = ir.Attr(
                    "mlo_max_iter", ir.AttributeType.INT, iterations
                )
                consumer.attributes["inFIFODepths"] = ir.Attr(
                    "inFIFODepths", ir.AttributeType.INTS, [2, 2]
                )
        elif LoopInputType == LoopBodyInputType.CONSTANT:
            const_indexes.append(i)

            # if input is constant push down into loop body graph
            # constant_producer       = nodes[0].inputs[i].producer()
            constant_producer_value = nodes[0].inputs[i]

            # build new node and value for the loop body graph
            new_const_prod_value = ir.Value(
                name=constant_producer_value.name + "_push_down",
                type=constant_producer_value.type,
                shape=constant_producer_value.shape,
                const_value=constant_producer_value.const_value,
            )
            new_const_prod_node = ir.Node(
                name=constant_producer_value.name + "_push_down_node",
                domain="",
                inputs=[],
                op_type="Constant",
                attributes=[
                    ir.Attr(
                        name="value",
                        type=ir.AttributeType.TENSOR,
                        value=new_const_prod_value.const_value,
                    )
                ],
                outputs=[new_const_prod_value],
            )
            # add new nodes to loop body
            LoopBody.function.append(new_const_prod_node)
            LoopBody.function.sort()

            for usage in LoopBody.function.inputs[i].uses():
                usage.node.replace_input_with(usage.idx, new_const_prod_value)

            LoopBody.function.sort()

        elif LoopInputType == LoopBodyInputType.ACTIVATION:
            cinp = osh.vdisconnect(copy.copy(nodes[0].inputs[i]))
            graph_inputs.append(cinp)
            loop_inputs.append(cinp)

    for i in reversed(const_indexes):
        del LoopBody.function.inputs[i]
        del LoopBody.signature[i]

    loop_outputs = []
    graph_outputs = []
    for out in LoopBody.function.outputs:
        output = osh.vdisconnect(copy.copy(out))
        loop_outputs.append(output)
        graph_outputs.append(output)

    g_loop_body = LoopBody.function._graph
    odt = g_loop_body.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"]
    idt = odt
    body_attr = ir.Attr(name="body", type=ir.AttributeType.GRAPH, value=LoopBody.function._graph)
    backend_attr = ir.Attr(name="backend", type=ir.AttributeType.STRING, value="fpgadataflow")
    iteration = ir.Attr(name="iteration", type=ir.AttributeType.INT, value=iterations)
    inputdatatype_attr = ir.Attr(name="inputDataType", type=ir.AttributeType.STRING, value=idt)
    outputdatatype_attr = ir.Attr(name="outputDataType", type=ir.AttributeType.STRING, value=odt)

    finn_loop_node = ir.Node(
        "finn.custom_op.fpgadataflow.rtl",
        "FINNLoop",
        inputs=loop_inputs,
        attributes=[body_attr, backend_attr, iteration, inputdatatype_attr, outputdatatype_attr],
        outputs=loop_outputs,
        graph=None,
    )

    graph_nodes.append(finn_loop_node)

    graph = ir.Graph(
        name="loop_replace", nodes=graph_nodes, inputs=graph_inputs, outputs=graph_outputs
    )

    graph.sort()

    return osh.ReplacementPatternGraph(graph)


class LoopExtraction(Transformation):
    def __init__(self, hierarchy_list: List[List[str]]):
        super().__init__()

        assert isinstance(hierarchy_list, list), "Hierarchy list must be a list of strings"
        for hlist in hierarchy_list:
            assert isinstance(hlist, list), "Each hierarchy entry must be a list of strings"
            assert all(
                isinstance(item, str) for item in hlist
            ), "All items in hierarchy sub-list must be strings"
        self.hierarchy_list = hierarchy_list
        self.loop_body_template = None

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        # Apply the loop extraction transformation
        # Extract the Loop Body from ONNX metadata
        model_ir = onnxscript.ir.serde.deserialize_model(model.model)
        graph = model_ir.graph

        P = osh.PytorchHierarchyNode()
        unadded_nodes = []
        for node in graph._nodes:
            added = P.add_node(node)
            if not added:
                unadded_nodes.append(node)
        # Handle the nodes that have no metadata from PyTorch
        # by finding a neighboring node and using its metadata
        # use predecessor first than successor nodes
        # probably should provide a more robust selection method
        # in the future
        for node in unadded_nodes:
            preds = node.predecessors()
            succs = node.successors()
            if len(preds) > 0:
                mnode = preds[0]
            elif len(succs) > 0:
                mnode = succs[0]
            else:
                print("error: could not find metadata for node")
                exit(1)

            node.metadata_props["pkg.torch.onnx.name_scopes"] = mnode.metadata_props[
                "pkg.torch.onnx.name_scopes"
            ]
            node.metadata_props["pkg.torch.onnx.class_hierarchy"] = mnode.metadata_props[
                "pkg.torch.onnx.class_hierarchy"
            ]

            assert P.add_node(node)
        graph.sort()
        for i, hierarchy in enumerate(self.hierarchy_list):
            if i == 0:
                nodes = P.get_nodes(hierarchy)
            else:
                nodes += P.get_nodes(hierarchy)

        loop_body_graph_view = osh.SubGraphView(graph, "loop-body", nodes)
        loop_body_model = onnxscript.ir.Model(
            loop_body_graph_view, ir_version=model.model.ir_version
        )
        proto = onnxscript.ir.serde.serialize_model(loop_body_model)

        onnx.save(proto, "loop-body-template.onnx")
        self.loop_body_template = LoopBodyTemplate("loop-body-template.onnx")

        # Replace instances of the loop body with a function call to the loop body
        change_layers_to_function_calls = pattern.RewriteRule(
            self.loop_body_template.pattern, self.loop_body_template.function_replace
        )

        model_layers_replaced = rewrite(
            model_ir, pattern_rewrite_rules=[change_layers_to_function_calls]
        )

        model_layers_replaced.functions[
            self.loop_body_template.function.identifier()
        ] = self.loop_body_template.function
        model_layers_replaced.graph.opset_imports["loop"] = 0

        model_proto = onnxscript.ir.serde.serialize_model(model_layers_replaced)
        model.model = model_proto

        return (model, False)


def add_finn_datatype_if_needed(tensor):
    if not tensor_has_finn_datatype(tensor):
        if "quant_parameter_tensor_names" not in tensor.meta:
            tensor.meta["quant_parameter_tensor_names"] = {}
        tensor.meta["quant_parameter_tensor_names"][
            "finn_datatype"
        ] = osh.tensor_type_to_finn_datatype_string(tensor.type)


def validate_loop_type(loop_node: ir.Node):
    assert loop_node.op_type == "FINNLoop", "Node is not a FINNLoop"


def validate_loop_attributes(loop_node: ir.Node):
    required_attrs = ["body", "backend", "iteration", "inputDataType", "outputDataType"]
    for attr in required_attrs:
        assert attr in loop_node.attributes, f"FINNLoop node missing required attribute: {attr}"
    assert (
        loop_node.attributes["backend"].value == "fpgadataflow"
    ), "FINNLoop backend attribute must be 'fpgadataflow'"
    assert (
        isinstance(loop_node.attributes["iteration"].value, int)
        and loop_node.attributes["iteration"].value > 0
    ), "FINNLoop iteration attribute must be a positive integer"
    idt = loop_node.attributes["inputDataType"].value
    odt = loop_node.attributes["outputDataType"].value
    assert idt == odt, "FINNLoop inputDataType and outputDataType must be the same"


def tensor_has_finn_datatype(tensor):
    return (
        "quant_parameter_tensor_names" in tensor.meta
        and "finn_datatype" in tensor.meta["quant_parameter_tensor_names"]
    )


def finn_datatypes_match(datatype_a, datatype_b):
    return datatype_a == datatype_b


def tensor_types_match(value_a, value_b):
    return value_a.type == value_b.type


def tensor_shapes_match(value_a, value_b):
    return value_a.shape == value_b.shape


def validate_loop_io_tensor_pair(tensor_a, tensor_b):
    assert tensor_types_match(
        tensor_a, tensor_b
    ), f"FINNLoop body activation input/output type mismatch {tensor_a.type} != {tensor_b.type}"
    assert tensor_shapes_match(
        tensor_a, tensor_b
    ), f"FINNLoop body activation input/output shape mismatch {tensor_a.shape} != {tensor_b.shape}"

    add_finn_datatype_if_needed(tensor_a)
    add_finn_datatype_if_needed(tensor_b)

    assert finn_datatypes_match(
        tensor_a.meta["quant_parameter_tensor_names"]["finn_datatype"],
        tensor_b.meta["quant_parameter_tensor_names"]["finn_datatype"],
    ), f"""FINNLoop body activation input/output finn_datatype mismatch
       {tensor_a.meta['quant_parameter_tensor_names']['finn_datatype']} !=
       {tensor_b.meta['quant_parameter_tensor_names']['finn_datatype']}"""


def validate_loop_io_tensors(loop_node: ir.Node):
    # Validate that loop body activation input and output types and shapes match
    body_graph = loop_node.attributes["body"].value
    for i in range(len(body_graph.outputs)):
        validate_loop_io_tensor_pair(loop_node.inputs[i], body_graph.inputs[i])
        validate_loop_io_tensor_pair(loop_node.outputs[i], body_graph.outputs[i])
        validate_loop_io_tensor_pair(body_graph.inputs[i], body_graph.outputs[i])


def validate_loop_node(loop_node: ir.Node):
    validate_loop_type(loop_node)
    validate_loop_attributes(loop_node)
    validate_loop_io_tensors(loop_node)


class LoopBodyInputType(Enum):
    UNDEFINED = 0
    ACTIVATION = 1
    CONSTANT = 2
    PARAMETER = 3
    ITERATOR = 4
    CONDITION = 5

    def __str__(self):
        return self.name


class LoopBodyTemplate:
    def __init__(self, filename):
        self.load(filename)
        self._ir_graph.sort()
        self.pattern = osh.direct_convert_ir_graph_to_pattern(self._ir_graph)
        self.function = self._build_ir_function()
        self.function_replace = self._build_function_replace_pattern()
        self.signature = [LoopBodyInputType.UNDEFINED] * len(self._ir_graph.inputs)

    def _build_ir_function(self):
        return ir.Function(
            domain="loop", name="fn_" + self._ir_graph.name, graph=self._ir_graph, attributes=[]
        )

    def _build_function_replace_pattern(self):
        inputs = [osh.vdisconnect(copy.copy(x)) for x in self._ir_graph.inputs]
        outputs = [osh.vdisconnect(copy.copy(x)) for x in self._ir_graph.outputs]

        node = ir.Node(
            domain=self.function.domain,
            version=0,
            op_type=self.function.name,
            inputs=inputs,
            outputs=outputs,
        )

        g = ir.Graph(inputs=inputs, outputs=outputs, nodes=[node])

        return osh.ReplacementPatternGraph(g)

    def build_function_match_pattern(self, graph, use_iteration_ext=True):
        graph.sort()
        nodes = osh.find_nodes_of_optype(graph, self.function.name)
        if use_iteration_ext:
            nodes.insert(0, graph.node("iteration_ext"))
            nodes.insert(0, graph.node("condition_ext"))

        ir_model = ir.Model(
            osh.SubGraphView(graph, "inlined_pipe_pattern", nodes),
            ir_version=self._model_proto.ir_version,
        )

        pattern = osh.direct_convert_ir_graph_to_pattern(ir_model.graph)

        return (pattern, nodes)

    def load(self, filename):
        self._model_proto = onnx.load(filename)
        self._ir_model = ir.serde.deserialize_model(self._model_proto)
        self._ir_graph = self._ir_model.graph

    def update(self):
        self._ir_model = ir.Model(self._ir_graph, ir_version=self._model_proto.ir_version)
        self._model_proto = ir.serde.serialize_model(self._ir_model)

    def save(self, filename):
        self.update()
        onnx.save(self._model_proto, filename)

    def set_signature_index(self, index, stype):
        self.signature[index] = stype

    @property
    def output_signature(self):
        # The output signature is the same as the input signature but without the iteration input
        return self.signature[1:]


class LoopRolling(Transformation):
    """Boilerplate Transformation for loop rolling in fpgadataflow."""

    def __init__(self, loop_body_template):
        super().__init__()
        self.loop_body_template = loop_body_template

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        model_ir = onnxscript.ir.serde.deserialize_model(model.model)
        graph = model_ir.graph
        LoopBody = self.loop_body_template

        #################################
        # I/O Normalization for Loop Body
        #################################
        graph.sort()

        # get the consecutive node layers
        # TODO: write a check to ensure that there is only one
        #       set of consecutive nodes.
        nodes = osh.find_nodes_of_optype(graph, LoopBody.function.name)
        # Loop through all the nodes (execept the last one) and
        # identify the input to output pairs

        # my loop rolling code assumes that the activation inputs are listed first and
        # that corresponding output activations have the same index as the input
        input_swaps = []
        if len(nodes) == 1:
            # find and label the activation inputs
            for i, input in enumerate(nodes[0].inputs):
                if not input.is_initializer():
                    if input.is_graph_input() or input.producer().op_type != "Constant":
                        input_swaps.append((len(input_swaps), i))
        else:
            for i in range(len(nodes) - 1):
                a_node = nodes[i]
                b_node = nodes[i + 1]

                for a_out in a_node.outputs:
                    # Require that outputs of a have a single use of b_node
                    assert len(a_out.uses()) == 1
                    assert a_out.uses()[0][0] is b_node

                    a_use_index = a_out.uses()[0][1]
                    input_swap = (a_out.index(), a_use_index)
                    if i == 0:
                        # add swaps from the first node
                        input_swaps.append(input_swap)
                    else:
                        # check that they are the same in the rest
                        assert input_swap in input_swaps

        # apply the input swaps to each nodes
        for node in nodes:
            for swap in input_swaps:
                a = node.inputs[swap[0]]
                b = node.inputs[swap[1]]
                node.replace_input_with(swap[0], b)
                node.replace_input_with(swap[1], a)

        # apply the input swaps to the function graph
        # mark swapped nodes as activations
        activations = 0
        for swap in input_swaps:
            a = LoopBody.function.inputs[swap[0]]
            b = LoopBody.function.inputs[swap[1]]
            LoopBody.function.inputs[swap[0]] = b
            LoopBody.function.inputs[swap[1]] = a
            LoopBody.signature[swap[0]] = LoopBodyInputType.ACTIVATION
            activations += 1

        # Next Label Inputs according to how they are produced.
        # Indexable inputs will have different constant or none producers
        # Constant values broadcast to all nodes will have the same producer
        # Skip the (all) Activation inputs (have been swapped to beginning of the list)
        for index in range(activations, len(nodes[0].inputs)):
            inputs = []
            for node in nodes:
                cinput = node.inputs[index]
                inputs.append(cinput)

            if osh.same(inputs) or same_values(inputs):
                # Constant with Respect to Loop
                LoopBody.signature[index] = LoopBodyInputType.CONSTANT
            else:
                # Must be Indexed
                LoopBody.signature[index] = LoopBodyInputType.PARAMETER

        ###################################################
        # End I/O Normalization for Loop Body
        ###################################################

        LoopMatchPattern, nodes = LoopBody.build_function_match_pattern(
            model_ir.graph, use_iteration_ext=False
        )

        loop_replace_pattern = build_loop_replace_pattern(model_ir.graph, LoopBody)

        change_function_calls_to_loop = pattern.RewriteRule(LoopMatchPattern, loop_replace_pattern)
        rewrite_set = pattern.RewriteRuleSet([change_function_calls_to_loop])
        count = rewrite_set.apply_to_model(model_ir, verbose=None)
        print(f"Rolled {count} function calls into a loop operator")

        for node in model_ir.graph._nodes:
            if node.op_type == "FINNLoop":
                validate_loop_node(node)

        model = onnxscript.ir.serde.serialize_model(model_ir)

        model_wrapper = ModelWrapper(model)

        # Allow operators in the loop body to adapt their attributes based on
        # the determined input signature (e.g., changing parameter styles from
        # "const" to "input" for streamed parameters)
        # This must be done after serialization so we can work with protobuf nodes

        from finn.util.basic import getHWCustomOp

        for loop_node in model_wrapper.get_nodes_by_op_type("FINNLoop"):
            loop_body = getHWCustomOp(loop_node).get_nodeattr("body")
            for node in loop_body.graph.node:
                if not is_custom_op(node.domain):
                    continue
                try:
                    inst = getHWCustomOp(node)
                    inst.adapt_for_loop_body(LoopBody.signature)
                except (KeyError, AttributeError):
                    # Operator doesn't need adaptation or doesn't support it
                    pass

        model = model_wrapper.transform(FoldConstants(), apply_to_subgraphs=True)

        return (model, False)
