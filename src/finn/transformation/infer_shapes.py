import onnx.shape_inference as si

import finn.custom_op.registry as registry
from finn.core.modelwrapper import ModelWrapper
from finn.transformation import Transformation


def _make_shape_compatible_op(node):
    """Return a shape-compatible non-FINN op for a given FINN op. Used for
    shape inference with custom ops."""
    assert node.domain == "finn"
    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type](node)
        return inst.make_shape_compatible_op(node)
    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)


def _hide_finn_ops(model):
    """Replace any FINN ops by shape-compatible ones, and return a dict that
    can be used to map the string representations of the new (shape-compatible)
    ops back to the old ops."""
    hidden_ops = {}
    node_ind = 0
    for node in model.graph.node:
        node_ind += 1
        if node.domain == "finn":
            new_node = _make_shape_compatible_op(node)
            hidden_ops[str(new_node)] = node
            model.graph.node.insert(node_ind, new_node)
            model.graph.node.remove(node)
    return hidden_ops


def _restore_finn_ops(model, hidden_ops):
    """Replace any shape-compatible ops with the FINN ops that originally
    generated them."""
    node_ind = 0
    for node in model.graph.node:
        node_ind += 1
        try:
            old_node = hidden_ops[str(node)]
            model.graph.node.insert(node_ind, old_node)
            model.graph.node.remove(node)
        except KeyError:
            pass


class InferShapes(Transformation):
    """Ensure every tensor in the model has a specified shape (ValueInfo)."""

    def apply(self, model):
        # hide your riches!
        hidden_ops = _hide_finn_ops(model)
        # call regular ONNX shape inference
        model = ModelWrapper(si.infer_shapes(model.model))
        # bring back hidden ops
        _restore_finn_ops(model, hidden_ops)
        return (model, False)
