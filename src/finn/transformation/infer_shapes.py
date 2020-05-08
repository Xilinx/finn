# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import onnx.shape_inference as si

import finn.custom_op.registry as registry
from finn.core.modelwrapper import ModelWrapper
from finn.transformation import Transformation


def _make_shape_compatible_op(node, model):
    """Return a shape-compatible non-FINN op for a given FINN op. Used for
    shape inference with custom ops."""
    assert node.domain == "finn", 'Node domain is not set to "finn".'
    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type](node)
        return inst.make_shape_compatible_op(model)
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
            new_node = _make_shape_compatible_op(node, model)
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
