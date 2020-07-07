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

import finn.custom_op.registry as registry
import finn.core.data_layout as DataLayout
from finn.transformation import Transformation
import warnings
from finn.util.basic import get_by_name


def _dims_to_layout(model, node, ndims):
    if ndims == 2:
        return DataLayout.NC
    else:
        if node.domain == "finn":
            if node.op_type == "MultiThreshold" or node.op_type == "QuantAvgPool2d":
                mt_inst = registry.getCustomOp(node)
                layout = mt_inst.get_nodeattr("data_layout")
                if layout == "NHWC" and ndims == 4:
                    return DataLayout.NHWC
                elif layout == "NCHW" and ndims == 4:
                    return DataLayout.NCHW
                else:
                    return DataLayout.UNKNOWN
            else:
                if ndims == 4:
                    return DataLayout.NHWC
                else:
                    return DataLayout.UNKNOWN
        else:
            # propagate input layout to output
            # TODO this won't work for concat, squeeze/unsqueeze/reshape...
            return model.get_tensor_layout(node.input[0])


def _infer_node_data_layout(model, node):
    """Infer output data layout annotation(s) for a particular node.
    Returns True if any changes were made."""
    old_layouts = list(map(lambda x: model.get_tensor_layout(x), node.output))
    if node.domain == "finn":
        # try to guess based on number of output dims
        for o in node.output:
            ndims = len(model.get_tensor_shape(o))
            new_layout = _dims_to_layout(model, node, ndims)
            model.set_tensor_layout(o, new_layout)
    else:
        if node.op_type == "Transpose":
            # grab input annotation and switch it around using perm
            perm = get_by_name(node.attribute, "perm").ints
            inp_layout = model.get_tensor_layout(node.input[0])
            out_layout = [inp_layout[i] for i in perm]
            model.set_tensor_layout(node.output[0], out_layout)
        else:
            # try to guess based on number of output dims
            for o in node.output:
                ndims = len(model.get_tensor_shape(o))
                model.set_tensor_layout(o, _dims_to_layout(model, node, ndims))
    # compare old and new output dtypes to see if anything changed
    new_layouts = list(map(lambda x: model.get_tensor_layout(x), node.output))
    graph_modified = new_layouts != old_layouts
    return graph_modified


class InferDataLayouts(Transformation):
    """Try to infer data layout annotations info for all input/intermediate/output
    tensors based on inputs and node type."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        # first, make sure that the global input has an annotation
        # this is really hard to do in general, so we do some bad guesswork
        inp_name = graph.input[0].name
        if model.get_tensor_layout(inp_name) is None:
            inp_shape = model.get_tensor_shape(inp_name)
            if len(inp_shape) == 4:
                warnings.warn("Assuming 4D input is NCHW")
                model.set_tensor_layout(inp_name, DataLayout.NCHW)
                graph_modified = True
            elif len(inp_shape) == 2:
                graph_modified = True
                warnings.warn("Assuming 2D input is NC")
                model.set_tensor_layout(inp_name, DataLayout.NC)
            else:
                raise Exception(
                    """Unknown number of dims for input, don't know
                how to annotate"""
                )
        for node in graph.node:
            graph_modified |= _infer_node_data_layout(model, node)
        return (model, graph_modified)
