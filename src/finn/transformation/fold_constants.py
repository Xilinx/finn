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

import finn.core.onnx_exec as oxe
from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes


class FoldConstants(Transformation):
    """Replace the output of a node with const-only inputs with a precomputed
    result."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        for n in graph.node:
            node_ind += 1
            node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
            node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
            node_out = n.output[0]
            is_all_constant_inputs = len(node_inp_dyn) == 0
            ishape = model.get_tensor_shape(n.input[0])
            is_const_shape = (n.op_type == "Shape") and (ishape is not None)
            if is_all_constant_inputs or is_const_shape:
                # this node has no dynamic inputs, only constant ones -- so we can
                # do constant folding.
                oxe.execute_node(n, execution_context, graph)
                # use the execution result as an initializer
                model.set_initializer(node_out, execution_context[node_out])
                # remove old node
                graph.node.remove(n)
                graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
        return (model, graph_modified)
