# Copyright (c) 2021, Xilinx
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

import warnings
from onnx import TensorProto, helper

from finn.transformation.base import Transformation

# ToDo: Move this transformation into finn-base?


class ExtractBiasFromConv(Transformation):
    """
    Extracts the (optional) Bias from a Conv node and inserts it behind the
    Conv node as an Add node.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv":
                # Check if the node has a bias input
                if len(n.input) > 2:
                    # Extract bias
                    bias = model.get_initializer(n.input[2])
                    if bias is None:
                        warnings.warn(
                            f"Could not extract bias from Conv node {n}, "
                            f"due to missing static initialization."
                        )
                        continue

                    # Insert bias as Add node behind the Conv node
                    out_shape = model.get_tensor_shape(n.output[0])
                    # Reshape bias tensor
                    add_shape = [1] * len(out_shape)
                    # ToDo: this must change to "add_shape[-1] = bias.shape[0]" when
                    #  channels last comes around
                    add_shape[1] = bias.shape[0]
                    model.set_initializer(n.input[2], bias.reshape(add_shape))

                    act_add_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        out_shape,
                    )
                    graph.value_info.append(act_add_tensor)

                    add_node = helper.make_node(
                        "Add",
                        [act_add_tensor.name, n.input[2]],
                        [n.output[0]],
                    )
                    graph.node.insert(node_ind, add_node)

                    # Repoint Conv output and remove bias tensor
                    n.output[0] = act_add_tensor.name
                    n.input.remove(n.input[2])

                    return model, True

        return model, False
