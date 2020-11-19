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

import numpy as np
from onnx import helper as oh

from finn.core.datatype import DataType
from finn.transformation.base import Transformation


class ConvertSignToThres(Transformation):
    """Convert Sign node instances to MultiThreshold with threshold at 0."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Sign":
                sign_in_name = n.input[0]
                sign_out_name = n.output[0]
                # find consumer
                consumer = model.find_consumer(sign_out_name)
                assert (
                    consumer is not None
                ), """There is no consumer of the
                sign_out tensor."""
                # create thresholds
                thres_param_name = model.make_new_valueinfo_name()
                thres_param = np.asarray([[0]], dtype=np.float32)
                model.set_initializer(thres_param_name, thres_param)
                # create a new node
                mt_node = oh.make_node(
                    "MultiThreshold",
                    [sign_in_name, thres_param_name],
                    [sign_out_name],
                    domain="finn.custom_op.general",
                    out_scale=2.0,
                    out_bias=-1.0,
                    out_dtype="BIPOLAR",
                )
                # remove old node, add new node to graph at correct position
                graph.node.insert(node_ind, mt_node)
                graph.node.remove(n)
                # add quantization annotations
                model.set_tensor_datatype(sign_out_name, DataType.BIPOLAR)
                graph_modified = True
        return (model, graph_modified)
