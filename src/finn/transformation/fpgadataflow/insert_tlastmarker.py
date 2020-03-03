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

from onnx import TensorProto
from onnx import helper as oh

from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation


class InsertTLastMarker(Transformation):
    """Ensure that the graph is terminated with a TLastMarker node, inserting
    one if necessary."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        # TODO only makes sense for a pure fpgadataflow graph -- check!
        graph_out_name = model.graph.output[0].name
        final_node = model.find_producer(graph_out_name)
        if final_node.op_type == "TLastMarker":
            # TODO maybe check the correctness of properties
            return (model, False)
        else:
            custom_op = getCustomOp(final_node)
            num_iters = int(custom_op.get_number_output_values())
            stream_width = int(custom_op.get_outstream_width())
            out_shape = model.get_tensor_shape(graph_out_name)
            out_dtype = model.get_tensor_datatype(graph_out_name)
            elem_width = out_dtype.bitwidth()
            # make new buffer
            final_node_out = oh.make_tensor_value_info(
                model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
            )
            model.graph.value_info.append(final_node_out)
            model.set_tensor_datatype(final_node_out.name, out_dtype)
            # reroute final node output to final_node_out_name
            final_node.output[0] = final_node_out.name
            tlast_node = oh.make_node(
                "TLastMarker",
                [final_node_out.name],
                [graph_out_name],
                NumIters=num_iters,
                StreamWidth=stream_width,
                ElemWidth=elem_width,
                domain="finn",
                backend="fpgadataflow",
            )
            model.graph.node.append(tlast_node)
            return (model, True)
