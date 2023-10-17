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

import math
import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.util.basic import get_by_name

from IPython.core.debugger import set_trace

class InsertACCL(Transformation):
    def __init__(self, world_size, rank, recv_from=None, send_to=None):
        self.world_size = world_size
        self.rank = rank
        self.recv_from = recv_from
        self.send_to = send_to


    def apply(self, model):
        modified = False
        # only makes sense for a pure fpgadataflow graph -- so we check!
        all_nodes = list(model.graph.node)
        assert all(
            get_by_name(x.attribute, "backend").s.decode("UTF-8") == "fpgadataflow"
            for x in all_nodes
        )

        if self.recv_from is not None:
            graph_in_names = [x.name for x in model.graph.input]
            for graph_in_name in graph_in_names:
                first_node = model.find_consumer(graph_in_name)
                if first_node.op_type == "ACCLIn":
                    continue
                else:
                    in_shape = model.get_tensor_shape(graph_in_name)
                    in_dtype = model.get_tensor_datatype(graph_in_name)

                    first_node_inst = getCustomOp(first_node)
                    in_folded_shape = first_node_inst.get_folded_input_shape()

                    # make new buffer
                    first_node_in = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, in_shape
                    )
                    model.graph.value_info.append(first_node_in)
                    model.set_tensor_datatype(first_node_in.name, in_dtype)
                    # reroute first node input
                    first_node.input[0] = first_node_in.name

                    accl_node = oh.make_node(
                        "ACCLIn",
                        [graph_in_name],
                        [first_node_in.name],
                        numInputVectors=in_folded_shape[:-1],
                        NumChannels=in_folded_shape[-1],
                        dataType=str(in_dtype),
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        rank=self.rank,
                        worldSize=self.world_size,
                        otherRank=self.recv_from,
                    )
                    model.graph.node.insert(0, accl_node)
                    modified = True
        if self.send_to is not None:
            graph_out_names = [x.name for x in model.graph.output]
            for graph_out_name in graph_out_names:
                final_node = model.find_producer(graph_out_name)
                if final_node.op_type == "ACCLOut":
                    continue
                else:
                    out_shape = model.get_tensor_shape(graph_out_name)
                    out_dtype = model.get_tensor_datatype(graph_out_name)

                    final_node_inst = getCustomOp(final_node)
                    out_folded_shape = final_node_inst.get_folded_output_shape()

                    # make new buffer
                    final_node_out = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                    )
                    model.graph.value_info.append(final_node_out)
                    model.set_tensor_datatype(final_node_out.name, out_dtype)
                    # reroute final node output to final_node_out_name
                    final_node.output[0] = final_node_out.name

                    dma_node = oh.make_node(
                        "ACCLOut",
                        [final_node_out.name],
                        [graph_out_name],
                        numInputVectors=out_folded_shape[:-1],
                        NumChannels=out_folded_shape[-1],
                        dataType=str(out_dtype),
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        rank=self.rank,
                        worldSize=self.world_size,
                        otherRank=self.send_to,
                    )
                    model.graph.node.append(dma_node)
                    modified = True

        if modified:
            model = model.transform(SortGraph())
        return (model, modified)
