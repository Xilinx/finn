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
from onnx import TensorProto
from onnx import helper as oh
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class InsertTLastMarker(Transformation):
    """Ensure that the graph is started/terminated with a TLastMarker node, inserting
    one if necessary.
    Use constructor args to determine type of TLastMarker to be inserted.
    More information available on the TLastMarker documentation.
    """

    def __init__(self, both=False, external=True, dynamic=True):
        super().__init__()
        self.dyniters = dynamic
        self.external = external
        self.both = both

    def apply(self, model):
        # TODO only makes sense for a pure fpgadataflow graph -- check!
        graph_out_name = model.graph.output[0].name
        final_node = model.find_producer(graph_out_name)
        graph_modified = False
        if final_node.op_type != "TLastMarker" and not (
            final_node.op_type == "IODMA"
            and get_by_name(final_node.attribute, "direction").s.decode("UTF-8")
            == "out"
        ):

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
                DynIters=(1 if self.dyniters else 0),
                Direction="out",
                Protocol=("external" if self.external else "internal"),
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
            )
            model.graph.node.append(tlast_node)
            graph_modified = True
        # if both is True, also insert marker on input
        if self.both:
            # detect and parse graph inputs
            insert_idx = 0
            graph_in_names = [x.name for x in model.graph.input]
            for graph_in_name in graph_in_names:
                first_node = model.find_consumers(graph_in_name)
                # skip if no consumers (this may be the case for unused initializers)
                # TODO: fix this with a cleanup transform
                if first_node == []:
                    continue
                assert len(first_node) == 1, "Input fans out to multiple nodes"
                first_node = first_node[0]
                # several scenarios exclude the node:
                # 1. node is a FC layer with internal weights, in which case
                #    the input is in the list of graph inputs because it has an
                #    initializer (TODO: fix this with a clean-up transform)
                if (
                    first_node.op_type == "MatrixVectorActivation"
                    and get_by_name(first_node.attribute, "mem_mode").s.decode("UTF-8")
                    != "external"
                ):
                    continue
                # 2. node is either a TLastMarker or an input IODMA
                if first_node.op_type != "TLastMarker" and not (
                    first_node.op_type == "IODMA"
                    and get_by_name(first_node.attribute, "direction").s.decode("UTF-8")
                    == "in"
                ):

                    custom_op = getCustomOp(first_node)
                    num_iters = np.prod(custom_op.get_folded_input_shape()[1:-1])
                    inp_idx = list(first_node.input).index(graph_in_name)
                    if inp_idx > 0:
                        if (
                            first_node.op_type == "MatrixVectorActivation"
                            and inp_idx == 1
                        ):
                            stream_width = int(custom_op.get_weightstream_width())
                        elif first_node.op_type == "AddStreams_Batch" and inp_idx == 1:
                            stream_width = int(custom_op.get_instream_width())
                        else:
                            raise Exception("No method to determine stream width")
                    else:
                        stream_width = int(custom_op.get_instream_width())
                    in_shape = model.get_tensor_shape(graph_in_name)
                    in_dtype = model.get_tensor_datatype(graph_in_name)
                    elem_width = in_dtype.bitwidth()
                    # make new buffer
                    first_node_in = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, in_shape
                    )
                    model.graph.value_info.append(first_node_in)
                    model.set_tensor_datatype(first_node_in.name, in_dtype)
                    ini = model.get_initializer(graph_in_name)
                    # copy initializer if it exists
                    if ini is not None:
                        model.set_initializer(first_node_in.name, ini)
                    # reroute final node output to first_node_in_name
                    first_node.input[inp_idx] = first_node_in.name
                    tlast_node = oh.make_node(
                        "TLastMarker",
                        [graph_in_name],
                        [first_node_in.name],
                        NumIters=num_iters,
                        StreamWidth=stream_width,
                        ElemWidth=elem_width,
                        DynIters=(1 if self.dyniters else 0),
                        Direction="in",
                        Protocol=("external" if self.external else "internal"),
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                    )
                    model.graph.node.insert(insert_idx, tlast_node)
                    graph_modified = True
                    insert_idx += 1
        return (model, graph_modified)
