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
import warnings
from onnx import TensorProto
from onnx import helper as oh
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.util.fpgadataflow import is_fpgadataflow_node


def _is_fifo_node(node):
    if node.op_type == "StreamingFIFO":
        return True
    else:
        return False


def _suitable_node(node):
    if node is not None:
        if is_fpgadataflow_node(node) is True:
            if _is_fifo_node(node) is False:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def _suitable_folded_shapes(ishape, oshape):
    matching_stream_width = ishape[-1] == oshape[-1]
    matching_size = np.prod(ishape) == np.prod(oshape)
    return matching_stream_width and matching_size


class InsertFIFO(Transformation):
    """Inserting FIFOs in the beginning and end of the graph as well as
    between fpgadataflow nodes.

    Takes the setting for the depth from the surrounding nodes by extracting
    node attribute 'outFIFODepth' of the previous and node attribute 'inFIFODepth'
    of the subsequent node. max() of these two values sets the FIFO depth.

    Constructor arguments:
    - max_qsrl_depth : FIFOs deeper than this will use Vivado IP instead of
                       Verilog FIFOs (Q_srl.v)
    - vivado_ram_style : the StreamingFIFO.ram_style attribute to be used for
                          large FIFOs implemented by Vivado
    - create_shallow_fifos : Normally, shallow-depth (<=2) FIFOs won't be created since
                            HLS streaming interfaces already have a degree of buffering.
                            Override with this parameter.


    The other node attributes necessary to create a FIFO node are taken from the
    node the FIFO node is inserted after: 'folded_shape' and 'dtype'"""

    def __init__(
        self, create_shallow_fifos=False, max_qsrl_depth=256, vivado_ram_style="auto"
    ):
        super().__init__()
        self.create_shallow_fifos = create_shallow_fifos
        self.max_qsrl_depth = max_qsrl_depth
        self.vivado_ram_style = vivado_ram_style

    def apply(self, model):
        graph = model.graph
        node_ind = -1
        graph_modified = False
        for first_node in graph.node:
            node_ind += 1
            if _suitable_node(first_node):
                for n_output in first_node.output:
                    consumers = model.find_consumers(n_output)
                    if consumers == []:
                        continue
                    if len(consumers) > 1:
                        warnings.warn(
                            first_node.name
                            + ": HLS node with fan-out higher than 1 cannot be stitched"
                        )
                    consumer = consumers[0]
                    if _suitable_node(consumer) is True:
                        n0 = getCustomOp(first_node)
                        # determine fifo node attributes
                        fld_shape = n0.get_folded_output_shape()
                        dtype = n0.get_output_datatype()

                        # check if folded_shape of output of first node and
                        # input of the second node is equal
                        n1 = getCustomOp(consumer)
                        for idx, inp in enumerate(consumer.input):
                            if inp == n_output:
                                if idx == 0:
                                    fld_shape_2 = n1.get_folded_input_shape()
                                else:
                                    fld_shape_2 = n1.get_folded_input_shape(ind=idx)
                        assert _suitable_folded_shapes(
                            fld_shape, fld_shape_2
                        ), """The
                        folded output shape of the first node is not the same as the
                        folded output shape of the second node. A streaming fifo can't
                        be implemented in between these nodes."""

                        # check if outFIFOdepth attribute of first node
                        # and inFIFOdepth attribute of consumer node is equal
                        n0_depth = n0.get_nodeattr("outFIFODepth")
                        n1_depth = n1.get_nodeattr("inFIFODepth")
                        if n0_depth == n1_depth:
                            fifo_depth = n0_depth
                        elif n0_depth != n1_depth:
                            fifo_depth = max(n0_depth, n1_depth)

                        if fifo_depth > 2 or self.create_shallow_fifos:
                            # assumption: HLS streaming components already have
                            # depth-2 FIFOs on inputs and outputs, so no point
                            # creating additional small FIFOs in between --
                            # we only create the larger FIFOs specified
                            # or unless create_shallow_fifos is specified
                            fifo_output_tensor = oh.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                n0.get_normal_output_shape(),
                            )
                            graph.value_info.append(fifo_output_tensor)
                            model.set_tensor_datatype(fifo_output_tensor.name, dtype)
                            impl_style = (
                                "vivado" if fifo_depth > self.max_qsrl_depth else "rtl"
                            )
                            fifo_node = oh.make_node(
                                "StreamingFIFO",
                                [n_output],
                                [fifo_output_tensor.name],
                                domain="finn.custom_op.fpgadataflow",
                                backend="fpgadataflow",
                                depth=fifo_depth,
                                folded_shape=fld_shape,
                                dataType=str(dtype.name),
                                impl_style=impl_style,
                                ram_style=self.vivado_ram_style,
                            )
                            # insert fifo
                            graph.node.insert(node_ind + 1, fifo_node)
                            # set fifo output tensor as new input tensor of second node
                            for idx, inp in enumerate(consumer.input):
                                if inp == n_output:
                                    consumer.input[idx] = fifo_output_tensor.name
                            # ensure created FIFO depth is reflected on both sides
                            n0.set_nodeattr("outFIFODepth", fifo_depth)
                            n1.set_nodeattr("inFIFODepth", fifo_depth)
                            graph_modified = True

        if graph_modified is False:
            graph_in_names = [x.name for x in model.graph.input]
            for graph_in_name in graph_in_names:
                first_node = model.find_consumer(graph_in_name)
                # insert FIFO as first node, except when first node is DMA
                if (
                    first_node.op_type != "StreamingFIFO"
                    and first_node.op_type != "IODMA"
                ):
                    inp_ind = list(first_node.input).index(graph_in_name)
                    n_input = first_node.input[inp_ind]
                    n0 = getCustomOp(first_node)
                    # determine fifo node attributes
                    if inp_ind == 0:
                        fld_shape = n0.get_folded_input_shape()
                        dtype = n0.get_input_datatype()
                    else:
                        fld_shape = n0.get_folded_input_shape(inp_ind)
                        dtype = n0.get_input_datatype(inp_ind)
                    fifo_depth = n0.get_nodeattr("inFIFODepth")

                    if fifo_depth <= 2:
                        warnings.warn("Overriding input FIFO depth to 32")
                        fifo_depth = 32

                    # create fifo node
                    fifo_output_tensor = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        n0.get_normal_input_shape(),
                    )
                    graph.value_info.append(fifo_output_tensor)
                    model.set_tensor_datatype(fifo_output_tensor.name, dtype)
                    impl_style = "vivado" if fifo_depth > self.max_qsrl_depth else "rtl"

                    fifo_node = oh.make_node(
                        "StreamingFIFO",
                        [n_input],
                        [fifo_output_tensor.name],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        depth=fifo_depth,
                        folded_shape=fld_shape,
                        dataType=str(dtype.name),
                        impl_style=impl_style,
                        ram_style=self.vivado_ram_style,
                    )
                    # insert fifo
                    graph.node.insert(0, fifo_node)

                    # set fifo output tensor as new input tensor of second node
                    first_node.input[inp_ind] = fifo_output_tensor.name

            # insert FIFO as last node, except when last node is DMA
            graph_out_names = [x.name for x in model.graph.output]
            for graph_out_name in graph_out_names:
                final_node = model.find_producer(graph_out_name)
                if (
                    final_node.op_type != "StreamingFIFO"
                    and final_node.op_type != "IODMA"
                ):
                    assert (
                        final_node.op_type != "TLastMarker"
                    ), """Insert tlast marker should be done
                        after inserting the FIFOs"""
                    n0 = getCustomOp(final_node)
                    # determine fifo node attributes
                    fld_shape = n0.get_folded_output_shape()
                    dtype = n0.get_output_datatype()
                    fifo_depth = n0.get_nodeattr("outFIFODepth")

                    if fifo_depth <= 2:
                        warnings.warn("Overriding output FIFO depth to 32")
                        fifo_depth = 32

                    # create fifo node
                    fifo_input_tensor = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        n0.get_normal_output_shape(),
                    )
                    graph.value_info.append(fifo_input_tensor)
                    model.set_tensor_datatype(fifo_input_tensor.name, dtype)
                    impl_style = "vivado" if fifo_depth > self.max_qsrl_depth else "rtl"
                    fifo_node = oh.make_node(
                        "StreamingFIFO",
                        [fifo_input_tensor.name],
                        [graph_out_name],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        depth=fifo_depth,
                        folded_shape=fld_shape,
                        dataType=str(dtype.name),
                        impl_style=impl_style,
                        ram_style=self.vivado_ram_style,
                    )
                    # insert fifo
                    graph.node.append(fifo_node)

                    # set fifo output tensor as new input tensor of second node
                    final_node.output[0] = fifo_input_tensor.name

        return (model, graph_modified)
