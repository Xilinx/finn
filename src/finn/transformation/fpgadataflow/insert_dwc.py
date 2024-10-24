# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

from onnx import helper as oh
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.util.fpgadataflow import is_fpgadataflow_node


def _is_dwc_node(node):
    return node.op_type.startswith("StreamingDataWidthConverter")


def _suitable_node(node):
    if node is not None:
        if is_fpgadataflow_node(node):
            if _is_dwc_node(node):
                # no DWC for DWCs
                return False
            elif node.op_type == "IODMA_hls":
                # IODMA data shapes/widths need special handling
                return False
            else:
                return True
        else:
            return False
    else:
        return False


class InsertDWC(Transformation):
    """Add data width converters between layers where necessary."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = -1
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if _suitable_node(n):
                for output_name in n.output:
                    consumers = model.find_consumers(output_name)
                    if consumers == []:
                        continue
                    assert len(consumers) == 1, (
                        n.name + ": HW node with fan-out higher than 1 cannot be stitched"
                    )
                    consumer = consumers[0]
                    if _suitable_node(consumer) is True:
                        n0 = getCustomOp(n)
                        n1 = getCustomOp(consumer)
                        n0_out_shape = n0.get_folded_output_shape()
                        # in some special cases, we need to get folded shapes of
                        # non-default inputs for the consumer
                        # - if FC and external mem, it could be connected to input 1
                        # - if concat, could be connected to any input
                        if (
                            consumer.op_type.startswith("MVAU")
                            and n1.get_nodeattr("mem_mode") == "external"
                        ) or (consumer.op_type.startswith("StreamingConcat")):
                            # get input idx
                            in_idx = None
                            for idx, n_input in enumerate(consumer.input):
                                if output_name == n_input:
                                    in_idx = idx
                            assert in_idx is not None, "Malformed model"
                            n1_in_shape = n1.get_folded_input_shape(in_idx)
                        else:
                            # use default folded input shape
                            n1_in_shape = n1.get_folded_input_shape()

                        if n0_out_shape[-1] != n1_in_shape[-1]:
                            graph_modified = True
                            # determine dwc inwidth
                            dwc_in_width = n0.get_outstream_width()
                            # determine dwc outwidth
                            dwc_out_width = n1.get_instream_width()
                            node_optype = "StreamingDataWidthConverter"

                            # determine shape for dwc
                            dwc_shape = n0.get_normal_output_shape()

                            # determine FINN dtype for dwc
                            dtype = n0.get_output_datatype()
                            # determine onnx tensor dtype for dwc
                            n0_otensor = model.get_tensor_valueinfo(output_name)
                            n0_tensor_dtype = n0_otensor.type.tensor_type.elem_type

                            dwc_output_tensor = oh.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                n0_tensor_dtype,
                                dwc_shape,
                            )
                            graph.value_info.append(dwc_output_tensor)

                            dwc_node = oh.make_node(
                                node_optype,
                                [output_name],
                                [dwc_output_tensor.name],
                                domain="finn.custom_op.fpgadataflow",
                                backend="fpgadataflow",
                                shape=dwc_shape,
                                inWidth=dwc_in_width,
                                outWidth=dwc_out_width,
                                dataType=str(dtype.name),
                            )
                            # insert dwc
                            graph.node.insert(node_ind + 1, dwc_node)

                            # set dwc output tensor as new input tensor of second node
                            for idx, inp in enumerate(consumer.input):
                                if inp == output_name:
                                    consumer.input[idx] = dwc_output_tensor.name

        return (model, graph_modified)
