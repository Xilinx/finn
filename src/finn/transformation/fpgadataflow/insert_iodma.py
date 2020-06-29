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

from finn.util.basic import get_by_name
from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation
from finn.transformation.general import SortGraph
import math
import numpy as np


class InsertIODMA(Transformation):
    """Insert DMA nodes on all inputs and outputs."""

    def __init__(self, max_intfwidth=32):
        super().__init__()
        assert (
            2 ** math.log2(max_intfwidth) == max_intfwidth
        ), "max_intfwidth must be a power of 2"
        self.max_intfwidth = max_intfwidth

    def apply(self, model):
        # only makes sense for a pure fpgadataflow graph -- so we check!
        all_nodes = list(model.graph.node)
        assert all(
            get_by_name(x.attribute, "backend").s.decode("UTF-8") == "fpgadataflow"
            for x in all_nodes
        )
        # parse streamingfclayers looking for external weights with no attached IODMA
        fc_extw_nodes = list(
            filter(
                lambda x: x.op_type == "StreamingFCLayer_Batch"
                and get_by_name(x.attribute, "mem_mode").s.decode("UTF-8") == "external"
                and model.find_producer(x.input[1]) is None,
                all_nodes,
            )
        )
        graph_in_name = model.graph.input[0].name
        first_node = model.find_consumer(graph_in_name)
        graph_out_name = model.graph.output[0].name
        final_node = model.find_producer(graph_out_name)
        if (
            final_node.op_type == "IODMA"
            and first_node.op_type == "IODMA"
            and len(fc_extw_nodes) == 0
        ):
            # TODO maybe check the correctness of properties
            return (model, False)
        else:
            if final_node.op_type != "IODMA":
                out_shape = model.get_tensor_shape(graph_out_name)
                out_dtype = model.get_tensor_datatype(graph_out_name)
                # determine the feasible interface width
                transfer_bits = np.prod(out_shape) * out_dtype.bitwidth()
                intfwidth = math.gcd(transfer_bits, self.max_intfwidth)
                assert (
                    intfwidth % 8 == 0
                ), "No feasible interface width for transfer size"
                # get width of stream input to DMA
                streamWidth = getCustomOp(final_node).get_outstream_width()
                # make new buffer
                final_node_out = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                model.graph.value_info.append(final_node_out)
                model.set_tensor_datatype(final_node_out.name, out_dtype)
                # reroute final node output to final_node_out_name
                final_node.output[0] = final_node_out.name
                dma_node = oh.make_node(
                    "IODMA",
                    [final_node_out.name],
                    [graph_out_name],
                    numInputVectors=out_shape[:-1],
                    NumChannels=out_shape[-1],
                    dataType=str(out_dtype.name),
                    intfWidth=intfwidth,
                    streamWidth=streamWidth,
                    direction="out",
                    domain="finn",
                    backend="fpgadataflow",
                )
                model.graph.node.append(dma_node)
            if first_node.op_type != "IODMA":
                in_shape = model.get_tensor_shape(graph_in_name)
                in_dtype = model.get_tensor_datatype(graph_in_name)
                # determine the feasible interface width
                transfer_bits = np.prod(in_shape) * in_dtype.bitwidth()
                intfwidth = math.gcd(transfer_bits, self.max_intfwidth)
                assert (
                    intfwidth % 8 == 0
                ), "No feasible interface width for transfer size"
                # get width of stream output from DMA
                streamWidth = getCustomOp(first_node).get_instream_width()
                # make new buffer
                first_node_in = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, in_shape
                )
                model.graph.value_info.append(first_node_in)
                model.set_tensor_datatype(first_node_in.name, in_dtype)
                # reroute final node output to final_node_out_name
                first_node.input[0] = first_node_in.name
                dma_node = oh.make_node(
                    "IODMA",
                    [graph_in_name],
                    [first_node_in.name],
                    numInputVectors=in_shape[:-1],
                    NumChannels=in_shape[-1],
                    dataType=str(in_dtype.name),
                    intfWidth=intfwidth,
                    streamWidth=streamWidth,
                    direction="in",
                    domain="finn",
                    backend="fpgadataflow",
                )
                model.graph.node.insert(0, dma_node)
            for fc_node in fc_extw_nodes:
                fc_w_name = fc_node.input[1]
                w_shape = model.get_tensor_shape(fc_w_name)
                w_dtype = model.get_tensor_datatype(fc_w_name)
                # determine the feasible interface width
                transfer_bits = np.prod(w_shape) * w_dtype.bitwidth()
                intfwidth = math.gcd(transfer_bits, self.max_intfwidth)
                assert (
                    intfwidth % 8 == 0
                ), "No feasible interface width for transfer size"
                # calculate width of stream output from DMA
                pe = get_by_name(fc_node.attribute, "PE").i
                simd = get_by_name(fc_node.attribute, "SIMD").i
                streamWidth = simd * pe * w_dtype.bitwidth()
                # make new buffer
                fc_node_in = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, w_shape
                )
                model.graph.value_info.append(fc_node_in)
                model.set_tensor_datatype(fc_node_in.name, w_dtype)
                dma_node = oh.make_node(
                    "IODMA",
                    [fc_w_name],
                    [fc_node_in.name],
                    numInputVectors=w_shape[:-1],
                    NumChannels=w_shape[-1],
                    dataType=str(w_dtype.name),
                    intfWidth=intfwidth,
                    streamWidth=streamWidth,
                    direction="in",
                    burstMode="wrap",
                    domain="finn",
                    backend="fpgadataflow",
                )
                fc_node.input[1] = fc_node_in.name
                model.graph.node.insert(0, dma_node)
            model = model.transform(SortGraph())
            return (model, True)
