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
from finn.transformation.base import Transformation
from finn.transformation.general import SortGraph
import finn.core.data_layout as DataLayout
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

    def get_mem_init(self, weights, pe, simd):
        """
        Returns matrix ready for pack_innermost_dim_as_hex_string with
        reverse=False (finn.util.data_packing) to return the memory init file
        little endian packed.
        That is, get_mem_init returns:
        elem(pe,simd)
        addr = 0: [(pe-1,simd-1),(pe-1,simd-2),...(0,1),(0,0)]
        addr = 1: [(pe-1,simd*2-1),.......(0,simd+1),(0,simd)]
        .
        """
        w_shape = weights.shape
        assert len(w_shape) == 2, "weights withincorrect number of dims"
        inp_w, out_w = w_shape

        assert out_w % pe == 0, "Malformed weight matrix"
        assert inp_w % simd == 0, "Malformed weight matrix"
        reshaped_w = np.zeros(inp_w * out_w).reshape(-1, pe * simd)

        addr = 0
        for fr in range(out_w // pe):
            for fc in range(inp_w // simd):
                tile = weights[
                    (fc * simd) : ((fc + 1) * simd), (fr * pe) : ((fr + 1) * pe)
                ]
                for p in range(pe):
                    reshaped_w[addr, (p * simd) : ((p + 1) * simd)] = tile[
                        :, p
                    ].transpose()
                addr += 1
        reshaped_w = np.flip(reshaped_w, axis=-1)
        return reshaped_w

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
                and get_by_name(x.attribute, "mem_mode") is not None
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
                final_node_inst = getCustomOp(final_node)
                out_folded_shape = final_node_inst.get_folded_output_shape()
                # take advantage of AXI stream width padding for DMA alignment
                # (AXI streams are always padded to 8 bits)
                # this is the width of stream input to DMA
                padded_outstream_width = final_node_inst.get_outstream_width_padded()
                padded_outstream_bytes = padded_outstream_width // 8
                # determine the feasible interface width
                transfer_bits = padded_outstream_width * np.prod(out_folded_shape[:-1])
                intfwidth = math.gcd(transfer_bits, self.max_intfwidth)
                assert (
                    intfwidth % 8 == 0
                ), "No feasible interface width for transfer size"
                # make new buffer
                final_node_out = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                model.graph.value_info.append(final_node_out)
                model.set_tensor_datatype(final_node_out.name, out_dtype)
                # reroute final node output to final_node_out_name
                final_node.output[0] = final_node_out.name
                # FIXME: currently always using 8-bit dtypes to work around the
                # padding problems for i/o DMA
                dma_node = oh.make_node(
                    "IODMA",
                    [final_node_out.name],
                    [graph_out_name],
                    numInputVectors=out_folded_shape[:-1],
                    NumChannels=padded_outstream_bytes,
                    dataType="UINT8",
                    intfWidth=intfwidth,
                    streamWidth=padded_outstream_width,
                    direction="out",
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                )
                model.graph.node.append(dma_node)
            if first_node.op_type != "IODMA":
                in_shape = model.get_tensor_shape(graph_in_name)
                in_dtype = model.get_tensor_datatype(graph_in_name)
                first_node_inst = getCustomOp(first_node)
                in_folded_shape = first_node_inst.get_folded_input_shape()
                # take advantage of AXI stream width padding for DMA alignment
                # (AXI streams are always padded to 8 bits)
                # this is the width of stream output expected from the DMA
                padded_instream_width = first_node_inst.get_instream_width_padded()
                padded_instream_bytes = padded_instream_width // 8
                # determine the feasible interface width
                transfer_bits = padded_instream_width * np.prod(in_folded_shape[:-1])
                intfwidth = math.gcd(transfer_bits, self.max_intfwidth)
                assert (
                    intfwidth % 8 == 0
                ), "No feasible interface width for transfer size"
                # make new buffer
                first_node_in = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, in_shape
                )
                model.graph.value_info.append(first_node_in)
                model.set_tensor_datatype(first_node_in.name, in_dtype)
                # reroute first node input
                # FIXME: currently always using 8-bit dtypes to work around the
                # padding problems for i/o DMA
                first_node.input[0] = first_node_in.name
                dma_node = oh.make_node(
                    "IODMA",
                    [graph_in_name],
                    [first_node_in.name],
                    numInputVectors=in_folded_shape[:-1],
                    NumChannels=padded_instream_bytes,
                    dataType="UINT8",
                    intfWidth=intfwidth,
                    streamWidth=padded_instream_width,
                    direction="in",
                    domain="finn.custom_op.fpgadataflow",
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
                W = model.get_initializer(fc_w_name)
                iodma_mem = self.get_mem_init(W, pe, simd)
                model.set_initializer(fc_w_name, iodma_mem)

                fc_node_in = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, iodma_mem.shape
                )
                model.graph.value_info.append(fc_node_in)
                model.set_tensor_datatype(fc_node_in.name, w_dtype)
                model.set_initializer(fc_node_in.name, W)
                dma_node = oh.make_node(
                    "IODMA",
                    [fc_w_name],
                    [fc_node_in.name],
                    numInputVectors=[iodma_mem.shape[0]],
                    NumChannels=pe * simd,
                    dataType=str(w_dtype.name),
                    intfWidth=intfwidth,
                    streamWidth=streamWidth,
                    direction="in",
                    burstMode="wrap",
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                )
                fc_node.input[1] = fc_node_in.name
                model.graph.node.insert(0, dma_node)
            model = model.transform(SortGraph())
            return (model, True)
