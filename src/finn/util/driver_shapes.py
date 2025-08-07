# Copyright (C) 2020 Xilinx, Inc.
# Copyright (C) 2025, Advanced Micro Devices, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor
from typing import Dict

from finn.kernels.kernel_registry import gkr
from .kernel_util import get_node_attr
from .data_packing import finnpy_to_packed_bytearray


def get_driver_shapes(model: ModelWrapper) -> Dict:
    idt = []
    idma_names = []
    ishape_normal = []
    ishape_folded = []
    ishape_packed = []
    for idma_ind, graph_in in enumerate(model.graph.input):
        i_tensor_name = graph_in.name
        # get inp tensor properties
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        i_tensor_shape_normal = tuple(model.get_tensor_shape(i_tensor_name))
        # go down into dataflow partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        i_consumer = model.find_consumer(i_tensor_name)
        assert (
            i_consumer.op_type == "StreamingDataflowPartition"
        ), """
            Ensure CreateDataflowPartition called before driver creation."""
        first_df_model = ModelWrapper(getCustomOp(i_consumer).get_nodeattr("model"))
        assert (
            first_df_model.graph.node[0].op_type == "IODMA"
        ), "First partition must hold input IODMA"
        successors = model.find_direct_successors(i_consumer)
        successor_input_num = list(successors[0].input).index(i_consumer.output[0])
        successor_sdp = getCustomOp(successors[0])
        successor_df_model = ModelWrapper(successor_sdp.get_nodeattr("model"))
        first_node = successor_df_model.find_consumer(
            successor_df_model.graph.input[successor_input_num].name
        )
        first_kernel = gkr.kernel(first_node.op_type, get_node_attr(first_node, successor_df_model))
        i_tensor_shape_folded = tuple(first_kernel.get_folded_input_shape())
        # generate dummy folded i/o tensors and their packed versions
        i_tensor_dummy_folded = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape_folded)
        i_tensor_dummy_packed = finnpy_to_packed_bytearray(i_tensor_dummy_folded, i_tensor_dt)
        i_tensor_shape_packed = i_tensor_dummy_packed.shape
        # append all input tensor info to relevant lists
        idt.append("DataType['%s']" % i_tensor_dt.name)
        ishape_normal.append(i_tensor_shape_normal)
        ishape_folded.append(i_tensor_shape_folded)
        ishape_packed.append(i_tensor_shape_packed)
        idma_names.append(getCustomOp(i_consumer).get_nodeattr("instance_name"))

    odt = []
    odma_names = []
    oshape_normal = []
    oshape_folded = []
    oshape_packed = []
    for odma_ind, graph_out in enumerate(model.graph.output):
        o_tensor_name = graph_out.name
        # get inp tensor properties
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        o_tensor_shape_normal = tuple(model.get_tensor_shape(o_tensor_name))
        # go down into IODMA partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        o_producer = model.find_producer(o_tensor_name)
        assert (
            o_producer.op_type == "StreamingDataflowPartition"
        ), """
            Ensure CreateDataflowPartition called before driver creation."""
        df_model = ModelWrapper(getCustomOp(o_producer).get_nodeattr("model"))
        assert df_model.graph.node[-1].op_type == "IODMA", "Partition must hold output IODMA"
        predecessors = model.find_direct_predecessors(o_producer)
        predecessor_output_num = list(predecessors[0].output).index(o_producer.input[0])
        predecessor_sdp = getCustomOp(predecessors[0])
        predecessor_df_model = ModelWrapper(predecessor_sdp.get_nodeattr("model"))
        last_node = predecessor_df_model.find_producer(
            predecessor_df_model.graph.output[predecessor_output_num].name
        )
        last_kernel = gkr.kernel(last_node.op_type, get_node_attr(last_node, successor_df_model))
        o_tensor_shape_folded = tuple(last_kernel.get_folded_output_shape())
        o_tensor_dummy_folded = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape_folded)
        o_tensor_dummy_packed = finnpy_to_packed_bytearray(o_tensor_dummy_folded, o_tensor_dt)
        o_tensor_shape_packed = o_tensor_dummy_packed.shape
        # append all output tensor info to relevant lists
        odt.append("DataType['%s']" % o_tensor_dt.name)
        oshape_normal.append(o_tensor_shape_normal)
        oshape_folded.append(o_tensor_shape_folded)
        oshape_packed.append(o_tensor_shape_packed)
        odma_names.append(getCustomOp(o_producer).get_nodeattr("instance_name"))

    return {
        "idt": idt,
        "idma_names": idma_names,
        "ishape_normal": ishape_normal,
        "ishape_folded": ishape_folded,
        "ishape_packed": ishape_packed,
        "odt": odt,
        "odma_names": odma_names,
        "oshape_normal": oshape_normal,
        "oshape_folded": oshape_folded,
        "oshape_packed": oshape_packed,
    }
