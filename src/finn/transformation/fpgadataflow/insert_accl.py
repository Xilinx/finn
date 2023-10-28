# Copyright (c) 2020, Xilinx # All rights reserved.
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
from finn.util.visualization import showInNetron

class InsertACCL(Transformation):
    def insert_at(self, model, tensor_name, producer, consumer):
        if producer.op_type == "ACCLOut":
            assert consumer.op_type == "ACCLIn", "Expect ACCLOut to comer after in"
            return False

        producer_inst = getCustomOp(producer)
        consumer_inst = getCustomOp(consumer)

        producer_rank = producer_inst.get_nodeattr("device_id")
        consumer_rank = consumer_inst.get_nodeattr("device_id")

        # Nodes are on same device, no need to insert accl nodes
        if producer_rank == consumer_rank: return False

        tensor_shape = model.get_tensor_shape(tensor_name)
        tensor_dtype = model.get_tensor_datatype(tensor_name)

        producer_out = oh.make_tensor_value_info(
            model.make_new_valueinfo_name(), TensorProto.FLOAT, tensor_shape
        )

        model.graph.value_info.append(producer_out)
        model.set_tensor_datatype(producer_out.name, tensor_dtype)

        consumer_in = oh.make_tensor_value_info(
            model.make_new_valueinfo_name(), TensorProto.FLOAT, tensor_shape
        )

        model.graph.value_info.append(consumer_in)
        model.set_tensor_datatype(consumer_in.name, tensor_dtype)

        producer_shape = producer_inst.get_folded_output_shape()

        for idx, out in enumerate(producer.output):
            if out == tensor_name:
                producer.output[idx] = producer_out.name

        world_size = int(model.get_metadata_prop("world_size"))

        accl_out = oh.make_node(
            "ACCLOut",
            [producer_out.name],
            [tensor_name],
            numInputVectors=producer_shape[:-1],
            NumChannels=producer_shape[-1],
            dataType=str(tensor_dtype),
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            worldSize=world_size,
            otherRank=consumer_rank,
        )

        getCustomOp(accl_out).set_nodeattr("device_id", producer_rank)

        model.graph.node.insert(0, accl_out)

        consumer_shape = producer_inst.get_folded_output_shape()

        accl_in = oh.make_node(
            "ACCLIn",
            [tensor_name],
            [consumer_in.name],
            numInputVectors=consumer_shape[:-1],
            NumChannels=consumer_shape[-1],
            dataType=str(tensor_dtype),
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            worldSize=world_size,
            otherRank=producer_rank,
        )

        getCustomOp(accl_in).set_nodeattr("device_id", consumer_rank)

        model.graph.node.insert(0, accl_in)

        for idx, inp in enumerate(consumer.input):
            if inp == tensor_name:
                consumer.input[idx] = consumer_in.name

        return True

    def apply(self, model):
        potential_comm_pairs = []

        for producer in model.graph.node:
            for tensor_name in producer.output:
                consumer = model.find_consumer(tensor_name)
                if consumer is None: continue
                potential_comm_pairs.append((tensor_name, producer, consumer))

        modified = False

        for tensor_name, producer, consumer in potential_comm_pairs:
            modified |= self.insert_at(model, tensor_name, producer, consumer)

        if modified:
            model = model.transform(SortGraph())

        return (model, modified)

