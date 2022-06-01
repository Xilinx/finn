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
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames

from finn.util.fpgadataflow import is_fpgadataflow_node


def _is_hook_node(node):
    if node.op_type in ["checksum"]:
        return True
    else:
        return False


def _suitable_node(node):
    if node is not None:
        if is_fpgadataflow_node(node) is True:
            if _is_hook_node(node) is False:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


class InsertHook(Transformation):
    """Inserting hook layer after each layer that has the node attribute
    'output_hook' specified"""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        list_supported_hooks = ["checksum"]
        graph = model.graph
        node_ind = -1
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if _suitable_node(n):
                for output_name in n.output:
                    consumers = model.find_consumers(output_name)
                    assert len(consumers) <= 1, (
                        n.name
                        + ": HLS node with fan-out higher than 1 cannot be stitched"
                    )
                    n0 = getCustomOp(n)
                    n0_hook = n0.get_nodeattr("output_hook")
                    if n0_hook in list_supported_hooks:
                        if n0_hook == "checksum":
                            if len(consumers) == 1:
                                if consumers[0].op_type == "checksum":
                                    continue
                            n0_normal_oshape = n0.get_normal_output_shape()
                            n0_folded_oshape = n0.get_folded_output_shape()
                            n0_odt = n0.get_output_datatype()
                            items_per_word = n0.get_nodeattr("PE")
                            words_per_frame = np.prod(n0_folded_oshape[:-1])
                            chk_otensor = oh.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                n0_normal_oshape,
                            )
                            chk_result = oh.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                [1],
                            )
                            chk_node = oh.make_node(
                                "checksum",
                                [output_name],
                                outputs=[chk_otensor.name, chk_result.name],
                                domain="finn.custom_op.fpgadataflow",
                                backend="fpgadataflow",
                                words_per_frame=words_per_frame,
                                items_per_word=items_per_word,
                                inputDataType=str(n0_odt.name),
                                folded_shape=n0_folded_oshape,
                            )
                            # insert checksum node
                            graph.node.insert(node_ind + 1, chk_node)
                            # insert newly-created tensors
                            graph.value_info.append(chk_otensor)
                            graph.value_info.append(chk_result)

                            # set chk output tensor as new input tensor of second node
                            if len(consumers) == 1:
                                consumers[0].input[0] = chk_otensor.name
                            else:
                                model.graph.output.pop()
                                model.graph.output.append(chk_otensor)
                                model = model.transform(GiveUniqueNodeNames())
                                model = model.transform(GiveReadableTensorNames())
                            graph_modified = True
                            return (model, graph_modified)

        return (model, graph_modified)
