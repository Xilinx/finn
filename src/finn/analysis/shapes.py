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

from finn.custom_op.registry import getCustomOp


def io_shapes(model):
    """Gathers shapes of graph IOs.

    Returns {node_name: shapes and direction}."""

    ret = {}
    for inp in model.graph.input:
        i_tensor_name = inp.name
        i_tensor_shape_normal = tuple(model.get_tensor_shape(inp))
        # extract HLSCustomOp instances to get folded i/o shapes
        first_node = getCustomOp(model.find_consumer(i_tensor_name))
        i_tensor_shape_folded = tuple(first_node.get_folded_input_shape())
        ret[i_tensor_name] = {
            "direction": "in",
            "shape_normal": i_tensor_shape_normal,
            "shape_folded": i_tensor_shape_folded,
        }

    for outp in model.graph.output:
        o_tensor_name = outp.name
        o_tensor_shape_normal = tuple(model.get_tensor_shape(outp))
        # extract HLSCustomOp instances to get folded i/o shapes
        last_node = getCustomOp(model.find_consumer(o_tensor_name))
        o_tensor_shape_folded = tuple(last_node.get_folded_input_shape())
        ret[o_tensor_name] = {
            "direction": "out",
            "shape_normal": o_tensor_shape_normal,
            "shape_folded": o_tensor_shape_folded,
        }

    return ret


def weight_shapes(model):
    """Gathers shapes/precision/values of StreamingFCLayer weights.
    Returns {node_name: DT, PE, SIMD, values (un-folded)}."""
    ret = {}
    for node in model.graph.node:
        if node.op_type != "StreamingFCLayer_Batch":
            continue
        node_inst = getCustomOp(node)
        pe = node_inst.get_nodeattr("PE")
        simd = node_inst.get_nodeattr("SIMD")
        wmem = node_inst.calc_wmem()
        dt = node_inst.get_weight_datatype()
        values = model.get_initializer(node.input[1])
        ret[node.name] = {
            "PE": pe,
            "SIMD": simd,
            "WMEM": wmem,
            "DataType": dt,
            "Values": values,
        }

    return ret
