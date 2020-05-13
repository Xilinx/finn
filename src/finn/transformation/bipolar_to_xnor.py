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

from finn.core.datatype import DataType
from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.util.basic import get_by_name
from finn.custom_op.registry import getCustomOp


class ConvertBipolarMatMulToXnorPopcount(Transformation):
    """Convert MatMul nodes with all-bipolar inputs to XnorPopcountMatMul
    and associated result correction."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                i_bp = model.get_tensor_datatype(mm_input) == DataType.BIPOLAR
                w_bp = model.get_tensor_datatype(mm_weight) == DataType.BIPOLAR
                if i_bp and w_bp:
                    # find producing threshold node and adjust output to binary
                    def find_prod_mt(x):
                        is_mt = x.op_type == "MultiThreshold"
                        is_bp = False
                        if is_mt:
                            dt = get_by_name(x.attribute, "out_dtype").s
                            is_bp = dt.decode("utf-8") == "BIPOLAR"
                        return is_mt and is_bp

                    mt_chain = model.find_upstream(mm_input, find_prod_mt)
                    if len(mt_chain) == 0:
                        raise Exception(
                            """Could not find upstream bipolar
                                            MultiThreshold"""
                        )
                    graph_modified = True
                    mt = mt_chain[-1]
                    mt_inst = getCustomOp(mt)
                    # ensure old scale/bias were correct for BIPOLAR
                    scale_ok = mt_inst.get_nodeattr("out_scale") == 2.0
                    bias_ok = mt_inst.get_nodeattr("out_bias") == -1.0
                    assert (
                        scale_ok and bias_ok
                    ), """Unexpected scale/bias
                    attributes for BIPOLAR MultiThreshold node."""
                    # start conversion, set MT output to binary
                    # (this is what XnorPopcountMatMul expects)
                    mt_inst.set_nodeattr("out_dtype", "BINARY")
                    mt_inst.set_nodeattr("out_scale", 1.0)
                    mt_inst.set_nodeattr("out_bias", 0.0)
                    model.set_tensor_datatype(mm_input, DataType.BINARY)
                    # change node type and domain
                    n.op_type = "XnorPopcountMatMul"
                    n.domain = "finn"
                    # convert weights into binary (-1,+1) -> (0,1)
                    Wbin = (model.get_initializer(mm_weight) + 1) / 2
                    # extract vector length (common matrix dim)
                    K = Wbin.shape[0]
                    model.set_initializer(mm_weight, Wbin)
                    model.set_tensor_datatype(mm_weight, DataType.BINARY)
                    # make new output node with correct shape
                    mm_out_shape = model.get_tensor_shape(mm_output)
                    xnorpcout = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, mm_out_shape
                    )
                    n.output[0] = xnorpcout.name
                    model.set_tensor_datatype(xnorpcout.name, DataType.UINT32)
                    # add mul-add nodes to produce correct dot product result
                    # need to derive P-N from P and K = P+N
                    # so we need 2*P-K
                    A = np.asarray([2.0], dtype=np.float32)
                    B = np.asarray([-K], dtype=np.float32)
                    # create value_info and initializers for Mul and Add constants
                    mul_const = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, A.shape
                    )
                    graph.value_info.append(mul_const)
                    model.set_initializer(mul_const.name, A)
                    mul_output = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, mm_out_shape
                    )
                    graph.value_info.append(mul_output)
                    add_const = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, B.shape
                    )
                    graph.value_info.append(add_const)
                    model.set_initializer(add_const.name, B)
                    # create Mul and Add nodes to replace the batchnorm
                    mul_node = oh.make_node(
                        "Mul", [xnorpcout.name, mul_const.name], [mul_output.name]
                    )
                    add_node = oh.make_node(
                        "Add", [mul_output.name, add_const.name], [mm_output]
                    )
                    # insert where the batchnorm is to preserve topological ordering
                    graph.node.insert(node_ind, mul_node)
                    graph.node.insert(node_ind + 1, add_node)
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
