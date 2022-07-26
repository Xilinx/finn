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
import pytest

import numpy as np
import onnx.parser as oprs
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import (
    MoveLinearPastFork,
    MoveTransposePastFork,
)


@pytest.mark.streamline
def test_move_past_fork_transpose():
    shp = [1, 3, 32, 32]
    shp_str = str(shp)
    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{shp_str} in0) => (float{shp_str} out0)
    {{
        t0_out = Transpose<perm=[0,2,3,1]>(in0)
        t1_out = Transpose<perm=[0,3,1,2]>(t0_out)
        t2_out = Transpose<perm=[0,3,1,2]>(t0_out)
        out0 = Add(t1_out, t2_out)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    new_model = model.transform(MoveTransposePastFork())
    new_model = new_model.transform(GiveUniqueNodeNames())
    nodes = new_model.graph.node
    assert oxe.compare_execution(
        model, new_model, {"in0": np.random.rand(*shp).astype(np.float32)}
    )
    assert len(nodes) == 5
    assert not new_model.is_fork_node(get_by_name(nodes, "Transpose_0"))


@pytest.mark.streamline
@pytest.mark.parametrize("ch", [64, 1])
# ifmdim
@pytest.mark.parametrize("ifmdim", [-1, 7])
def test_move_past_fork_linear(ch, ifmdim):
    if ifmdim == -1:
        shp = [1, ch]
    else:
        shp = [1, ch, ifmdim, ifmdim]
    shp_str = str(shp)
    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{shp_str} in0) => (float{shp_str} out0)
    <
        float{shp_str} add0_param,
        float{shp_str} mul_shared_param,
        float{shp_str} add2_param,
        float{shp_str} mul2_param,
        float{shp_str} add3_param,
        float{shp_str} add4_param,
        float{shp_str} mul3_param,
        float{shp_str} add6_param
    >
    {{

        add0_out = Add(in0, add0_param)
        mul0_out = Mul(add0_out, mul_shared_param)
        mul1_out = Mul(add0_out, mul_shared_param)
        add1_out = Add(mul0_out, mul1_out)
        add2_out = Add(add1_out, add2_param)
        mul2_out = Mul(add2_out, mul2_param)
        add3_out = Add(mul2_out, add3_param)
        add4_out = Add(mul2_out, add4_param)
        add5_out = Add(add3_out, add4_out)
        mul3_out = Mul(add5_out, mul3_param)
        out0 = Add(mul3_out, add6_param)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())

    np.random.seed(0)
    for tensor_name in model.get_all_tensor_names():
        if tensor_name.endswith("_param"):
            pshape = model.get_tensor_shape(tensor_name)
            model.set_initializer(
                tensor_name, np.random.rand(*pshape).astype(np.float32)
            )
    model = model.transform(GiveUniqueNodeNames())
    # Transform
    new_model = model.transform(MoveLinearPastFork())
    new_model = new_model.transform(GiveUniqueNodeNames())
    inp_dict = {"top_in": np.random.rand(*shp).astype(np.float32)}
    # Test
    assert oxe.compare_execution(model, new_model, inp_dict)
    nodes = new_model.graph.node
    assert len(new_model.get_nodes_by_op_type("Add")) == 9
    assert len(new_model.get_nodes_by_op_type("Mul")) == 5
    assert not new_model.is_fork_node(get_by_name(nodes, "Add_0"))
    assert new_model.is_join_node(get_by_name(nodes, "Add_2"))
    assert not new_model.is_fork_node(get_by_name(nodes, "Mul_2"))
    assert not new_model.is_join_node(get_by_name(nodes, "Add_5"))
    assert len(new_model.graph.node) == 14
