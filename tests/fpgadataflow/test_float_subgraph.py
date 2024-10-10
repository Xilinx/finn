# Copyright (c) 2024, Advanced Micro Devices, Inc.
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

import onnx.parser as oprs
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.fpgadataflow import is_fpgadataflow_node


def create_matmul_mul_add_subgraph():
    tensors = {
        # tensor_name : [shape, dtype, needs_init]
        "in0": [(1, 4), DataType["UINT4"], False],
        "param_mm": [(4, 8), DataType["INT4"], True],
        "param_mul": [(1, 8), DataType["FLOAT32"], True],
        "param_add": [(1, 8), DataType["FLOAT32"], True],
        "out0": [(1, 8), DataType["FLOAT32"], False],
    }
    t_decl_list = [
        f"float{str(list(spec[0]))} {tname}" for tname, spec in tensors.items() if spec[-1] is True
    ]
    ishp_str = str(list(tensors["in0"][0]))
    oshp_str = str(list(tensors["out0"][0]))

    input_str = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    <
        {", ".join(t_decl_list)}
    >
    {{
        matmul0_out0 = MatMul(in0, param_mm)
        mul0_out0 = Mul(matmul0_out0, param_mul)
        out0 = Add(mul0_out0, param_add)
    }}
    """
    model = oprs.parse_model(input_str)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    for tname, spec in tensors.items():
        model.set_tensor_datatype(tname, spec[1])
        if spec[-1]:
            model.set_initializer(tname, gen_finn_dt_tensor(spec[1], spec[0]))
    model = model.transform(InferDataTypes())
    return model, tensors


# Specializes all nodes to be implemented as HLS backend
def specialize_hls(model: ModelWrapper):
    # Mark all nodes to be specialized as HLS backend implementations
    for node in model.graph.node:  # noqa: Duplicate test setup code
        # Skip non-fpgadataflow backend operators as these do not have the
        # preferred_impl_style attribute
        if is_fpgadataflow_node(node):
            # Get the CustomOp instance of the node to get access to the node
            # attributes
            inst = getCustomOp(node)
            # Note: only HLS-based layers execute C++ Simulation
            inst.set_nodeattr("preferred_impl_style", "hls")
    # Turn all HWCustomOp layers into HLS specializations
    return model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))


def test_create_matmul_mul_add_subgraph():
    model, tensors = create_matmul_mul_add_subgraph()
    inp = gen_finn_dt_tensor(tensors["in0"][1], tensors["in0"][0])
    idict = {"in0": inp}
    golden = execute_onnx(model, idict)["out0"]
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = specialize_hls(model)
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 10))  # noqa
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    produced = execute_onnx(model, idict)["out0"]
    assert (golden == produced).all()
