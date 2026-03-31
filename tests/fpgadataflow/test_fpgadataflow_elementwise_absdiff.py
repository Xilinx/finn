# Copyright (c) 2022, Xilinx
# Copyright (C) 2022-2026, Advanced Micro Devices, Inc.
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

"""
Test for ElementwiseAbsDiff operation.

This tests the Sub -> Abs pattern fusion into ElementwiseAbsDiff.
"""

import pytest

import numpy as np
import onnx.parser as oprs
import qonnx.core.data_layout as dl
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import getHWCustomOp


def build_absdiff_model(shp, dt0, dt1):
    """Build a model with Sub -> Abs pattern that will be fused to AbsDiff."""
    np.random.seed(0)
    shp_str = str(shp)
    graph = """
    sub_out = Sub(in0, in1)
    out0 = Abs(sub_out)
    """

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{shp_str} in0, float{shp_str} in1) => (float{shp_str} out0)
    {{
        {graph}
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_tensor_datatype("in0", dt0)
    model.set_tensor_datatype("in1", dt1)
    model.set_tensor_layout("in0", dl.NHWC)
    model.set_tensor_layout("in1", dl.NHWC)
    model = model.transform(InferShapes())
    return model


# input datatypes
@pytest.mark.parametrize(
    "dt0,dt1",
    [
        (DataType["UINT4"], DataType["UINT8"]),
        (DataType["INT8"], DataType["INT8"]),
        (DataType["FLOAT32"], DataType["FLOAT32"]),
    ],
)
# channels
@pytest.mark.parametrize("ch", [1, 64])
# folding
@pytest.mark.parametrize("fold", [-1, 2, 1])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_elementwise_absdiff(dt0, dt1, ch, fold, exec_mode):
    """Test ElementwiseAbsDiff operation via Sub -> Abs pattern fusion."""
    if fold == -1:
        pe = 1
    else:
        pe = max(1, ch // fold)
    assert ch % pe == 0
    shp = [1, 4, 2, ch]
    model = build_absdiff_model(shp, dt0, dt1)
    in0 = gen_finn_dt_tensor(dt0, shp)
    in1 = gen_finn_dt_tensor(dt1, shp)
    idict = {"in0": in0, "in1": in1}
    y_expected = execute_onnx(model, idict)["out0"]

    # Apply transformation - should fuse Sub -> Abs into ElementwiseAbsDiff
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwiseAbsDiff"

    y_produced = execute_onnx(model, idict)["out0"]
    if dt0 in [DataType["FLOAT32"], DataType["FLOAT16"]]:
        assert np.allclose(
            y_produced, y_expected, rtol=1e-3, atol=1e-5
        ), "HW layer execution failed"
    else:
        assert (y_produced == y_expected).all(), "HW layer execution failed"

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwiseAbsDiff_hls"
    getHWCustomOp(model.graph.node[0]).set_nodeattr("PE", pe)

    model = model.transform(MinimizeAccumulatorWidth())

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    y_produced = execute_onnx(model, idict)["out0"]
    if dt0 in [DataType["FLOAT32"], DataType["FLOAT16"]]:
        assert np.allclose(y_produced, y_expected, rtol=1e-3, atol=1e-5), exec_mode + " failed"
    else:
        assert (y_produced == y_expected).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("ElementwiseAbsDiff_hls")[0]
        inst = getHWCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0
