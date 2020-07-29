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

from onnx import TensorProto, helper
import numpy as np
import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.general import GiveUniqueNodeNames
from finn.custom_op.registry import getCustomOp
from finn.util.basic import gen_finn_dt_tensor
from finn.transformation.infer_shapes import InferShapes
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer


def make_single_maxpool_modelwrapper(k, stride, pad, ifm_ch, ifm_dim, ofm_dim, idt):
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ifm_ch, ofm_dim, ofm_dim]
    )

    mp_node = helper.make_node(
        "MaxPool",
        ["inp"],
        ["outp"],
        kernel_shape=[k, k],
        pads=[pad, pad, pad, pad],
        strides=[stride, stride],
    )
    graph = helper.make_graph(
        nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="mp-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model = model.transform(InferShapes())

    return model


def make_single_quantavpool_modelwrapper(k, stride, ifm_ch, ifm_dim, ofm_dim, idt, odt):
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ifm_ch, ofm_dim, ofm_dim]
    )

    mp_node = helper.make_node(
        "QuantAvgPool2d",
        ["inp"],
        ["outp"],
        domain="finn",
        stride=stride,
        kernel=k,
        ibits=idt.bitwidth(),
        obits=odt.bitwidth(),
        signed=1 if idt.signed() else 0,
        data_layout="NCHW",
    )
    graph = helper.make_graph(
        nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="mp-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model = model.transform(InferShapes())

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# input datatype
@pytest.mark.parametrize("idt", [DataType.UINT4, DataType.INT4, DataType.INT8])
# output datatype
@pytest.mark.parametrize("odt", [DataType.UINT4, DataType.INT4])
# pool configuration:                   ( k,stride, pad, ifm_dim )
@pytest.mark.parametrize("pool_config", [(7, 7, 0, 7), (3, 2, 1, 5)])
# input channels
@pytest.mark.parametrize("ifm_ch", [1, 4])
# number of out channel computed in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# pool type
@pytest.mark.parametrize("op_type", ["QuantAvgPool2d", "MaxPool"])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.slow
@pytest.mark.vivado
def test_convert_to_hls_pool_batch(
    idt, odt, pool_config, ifm_ch, pe, op_type, exec_mode
):
    k, stride, pad, ifm_dim = pool_config

    if ifm_ch % pe != 0:
        pytest.skip("ifm_ch%pe != 0. Skipping")

    if pad != 0 and idt.signed():
        pytest.skip("No support for pal_val != 0. Skipping")

    np.random.seed(0)
    ofm_dim = int(((ifm_dim + 2 * pad - k) / stride) + 1)

    x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_dim, ifm_dim))
    # prepare input data
    input_dict = prepare_inputs(x)
    if op_type == "MaxPool":
        # if idt.signed():
        #     pytest.skip("""No support for signed input (see accu initialization
        #         in Pool_batch HLSLIB function). Skipping""")

        if idt != odt:
            pytest.skip("Skipping Maxpool with idt != odt")

        model = make_single_maxpool_modelwrapper(
            k, stride, pad, ifm_ch, ifm_dim, ofm_dim, idt
        )
    elif op_type == "QuantAvgPool2d":
        if pad != 0:
            pytest.skip("No padding support for QuantAvgPool2d. Skipping")

        if idt.signed() != odt.signed():
            pytest.skip("Skipping QuantAvgPool2d with idt.signed() != odt.signed()")
        model = make_single_quantavpool_modelwrapper(
            k, stride, ifm_ch, ifm_dim, ofm_dim, idt, odt
        )
    else:
        assert False, "{} is not a supported op_type".format(op_type)

    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    new_model = model.transform(to_hls.InferPool_Batch())
    new_model = new_model.transform(GiveUniqueNodeNames())

    if ifm_ch != pe:
        new_model = new_model.transform(to_hls.InferConvInpGen())
        # Folding
        for n in new_model.graph.node:
            if n.op_type == "ConvolutionInputGenerator":
                inst = getCustomOp(n)
                inst.set_nodeattr("SIMD", pe)
            elif n.op_type == "Pool_Batch":
                inst = getCustomOp(n)
                inst.set_nodeattr("PE", pe)

    if exec_mode == "cppsim":
        new_model = new_model.transform(SetExecMode("cppsim"))
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # execute new_model
    y_produced = oxe.execute_onnx(new_model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
    if stride <= k:
        if pad == 0 or ifm_ch == pe:
            assert len(new_model.graph.node) == 4
        else:
            assert len(new_model.graph.node) == 5
    else:
        assert len(new_model.graph.node) == 1

    if exec_mode == "rtlsim":
        node = new_model.get_nodes_by_op_type("Pool_Batch")[0]
        inst = getCustomOp(node)
        sim_cycles = inst.get_nodeattr("sim_cycles")
        exp_cycles_dict = new_model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[str(node)]
        assert np.isclose(exp_cycles, sim_cycles, atol=10)
