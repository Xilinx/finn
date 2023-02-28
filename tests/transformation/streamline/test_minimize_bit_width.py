# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
from typing import Optional
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType, IntType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor

from finn.custom_op.fpgadataflow.vectorvectoractivation import VectorVectorActivation
from finn.custom_op.fpgadataflow.matrixvectoractivation import MatrixVectorActivation
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth


def make_unit_test_model(wdt: DataType, idt: DataType, tdt: Optional[DataType] = None):
    """Creates a toy finn-onnx model for unit testing. The VVAU-MVAU pair is based
    on the first pair of MobileNetV1"""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 32, 32, 288])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 32, 32, 64])
    layer1 = helper.make_node(
        "VectorVectorActivation",
        ["inp", "params0", "thresh"] if tdt is not None else ["inp", "params0"],
        ["hid"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=1,
        Channels=32,
        Dim=(32, 32),
        Kernel=(3,3),
        inputDataType=idt.name,
        outputDataType=idt.name,
        weightDataType=wdt.name,
        noActivation=tdt.min() if tdt is not None else 0,
        ActVal=0 if tdt is not None else 1,
    )
    layer2 = helper.make_node(
        "MatrixVectorActivation",
        ["hid", "params1", "thresh"] if tdt is not None else ["hid", "params1"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=32, # matrix_width (num_inputs)
        MH=64, # matrix_height (num_outputs)
        SIMD=1,
        PE=1,
        inputDataType=idt.name,
        outputDataType=idt.name,
        weightDataType=wdt.name,
        noActivation=tdt.min() if tdt is not None else 0,
        ActVal=0 if tdt is not None else 1,
        binaryXnorMode=0
    )
    graph = helper.make_graph(
        nodes=[layer1, layer2], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    model.set_tensor_datatype("hid", idt)
    model.set_tensor_datatype("params0", wdt)
    model.set_tensor_datatype("params1", wdt)
    model.set_initializer("params0",
        gen_finn_dt_tensor(wdt, (32, 1, 3, 3))
    )
    model.set_initializer("params1",
                          gen_finn_dt_tensor(wdt, (32, 64))
    )
    if tdt is not None:
        model.set_tensor_datatype("thresh", tdt)
        # model.set_initializer("thresh", thresholds)
    return model


weight_data_types = [
    DataType['INT8'],
    DataType['UINT8'],
    DataType['INT7'],
    DataType['UINT7'],
    DataType['INT3'],
    DataType['UINT3'],
    DataType["BIPOLAR"],
    DataType["TERNARY"],
]

@pytest.mark.parametrize("wdt", weight_data_types)
@pytest.mark.parametrize("rww", [True, False])
def test_minimize_weight_bit_width(wdt: DataType, rww: bool):
    """Testing MinimizeWeightBitWidth for VVAU and MVAU.
    
    :param wdt: (DataType) The data type that we are testing for the weights
    :param rww: (bool) Whether or not to use runtime-writeable weights"""

    # Create a w8a8 model
    def_wdt = DataType['UINT8'] 
    model = make_unit_test_model(def_wdt, DataType['INT8'])
    
    # Create new weights for the model based on wdt
    params0 = gen_finn_dt_tensor(wdt, (32, 1, 3, 3))
    params1 = gen_finn_dt_tensor(wdt, (32, 64))
    model.set_initializer("params0", params0)
    model.set_initializer("params1", params1)

    # If runtime-writeable weights, specify as a node attribute
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, (MatrixVectorActivation, VectorVectorActivation)):
            inst.set_nodeattr("runtime_writeable_weights", int(rww))

    # Apply the optimization
    model = model.transform(MinimizeWeightBitWidth())

    # Iterate through each node to make sure it functioned properly 
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, (MatrixVectorActivation, VectorVectorActivation)):
            cur_wdt = DataType[inst.get_nodeattr("weightDataType")]
            exp_wdt = def_wdt if rww else wdt
            assert cur_wdt.bitwidth() == exp_wdt.bitwidth(), "Mismatched data types"
