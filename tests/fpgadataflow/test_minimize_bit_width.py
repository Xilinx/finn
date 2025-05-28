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

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import BipolarType, DataType, IntType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor, roundup_to_integer_multiple
from typing import Optional, Union

from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)


def make_unit_test_model(wdt: DataType, idt: DataType, tdt: Optional[DataType] = None):
    """Creates a toy finn-onnx model for unit testing. The VVAU-MVAU pair is based
    on the first pair of MobileNetV1"""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 32, 32, 288])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 32, 32, 64])
    layer1 = helper.make_node(
        "VVAU",
        ["inp", "params0", "thresh0"] if tdt is not None else ["inp", "params0"],
        ["hid"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=1,
        Channels=32,
        Dim=(32, 32),
        Kernel=(3, 3),
        inputDataType=idt.name,
        outputDataType=idt.name,
        weightDataType=wdt.name,
        ActVal=tdt.min() if tdt is not None else 0,
        noActivation=0 if tdt is not None else 1,
    )
    layer2 = helper.make_node(
        "MVAU",
        ["hid", "params1", "thresh1"] if tdt is not None else ["hid", "params1"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=32,  # matrix_width (num_inputs)
        MH=64,  # matrix_height (num_outputs)
        SIMD=1,
        PE=1,
        inputDataType=idt.name,
        outputDataType=idt.name,
        weightDataType=wdt.name,
        ActVal=tdt.min() if tdt is not None else 0,
        noActivation=0 if tdt is not None else 1,
        binaryXnorMode=0,
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
    model.set_initializer("params0", gen_finn_dt_tensor(wdt, (32, 1, 3, 3)))
    model.set_initializer("params1", gen_finn_dt_tensor(wdt, (32, 64)))
    # if the threshold data type is specified, then we need to generate
    # some dummy threshold values
    if tdt is not None:
        model.set_tensor_datatype("thresh0", tdt)
        model.set_tensor_datatype("thresh1", tdt)
        # Create threshold tensors
        n_steps: int = idt.get_num_possible_values() - 1
        thresholds: Optional[np.ndarray] = np.random.randint(
            tdt.min(), tdt.max() - 1, (32, n_steps)
        ).astype(
            np.float32
        )  # generate thresholds for the activations
        thresholds = np.sort(thresholds, axis=1)  # provide non-decreasing thresholds
        model.set_initializer("thresh0", thresholds)
        thresholds: Optional[np.ndarray] = np.random.randint(
            tdt.min(), tdt.max() - 1, (64, n_steps)
        ).astype(
            np.float32
        )  # generate thresholds for the activations
        thresholds = np.sort(thresholds, axis=1)  # provide non-decreasing thresholds
        model.set_initializer("thresh1", thresholds)
    return model


weight_data_types = [
    DataType["INT8"],
    DataType["UINT8"],
    DataType["INT7"],
    DataType["UINT7"],
    DataType["INT3"],
    DataType["UINT3"],
    # DataType["BIPOLAR"], # TODO - add support for bipolar weights
    DataType["TERNARY"],
]


input_data_types = [
    DataType["INT8"],
    DataType["UINT8"],
    DataType["INT3"],
    DataType["UINT3"],
    DataType["BIPOLAR"],
    DataType["TERNARY"],
]


@pytest.mark.parametrize("wdt", weight_data_types)
@pytest.mark.parametrize("rww", [True, False])
@pytest.mark.fpgadataflow
def test_minimize_weight_bit_width(wdt: DataType, rww: bool):
    """Testing MinimizeWeightBitWidth for VVAU and MVAU.

    :param wdt: (DataType) The data type that we are testing for the weights
    :param rww: (bool) Whether or not to use runtime-writeable weights"""
    if isinstance(wdt, BipolarType):
        # current MinimizeWeightBitWidth sets {-1,1} to INT2, need to check
        # for 0 in weights to minimize weight bit width to bipolar
        pytest.skip("Not well-supported for this optimization")

    # Create a w8a8 model
    def_wdt = DataType["UINT8"]
    model = make_unit_test_model(def_wdt, DataType["INT8"])

    # Create new weights for the model based on wdt
    params0 = gen_finn_dt_tensor(wdt, (32, 1, 3, 3))
    params1 = gen_finn_dt_tensor(wdt, (32, 64))
    model.set_initializer("params0", params0)
    model.set_initializer("params1", params1)

    # If runtime-writeable weights, specify as a node attribute
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, (MVAU, VVAU)):
            inst.set_nodeattr("runtime_writeable_weights", int(rww))

    # Apply the optimization
    model = model.transform(MinimizeWeightBitWidth())

    # Iterate through each node to make sure it functioned properly
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, (MVAU, VVAU)):
            cur_wdt = DataType[inst.get_nodeattr("weightDataType")]
            exp_wdt = def_wdt if rww else wdt
            assert cur_wdt.bitwidth() == exp_wdt.bitwidth(), "Mismatched data types"


def calculate_accumulator_bit_width(
    inst: Union[MVAU, VVAU], model: ModelWrapper
) -> Union[DataType, IntType]:
    """Calculate the accumulator bit width using the closed-form expressions
    derived in `Quantized Neural Networks for Low-Precision Accumulation
    with Guaranteed Overflow Avoidance` (2023) by I.Colbert, A. Pappalardo,
    J. Petri-Koenig

    :param inst: (HLSCustomOp) The instance of the MVAU or VVAU
    :param model: (ModelWrapper) The instance of the whole model
    """

    def phi(x: float) -> float:
        return np.log2(1 + pow(2, -x))

    weights = model.get_initializer(inst.onnx_node.input[1])
    # since in the calculation the values of the weight matrix are used,
    # for the bipolar case they need to be converted to bipolar
    if inst.get_nodeattr("binaryXnorMode"):
        weights = 2 * weights - 1
    # modify the weights based on if the node is a VVAU or MVAU
    if isinstance(inst, MVAU):
        K = inst.get_nodeattr("MW")  # matrix_width = num_inputs
    elif isinstance(inst, VVAU):
        k_h, k_w = inst.get_nodeattr("Kernel")
        K = k_h * k_w  # size of kernels = num_inputs
        fm = inst.get_nodeattr("Channels")
        # put weights into the shape expected by calculate_matvec_accumulator_range
        weights = weights.reshape(fm, k_h * k_w).transpose()
    else:
        raise Exception("Considering only MVAU and VVAU currently")
    # collect attributes used to determine the accumulator bit width bound
    wdt = inst.get_input_datatype(1)
    idt = inst.get_input_datatype(0)
    rww = inst.get_nodeattr("runtime_writeable_weights")
    # if runtime-writeable weights, then use the lower bound on the accumulator bit
    # width as determined by the input and weight data types and size of dot product
    if rww:
        alpha = np.log2(K) + idt.bitwidth() + wdt.bitwidth() - 1.0 - float(idt.signed())
        P = np.ceil(alpha + phi(alpha) + 1.0)
    # if not runtime-writable weights, then use the tighter bound on the accumulator
    # bit width as determined by the weight values themselves
    else:
        beta = np.log2(abs(weights).sum(axis=0).max()) + idt.bitwidth() - float(idt.signed())
        P = np.ceil(beta + phi(beta) + 1.0)
    # if the node is the last in the graph, then round up to the nearest 8 bits
    if model.find_direct_successors(inst.onnx_node) is None:
        P = roundup_to_integer_multiple(P, 8)
    return DataType[f"INT{int(P)}"]


thresh_data_types = [
    None,
    DataType["INT32"],
    DataType["INT24"],
    DataType["INT16"],
]

# Removing unsigned data types fro weights
weight_data_types = [
    DataType["INT8"],
    DataType["INT7"],
    DataType["INT3"],
    # DataType["BIPOLAR"], # TODO - add support for bipolar weights
    DataType["TERNARY"],
]


@pytest.mark.parametrize("wdt", weight_data_types)
@pytest.mark.parametrize("idt", input_data_types)
@pytest.mark.parametrize("tdt", thresh_data_types)
@pytest.mark.parametrize("rww", [True, False])
@pytest.mark.fpgadataflow
def test_minimize_accumulator_width(wdt: DataType, idt: DataType, tdt: DataType, rww: bool):
    """Testing MinimizeAccumulatorWidth for VVAU and MVAU.

    :param wdt: (DataType) The data type that we are testing for the weights
    :param idt: (DataType) The data type that we are testing for the activations
    :param tdt: (DataType) The data type that we are testing for the thresholds
    :param rww: (bool) Whether or not to use runtime-writeable weights"""
    if (not wdt.signed()) or isinstance(wdt, BipolarType):
        pytest.skip("Closed-form accumulator calculation is designed to consider signed weights")

    # Create uniform-precision model
    model = make_unit_test_model(wdt, idt, tdt)
    def_adt = DataType["INT32"]

    # If runtime-writeable weights, specify as a node attribute
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, (MVAU, VVAU)):
            inst.set_nodeattr("runtime_writeable_weights", int(rww))
            cur_adt = DataType[inst.get_nodeattr("accDataType")]
            assert cur_adt.bitwidth() == def_adt.bitwidth(), "Default data type is incorrect"

    # Apply the optimization
    model = model.transform(MinimizeAccumulatorWidth())

    # Iterate through each node to make sure it functioned properly
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, (MVAU, VVAU)):
            cur_adt = DataType[inst.get_nodeattr("accDataType")]
            cur_odt = DataType[inst.get_nodeattr("outputDataType")]
            # Calculating expected accumulator bit width using a closed-form expression
            # that is a slight over-approximation of the lower bound. The accumulator
            # bit width minimization logic in the MVAU and VVAU is exact and should be
            # less than or equal to this calculation
            exp_adt = calculate_accumulator_bit_width(inst, model)
            assert cur_adt.bitwidth() <= exp_adt.bitwidth(), "Mismatched accumulation data types"

            # if there is no activation, outputDataType = accDataType and if it is the last node
            # it needs to be divisible by 8
            if inst.get_nodeattr("noActivation"):
                assert (
                    cur_adt.bitwidth() == cur_odt.bitwidth()
                ), "outputDataType and accDataType should be equal"
                if model.find_direct_successors(inst.onnx_node) is None:
                    assert (
                        cur_adt.bitwidth() % 8
                    ) == 0, "bit width of last node needs to be divisible by 8"
