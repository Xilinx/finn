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

from finn.custom_op.fpgadataflow.dotprod import DotProd
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds


def make_unit_test_model(
    wdt: DataType,
    idt: DataType,
    tdt: Optional[DataType] = None,
    # DotProd-specific parameters (only used if layer_type="dotprod")
    layer_type: str = "depthwise_fc",
    mw: Optional[int] = None,
    mh: Optional[int] = None,
    odt: Optional[DataType] = None,
    adt: Optional[DataType] = None,
    weights: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    add_successor: bool = False,
):
    """Creates a toy finn-onnx model for unit testing.

    :param wdt: Weight datatype
    :param idt: Input datatype
    :param tdt: Threshold datatype (None for no thresholds)
    :param layer_type: Type of model to create - "depthwise_fc"
    (default, MobileNetV1-based pair) or "dotprod" (single DotProd)
    :param mw: Matrix width for DotProd (only used if layer_type="dotprod")
    :param mh: Matrix height for DotProd (only used if layer_type="dotprod")
    :param odt: Output datatype (defaults to idt if tdt, else INT32)
    :param adt: Accumulator datatype (defaults to INT32)
    :param weights: Weight values (auto-generated if None, only for layer_type="dotprod")
    :param thresholds: Threshold values (auto-generated if None and tdt is not None)
    :param add_successor: If True, adds a dummy successor node to prevent rounding
    (only for layer_type="dotprod")
    :return: ModelWrapper with configured model
    """
    if layer_type in ("vvau_mvau", "depthwise_fc"):
        # DotProd pair: depthwise (depthwise=1) + FC (depthwise=0), MobileNetV1-based
        return _make_depthwise_fc_model(wdt, idt, tdt)
    elif layer_type in ("mvau", "dotprod"):
        # Single DotProd node with flexible configuration
        assert mw is not None and mh is not None, "mw and mh required for layer_type='dotprod'"
        return _make_dotprod_model(mw, mh, wdt, idt, tdt, odt, adt, weights, thresholds, add_successor)
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")


def _make_dotprod_model(
    mw: int,
    mh: int,
    wdt: DataType,
    idt: DataType,
    tdt: Optional[DataType],
    odt: Optional[DataType],
    adt: Optional[DataType],
    weights: Optional[np.ndarray],
    thresholds: Optional[np.ndarray],
    add_successor: bool,
):
    """Helper to create a single DotProd node, optionally followed by MultiThreshold."""
    if odt is None:
        odt = idt if tdt is not None else DataType["INT32"]
    if adt is None:
        adt = DataType["INT32"]
    if weights is None:
        weights = gen_finn_dt_tensor(wdt, (mw, mh))

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])

    # DotProd always outputs the accumulator; if thresholds are needed,
    # a downstream MultiThreshold handles the quantization.
    if tdt is not None:
        dotprod_out = "acc0"
    elif add_successor:
        dotprod_out = "hid"
    else:
        dotprod_out = "outp"

    # Create DotProd node (always a pure accumulator, no thresholds)
    main_node = helper.make_node(
        "DotProd",
        ["inp", "weights"],
        [dotprod_out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        IFMChannels=mw,
        OFMChannels=mh,
        IFMDim=[1, 1],
        OFMDim=[1, 1],
        KernelSize=[1, 1],
        Stride=[1, 1],
        Dilation=[1, 1],
        Padding=[0, 0, 0, 0],
        depthwise=0,
        SIMD=1,
        PE=1,
        inputDataType=idt.name,
        outputDataType=adt.name,
        weightDataType=wdt.name,
        accDataType=adt.name,
        mem_mode="internal_decoupled",
    )

    nodes = [main_node]
    initializers = [
        helper.make_tensor("weights", TensorProto.FLOAT, weights.shape, weights.flatten())
    ]

    # Add MultiThreshold if thresholds are needed
    if tdt is not None:
        if thresholds is None:
            n_steps = idt.get_num_possible_values() - 1
            thresholds = np.random.randint(tdt.min(), tdt.max() - 1, (mh, n_steps)).astype(
                np.float32
            )
            thresholds = np.sort(thresholds, axis=1)

        thresh_out = "outp" if not add_successor else "hid"
        thresh_node = helper.make_node(
            "MultiThreshold",
            [dotprod_out, "thresh"],
            [thresh_out],
            domain="qonnx.custom_op.general",
            out_dtype=odt.name,
        )
        nodes.append(thresh_node)
        initializers.append(
            helper.make_tensor("thresh", TensorProto.FLOAT, thresholds.shape, thresholds.flatten())
        )

    # Add successor DotProd node if requested
    if add_successor:
        weights2 = gen_finn_dt_tensor(wdt, (mh, mh))
        successor_node = helper.make_node(
            "DotProd",
            ["hid", "weights2"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            IFMChannels=mh,
            OFMChannels=mh,
            IFMDim=[1, 1],
            OFMDim=[1, 1],
            KernelSize=[1, 1],
            Stride=[1, 1],
            Dilation=[1, 1],
            Padding=[0, 0, 0, 0],
            depthwise=0,
            SIMD=1,
            PE=1,
            inputDataType=odt.name,
            outputDataType=adt.name,
            weightDataType=wdt.name,
            accDataType=adt.name,
            mem_mode="internal_decoupled",
        )
        nodes.append(successor_node)
        initializers.append(
            helper.make_tensor("weights2", TensorProto.FLOAT, weights2.shape, weights2.flatten())
        )

    graph = helper.make_graph(
        nodes=nodes,
        name="dotprod_test",
        inputs=[inp],
        outputs=[outp],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="dotprod-test")
    model = ModelWrapper(model)

    # Set datatypes
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("weights", wdt)

    if tdt is not None:
        model.set_tensor_datatype(dotprod_out, adt)
        model.set_tensor_datatype("thresh", tdt)
        thresh_out = "outp" if not add_successor else "hid"
        model.set_tensor_datatype(thresh_out, odt)
    elif add_successor:
        model.set_tensor_datatype("hid", adt)
    else:
        model.set_tensor_datatype("outp", adt)

    if add_successor:
        model.set_tensor_datatype("outp", adt)
        model.set_tensor_datatype("weights2", wdt)

    return model


def _make_depthwise_fc_model(wdt: DataType, idt: DataType, tdt: Optional[DataType] = None):
    """Creates a depthwise DotProd + FC DotProd pair based on the first pair of MobileNetV1.

    When tdt is not None, each DotProd is followed by a MultiThreshold node.
    """
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 32, 32, 288])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 32, 32, 64])

    nodes = []
    initializers = []

    # Depthwise DotProd: depthwise=1
    dp0_out = "acc0" if tdt is not None else "hid"
    layer1 = helper.make_node(
        "DotProd",
        ["inp", "params0"],
        [dp0_out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=1,
        SIMD=1,
        IFMChannels=32,
        OFMChannels=32,
        IFMDim=[32, 32],
        OFMDim=[32, 32],
        KernelSize=[3, 3],
        Stride=[1, 1],
        Dilation=[1, 1],
        Padding=[0, 0, 0, 0],
        depthwise=1,
        inputDataType=idt.name,
        outputDataType=DataType["INT32"].name,
        weightDataType=wdt.name,
        accDataType=DataType["INT32"].name,
        mem_mode="internal_decoupled",
    )
    nodes.append(layer1)

    # If thresholds, add MultiThreshold after depthwise DotProd
    if tdt is not None:
        mt0 = helper.make_node(
            "MultiThreshold",
            [dp0_out, "thresh0"],
            ["hid"],
            domain="qonnx.custom_op.general",
            out_dtype=idt.name,
        )
        nodes.append(mt0)

    # FC DotProd: depthwise=0
    dp1_out = "acc1" if tdt is not None else "outp"
    layer2 = helper.make_node(
        "DotProd",
        ["hid", "params1"],
        [dp1_out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        IFMChannels=32,
        OFMChannels=64,
        IFMDim=[1, 1],
        OFMDim=[1, 1],
        KernelSize=[1, 1],
        Stride=[1, 1],
        Dilation=[1, 1],
        Padding=[0, 0, 0, 0],
        depthwise=0,
        SIMD=1,
        PE=1,
        inputDataType=idt.name,
        outputDataType=DataType["INT32"].name,
        weightDataType=wdt.name,
        accDataType=DataType["INT32"].name,
        mem_mode="internal_decoupled",
    )
    nodes.append(layer2)

    # If thresholds, add MultiThreshold after FC DotProd
    if tdt is not None:
        mt1 = helper.make_node(
            "MultiThreshold",
            [dp1_out, "thresh1"],
            ["outp"],
            domain="qonnx.custom_op.general",
            out_dtype=idt.name,
        )
        nodes.append(mt1)

    graph = helper.make_graph(
        nodes=nodes, name="dotprod_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="dotprod-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    model.set_tensor_datatype("hid", idt)
    model.set_tensor_datatype("params0", wdt)
    model.set_tensor_datatype("params1", wdt)
    # Weight shape: (K_H*K_W, Channels) = (9, 32) for depthwise DotProd
    model.set_initializer("params0", gen_finn_dt_tensor(wdt, (9, 32)))
    model.set_initializer("params1", gen_finn_dt_tensor(wdt, (32, 64)))

    if tdt is not None:
        model.set_tensor_datatype(dp0_out, DataType["INT32"])
        model.set_tensor_datatype(dp1_out, DataType["INT32"])
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
@pytest.mark.parametrize("external", [True, False])
@pytest.mark.fpgadataflow
def test_minimize_weight_bit_width(wdt: DataType, rww: bool, external: bool):
    """Testing MinimizeWeightBitWidth for DotProd nodes.

    :param wdt: (DataType) The data type that we are testing for the weights
    :param rww: (bool) Whether or not to use runtime-writeable weights
    :param_external: (bool) Whether or not the mem_mode is external."""
    if isinstance(wdt, BipolarType):
        # current MinimizeWeightBitWidth sets {-1,1} to INT2, need to check
        # for 0 in weights to minimize weight bit width to bipolar
        pytest.skip("Not well-supported for this optimization")

    # Create a w8a8 model
    def_wdt = DataType["UINT8"]
    model = make_unit_test_model(def_wdt, DataType["INT8"])

    # Create new weights for the model based on wdt
    # Weight shape (K_H*K_W, Channels) = (9, 32) for depthwise DotProd
    params0 = gen_finn_dt_tensor(wdt, (9, 32))
    params1 = gen_finn_dt_tensor(wdt, (32, 64))
    model.set_initializer("params0", params0)
    model.set_initializer("params1", params1)

    # If runtime-writeable weights, specify as a node attribute
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, DotProd):
            inst.set_nodeattr("runtime_writeable_weights", int(rww))
            if external:
                inst.set_nodeattr("mem_mode", "external")

    # Apply the optimization
    model = model.transform(MinimizeWeightBitWidth())

    # Iterate through each node to make sure it functioned properly
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, DotProd):
            cur_wdt = DataType[inst.get_nodeattr("weightDataType")]
            exp_wdt = def_wdt if (rww or external) else wdt
            assert cur_wdt.bitwidth() == exp_wdt.bitwidth(), "Mismatched data types"


def calculate_accumulator_bit_width(
    inst: DotProd, model: ModelWrapper
) -> Union[DataType, IntType]:
    """Calculate the accumulator bit width using the closed-form expressions
    derived in `Quantized Neural Networks for Low-Precision Accumulation
    with Guaranteed Overflow Avoidance` (2023) by I.Colbert, A. Pappalardo,
    J. Petri-Koenig

    :param inst: (DotProd) The instance of the DotProd node
    :param model: (ModelWrapper) The instance of the whole model
    """

    def phi(x: float) -> float:
        return np.log2(1 + pow(2, -x))

    weights = model.get_initializer(inst.onnx_node.input[1])
    # MW is a derived property that works for both depthwise and FC DotProd
    K = inst.MW
    # Reshape weights to (MW, MH) for the accumulator range calculation
    weights = weights.reshape(inst.MW, inst.MH)
    # collect attributes used to determine the accumulator bit width bound
    wdt = inst.get_input_datatype(1)
    idt = inst.get_input_datatype(0)
    rww = inst.get_nodeattr("runtime_writeable_weights")
    external = inst.get_nodeattr("mem_mode") == "external"
    # if runtime-writeable weights, then use the lower bound on the accumulator bit
    # width as determined by the input and weight data types and size of dot product
    if rww or external:
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
@pytest.mark.parametrize("external", [True, False])
@pytest.mark.fpgadataflow
def test_minimize_accumulator_width(
    wdt: DataType, idt: DataType, tdt: DataType, rww: bool, external: bool
):
    """Testing MinimizeAccumulatorWidth for DotProd nodes.

    :param wdt: (DataType) The data type that we are testing for the weights
    :param idt: (DataType) The data type that we are testing for the activations
    :param tdt: (DataType) The data type that we are testing for the thresholds
    :param rww: (bool) Whether or not to use runtime-writeable weights
    :param_external: (bool) Whether or not the mem_mode is external."""
    if (not wdt.signed()) or isinstance(wdt, BipolarType):
        pytest.skip("Closed-form accumulator calculation is designed to consider signed weights")

    # Create uniform-precision model
    model = make_unit_test_model(wdt, idt, tdt)
    def_adt = DataType["INT32"]

    # If runtime-writeable weights, specify as a node attribute
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, DotProd):
            inst.set_nodeattr("runtime_writeable_weights", int(rww))
            if external:
                inst.set_nodeattr("mem_mode", "external")
            cur_adt = DataType[inst.get_nodeattr("accDataType")]
            assert cur_adt.bitwidth() == def_adt.bitwidth(), "Default data type is incorrect"

    # Apply the optimization
    model = model.transform(MinimizeAccumulatorWidth())

    # Iterate through each node to make sure it functioned properly
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, DotProd):
            cur_adt = DataType[inst.get_nodeattr("accDataType")]
            cur_odt = DataType[inst.get_nodeattr("outputDataType")]
            # Calculating expected accumulator bit width using a closed-form expression
            # that is a slight over-approximation of the lower bound. The accumulator
            # bit width minimization logic in DotProd is exact and should be
            # less than or equal to this calculation
            exp_adt = calculate_accumulator_bit_width(inst, model)
            assert cur_adt.bitwidth() <= exp_adt.bitwidth(), "Mismatched accumulation data types"

            # DotProd is always a pure accumulator: outputDataType = accDataType
            # If the DotProd is the last fpgadataflow node (no DotProd successor),
            # it needs to be divisible by 8
            assert (
                cur_adt.bitwidth() == cur_odt.bitwidth()
            ), "outputDataType and accDataType should be equal for DotProd"
            if model.find_direct_successors(inst.onnx_node) is None:
                assert (
                    cur_adt.bitwidth() % 8
                ) == 0, "bit width of last node needs to be divisible by 8"


@pytest.mark.parametrize("tdt", [DataType["INT16"], DataType["INT24"], DataType["INT32"]])
@pytest.mark.fpgadataflow
def test_minimize_weight_bit_width_threshold_boundary_values(tdt: DataType):
    """Test minimize_weight_bit_width with boundary threshold values for Thresholding layers.
    This tests that threshold datatypes can be minimized based on actual threshold values."""

    idt = DataType["INT15"]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 1, 1, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 1, 1, 4])

    boundary_thresholds = np.array(
        [
            [tdt.max() - 1, tdt.max() - 2, tdt.max() - 3],
            [tdt.min(), tdt.min() + 1, tdt.min() + 2],
            [tdt.max() - 7, tdt.max() - 6, tdt.max() - 5],
            [tdt.min() + 7, tdt.min() + 6, tdt.min() + 5],
        ],
        dtype=np.float32,
    )

    node = helper.make_node(
        "Thresholding",
        ["inp", "thresh"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=4,
        PE=1,
        inputDataType=idt.name,
        weightDataType=tdt.name,
        outputDataType="UINT2",
        numInputVectors=[1, 1, 1],
        numSteps=3,
    )

    graph = helper.make_graph(
        nodes=[node],
        name="threshold_test",
        inputs=[inp],
        outputs=[outp],
        initializer=[
            helper.make_tensor(
                "thresh",
                TensorProto.FLOAT,
                boundary_thresholds.shape,
                boundary_thresholds.flatten(),
            )
        ],
    )

    model = helper.make_model(graph, producer_name="threshold-boundary-test")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", DataType["UINT2"])
    model.set_tensor_datatype("thresh", tdt)

    inst = getCustomOp(model.graph.node[0])
    result_dt = inst.minimize_weight_bit_width(model)

    assert result_dt is not None

    threshold_tensor = inst.get_hw_compatible_threshold_tensor(boundary_thresholds)
    assert np.vectorize(result_dt.allowed)(threshold_tensor).all()

    # Result should be able to represent all threshold values
    assert result_dt.bitwidth() >= tdt.bitwidth()


@pytest.mark.parametrize("wdt", [DataType["INT4"], DataType["INT8"]])
@pytest.mark.parametrize("idt", [DataType["INT8"], DataType["UINT8"]])
@pytest.mark.fpgadataflow
def test_minimize_accumulator_width_independent_of_thresholds(wdt: DataType, idt: DataType):
    """Test that minimize_accumulator_width for DotProd is independent of threshold values.
    The accumulator width should be determined solely by weights and inputs, not thresholds.
    Thresholds are handled by a separate downstream MultiThreshold node.

    :param wdt: (DataType) Weight datatype
    :param idt: (DataType) Input datatype
    """
    mw = 4
    mh = 8
    weights = gen_finn_dt_tensor(wdt, (mw, mh))

    # Create DotProd WITH downstream MultiThreshold
    n_steps = 7
    thresholds = np.random.randint(-10000, 10000, (mh, n_steps)).astype(np.float32)
    thresholds = np.sort(thresholds, axis=1)

    model_with_thresh = make_unit_test_model(
        wdt,
        idt,
        tdt=DataType["INT32"],
        layer_type="dotprod",
        mw=mw,
        mh=mh,
        odt=DataType["UINT4"],
        weights=weights,
        thresholds=thresholds,
    )

    # Create DotProd WITHOUT thresholds
    # Add a dummy successor node so accumulator doesn't get rounded to multiple of 8
    model_no_thresh = make_unit_test_model(
        wdt,
        idt,
        tdt=None,
        layer_type="dotprod",
        mw=mw,
        mh=mh,
        weights=weights,
        add_successor=True,
    )

    # Create a third model WITHOUT dummy successor to verify rounding behavior
    model_last = make_unit_test_model(
        wdt, idt, tdt=None, layer_type="dotprod", mw=mw, mh=mh, weights=weights
    )

    # Run MinimizeAccumulatorWidth on all models
    model_with_thresh = model_with_thresh.transform(MinimizeAccumulatorWidth())
    model_no_thresh = model_no_thresh.transform(MinimizeAccumulatorWidth())
    model_last = model_last.transform(MinimizeAccumulatorWidth())

    # Get accumulator datatypes — DotProd is always node[0]
    inst_with_thresh = getCustomOp(model_with_thresh.graph.node[0])
    inst_no_thresh = getCustomOp(model_no_thresh.graph.node[0])
    inst_last = getCustomOp(model_last.graph.node[0])

    adt_with_thresh = DataType[inst_with_thresh.get_nodeattr("accDataType")]
    adt_no_thresh = DataType[inst_no_thresh.get_nodeattr("accDataType")]
    adt_last = DataType[inst_last.get_nodeattr("accDataType")]

    # Verify that last node gets rounded to multiple of 8
    assert (
        adt_last.bitwidth() % 8 == 0
    ), f"Last node should have accumulator rounded to multiple of 8, got {adt_last}"

    # Critical assertion: accumulator widths should be IDENTICAL
    # Downstream MultiThreshold should NOT influence DotProd accumulator datatype
    assert adt_with_thresh == adt_no_thresh, (
        f"Accumulator width must be independent of threshold values. "
        f"With thresholds: {adt_with_thresh}, without thresholds: {adt_no_thresh}"
    )


@pytest.mark.parametrize("tdt", [DataType["INT16"], DataType["INT24"], DataType["INT32"]])
@pytest.mark.fpgadataflow
def test_minimize_accumulator_width_dotprod_threshold_boundary_values(tdt: DataType):
    """Test DotProd + MultiThreshold with boundary threshold values (near INT24/INT32 limits)
    to ensure deterministic behavior and avoid float32 precision issues.
    This test verifies the full pipeline handles large threshold values correctly.

    :param tdt: (DataType) The data type for the thresholds (INT16/INT24/INT32)
    """
    wdt = DataType["INT8"]
    idt = DataType["INT8"]
    mw = 4
    mh = 8

    n_steps = idt.get_num_possible_values() - 1
    boundary_thresholds = np.zeros((mh, n_steps), dtype=np.float32)

    # Mix boundary values - test near min/max of large datatypes
    for i in range(mh):
        if i % 3 == 0:
            # Near maximum - critical for float32 precision testing
            boundary_thresholds[i] = np.linspace(tdt.max() - n_steps, tdt.max() - 1, n_steps)
        elif i % 3 == 1:
            # Near minimum
            boundary_thresholds[i] = np.linspace(tdt.min(), tdt.min() + n_steps - 1, n_steps)
        else:
            # Mid-range
            mid_point = (tdt.max() + tdt.min()) // 2
            boundary_thresholds[i] = np.linspace(
                mid_point - n_steps // 2, mid_point + n_steps // 2, n_steps
            )

    boundary_thresholds = np.sort(boundary_thresholds, axis=1)

    model = make_unit_test_model(
        wdt, idt, tdt=tdt, layer_type="dotprod", mw=mw, mh=mh, odt=idt,
        thresholds=boundary_thresholds,
    )

    # Run MinimizeAccumulatorWidth - should not be affected by large threshold values
    model = model.transform(MinimizeAccumulatorWidth())

    # DotProd is always node[0]
    inst = getCustomOp(model.graph.node[0])
    result_adt = DataType[inst.get_nodeattr("accDataType")]

    assert result_adt is not None, "Accumulator data type should be set"

    # Verify accumulator datatype is reasonable (not expanded to accommodate thresholds)
    # For INT8 x INT8, accumulator should be around INT16-INT18, NOT INT24/INT32
    assert (
        result_adt.bitwidth() < 20
    ), f"Accumulator too wide ({result_adt}), likely incorrectly expanded for thresholds"

    # RoundAndClipThresholds handles MultiThreshold nodes (node[1] is the MultiThreshold)
    model = model.transform(RoundAndClipThresholds())

    # Verify thresholds were clipped to accumulator range
    # The MultiThreshold input is the accumulator output, so thresholds are clipped
    # to the accumulator's data type range
    final_thresholds = model.get_initializer("thresh")
    assert final_thresholds.min() >= result_adt.min(), "Thresholds not clipped to accumulator min"
    assert (
        final_thresholds.max() <= result_adt.max() + 1
    ), "Thresholds not clipped to accumulator max"

    # Verify all thresholds are integers (rounded up)
    assert all(
        x.is_integer() for x in final_thresholds.flatten()
    ), "Thresholds not rounded to integers"

    # Verify threshold datatype was set correctly (one bit wider than accumulator)
    final_thresh_dt = model.get_tensor_datatype("thresh")
    if not result_adt.signed():
        expected_thresh_dt = DataType.get_smallest_possible(result_adt.max() + 1)
    else:
        expected_thresh_dt = DataType.get_smallest_possible(-(result_adt.max() + 1) - 1)

    assert (
        final_thresh_dt == expected_thresh_dt
    ), f"Threshold datatype {final_thresh_dt} should be {expected_thresh_dt}"


@pytest.mark.fpgadataflow
def test_full_bit_width_optimization_pipeline():
    """Test the full bit width optimization pipeline as it runs in step_minimize_bit_width.

    Pipeline:
    1. MinimizeWeightBitWidth (first pass) - minimize DotProd weights
    2. MinimizeAccumulatorWidth - minimize accumulators based on weights/inputs
    3. RoundAndClipThresholds - round/clip thresholds to accumulator range
    4. MinimizeWeightBitWidth (second pass) - minimize threshold datatypes

    This test verifies:
    - Thresholds are rounded up (ceil) to integers
    - Thresholds are clipped to accumulator range
    - Threshold datatypes are minimized based on rounded/clipped values
    - Final datatypes are optimal
    """
    wdt = DataType["INT8"]
    idt = DataType["INT8"]
    mw = 8
    mh = 16

    # Generate INT8 weights (values fit in INT8) but we'll set datatype to INT32
    weights = gen_finn_dt_tensor(wdt, (mw, mh))

    # Create thresholds with non-integer values that will need rounding
    # Start with wide range including values outside accumulator range
    thresholds = np.array(
        [
            [0.5, 10.7, 20.3, 30.9, 40.1, 50.6, 60.2],
            [-5.3, 5.8, 15.2, 25.7, 35.4, 45.9, 55.1],
            [2.1, 12.4, 22.8, 32.5, 42.3, 52.7, 62.9],
            [1.9, 11.2, 21.6, 31.4, 41.8, 51.3, 61.5],
            [3.7, 13.1, 23.5, 33.2, 43.6, 53.8, 63.4],
            [-2.4, 7.9, 17.3, 27.8, 37.2, 47.5, 57.6],
            [4.2, 14.6, 24.1, 34.7, 44.9, 54.2, 64.8],
            [0.8, 10.3, 20.9, 30.4, 40.7, 50.1, 60.5],
            [-1.6, 8.5, 18.7, 28.2, 38.9, 48.4, 58.3],
            [5.4, 15.8, 25.3, 35.6, 45.2, 55.7, 65.1],
            [2.7, 12.9, 22.4, 32.8, 42.5, 52.3, 62.6],
            [-3.1, 6.2, 16.8, 26.3, 36.7, 46.1, 56.4],
            [4.9, 14.3, 24.7, 34.2, 44.6, 54.9, 64.3],
            [1.3, 11.7, 21.2, 31.9, 41.4, 51.8, 61.7],
            [3.5, 13.6, 23.9, 33.3, 43.1, 53.4, 63.8],
            [-4.8, 5.1, 15.4, 25.9, 35.8, 45.3, 55.9],
        ],
        dtype=np.float32,
    )

    model = make_unit_test_model(
        DataType["INT32"],  # Wide initial weight datatype (but actual values are INT8)
        idt,
        tdt=DataType["INT32"],  # Wide initial threshold datatype
        layer_type="dotprod",
        mw=mw,
        mh=mh,
        odt=DataType["UINT3"],
        adt=DataType["INT32"],  # Wide initial accumulator datatype
        weights=weights,  # Pass INT8 weights explicitly
        thresholds=thresholds,
    )

    # Store original threshold values
    original_thresholds = model.get_initializer("thresh").copy()

    # Step 1: MinimizeWeightBitWidth (first pass)
    model = model.transform(MinimizeWeightBitWidth())

    inst = getCustomOp(model.graph.node[0])
    wdt_after_first = DataType[inst.get_nodeattr("weightDataType")]

    # Weight datatype should be minimized
    assert (
        wdt_after_first.bitwidth() < DataType["INT32"].bitwidth()
    ), "Weight datatype should be minimized in first pass"
    assert wdt_after_first == wdt, f"Expected {wdt}, got {wdt_after_first}"

    # Step 2: MinimizeAccumulatorWidth
    model = model.transform(MinimizeAccumulatorWidth())

    inst = getCustomOp(model.graph.node[0])  # Refresh inst after transformation
    acc_dt = DataType[inst.get_nodeattr("accDataType")]

    # Accumulator should be minimized (not INT32)
    assert (
        acc_dt.bitwidth() < DataType["INT32"].bitwidth()
    ), f"Accumulator datatype should be minimized, got {acc_dt}"

    # Thresholds should NOT have been modified yet
    current_thresholds = model.get_initializer("thresh")
    assert np.allclose(
        current_thresholds, original_thresholds
    ), "Thresholds should not be modified by MinimizeAccumulatorWidth"

    # Step 3: RoundAndClipThresholds
    model = model.transform(RoundAndClipThresholds())

    rounded_thresholds = model.get_initializer("thresh")

    # Verify thresholds are rounded up (ceil)
    assert all(
        x.is_integer() for x in rounded_thresholds.flatten()
    ), "All thresholds should be integers after rounding"

    # Verify rounding was up (ceil), not nearest
    for i in range(len(original_thresholds.flatten())):
        orig = original_thresholds.flatten()[i]
        rounded = rounded_thresholds.flatten()[i]
        if not orig.is_integer():
            assert rounded == np.ceil(orig), f"Expected ceil({orig})={np.ceil(orig)}, got {rounded}"

    # Verify thresholds are clipped to accumulator range
    assert (
        rounded_thresholds.min() >= acc_dt.min()
    ), f"Min threshold {rounded_thresholds.min()} below accumulator min {acc_dt.min()}"
    assert (
        rounded_thresholds.max() <= acc_dt.max() + 1
    ), f"Max threshold {rounded_thresholds.max()} above accumulator max+1 {acc_dt.max() + 1}"

    # Verify threshold datatype is set (one bit wider than accumulator)
    thresh_dt_after_round = model.get_tensor_datatype("thresh")
    expected_thresh_dt = DataType.get_smallest_possible(-(acc_dt.max() + 1) - 1)
    assert (
        thresh_dt_after_round == expected_thresh_dt
    ), f"Expected threshold datatype {expected_thresh_dt}, got {thresh_dt_after_round}"

    # Step 4: MinimizeWeightBitWidth (second pass) - minimize threshold datatypes
    model = model.transform(MinimizeWeightBitWidth())

    final_thresh_dt = model.get_tensor_datatype("thresh")

    # Threshold datatype might be further minimized based on actual rounded values
    assert (
        final_thresh_dt.bitwidth() <= thresh_dt_after_round.bitwidth()
    ), "Second MinimizeWeightBitWidth pass should not increase threshold datatype"

    # Verify thresholds can be represented in final datatype
    final_thresholds = model.get_initializer("thresh")
    assert (
        final_thresholds.min() >= final_thresh_dt.min()
    ), f"Min threshold {final_thresholds.min()} cannot be represented in {final_thresh_dt}"
    assert (
        final_thresholds.max() <= final_thresh_dt.max()
    ), f"Max threshold {final_thresholds.max()} cannot be represented in {final_thresh_dt}"

    # Overall verification: all datatypes should be optimized
    final_wdt = DataType[inst.get_nodeattr("weightDataType")]
    final_acc_dt = DataType[inst.get_nodeattr("accDataType")]

    assert final_wdt.bitwidth() <= DataType["INT32"].bitwidth(), "Weight datatype optimized"
    assert final_acc_dt.bitwidth() <= DataType["INT32"].bitwidth(), "Accumulator datatype optimized"
    assert (
        final_thresh_dt.bitwidth() <= DataType["INT32"].bitwidth()
    ), "Threshold datatype optimized"


# --- Tests for Pool, GlobalAccPool, Lookup minimize methods ---


def make_pool_model(idt: DataType, k: int, function: str):
    """Create a simple Pool node for testing."""
    channels, odim = 4, 4
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, odim, odim, k * k * channels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim, odim, channels])

    node = helper.make_node(
        "Pool",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        Channels=channels,
        PE=1,
        KernelSize=[k, k],
        Function=function,
        OutImgDims=[odim, odim],
        InputDataType=idt.name,
        OutputDataType=idt.name,
        AccumBits=32,
        BatchSize=1,
    )

    graph = helper.make_graph(nodes=[node], name="pool_test", inputs=[inp], outputs=[outp])
    model = helper.make_model(graph, producer_name="pool-test")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    return model


def make_globalaccpool_model(idt: DataType, h: int, w: int):
    """Create a simple GlobalAccPool node for testing."""
    channels = 4
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, h, w, channels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 1, 1, channels])

    node = helper.make_node(
        "GlobalAccPool",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=channels,
        PE=1,
        inputDataType=idt.name,
        numInputVectors=[1, h, w],
    )

    graph = helper.make_graph(nodes=[node], name="globalaccpool_test", inputs=[inp], outputs=[outp])
    model = helper.make_model(graph, producer_name="globalaccpool-test")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    return model


def make_lookup_model(edt: DataType, mem_mode: str = "internal_embedded"):
    """Create a simple Lookup node for testing."""
    num_embeddings, embedding_dim = 16, 8
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4, embedding_dim])

    node = helper.make_node(
        "Lookup",
        ["inp", "embeddings"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumEmbeddings=num_embeddings,
        EmbeddingDim=embedding_dim,
        EmbeddingType=edt.name,
        InputType="UINT4",
        InputShape=[1, 4],
        mem_mode=mem_mode,
    )

    embeddings = gen_finn_dt_tensor(edt, (num_embeddings, embedding_dim))

    graph = helper.make_graph(
        nodes=[node],
        name="lookup_test",
        inputs=[inp],
        outputs=[outp],
        initializer=[
            helper.make_tensor(
                "embeddings", TensorProto.FLOAT, embeddings.shape, embeddings.flatten()
            )
        ],
    )
    model = helper.make_model(graph, producer_name="lookup-test")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", DataType["UINT4"])
    model.set_tensor_datatype("embeddings", edt)
    model.set_tensor_datatype("outp", edt)
    model.set_initializer("embeddings", embeddings)
    return model


@pytest.mark.parametrize(
    "idt", [DataType["INT8"], DataType["UINT8"], DataType["INT4"], DataType["UINT4"]]
)
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("function", ["QuantAvgPool", "MaxPool"])
@pytest.mark.fpgadataflow
def test_pool_minimize_accumulator_width(idt: DataType, k: int, function: str):
    """Test minimize_accumulator_width for Pool (QuantAvgPool and MaxPool)."""
    model = make_pool_model(idt, k=k, function=function)
    inst = getCustomOp(model.graph.node[0])
    initial_accum_bits = inst.get_nodeattr("AccumBits")

    result_dt = inst.minimize_accumulator_width(model)

    if function == "MaxPool":
        # MaxPool has no accumulator - should be no-op
        assert result_dt is None
        assert inst.get_nodeattr("AccumBits") == initial_accum_bits
    else:
        # QuantAvgPool should minimize AccumBits
        assert inst.get_nodeattr("AccumBits") < 32
        # Verify accumulator can hold worst-case sum
        kernel_elements = k * k
        if idt.signed():
            worst_min = kernel_elements * idt.min()
            worst_max = kernel_elements * idt.max()
            assert result_dt.min() <= worst_min and result_dt.max() >= worst_max
        else:
            assert result_dt.max() >= kernel_elements * idt.max()


@pytest.mark.parametrize(
    "idt", [DataType["INT8"], DataType["UINT8"], DataType["INT4"], DataType["UINT4"]]
)
@pytest.mark.parametrize("h,w", [(2, 2), (4, 4), (8, 8), (4, 8)])
@pytest.mark.fpgadataflow
def test_globalaccpool_minimize_accumulator_width(idt: DataType, h: int, w: int):
    """Test minimize_accumulator_width for GlobalAccPool."""
    model = make_globalaccpool_model(idt, h=h, w=w)
    inst = getCustomOp(model.graph.node[0])

    result_dt = inst.minimize_accumulator_width(model)

    # Verify output tensor datatype was set
    assert model.get_tensor_datatype("outp") == result_dt

    # Verify datatype can hold worst-case accumulation
    npixels = h * w
    if idt.signed():
        assert result_dt.min() <= npixels * idt.min()
        assert result_dt.max() >= npixels * idt.max()
    else:
        assert result_dt.max() >= npixels * idt.max()


@pytest.mark.parametrize(
    "edt", [DataType["INT8"], DataType["UINT8"], DataType["INT4"], DataType["INT16"]]
)
@pytest.mark.parametrize("mem_mode", ["internal_embedded", "external"])
@pytest.mark.parametrize("datatype_only", [True, False])
@pytest.mark.fpgadataflow
def test_lookup_minimize_weight_bit_width(edt: DataType, mem_mode: str, datatype_only: bool):
    """Test minimize_weight_bit_width for Lookup with various configurations."""
    model = make_lookup_model(edt, mem_mode=mem_mode)
    inst = getCustomOp(model.graph.node[0])

    result_dt = inst.minimize_weight_bit_width(model, datatype_only=datatype_only)

    if mem_mode == "external" or datatype_only:
        # Should return original datatype without modification
        assert result_dt == edt
    else:
        # Should minimize based on actual embedding values
        new_edt = DataType[inst.get_nodeattr("EmbeddingType")]
        assert new_edt == result_dt
        # Verify embeddings fit in new datatype
        embeddings = model.get_initializer("embeddings")
        assert embeddings.min() >= result_dt.min()
        assert embeddings.max() <= result_dt.max()


@pytest.mark.fpgadataflow
def test_lookup_minimize_weight_bit_width_narrows_wide_datatype():
    """Test that Lookup narrows INT32 to smaller type when values are small."""
    model = make_lookup_model(DataType["INT32"])
    inst = getCustomOp(model.graph.node[0])

    # Replace embeddings with small values that fit in INT4
    small_embeddings = np.random.randint(-8, 7, (16, 8)).astype(np.float32)
    model.set_initializer("embeddings", small_embeddings)

    result_dt = inst.minimize_weight_bit_width(model)

    assert (
        result_dt.bitwidth() <= 8
    ), f"Expected INT8 or smaller for values in [-8,7], got {result_dt}"
