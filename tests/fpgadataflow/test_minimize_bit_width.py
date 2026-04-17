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


def make_unit_test_model(
    wdt: DataType,
    idt: DataType,
    tdt: Optional[DataType] = None,
    # MVAU-specific parameters (only used if layer_type="mvau")
    layer_type: str = "vvau_mvau",
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
    :param tdt: Threshold datatype (None for noActivation=1)
    :param layer_type: Type of model to create - "vvau_mvau"
    (default, MobileNetV1-based pair) or "mvau" (single MVAU)
    :param mw: Matrix width for MVAU (only used if layer_type="mvau")
    :param mh: Matrix height for MVAU (only used if layer_type="mvau")
    :param odt: Output datatype (defaults to idt if tdt, else INT32)
    :param adt: Accumulator datatype (defaults to INT32)
    :param weights: Weight values (auto-generated if None, only for layer_type="mvau")
    :param thresholds: Threshold values (auto-generated if None and tdt is not None)
    :param add_successor: If True, adds a dummy successor node to prevent rounding
    (only for layer_type="mvau")
    :return: ModelWrapper with configured model
    """
    if layer_type == "vvau_mvau":
        # Original VVAU-MVAU pair logic (MobileNetV1-based)
        return _make_vvau_mvau_model(wdt, idt, tdt)
    elif layer_type == "mvau":
        # Single MVAU node with flexible configuration
        assert mw is not None and mh is not None, "mw and mh required for layer_type='mvau'"
        return _make_mvau_model(mw, mh, wdt, idt, tdt, odt, adt, weights, thresholds, add_successor)
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")


def _make_mvau_model(
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
    """Helper to create a single MVAU node or MVAU with successor."""
    if odt is None:
        odt = idt if tdt is not None else DataType["INT32"]
    if adt is None:
        adt = DataType["INT32"]
    if weights is None:
        weights = gen_finn_dt_tensor(wdt, (mw, mh))

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])

    output_name = "outp" if not add_successor else "hid"

    # Create main MVAU node
    node_inputs = ["inp", "weights"]
    if tdt is not None:
        node_inputs.append("thresh")

    main_node = helper.make_node(
        "MVAU",
        node_inputs,
        [output_name],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType=idt.name,
        outputDataType=odt.name,
        weightDataType=wdt.name,
        accDataType=adt.name,
        ActVal=tdt.min() if tdt is not None else 0,
        noActivation=0 if tdt is not None else 1,
        binaryXnorMode=0,
    )

    nodes = [main_node]
    initializers = [
        helper.make_tensor("weights", TensorProto.FLOAT, weights.shape, weights.flatten())
    ]

    # Add thresholds if provided
    if tdt is not None:
        if thresholds is None:
            n_steps = idt.get_num_possible_values() - 1
            thresholds = np.random.randint(tdt.min(), tdt.max() - 1, (mh, n_steps)).astype(
                np.float32
            )
            thresholds = np.sort(thresholds, axis=1)
        initializers.append(
            helper.make_tensor("thresh", TensorProto.FLOAT, thresholds.shape, thresholds.flatten())
        )

    # Add successor node if requested
    if add_successor:
        weights2 = gen_finn_dt_tensor(wdt, (mh, mh))
        successor_node = helper.make_node(
            "MVAU",
            ["hid", "weights2"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            MW=mh,
            MH=mh,
            SIMD=1,
            PE=1,
            inputDataType=odt.name,
            outputDataType=odt.name,
            weightDataType=wdt.name,
            accDataType=adt.name,
            ActVal=0,
            noActivation=1,
            binaryXnorMode=0,
        )
        nodes.append(successor_node)
        initializers.append(
            helper.make_tensor("weights2", TensorProto.FLOAT, weights2.shape, weights2.flatten())
        )

    graph = helper.make_graph(
        nodes=nodes,
        name="mvau_test",
        inputs=[inp],
        outputs=[outp],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="mvau-test")
    model = ModelWrapper(model)

    # Set datatypes
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype(output_name, odt)
    model.set_tensor_datatype("weights", wdt)

    if add_successor:
        model.set_tensor_datatype("outp", odt)
        model.set_tensor_datatype("weights2", wdt)

    if tdt is not None:
        model.set_tensor_datatype("thresh", tdt)

    return model


def _make_vvau_mvau_model(wdt: DataType, idt: DataType, tdt: Optional[DataType] = None):
    """Creates a VVAU-MVAU pair based on the first pair of MobileNetV1."""
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
@pytest.mark.parametrize("external", [True, False])
@pytest.mark.fpgadataflow
def test_minimize_weight_bit_width(wdt: DataType, rww: bool, external: bool):
    """Testing MinimizeWeightBitWidth for VVAU and MVAU.

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
    params0 = gen_finn_dt_tensor(wdt, (32, 1, 3, 3))
    params1 = gen_finn_dt_tensor(wdt, (32, 64))
    model.set_initializer("params0", params0)
    model.set_initializer("params1", params1)

    # If runtime-writeable weights, specify as a node attribute
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, (MVAU, VVAU)):
            inst.set_nodeattr("runtime_writeable_weights", int(rww))
            if external:
                inst.set_nodeattr("mem_mode", "external")

    # Apply the optimization
    model = model.transform(MinimizeWeightBitWidth())

    # Iterate through each node to make sure it functioned properly
    for node in model.graph.node:
        inst = getCustomOp(node)
        if isinstance(inst, (MVAU, VVAU)):
            cur_wdt = DataType[inst.get_nodeattr("weightDataType")]
            exp_wdt = def_wdt if (rww or external) else wdt
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
    """Testing MinimizeAccumulatorWidth for VVAU and MVAU.

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
        if isinstance(inst, (MVAU, VVAU)):
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
    """Test that minimize_accumulator_width for MVAU is independent of threshold values.
    The accumulator width should be determined solely by weights and inputs, not thresholds.
    Thresholds are handled separately by RoundAndClipThresholds transformation.

    :param wdt: (DataType) Weight datatype
    :param idt: (DataType) Input datatype
    """
    mw = 4
    mh = 8
    weights = gen_finn_dt_tensor(wdt, (mw, mh))

    # Create MVAU WITH thresholds (noActivation=0)
    n_steps = 7
    thresholds = np.random.randint(-10000, 10000, (mh, n_steps)).astype(np.float32)
    thresholds = np.sort(thresholds, axis=1)

    model_with_thresh = make_unit_test_model(
        wdt,
        idt,
        tdt=DataType["INT32"],
        layer_type="mvau",
        mw=mw,
        mh=mh,
        odt=DataType["UINT4"],
        weights=weights,
        thresholds=thresholds,
    )

    # Create MVAU WITHOUT thresholds (noActivation=1)
    # Add a dummy successor node so accumulator doesn't get rounded to multiple of 8
    model_no_thresh = make_unit_test_model(
        wdt,
        idt,
        tdt=None,
        layer_type="mvau",
        mw=mw,
        mh=mh,
        weights=weights,
        add_successor=True,
    )

    # Create a third model WITHOUT dummy successor to verify rounding behavior
    model_last = make_unit_test_model(
        wdt, idt, tdt=None, layer_type="mvau", mw=mw, mh=mh, weights=weights
    )

    # Run MinimizeAccumulatorWidth on all models
    model_with_thresh = model_with_thresh.transform(MinimizeAccumulatorWidth())
    model_no_thresh = model_no_thresh.transform(MinimizeAccumulatorWidth())
    model_last = model_last.transform(MinimizeAccumulatorWidth())

    # Get accumulator datatypes
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
    # Thresholds should NOT influence accumulator datatype
    assert adt_with_thresh == adt_no_thresh, (
        f"Accumulator width must be independent of threshold values. "
        f"With thresholds: {adt_with_thresh}, without thresholds: {adt_no_thresh}"
    )


@pytest.mark.parametrize("tdt", [DataType["INT16"], DataType["INT24"], DataType["INT32"]])
@pytest.mark.fpgadataflow
def test_minimize_accumulator_width_mvau_threshold_boundary_values(tdt: DataType):
    """Test MVAU with boundary threshold values (near INT24/INT32 limits) to ensure
    deterministic behavior and avoid float32 precision issues.
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
        wdt, idt, tdt=tdt, layer_type="mvau", mw=mw, mh=mh, odt=idt, thresholds=boundary_thresholds
    )

    # Run MinimizeAccumulatorWidth - should not be affected by large threshold values
    model = model.transform(MinimizeAccumulatorWidth())

    inst = getCustomOp(model.graph.node[0])
    result_adt = DataType[inst.get_nodeattr("accDataType")]

    assert result_adt is not None, "Accumulator data type should be set"

    # Verify accumulator datatype is reasonable (not expanded to accommodate thresholds)
    # For INT8 x INT8, accumulator should be around INT16-INT18, NOT INT24/INT32
    assert (
        result_adt.bitwidth() < 20
    ), f"Accumulator too wide ({result_adt}), likely incorrectly expanded for thresholds"

    # Now test that RoundAndClipThresholds handles boundary values correctly
    from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

    model = model.transform(RoundAndClipThresholds())

    # Verify thresholds were clipped to accumulator range
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
    1. MinimizeWeightBitWidth (first pass) - minimize MVAU/VVAU weights
    2. MinimizeAccumulatorWidth - minimize accumulators based on weights/inputs
    3. RoundAndClipThresholds - round/clip thresholds to accumulator range
    4. MinimizeWeightBitWidth (second pass) - minimize threshold datatypes

    This test verifies:
    - Thresholds are rounded up (ceil) to integers
    - Thresholds are clipped to accumulator range
    - Threshold datatypes are minimized based on rounded/clipped values
    - Final datatypes are optimal
    """
    from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

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
        layer_type="mvau",
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
