# Copyright (c) 2021, Xilinx
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

import importlib_resources as importlib
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
from brevitas.export import export_qonnx
from onnx import TensorProto, helper
from pkgutil import get_data
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup
from tempfile import TemporaryDirectory

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.test import get_test_model_trained


def get_brev_model_and_sample_inputs(model_name, wbits, abits):
    if "FC" in model_name:
        in_shape = (1, 1, 28, 28)
        raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
        input_tensor = onnx.load_tensor_from_string(raw_i)
        input_tensor = nph.to_array(input_tensor)
        brev_model = get_test_model_trained(model_name, wbits, abits)
    elif model_name == "CNV":
        in_shape = (1, 3, 32, 32)
        ref = importlib.files("finn.qnn-data") / "cifar10/cifar10-test-data-class3.npz"
        with importlib.as_file(ref) as fn:
            input_tensor = np.load(fn)["arr_0"].astype(np.float32)
        input_tensor = input_tensor / 255
        brev_model = get_test_model_trained(model_name, wbits, abits)
    elif model_name == "mobilenet":
        in_shape = (1, 3, 224, 224)
        np.random.seed(42)
        input_tensor = np.random.normal(size=in_shape).astype(dtype=np.float32)
        brev_model = get_test_model_trained(model_name, 4, 4)
    else:
        raise RuntimeError(f"The model with the name {model_name} is not supported.")

    return brev_model, in_shape, input_tensor


def analysis_testing_for_no_quant_nodes(model):
    # Test that all Quant nodes have been converted to MultiThreshold nodes
    # or folded into tensor initializers.

    for op_type in ["BinaryQuant", "Quant", "Trunc"]:
        q_count = len(model.get_nodes_by_op_type(op_type))
        if q_count > 0:
            raise ValueError(f"There should be no {op_type} nodes left in the graph.")

    return dict()


@pytest.mark.transform
# This test currently takes about 4 min and 20 seconds
@pytest.mark.parametrize("abits", [1, 2])
@pytest.mark.parametrize("wbits", [1, 2])
@pytest.mark.parametrize("model_name", ["TFC", "SFC", "LFC", "CNV", "mobilenet"])
def test_QONNX_to_FINN(model_name, wbits, abits):
    if model_name == "mobilenet":
        pytest.xfail("MobileNet test is temporarily excluded from QONNX testing.")
    if wbits > abits:
        pytest.skip("No wbits > abits cases at the moment")
    if model_name == "LFC" and wbits == 2 and abits == 2:
        pytest.skip("No LFC-w2a2 present at the moment")
    if model_name == "mobilenet" and (wbits != 2 or abits != 2):
        pytest.skip("Mobilenet only runs at W2A2, though it's technically W4A4.")

    # Get test config and model
    ATOL = 1e-6
    brev_model, in_shape, input_tensor = get_brev_model_and_sample_inputs(model_name, wbits, abits)
    temp_dir = TemporaryDirectory()
    qonnx_base_path = temp_dir.name + "/qonnx_{}.onnx"

    # Get Brevitas output
    torch_input_tensor = torch.from_numpy(input_tensor).float()
    brev_output = brev_model.forward(torch_input_tensor).detach().numpy()

    # Get QONNX model
    _ = export_qonnx(brev_model, torch.randn(in_shape), qonnx_base_path.format("raw"))
    cleanup(qonnx_base_path.format("raw"), out_file=qonnx_base_path.format("clean"))

    # Compare output
    model = ModelWrapper(qonnx_base_path.format("clean"))
    input_dict = {model.get_first_global_in(): input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict, False)
    qonnx_export_output = output_dict[model.get_first_global_out()]
    assert np.isclose(
        brev_output, qonnx_export_output, atol=ATOL
    ).all(), "The output of the Brevitas model and the QONNX model should match."

    # Run QONNX to FINN conversion
    model = ModelWrapper(qonnx_base_path.format("clean"))
    model = model.transform(ConvertQONNXtoFINN())
    model.save(qonnx_base_path.format("whole_trafo"))

    # Compare output
    model = ModelWrapper(qonnx_base_path.format("whole_trafo"))
    input_dict = {model.get_first_global_in(): input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict, False)
    test_output = output_dict[model.get_first_global_out()]
    assert np.isclose(test_output, qonnx_export_output, atol=ATOL).all(), (
        "The output of the FINN model " "and the QONNX -> FINN converted model should match."
    )

    # Run analysis passes on the converted model
    model = ModelWrapper(qonnx_base_path.format("whole_trafo"))
    _ = model.analysis(analysis_testing_for_no_quant_nodes)

    temp_dir.cleanup()


def _make_single_quant_model(x, scale, zeropt, bitwidth, signed, narrow, rounding):
    """Minimal graph: global_in -> Quant -> global_out, params as initializers."""
    ish = list(x.shape)
    gi = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, ish)
    go = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, ish)

    def _init(name, arr):
        return nph.from_array(np.asarray(arr, dtype=np.float32), name)

    inits = [_init("scale", scale), _init("zeropt", zeropt), _init("bitwidth", bitwidth)]
    quant = helper.make_node(
        "Quant",
        ["global_in", "scale", "zeropt", "bitwidth"],
        ["global_out"],
        domain="qonnx.custom_op.general",
        signed=int(signed),
        narrow=int(narrow),
        rounding_mode=str(rounding),
    )
    graph = helper.make_graph([quant], "single_quant", [gi], [go], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.opset_import.append(helper.make_opsetid("qonnx.custom_op.general", 1))
    m = ModelWrapper(model)
    return m.transform(InferShapes()).transform(InferDataTypes())


@pytest.mark.transform
def test_QONNX_to_FINN_threshold_precision():
    """Regression test for a float32 accumulation error in MultiThreshold
    threshold derivation (QuantActBaseHandler._calculate_thresholds).

    Each threshold used to be accumulated as ``min_threshold + step * t`` in
    float32. Because ``min_threshold`` grows to ~-step*(num_thresholds/2), that
    accumulation carried ~1 ulp of error at the base's magnitude, which could
    place a boundary slightly below an input that Quant legitimately rounds down.
    MultiThreshold's exact ``>=`` compare then stepped up where Quant did not,
    giving a one-step mismatch.

    The input below sits just under the ideal 1->2 boundary (1.5*scale), inside
    the sliver the float32 error opened up: x/scale = 1.4999988 rounds to level 1
    under Quant, so the converted graph must agree.
    """
    x = np.array([[[0.39990925788879395]]], dtype=np.float32)
    scale = np.array([0.2666063904762268], dtype=np.float32)
    zeropt = np.array([0.0], dtype=np.float32)
    bitwidth = np.array([6.0], dtype=np.float32)

    ref_m = _make_single_quant_model(x, scale, zeropt, bitwidth, 1, 0, "ROUND")
    conv_m = ref_m.transform(ConvertQONNXtoFINN())

    ref = oxe.execute_onnx(ref_m, {"global_in": x.copy()})["global_out"]
    conv = oxe.execute_onnx(conv_m, {"global_in": x.copy()})["global_out"]

    assert np.allclose(ref, conv, atol=1e-6), (
        "Quant and ConvertQONNXtoFINN disagree on a boundary input: "
        f"Quant={ref.ravel()[0]!r} vs MultiThreshold={conv.ravel()[0]!r} "
        "(threshold derivation precision regression)"
    )
