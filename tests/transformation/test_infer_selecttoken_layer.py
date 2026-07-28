# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferSelectTokenLayer


def make_selecttoken_gather_modelwrapper(indices):
    indices = np.asarray(indices, dtype=np.int64)
    output_shape = [1, 4] if indices.ndim == 0 else [1, len(indices), 4]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, output_shape)
    gather_node = helper.make_node("Gather", ["inp", "indices"], ["outp"], axis=1)
    graph = helper.make_graph(
        [gather_node],
        "selecttoken-gather-model",
        [inp],
        [outp],
        [numpy_helper.from_array(indices, name="indices")],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="selecttoken-gather-model"))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("outp", DataType["INT8"])
    return model


@pytest.mark.transform
def test_infer_selecttoken_layer_rejects_nonscalar_gather():
    indices = [1, 2]
    input_tensor = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    input_dict = {"inp": input_tensor}
    model = make_selecttoken_gather_modelwrapper(indices)
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    model = model.transform(InferSelectTokenLayer())
    assert model.graph.node[0].op_type == "Gather"

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
