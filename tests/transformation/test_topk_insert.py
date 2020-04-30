import onnx
from finn.util.test import get_test_model_trained
import brevitas.onnx as bo
import numpy as np
import onnx.numpy_helper as nph
import torch

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.insert_topk import InsertTopK

import finn.core.onnx_exec as oxe
from pkgutil import get_data

import pytest

export_onnx_path = "test_output_lfc.onnx"


@pytest.mark.parametrize("k", [1, 5, 10])
def test_topk_insert(k):
    tfc = get_test_model_trained("TFC", 1, 1)
    bo.export_finn_onnx(tfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)

    # do transformations (no topk)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())

    # verification: generate random input, run through net, streamline,
    # run again, check that output is top-k
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_brevitas = torch.from_numpy(nph.to_array(input_tensor)).float()
    output_golden = tfc.forward(input_brevitas).detach().numpy()
    output_golden_topk = np.flip(output_golden.flatten().argsort())[:k]
    output_golden_topk = output_golden_topk.flatten()

    input_dict = {"global_in": nph.to_array(input_tensor)}

    # insert top-k
    model = model.transform(InsertTopK(k))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())

    # verify output of top-k
    output_dict_topk = oxe.execute_onnx(model, input_dict)
    output_pysim_topk = output_dict_topk[list(output_dict_topk.keys())[0]]
    output_pysim_topk = output_pysim_topk.astype(np.int).flatten()

    assert np.array_equal(output_golden_topk, output_pysim_topk)
