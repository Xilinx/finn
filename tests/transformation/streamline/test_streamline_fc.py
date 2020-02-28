from pkgutil import get_data

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import pytest

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.util.test import get_test_model_trained
from finn.util.basic import make_build_dir

export_onnx_path = make_build_dir("test_streamline_fc_")

# activation: None or DataType
@pytest.mark.parametrize("size", ["TFC", "SFC", "LFC"])
# weight bits
@pytest.mark.parametrize("wbits", [1])
# act bits
@pytest.mark.parametrize("abits", [1, 2])
def test_streamline_fc(size, wbits, abits):
    nname = "%s_%dW%dA" % (size, wbits, abits)
    finn_onnx = export_onnx_path + "/%s.onnx" % nname
    fc = get_test_model_trained(size, wbits, abits)
    bo.export_finn_onnx(fc, (1, 1, 28, 28), finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {"global_in": nph.to_array(input_tensor)}
    expected_ctx = oxe.execute_onnx(model, input_dict, True)
    expected = expected_ctx[model.graph.output[0].name]
    model = model.transform(Streamline())
    produced_ctx = oxe.execute_onnx(model, input_dict, True)
    produced = produced_ctx[model.graph.output[0].name]
    assert np.isclose(expected, produced, atol=1e-3).all()
