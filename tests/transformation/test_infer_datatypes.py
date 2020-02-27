import os

import brevitas.onnx as bo

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.util.test import get_test_model_trained

export_onnx_path = "test_output_lfc.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_lfc_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/LFC_1W1A/checkpoints/best.tar"
)


def test_infer_datatypes():
    lfc = get_test_model_trained("LFC", 1, 1)
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("MatMul_0_out0") == DataType.INT32
    assert model.get_tensor_datatype("MatMul_1_out0") == DataType.INT32
    assert model.get_tensor_datatype("MatMul_2_out0") == DataType.INT32
    assert model.get_tensor_datatype("MatMul_3_out0") == DataType.INT32
    assert model.get_tensor_datatype("Sign_0_out0") == DataType.BIPOLAR
    assert model.get_tensor_datatype("Sign_1_out0") == DataType.BIPOLAR
    assert model.get_tensor_datatype("Sign_2_out0") == DataType.BIPOLAR
    assert model.get_tensor_datatype("Sign_3_out0") == DataType.BIPOLAR
    os.remove(export_onnx_path)
