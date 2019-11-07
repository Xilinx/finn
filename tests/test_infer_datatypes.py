import os

import brevitas.onnx as bo
import torch
from models.LFC import LFC

import finn.transformation.fold_constants as fc
import finn.transformation.general as tg
import finn.transformation.infer_datatypes as id
import finn.transformation.infer_shapes as si
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper

export_onnx_path = "test_output_lfc.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_lfc_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/LFC_1W1A/checkpoints/best.tar"
)


def test_infer_datatypes():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    checkpoint = torch.load(trained_lfc_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform_single(si.infer_shapes)
    model = model.transform_repeated(fc.fold_constants)
    model = model.transform_single(tg.give_unique_node_names)
    model = model.transform_single(tg.give_readable_tensor_names)
    model = model.transform_repeated(id.infer_datatypes)
    assert model.get_tensor_datatype("MatMul_0_out0") == DataType.INT32
    assert model.get_tensor_datatype("MatMul_1_out0") == DataType.INT32
    assert model.get_tensor_datatype("MatMul_2_out0") == DataType.INT32
    assert model.get_tensor_datatype("MatMul_3_out0") == DataType.INT32
    assert model.get_tensor_datatype("Sign_0_out0") == DataType.BIPOLAR
    assert model.get_tensor_datatype("Sign_1_out0") == DataType.BIPOLAR
    assert model.get_tensor_datatype("Sign_2_out0") == DataType.BIPOLAR
    assert model.get_tensor_datatype("Sign_3_out0") == DataType.BIPOLAR
    os.remove(export_onnx_path)
