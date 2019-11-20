import os

import brevitas.onnx as bo
import torch
from models.LFC import LFC

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline

export_onnx_path = "test_output_lfc.onnx"
# TODO get from config instead, hardcoded to Docker path for now
trained_lfc_checkpoint = (
    "/workspace/brevitas_cnv_lfc/pretrained_models/LFC_1W1A/checkpoints/best.tar"
)


def test_convert_to_hls_layers_lfc_w1a1():
    lfc = LFC(weight_bit_width=1, act_bit_width=1, in_bit_width=1)
    checkpoint = torch.load(trained_lfc_checkpoint, map_location="cpu")
    lfc.load_state_dict(checkpoint["state_dict"])
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(Streamline())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(to_hls.InferBinaryStreamingFCLayer())
    fc0 = model.graph.node[2]
    assert fc0.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc0.input[0]) == [1, 784, 1]
    assert model.get_tensor_shape(fc0.input[1]) == [1, 784 * 1024, 1]
    assert model.get_tensor_shape(fc0.input[2]) == [1, 1024, 1]
    fc1 = model.graph.node[3]
    assert fc1.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc1.input[0]) == [1, 1024, 1]
    assert model.get_tensor_shape(fc1.input[1]) == [1, 1024 * 1024, 1]
    assert model.get_tensor_shape(fc1.input[2]) == [1, 1024, 1]
    fc2 = model.graph.node[4]
    assert fc2.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc2.input[0]) == [1, 1024, 1]
    assert model.get_tensor_shape(fc2.input[1]) == [1, 1024 * 1024, 1]
    assert model.get_tensor_shape(fc2.input[2]) == [1, 1024, 1]
    os.remove(export_onnx_path)
