import os
from pkgutil import get_data

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch
from models.LFC import LFC

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.streamingfclayer_batch import StreamingFCLayer_Batch
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.fpgadataflow.codegen_npysim import CodeGen_npysim
from finn.transformation.fpgadataflow.compile import Compile
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

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
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(to_hls.InferBinaryStreamingFCLayer())
    fc0 = model.graph.node[2]
    assert fc0.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc0.input[0]) == [1, 784]
    assert model.get_tensor_shape(fc0.input[1]) == [784, 1024]
    assert model.get_tensor_shape(fc0.input[2]) == [1024, 1]
    fc1 = model.graph.node[3]
    assert fc1.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc1.input[0]) == [1, 1024]
    assert model.get_tensor_shape(fc1.input[1]) == [1024, 1024]
    assert model.get_tensor_shape(fc1.input[2]) == [1024, 1]
    fc2 = model.graph.node[4]
    assert fc2.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc2.input[0]) == [1, 1024]
    assert model.get_tensor_shape(fc2.input[1]) == [1024, 1024]
    assert model.get_tensor_shape(fc2.input[2]) == [1024, 1]
    fc3 = model.graph.node[5]
    assert fc3.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc3.input[0]) == [1, 1024]
    assert model.get_tensor_shape(fc3.input[1]) == [1024, 10]
    os.remove(export_onnx_path)

    fc0w = StreamingFCLayer_Batch(fc0)
    fc0w.set_nodeattr("SIMD", 784)
    fc0w.set_nodeattr("PE", 32)

    fc1w = StreamingFCLayer_Batch(fc1)
    fc1w.set_nodeattr("SIMD", 1024)
    fc1w.set_nodeattr("PE", 32)

    fc2w = StreamingFCLayer_Batch(fc2)
    fc2w.set_nodeattr("SIMD", 1024)
    fc2w.set_nodeattr("PE", 32)

    fc3w = StreamingFCLayer_Batch(fc3)
    fc3w.set_nodeattr("SIMD", 1024)
    fc3w.set_nodeattr("PE", 10)

    model = model.transform(CodeGen_npysim())
    model = model.transform(Compile())

    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {"global_in": nph.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # run using PyTorch/Brevitas
    input_tensor = torch.from_numpy(nph.to_array(input_tensor)).float()
    assert input_tensor.shape == (1, 1, 28, 28)
    # do forward pass in PyTorch/Brevitas
    expected = lfc.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
