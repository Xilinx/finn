import os
import csv
from PIL import Image
import numpy as np
import brevitas.onnx as bo
import torch
from finn.util.basic import make_build_dir
from finn.util.pytorch import NormalizePreProc
from finn.util.test import get_test_model_trained, resize_smaller_side, crop_center
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from finn.transformation.merge_onnx_models import MergeONNXModels
import finn.transformation.streamline.absorb as absorb
from finn.transformation.insert_topk import InsertTopK
import finn.core.onnx_exec as oxe
import finn.util.imagenet as imagenet_util
import pytest

n_images = 1


@pytest.mark.slow
def test_brevitas_validate_mobilenet():
    if "IMAGENET_VAL_PATH" not in os.environ.keys():
        pytest.skip("Can't do validation without IMAGENET_VAL_PATH")
    export_onnx_path = make_build_dir("test_brevitas_mobilenet-v1_")
    # export preprocessing
    preproc_onnx = export_onnx_path + "/quant_mobilenet_v1_4b_preproc.onnx"
    mean = [0.485, 0.456, 0.406]
    std = 0.226
    ch = 3
    preproc = NormalizePreProc(mean, std, ch)
    bo.export_finn_onnx(preproc, (1, 3, 224, 224), preproc_onnx)
    preproc_model = ModelWrapper(preproc_onnx)
    preproc_model = preproc_model.transform(InferShapes())
    preproc_model = preproc_model.transform(GiveUniqueNodeNames())
    preproc_model = preproc_model.transform(GiveUniqueParameterTensors())
    preproc_model = preproc_model.transform(GiveReadableTensorNames())
    # export the actual MobileNet-v1
    finn_onnx = export_onnx_path + "/quant_mobilenet_v1_4b.onnx"
    mobilenet = get_test_model_trained("mobilenet", 4, 4)
    bo.export_finn_onnx(mobilenet, (1, 3, 224, 224), finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(InsertTopK())
    # get initializer from Mul that will be absorbed into topk
    a0 = model.get_initializer(model.graph.node[-2].input[1])
    model = model.transform(absorb.AbsorbScalarMulIntoTopK())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveReadableTensorNames())
    model.save(export_onnx_path + "/quant_mobilenet_v1_4b_wo_preproc.onnx")
    # create merged preprocessing + MobileNet-v1 model
    model = model.transform(MergeONNXModels(preproc_model))
    model.save(export_onnx_path + "/quant_mobilenet_v1_4b.onnx")

    with open("mobilenet_validation.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "goldenID",
                "brevitasTop5",
                "brevitasTop5[%]",
                "finnTop5",
                "finnTop5[%]" "top5equal",
                "top5%equal",
            ]
        )
        csvfile.flush()
        workload = imagenet_util.get_val_images(n_images)
        for (img_path, target_id) in workload:
            # get single image as input and prepare image
            img = Image.open(img_path)
            # resize smallest side of the image to 256 pixels and resize larger side
            # with same ratio
            img = resize_smaller_side(256, img)
            # crop central 224*224 window
            img = crop_center(224, img)
            # save image as numpy array and as torch tensor to enable testing in
            # brevitas/pytorch and finn and transpose from (H, W, C) to (C, H, W)
            img_np = np.asarray(img).copy().astype(np.float32).transpose(2, 0, 1)
            img_np = img_np.reshape(1, 3, 224, 224)
            img_torch = torch.from_numpy(img_np).float()
            # do forward pass in PyTorch/Brevitas
            input_tensor = preproc.forward(img_torch)
            expected = mobilenet.forward(input_tensor).detach().numpy()
            expected_topk = expected.flatten()
            expected_top5 = np.argsort(expected_topk)[-5:]
            expected_top5 = np.flip(expected_top5)
            expected_top5_prob = []
            for index in expected_top5:
                expected_top5_prob.append(expected_topk[index])
            idict = {model.graph.input[0].name: img_np}
            odict = oxe.execute_onnx(model, idict, True)
            produced = odict[model.graph.output[0].name]
            produced_prob = odict["TopK_0_out0"] * a0
            writer.writerow(
                [
                    str(target_id),
                    str(expected_top5),
                    str(expected_top5_prob),
                    str(produced.flatten()),
                    str(produced_prob.flatten()),
                    str((produced.flatten() == expected_top5).all()),
                    str(np.isclose(produced_prob.flatten(), expected_top5_prob).all()),
                ]
            )
            csvfile.flush()
