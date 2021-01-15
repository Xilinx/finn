import os
import csv
import numpy as np
import brevitas.onnx as bo
import torch
from finn.util.basic import make_build_dir
from finn.util.pytorch import NormalizePreProc
from finn.util.test import get_test_model_trained
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import RemoveStaticGraphInputs
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
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# normalization (preprocessing) settings for MobileNet-v1 w4a4
mean = [0.485, 0.456, 0.406]
std = 0.226
ch = 3


def test_brevitas_mobilenet_preproc():
    if "IMAGENET_VAL_PATH" not in os.environ.keys():
        pytest.skip("Can't do validation without IMAGENET_VAL_PATH")
    n_images = 1000
    # Brevitas-style: use torchvision pipeline
    std_arr = [std, std, std]
    normalize = transforms.Normalize(mean=mean, std=std_arr)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            os.environ["IMAGENET_VAL_PATH"] + "/../",
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    # FINN-style: load_resize_crop then normalization as PyTorch graph
    preproc = NormalizePreProc(mean, std, ch)
    finn_loader = imagenet_util.get_val_images(n_images)
    val_loader = iter(val_loader)
    for i in range(n_images):
        (img_path, finn_target) = next(finn_loader)
        finn_img = imagenet_util.load_resize_crop(img_path)
        finn_img = preproc.forward(torch.from_numpy(finn_img).float())
        (pyt_img, pyt_target) = next(val_loader)
        assert finn_img.shape == pyt_img.shape
        assert (finn_img == pyt_img).all()


@pytest.mark.slow
# marked as XFAIL until Brevitas export issues are resolved:
# https://github.com/Xilinx/brevitas/issues/173
@pytest.mark.xfail
def test_brevitas_compare_exported_mobilenet():
    if "IMAGENET_VAL_PATH" not in os.environ.keys():
        pytest.skip("Can't do validation without IMAGENET_VAL_PATH")
    n_images = 10
    debug_mode = False
    export_onnx_path = make_build_dir("test_brevitas_mobilenet-v1_")
    # export preprocessing
    preproc_onnx = export_onnx_path + "/quant_mobilenet_v1_4b_preproc.onnx"
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
    if debug_mode:
        dbg_hook = bo.enable_debug(mobilenet)
    bo.export_finn_onnx(mobilenet, (1, 3, 224, 224), finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(InsertTopK())
    # get initializer from Mul that will be absorbed into topk

    a0 = model.get_initializer(model.get_nodes_by_op_type("Mul")[-1].input[1])
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
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

    with open(
        export_onnx_path + "/mobilenet_validation.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "goldenID",
                "brevitasTop5",
                "brevitasTop5[%]",
                "finnTop5",
                "finnTop5[%]",
                "top5equal",
                "top5%equal",
            ]
        )
        csvfile.flush()
        workload = imagenet_util.get_val_images(n_images, interleave_classes=True)
        all_inds_ok = True
        all_probs_ok = True
        for (img_path, target_id) in workload:
            img_np = imagenet_util.load_resize_crop(img_path)
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
            odict = oxe.execute_onnx(model, idict, return_full_exec_context=True)
            produced = odict[model.graph.output[0].name]
            produced_prob = odict["TopK_0_out0"] * a0
            inds_ok = (produced.flatten() == expected_top5).all()
            probs_ok = np.isclose(produced_prob.flatten(), expected_top5_prob).all()
            all_inds_ok = all_inds_ok and inds_ok
            all_probs_ok = all_probs_ok and probs_ok
            writer.writerow(
                [
                    str(target_id),
                    str(expected_top5),
                    str(expected_top5_prob),
                    str(produced.flatten()),
                    str(produced_prob.flatten()),
                    str(inds_ok),
                    str(probs_ok),
                ]
            )
            csvfile.flush()
            if ((not inds_ok) or (not probs_ok)) and debug_mode:
                print("Results differ for %s" % img_path)
                # check all tensors at debug markers
                names_brevitas = set(dbg_hook.values.keys())
                names_finn = set(odict.keys())
                names_common = names_brevitas.intersection(names_finn)
                for dbg_name in names_common:
                    if not np.isclose(
                        dbg_hook.values[dbg_name].detach().numpy(),
                        odict[dbg_name],
                        atol=1e-3,
                    ).all():
                        print("Tensor %s differs between Brevitas and FINN" % dbg_name)
        assert all_inds_ok and all_probs_ok
