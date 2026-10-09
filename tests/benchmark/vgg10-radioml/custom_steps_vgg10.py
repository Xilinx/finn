# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors

import finn.transformation.streamline.absorb as absorb
from finn.builder.build_dataflow_config import DataflowBuildConfig


def step_pre_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    # RadioML is a 1D-conv signal model: lift 3D tensors to 4D so the standard
    # streamline/convert phases can handle them, and fold the scalar mul/add on
    # the classifier output into the TopK.
    model = model.transform(Change3DTo4DTensors())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    return model
