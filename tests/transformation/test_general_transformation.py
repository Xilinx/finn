############################################################################
# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import json
import os
from pkgutil import get_data
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import ApplyConfig, GiveUniqueNodeNames
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul


@pytest.mark.transform
def test_apply_config():
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(GiveUniqueNodeNames())
    # set up a config in a dict, then dump it to JSON
    config = {}
    config["Defaults"] = {"kernel_size": [[3, 3], ["Im2Col"]]}
    config["Im2Col_0"] = {"kernel_size": [7, 7]}
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    model = model.transform(ApplyConfig("config.json"))
    # check model
    assert getCustomOp(model.graph.node[2]).get_nodeattr("kernel_size") == [7, 7]
    assert getCustomOp(model.graph.node[9]).get_nodeattr("kernel_size") == [3, 3]
    os.remove("config.json")
