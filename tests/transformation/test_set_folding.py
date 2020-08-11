# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest
from pkgutil import get_data
from finn.custom_op.registry import getCustomOp
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.util.test import load_test_checkpoint_or_skip


# desired frames per second
@pytest.mark.parametrize("target_fps", [30, 10 ** 5, 10 ** 7])
def test_set_folding(target_fps):

    # test with tfcW1A1
    raw_m = get_data(
        "finn", "data/onnx/finn-hls-model/tfc_w1_a1_after_conv_to_hls.onnx"
    )
    model = ModelWrapper(raw_m)
    model = model.transform(GiveUniqueNodeNames())
    parent_model = model.transform(CreateDataflowPartition())
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    clk_ns = 5
    target_cycles_per_frame = int((10 ** 9 / clk_ns) / target_fps)

    dataflow_model = dataflow_model.transform(SetFolding(target_cycles_per_frame))

    exp_cycles_dict = dataflow_model.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame = max(exp_cycles_dict.values())

    # if achieved is more than target, this means we can't fold enough; re-fold with
    # target set to the largest cycle delay of all layers, to minimize resource usage
    if target_cycles_per_frame < achieved_cycles_per_frame:
        dataflow_model = dataflow_model.transform(SetFolding(achieved_cycles_per_frame))
        exp_cycles_dict = dataflow_model.analysis(exp_cycles_per_layer)

    assert achieved_cycles_per_frame <= max(
        28, target_cycles_per_frame
    ), "Folding target not met"
