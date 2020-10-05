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

import os.path
from pkgutil import get_data


from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.util.basic import make_build_dir
from finn.util.test import load_test_checkpoint_or_skip

build_dir = make_build_dir("test_dataflow_partition_")


def test_dataflow_partition_create():
    # load the onnx model
    raw_m = get_data(
        "finn.qnn-data", "onnx/finn-hls-model/tfc_w1_a1_after_conv_to_hls.onnx"
    )
    model = ModelWrapper(raw_m)
    model = model.transform(CreateDataflowPartition())
    assert model.graph.node[2].op_type == "StreamingDataflowPartition"
    sdp_node = getCustomOp(model.graph.node[2])
    assert sdp_node.__class__.__name__ == "StreamingDataflowPartition"
    assert os.path.isfile(sdp_node.get_nodeattr("model"))
    model.save(build_dir + "/test_dataflow_partition_create.onnx")


def test_dataflow_partition_tlastmarker():
    model = load_test_checkpoint_or_skip(
        build_dir + "/test_dataflow_partition_create.onnx"
    )
    model_path = getCustomOp(model.graph.node[2]).get_nodeattr("model")
    model = ModelWrapper(model_path)
    model = model.transform(InsertTLastMarker())
    assert model.graph.node[-1].op_type == "TLastMarker"
    assert model.graph.node[-1].domain == "finn"
    tl_node = getCustomOp(model.graph.node[-1])
    assert tl_node.get_nodeattr("NumIters") == 1
    assert tl_node.get_nodeattr("StreamWidth") == 320
    assert tl_node.get_nodeattr("ElemWidth") == 32
    model.save(build_dir + "/test_dataflow_partition_tlastmarker.onnx")
    model = model.transform(InsertTLastMarker())
    model.save(build_dir + "/test_dataflow_partition_tlastmarker2.onnx")
