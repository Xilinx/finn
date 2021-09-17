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

import numpy as np
from onnx import TensorProto, helper

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.test import load_test_checkpoint_or_skip


def make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes):

    W = np.random.randint(wdt.min(), wdt.max() + 1, size=(ch, ch))
    W = W.astype(np.float32)

    T = np.random.randint(tdt.min(), tdt.max() + 1, size=(ch, 2 ** adt.bitwidth() - 1))
    T = T.astype(np.float32)

    tensors = []
    tensors.append(helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch]))
    for i in range(1, nnodes):
        inter = helper.make_tensor_value_info(
            "inter_" + str(i), TensorProto.FLOAT, [1, ch]
        )
        tensors.append(inter)
    tensors.append(helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch]))

    FCLayer_nodes = []
    for i in range(nnodes):
        pe = 1
        simd = 1
        FCLayer_nodes += [
            helper.make_node(
                "StreamingFCLayer_Batch",
                [tensors[i].name, "weights_" + str(i), "thresh_" + str(i)],
                [tensors[i + 1].name],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                MW=ch,
                MH=ch,
                SIMD=simd,
                PE=pe,
                inputDataType=adt.name,
                weightDataType=wdt.name,
                outputDataType=adt.name,
                ActVal=0,
                binaryXnorMode=0,
                noActivation=0,
            )
        ]

    graph = helper.make_graph(
        nodes=FCLayer_nodes,
        name="fclayer_graph",
        inputs=[tensors[0]],
        outputs=[tensors[-1]],
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", adt)
    model.set_tensor_datatype("outp", adt)

    for i in range(1, nnodes + 1):
        if tensors[i].name != "outp":
            model.graph.value_info.append(tensors[i])
        model.set_initializer("weights_" + str(i - 1), W)
        model.set_initializer("thresh_" + str(i - 1), T)
        model.set_tensor_datatype("weights_" + str(i - 1), wdt)
        model.set_tensor_datatype("thresh_" + str(i - 1), tdt)

    return model


# desired frames per second
@pytest.mark.parametrize("target_fps", [30, 10 ** 5, 10 ** 7])
# target chip or board
@pytest.mark.parametrize("platform", ["Pynq-Z1", "Ultra96", "U200"])
def test_set_folding(target_fps, platform):

    model = make_multi_fclayer_model(
        128, DataType.INT4, DataType.INT2, DataType.INT16, 5
    )

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

    min_cycles = dict()
    min_cycles["Pynq-Z1"] = 128
    min_cycles["Ultra96"] = 64
    min_cycles["U200"] = 1

    assert achieved_cycles_per_frame <= max(
        min_cycles[platform], target_cycles_per_frame
    ), "Folding target not met"
