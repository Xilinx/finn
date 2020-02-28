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

from pkgutil import get_data

import numpy as np
from onnx import TensorProto, helper

import finn.util.basic as util
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes


def test_infer_shapes():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    graph = model.graph

    # multi-thresholding node to be inserted between the first Relu and MaxPool node

    # get Relu node to use data
    Relu_node = graph.node[3]
    assert Relu_node.op_type == "Relu", "The wrong model was chosen for the check"

    # create thresholds tensor as constant
    mt_thresh0 = helper.make_tensor_value_info("mt_thresh0", TensorProto.FLOAT, [8, 7])

    # random numbers for the thresholds
    # thresholds for one channel have to be sorted to guarantee the correct behavior
    mt_thresh0_values = np.empty([8, 7], dtype=np.float32)
    for i in range(len(mt_thresh0_values)):
        mt_thresh0_values[i] = np.sort(np.random.random_sample(7) * 10)

    model.set_initializer(mt_thresh0.name, mt_thresh0_values)

    # add multi-thresholding node and change Relu node
    mt_node = helper.make_node(
        "MultiThreshold", ["mt_v0", "mt_thresh0"], [Relu_node.output[0]], domain="finn"
    )
    Relu_node.output[0] = "mt_v0"

    # explicitly remove any present shape from ReLU and MultiThreshold outputs
    util.remove_by_name(model.graph.value_info, Relu_node.output[0])
    util.remove_by_name(model.graph.value_info, mt_node.output[0])
    graph.node.insert(4, mt_node)

    # first check routine
    # check if at least one shape is not specified
    assert not (
        model.check_all_tensor_shapes_specified()
    ), "All tensors are already specified before the shape inference execution"

    # perform shape inference on mixed model
    model = model.transform(InferShapes())

    # second check routine
    # now all shapes should be specified and mt_node output shape is (1,8,28,28)
    assert (
        model.check_all_tensor_shapes_specified()
    ), "There are still tensors that are not specified"
    assert (model.get_tensor_shape(mt_node.output[0])) == (
        [1, 8, 28, 28]
    ), "output of multi-thresholding node has wrong shape"
