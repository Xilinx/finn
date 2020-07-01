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
import copy

from onnx import helper

from finn.transformation import Transformation
from finn.core.modelwrapper import ModelWrapper
import finn.util.basic as util
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)


def _make_model_values_unique(model1, model2):
    # ensure that tensor and node names are different in each model
    # tensors
    names1 = model1.get_all_tensor_names()
    names2 = model2.get_all_tensor_names()
    duplicates = list(set(names1).intersection(names2))
    # if there are duplicates in the tensor names rename these tensors
    if duplicates:
        used_names = names1 + names2
        for name in duplicates:
            # model1
            new_name = util.random_string()
            while new_name in used_names:
                new_name = util.random_string()
            model1.rename_tensor(name, new_name)
            used_names.append(new_name)

            # model2
            new_name = util.random_string()
            while new_name in used_names:
                new_name = util.random_string()
            model2.rename_tensor(name, new_name)
            used_names.append(new_name)

    # nodes
    names1 = [x.name for x in model1.graph.node]
    names1 = list(filter(None, names1))  # filter out empty node names
    names2 = [x.name for x in model2.graph.node]
    names2 = list(filter(None, names2))
    duplicates = list(set(names1).intersection(names2))
    # if there are duplicates erase all node names
    if duplicates:
        for n in model1.graph.node:
            n.name = ""
        for n in model2.graph.node:
            n.name = ""

    return (model1, model2)


class MergeONNXModels(Transformation):
    """Merges two models. The model passed in the transformation will be inserted before
    the model the transformation is applied on, the resulting model is returned."""

    def __init__(self, pre_model):
        super().__init__()
        # use deep copy of model that should be inserted in the beginning of
        # the other model to ensure that it stays unchanged
        self.pre_model = copy.deepcopy(pre_model)

    def apply(self, model):
        graph_modified = False
        pre_model = self.pre_model

        (pre_model, model) = _make_model_values_unique(pre_model, model)

        # check if models can be merged
        output_model_a = pre_model.graph.output[0].name
        input_model_b = model.graph.input[0].name
        output_a_shape = pre_model.get_tensor_shape(output_model_a)
        input_b_shape = model.get_tensor_shape(input_model_b)
        assert (
            output_a_shape == input_b_shape
        ), "Models can't be merged! Shapes don't match."
        for n in pre_model.graph.node:
            if output_model_a == n.output[0]:
                n.output[0] = input_model_b

        node_list_a = pre_model.graph.node
        node_list_b = model.graph.node

        node_list = node_list_a
        for node in node_list_b:
            node_list.append(node)
        inp = pre_model.graph.input[0]
        outp = model.graph.output[0]
        new_graph = helper.make_graph(
            nodes=node_list,
            name="fuse-graph",
            inputs=[inp],
            outputs=[outp],
            value_info=[],
        )

        new_model = helper.make_model(new_graph, producer_name="fuse_model")
        new_model = ModelWrapper(new_model)
        vi_preproc = [x for x in pre_model.graph.input]
        vi_preproc += [x for x in pre_model.graph.output]
        vi_preproc += [x for x in pre_model.graph.value_info]
        for vi in vi_preproc:
            if vi == inp:
                continue
            new_model.graph.value_info.append(vi)
            init_val = pre_model.get_initializer(vi.name)
            if init_val is not None:
                new_model.set_initializer(vi.name, init_val)
        vi_model = [x for x in model.graph.input]
        vi_model += [x for x in model.graph.output]
        vi_model += [x for x in model.graph.value_info]
        for vi in vi_model:
            if vi == outp:
                continue
            new_model.graph.value_info.append(vi)
            init_val = model.get_initializer(vi.name)
            if init_val is not None:
                new_model.set_initializer(vi.name, init_val)

        model = new_model
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(GiveReadableTensorNames())
        return (model, graph_modified)
