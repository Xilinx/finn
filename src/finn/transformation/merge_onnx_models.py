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
import warnings
from onnx import helper

from finn.transformation import Transformation
from finn.core.modelwrapper import ModelWrapper
import finn.util.basic as util
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)


class MergeONNXModels(Transformation):
    """Merges two models. The model passed in the transformation will be inserted before
    the model the transformation is applied on, the resulting model is returned.
    This transformation will try to connect graph.output[0] of the pre model and
    graph.input[0] of the post model.
    If more than one input or output exists, a warning is raised."""

    def __init__(self, pre_model):
        super().__init__()
        # use deep copy of model that should be inserted in the beginning of
        # the other model to ensure that it stays unchanged
        self.pre_model = copy.deepcopy(pre_model)

    def apply(self, model):
        graph_modified = False
        pre_model = self.pre_model
        post_model = copy.deepcopy(model)

        # check for dynamic outputs of pre model
        dyn_outp = []
        for outp in pre_model.graph.output:
            init_val = pre_model.get_initializer(outp.name)
            if init_val is None:
                dyn_outp.append(outp)

        if len(dyn_outp) != 1:
            warnings.warn(
                "The pre model has more than one dynamic output! The transformation "
                "tries to connect the first dynamic output to the first dynamic input "
                "of the post model."
            )

        # check for dynamic inputs of post model
        dyn_inp = []
        for inp in post_model.graph.input:
            init_val = post_model.get_initializer(inp.name)
            if init_val is None:
                dyn_inp.append(inp)

        if len(dyn_inp) != 1:
            warnings.warn(
                "The post model has more than one dynamic input! The transformation "
                "tries to connect the first dynamic input to the first dynamic output "
                "of the pre model."
            )

        # erase all node names to avoid conflict
        for n in pre_model.graph.node:
            n.name = ""
        for n in post_model.graph.node:
            n.name = ""

        # randomize all tensor names
        names1 = pre_model.get_all_tensor_names()
        names2 = post_model.get_all_tensor_names()
        used_names = names1 + names2

        # pre_model
        for tensor_name in names1:
            new_name = util.random_string()
            while new_name in used_names:
                new_name = util.random_string()
            pre_model.rename_tensor(tensor_name, new_name)
            used_names.append(new_name)

        # post_model
        for tensor in names2:
            new_name = util.random_string()
            while new_name in used_names:
                new_name = util.random_string()
            post_model.rename_tensor(tensor_name, new_name)
            used_names.append(new_name)

        # check if models can be merged
        output_model_a = dyn_outp[0].name
        input_model_b = dyn_inp[0].name
        output_a_shape = pre_model.get_tensor_shape(output_model_a)
        input_b_shape = post_model.get_tensor_shape(input_model_b)
        assert (
            output_a_shape == input_b_shape
        ), "Models can't be merged! Shapes don't match."

        # connect output of one model to input of the other
        for n in pre_model.graph.node:
            if output_model_a == n.output[0]:
                n.output[0] = input_model_b

        # extract information for new model

        # nodes
        node_list_a = pre_model.graph.node
        node_list_b = post_model.graph.node

        node_list = node_list_a
        for node in node_list_b:
            node_list.append(node)

        # in and output
        inp = pre_model.graph.input[0]
        outp = post_model.graph.output[0]

        # create new graph and model
        new_graph = helper.make_graph(
            nodes=node_list,
            name="fuse-graph",
            inputs=[inp],
            outputs=[outp],
            value_info=[],
        )

        new_model = helper.make_model(new_graph, producer_name="fuse_model")
        new_model = ModelWrapper(new_model)

        # add value info from both models to new model
        # pre model
        vi_pre = [x for x in pre_model.graph.input]
        vi_pre += [x for x in pre_model.graph.output]
        vi_pre += [x for x in pre_model.graph.value_info]
        for vi in vi_pre:
            # preserve intializers, quantization/sparsity annotation, etc.
            # initializer
            init_val = pre_model.get_initializer(vi.name)
            if init_val is not None:
                new_model.set_initializer(vi.name, init_val)
            # FINN datatype
            dtype = pre_model.get_tensor_datatype(vi.name)
            new_model.set_tensor_datatype(vi.name, dtype)
            # data layout
            data_layout = pre_model.get_tensor_layout(vi.name)
            if data_layout is not None:
                new_model.set_tensor_layout(vi.name, data_layout)
            # sparsity
            sparsity = pre_model.get_tensor_sparsity(vi.name)
            if sparsity is not None:
                new_model.set_tensor_sparsity(vi.name, sparsity)
            # graph input should not be part of graph.value_info, so don't insert
            # if current vi == inp, but the quantization annotation is preserved
            if vi == inp:
                continue
            new_model.graph.value_info.append(vi)

        # post model
        vi_model = [x for x in post_model.graph.input]
        vi_model += [x for x in post_model.graph.output]
        vi_model += [x for x in post_model.graph.value_info]
        for vi in vi_model:
            # preserve intializers, quantization/sparsity annotation, etc.
            # initializer
            init_val = post_model.get_initializer(vi.name)
            if init_val is not None:
                new_model.set_initializer(vi.name, init_val)
            # FINN datatype
            dtype = post_model.get_tensor_datatype(vi.name)
            new_model.set_tensor_datatype(vi.name, dtype)
            # data layout
            data_layout = post_model.get_tensor_layout(vi.name)
            if data_layout is not None:
                new_model.set_tensor_layout(vi.name, data_layout)
            # sparsity
            sparsity = post_model.get_tensor_sparsity(vi.name)
            if sparsity is not None:
                new_model.set_tensor_sparsity(vi.name, sparsity)
            # graph output should not be part of graph.value_info, so don't insert
            # if current vi == outp, but the quantization annotation is preserved
            if vi == outp:
                continue
            new_model.graph.value_info.append(vi)

        # tidy-up new model
        model = new_model
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(GiveReadableTensorNames())

        return (model, graph_modified)
