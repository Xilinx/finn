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

import onnx
import onnx.helper as oh
import onnx.numpy_helper as np_helper
from onnx import TensorProto

import finn.util.basic as util
import finn.util.onnx as onnxutil
from finn.core.datatype import DataType


class ModelWrapper:
    """A wrapper around ONNX ModelProto that exposes some useful utility
    functions for graph manipulation and exploration."""

    def __init__(self, onnx_model_proto, make_deepcopy=False):
        """Creates a ModelWrapper instance.
        onnx_model_proto can be either a ModelProto instance, or a string
        with the path to a stored .onnx file on disk, or serialized bytes.
        The make_deepcopy option controls whether a deep copy of the ModelProto
        is made internally.
        """
        if isinstance(onnx_model_proto, str):
            self._model_proto = onnx.load(onnx_model_proto)
        elif isinstance(onnx_model_proto, bytes):
            self._model_proto = onnx.load_from_string(onnx_model_proto)
        else:
            if make_deepcopy:
                self._model_proto = copy.deepcopy(onnx_model_proto)
            else:
                self._model_proto = onnx_model_proto

    @property
    def graph(self):
        """Returns the graph of the model."""
        return self._model_proto.graph

    @graph.setter
    def graph(self, value):
        """Sets the graph of the model according to value"""
        self._model_proto.graph = value

    @property
    def model(self):
        """Returns the model."""
        return self._model_proto

    @model.setter
    def model(self, value):
        """Sets the model according to value."""
        self._model_proto = value

    def save(self, filename):
        """Saves the wrapper ONNX ModelProto into a file with given name."""
        onnx.save(self._model_proto, filename)

    def analysis(self, analysis_fxn):
        """Runs given anaylsis_fxn on this model and return resulting dict."""
        return analysis_fxn(self)

    def transform(self, transformation, make_deepcopy=True):
        """Applies given Transformation repeatedly until no more changes can be made
        and returns a transformed ModelWrapper instance.

        If make_deepcopy is specified, operates on a new (deep)copy of model.
        """
        transformed_model = self
        if make_deepcopy:
            transformed_model = copy.deepcopy(self)
        model_was_changed = True
        while model_was_changed:
            (transformed_model, model_was_changed) = transformation.apply(
                transformed_model
            )
        return transformed_model

    def check_compatibility(self):
        """Checks this model for FINN compatibility:

        * no embedded subgraphs

        * all tensor shapes are specified, including activations

        * all constants are initializers
        """
        # TODO check for no embedded subgraphs
        # TODO check that all shapes are inferred
        # TODO check that all constants are initializers
        return True

    def get_tensor_datatype(self, tensor_name):
        """Returns the FINN DataType of tensor with given name."""
        graph = self._model_proto.graph
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, tensor_name, "tensor_name")
        if ret is not None:
            ret = util.get_by_name(
                ret.quant_parameter_tensor_names, "finn_datatype", "key"
            )
            if ret is not None:
                return DataType[ret.value]
        # TODO maybe use native ONNX tensor type instead of assuming fp32?
        return DataType["FLOAT32"]

    def set_tensor_datatype(self, tensor_name, datatype):
        """Sets the FINN DataType of tensor with given name."""
        graph = self._model_proto.graph
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, tensor_name, "tensor_name")
        if ret is not None:
            ret = util.get_by_name(
                ret.quant_parameter_tensor_names, "finn_datatype", "key"
            )
            if ret is not None:
                ret.value = datatype.name
        else:
            qa = onnx.TensorAnnotation()
            dt = onnx.StringStringEntryProto()
            dt.key = "finn_datatype"
            dt.value = datatype.name
            qa.tensor_name = tensor_name
            qa.quant_parameter_tensor_names.append(dt)
            qnt_annotations.append(qa)

    def get_tensor_valueinfo(self, tensor_name):
        """Returns ValueInfoProto of tensor with given name, if it has one."""
        graph = self._model_proto.graph
        vi_names = [(x.name, x) for x in graph.input]
        vi_names += [(x.name, x) for x in graph.output]
        vi_names += [(x.name, x) for x in graph.value_info]
        try:
            vi_ind = [x[0] for x in vi_names].index(tensor_name)
            vi = vi_names[vi_ind][1]
            return vi
        except ValueError:
            return None

    def get_tensor_shape(self, tensor_name):
        """Returns the shape of tensor with given name, if it has ValueInfoProto."""
        graph = self._model_proto.graph
        vi_names = [(x.name, x) for x in graph.input]
        vi_names += [(x.name, x) for x in graph.output]
        vi_names += [(x.name, x) for x in graph.value_info]
        try:
            vi_ind = [x[0] for x in vi_names].index(tensor_name)
            vi = vi_names[vi_ind][1]
            dims = [x.dim_value for x in vi.type.tensor_type.shape.dim]
            return dims
        except ValueError:
            return None

    def set_tensor_shape(self, tensor_name, tensor_shape, dtype=TensorProto.FLOAT):
        """Assigns shape in ValueInfoProto for tensor with given name."""
        new_vi = oh.make_tensor_value_info(tensor_name, dtype, tensor_shape)
        # find what container tis tensor's ValueInfo lives in
        # if not found anywhere, we assume it's a new value_info
        target_container = self.graph.value_info
        if util.get_by_name(self.graph.input, tensor_name) is not None:
            target_container = self.graph.input
        if util.get_by_name(self.graph.output, tensor_name) is not None:
            target_container = self.graph.output
        # remove from target container and add new
        util.remove_by_name(target_container, tensor_name)
        target_container.append(new_vi)

    def set_initializer(self, tensor_name, tensor_value):
        """Sets the initializer value for tensor with given name."""
        graph = self._model_proto.graph
        # convert tensor_value (numpy array) into TensorProto w/ correct name
        tensor_init_proto = np_helper.from_array(tensor_value)
        tensor_init_proto.name = tensor_name
        # first, remove if an initializer already exists
        init_names = [x.name for x in graph.initializer]
        try:
            init_ind = init_names.index(tensor_name)
            init_old = graph.initializer[init_ind]
            graph.initializer.remove(init_old)
        except ValueError:
            pass
        # create and insert new initializer
        graph.initializer.append(tensor_init_proto)
        # set shape
        dtype = tensor_init_proto.data_type
        self.set_tensor_shape(tensor_name, list(tensor_value.shape), dtype)

    def rename_tensor(self, old_name, new_name):
        """Renames a tensor from old_name to new_name."""
        graph = self.graph
        # sweep over inputs
        if util.get_by_name(graph.input, old_name) is not None:
            util.get_by_name(graph.input, old_name).name = new_name
        # sweep over outputs
        if util.get_by_name(graph.output, old_name) is not None:
            util.get_by_name(graph.output, old_name).name = new_name
        # sweep over value_info
        if util.get_by_name(graph.value_info, old_name) is not None:
            util.get_by_name(graph.value_info, old_name).name = new_name
        # sweep over initializers
        if util.get_by_name(graph.initializer, old_name) is not None:
            util.get_by_name(graph.initializer, old_name).name = new_name
        # sweep over quantization annotations
        if (
            util.get_by_name(graph.quantization_annotation, old_name, "tensor_name")
            is not None
        ):
            util.get_by_name(
                graph.quantization_annotation, old_name, "tensor_name"
            ).tensor_name = new_name
        # sweep over node i/o
        for n in graph.node:
            if old_name in n.input:
                n.input[list(n.input).index(old_name)] = new_name
            if old_name in n.output:
                n.output[list(n.output).index(old_name)] = new_name

    def get_initializer(self, tensor_name):
        """Gets the initializer value for tensor with given name, if any."""
        graph = self._model_proto.graph
        init_names = [x.name for x in graph.initializer]
        try:
            init_ind = init_names.index(tensor_name)
            return np_helper.to_array(graph.initializer[init_ind])
        except ValueError:
            return None

    def find_producer(self, tensor_name):
        """Finds and returns the node that produces the tensor with given name.
        Currently only works for linear graphs."""
        all_outputs = [x.output[0] for x in self._model_proto.graph.node]
        try:
            producer_ind = all_outputs.index(tensor_name)
            return self._model_proto.graph.node[producer_ind]
        except ValueError:
            return None

    def find_upstream(self, tensor_name, finder_fxn):
        """Follow the producer chain upstream, calling finder_fxn on each upstream
        node until it returns True or there are no nodes left. Returns the list
        of nodes visited, or None if finder_fxn did not return True."""
        visit_list = []
        current_tensor = tensor_name
        while True:
            current_producer = self.find_producer(current_tensor)
            if current_producer is None:
                return []
            else:
                found = finder_fxn(current_producer)
                visit_list.append(current_producer)
                if found:
                    return visit_list
                else:
                    current_tensor = current_producer.input[0]

    def find_consumer(self, tensor_name):
        """Finds and returns the node that consumes the tensor with given name.
        Currently only works for linear graphs."""
        all_inputs = [x.input[0] for x in self._model_proto.graph.node]
        try:
            consumer_ind = all_inputs.index(tensor_name)
            return self._model_proto.graph.node[consumer_ind]
        except ValueError:
            return None

    def find_consumers(self, tensor_name):
        consumers = []
        try:
            for n in self._model_proto.graph.node:
                for inp_tensor in n.input:
                    if inp_tensor == tensor_name:
                        consumers.append(n)
            return consumers
        except ValueError:
            return None

    def find_successors(self, node):
        successors = []
        try:
            for outp_tensor in node.output:
                tensor_consumer_list = self.find_consumers(outp_tensor)
                for consumer in tensor_consumer_list:
                    successors.append(consumer)
            return successors
        except ValueError:
            return None

    def find_predecessors(self, node):
        predecessors = []
        try:
            for inp_tensor in node.input:
                producer = self.find_producer(inp_tensor)
                predecessors.append(producer)
            return predecessors
        except ValueError:
            return None

    def get_all_tensor_names(self):
        """Returns a list of all (input, output and value_info) tensor names
        in the graph."""
        graph = self.graph
        names = [x.name for x in graph.value_info]
        names += [x.name for x in graph.input]
        names += [x.name for x in graph.output]
        return names

    def make_new_valueinfo_name(self):
        """Returns a name that can be used for a new value_info."""
        names = self.get_all_tensor_names()
        candidate = util.random_string()
        while candidate in names:
            candidate = util.random_string()
        return candidate

    def make_empty_exec_context(self):
        """Creates an empty execution context for this model.

        The execution context is a dictionary of all tensors used for the
        inference computation. Any initializer values will be taken into
        account, all other tensors will be zero."""
        execution_context = dict()
        graph = self._model_proto.graph
        # make empty tensors for all the graph inputs and outputs
        for vi in graph.input:
            new_tensor = onnxutil.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        for vi in graph.output:
            new_tensor = onnxutil.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        # make empty tensors for all intermediate buffers
        for vi in graph.value_info:
            new_tensor = onnxutil.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        # fill in the constants provided by the initializers (TensorProto to npy)
        for t in graph.initializer:
            execution_context[t.name] = np_helper.to_array(t)
        return execution_context

    def check_all_tensor_shapes_specified(self):
        """Checks whether all tensors have a specified shape (ValueInfo).
        The ONNX standard allows for intermediate activations to have no
        associated ValueInfo, but FINN expects this."""
        graph = self._model_proto.graph
        ret = True
        for n in graph.node:
            for i in n.input:
                ret = ret and (self.get_tensor_shape(i) is not None)
            for o in n.output:
                ret = ret and (self.get_tensor_shape(o) is not None)
        return ret

    def get_tensor_fanout(self, tensor_name):
        """Returns the number of nodes for which the tensor with given name is
        as input."""
        graph = self.graph
        fanout = 0
        for n in graph.node:
            if tensor_name in n.input:
                fanout += 1
        return fanout

    def get_metadata_prop(self, key):
        """Returns the value associated with metadata_prop with given key,
        or None otherwise."""
        metadata_prop = util.get_by_name(self.model.metadata_props, key, "key")
        if metadata_prop is None:
            return None
        else:
            return metadata_prop.value

    def set_metadata_prop(self, key, value):
        """Sets metadata property with given key to the given value."""
        metadata_prop = util.get_by_name(self.model.metadata_props, key, "key")
        if metadata_prop is None:
            metadata_prop = onnx.StringStringEntryProto()
            metadata_prop.key = key
            metadata_prop.value = value
            self.model.metadata_props.append(metadata_prop)
        else:
            metadata_prop.value = value

    def get_nodes_by_op_type(self, op_type):
        """Returns a list of nodes with specified op_type."""
        return list(filter(lambda x: x.op_type == op_type, self.graph.node))

    def get_finn_nodes(self):
        """Returns a list of nodes where domain == 'finn'."""
        return list(filter(lambda x: x.domain == "finn", self.graph.node))

    def get_non_finn_nodes(self):
        """Returns a list of nodes where domain != 'finn'."""
        return list(filter(lambda x: x.domain != "finn", self.graph.node))
