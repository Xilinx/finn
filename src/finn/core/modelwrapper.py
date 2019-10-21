import copy

import onnx
import onnx.numpy_helper as np_helper

import finn.core.utils as util


class ModelWrapper:
    """A wrapper around ONNX ModelProto that exposes some useful utility
    functions for graph manipulation and exploration."""

    def __init__(self, onnx_model_proto, make_deepcopy=False):
        """Creates a ModelWrapper instance.
        onnx_model_proto can be either a ModelProto instance, or a string
        with the path to a stored .onnx file on disk.
        The make_deepcopy option controls whether a deep copy of the ModelProto
        is made internally.
        """
        if isinstance(onnx_model_proto, str):
            self._model_proto = onnx.load(onnx_model_proto)
        else:
            if make_deepcopy:
                self._model_proto = copy.deepcopy(onnx_model_proto)
            else:
                self._model_proto = onnx_model_proto

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

    def set_initializer(self, tensor_name, tensor_value):
        """Set the initializer value for tensor with given name."""
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

    def get_initializer(self, tensor_name):
        """Get the initializer value for tensor with given name, if any."""
        graph = self._model_proto.graph
        init_names = [x.name for x in graph.initializer]
        try:
            init_ind = init_names.index(tensor_name)
            return np_helper.to_array(graph.initializer[init_ind])
        except ValueError:
            return None

    def find_producer(self, tensor_name):
        """Find and return the node that produces the tensor with given name.
        Currently only works for linear graphs."""
        all_outputs = [x.output[0] for x in self._model_proto.graph.node]
        try:
            producer_ind = all_outputs.index(tensor_name)
            return self._model_proto.graph.node[producer_ind]
        except ValueError:
            return None

    def find_consumer(self, tensor_name):
        """Find and return the node that consumes the tensor with given name.
        Currently only works for linear graphs."""
        all_inputs = [x.input[0] for x in self._model_proto.graph.node]
        try:
            consumer_ind = all_inputs.index(tensor_name)
            return self._model_proto.graph.node[consumer_ind]
        except ValueError:
            return None

    def make_new_valueinfo_name(self):
        """Returns a name that can be used for a new value_info."""
        graph = self._model_proto.graph
        names = [x.name for x in graph.value_info]
        names += [x.name for x in graph.input]
        names += [x.name for x in graph.output]
        candidate = str(len(names) + 1)
        while candidate in names:
            candidate = str(int(candidate) + 1)
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
            new_tensor = util.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        for vi in graph.output:
            new_tensor = util.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        # make empty tensors for all intermediate buffers
        for vi in graph.value_info:
            new_tensor = util.valueinfo_to_tensor(vi)
            execution_context[vi.name] = new_tensor
        # fill in the constants provided by the initializers (TensorProto to npy)
        for t in graph.initializer:
            execution_context[t.name] = np_helper.to_array(t)
        return execution_context
