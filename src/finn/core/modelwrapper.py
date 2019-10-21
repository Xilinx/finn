import copy

import onnx.numpy_helper as np_helper


class ModelWrapper:
    """A wrapper around ONNX ModelProto that exposes some useful utility
    functions for graph manipulation and exploration."""

    def __init__(self, onnx_model_proto, make_deepcopy=False):
        if make_deepcopy:
            self._model_proto = copy.deepcopy(onnx_model_proto)
        else:
            self._model_proto = onnx_model_proto

    def get_tensor_shape(self):
        """Returns the shape of tensor with given name, if it has ValueInfoProto."""
        graph = self._model_proto.graph
        vi_names = [(x.name, x) for x in graph.input]
        vi_names += [(x.name, x) for x in graph.output]
        vi_names += [(x.name, x) for x in graph.value_info]
        try:
            vi_ind = [x[0] for x in vi_names].index(self)
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
            init_ind = init_names.index(self)
            init_old = graph.initializer[init_ind]
            graph.initializer.remove(init_old)
        except ValueError:
            pass
        # create and insert new initializer
        graph.initializer.append(tensor_init_proto)

    def get_initializer(self):
        """Get the initializer value for tensor with given name, if any."""
        graph = self._model_proto.graph
        init_names = [x.name for x in graph.initializer]
        try:
            init_ind = init_names.index(self)
            return np_helper.to_array(graph.initializer[init_ind])
        except ValueError:
            return None

    def find_producer(self):
        """Find and return the node that produces the tensor with given name.
        Currently only works for linear graphs."""
        all_outputs = [x.output[0] for x in self._model_proto.graph.node]
        try:
            producer_ind = all_outputs.index(self)
            return self._model_proto.graph.node[producer_ind]
        except ValueError:
            return None

    def find_consumer(self):
        """Find and return the node that consumes the tensor with given name.
        Currently only works for linear graphs."""
        all_inputs = [x.input[0] for x in self._model_proto.graph.node]
        try:
            consumer_ind = all_inputs.index(self)
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
