import numpy as np


def is_linear(model):
    """Checks whether the given model graph is linear. This is done by looking
    at the fan-out of each tensor. All tensors have a fan-out <= 1 in a linear
    graph. 
    
    Returns {"is_linear": Bool}."""
    per_tensor_fanouts = get_per_tensor_fanouts(model)
    # check for tensors that have fanout > 1
    multi_fanouts = list(filter(lambda x: x[1] > 1, per_tensor_fanouts.items()))
    return {"is_linear": len(multi_fanouts) == 0}


def get_per_tensor_fanouts(model):
    """Returns a dictionary of {tensor_name: tensor_fanout} for the model."""
    # make execution context to get a list of tensors
    per_tensor_fanouts = model.make_empty_exec_context()
    # replace every tensor with its fanout
    for tensor_name in per_tensor_fanouts.keys():
        per_tensor_fanouts[tensor_name] = model.get_tensor_fanout(tensor_name)
    return per_tensor_fanouts


def all_tensors_f32(model):
    """Checks whether all tensors have a float32 dtype, extra quantization
    annotations notwithstanding. 
    
    Returns {"all_tensors_f32": Bool}."""
    all_tensors = model.make_empty_exec_context().items()
    non_f32_tensors = filter(lambda x: x[1].dtype != np.float32, all_tensors)
    return {"all_tensors_f32": len(list(non_f32_tensors)) == 0}


def node_inputs_in_expected_order(model):
    """Verifies that the node inputs are ordered in the way that FINN expects
    them. When a node has a mixture of static (= constant, initialized) inputs
    and dynamic inputs, the dynamic input should come first, followed by the
    static one. Only verifiable for a small subset of op_types for now. 
    
    Returns {"node_inputs_in_expected_order": Bool}."""
    op_types = ["MatMul", "Conv", "Add", "Mul"]
    nodes = filter(lambda x: x.op_type in op_types, model.graph.node)
    all_OK = True
    for n in nodes:
        all_OK = all_OK and len(list(n.input)) == 2
        # input 0 should be dynamic, no initializer
        all_OK = all_OK and (model.get_initializer(n.input[0]) is None)
        # input 1 should be static (unless eltwise add)
        if n.op_type != "Add":
            all_OK = all_OK and (model.get_initializer(n.input[1]) is not None)
    return {"node_inputs_in_expected_order": all_OK}
