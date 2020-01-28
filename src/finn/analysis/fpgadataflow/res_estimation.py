import finn.core.utils as util
import finn.custom_op.registry as registry


def res_estimation(model):
    """Estimates the resources needed for the given model.
    Returns {node name : resource estimation}"""

    res_dict = {}
    for node in model.graph.node:
        if node.domain == "finn":
            backend_attribute = util.get_by_name(node.attribute, "backend")
            if backend_attribute is None:
                continue
            backend_value = backend_attribute.s.decode("UTF-8")
            if backend_value == "fpgadataflow":
                op_type = node.op_type
                inst = registry.custom_op[op_type](node)
                res_dict[node.name] = inst.node_res_estimation()

    return res_dict
