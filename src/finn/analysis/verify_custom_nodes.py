import finn.custom_op.registry as registry


def verify_nodes(model):
    """Checks if custom ops in graph are correctly built, with all attributes
    and inputs. Returns {node op_type : info_messages}
    *info_messages is list of strings about the result of the verification"""

    verification_dict = {}
    for node in model.graph.node:
        if node.domain == "finn":
            op_type = node.op_type
            inst = registry.custom_op[op_type](node)
            verification_dict[op_type] = inst.verify_node()

    return verification_dict
