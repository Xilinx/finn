import finn.custom_op.registry as registry


def code_gen_transformation(node):
    """Call custom implementation to generate code for single custom node
    and create folder that contains all the generated files"""
    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type](node)

        # get the path of the code generation directory if already set
        # check instance and check node attributes for value
        code_gen_dir = inst.code_gen_dir

        # parameter is empty
        if not code_gen_dir:
            print("parameter is empty")
            # create new directory, set the value and generate the code

        else:
            print("parameter contains value")
            # check directory for files if empty,
            # delete directory and create new one
            # otherwise just leave it that way

    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)
