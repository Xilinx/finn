import os
import tempfile as tmp

import finn.custom_op.registry as registry


def code_gen_transformation(node, context):
    """Call custom implementation to generate code for single custom node
    and create folder that contains all the generated files"""
    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type](node)

        # get the path of the code generation directory if already set
        # check instance and check node attributes for value
        code_gen_dir = inst.code_gen_dir
        code_gen_dir = code_gen_dir.s.decode("UTF-8")

        # parameter is empty
        if not code_gen_dir:
            tmp_dir = tmp.mkdtemp(prefix="code_gen_" + str(node.op_type) + "_")
            inst.tmp_dir = tmp_dir
            inst.code_generation(context)
            # check if directory exists
            if os.path.isdir(tmp_dir):
                if len(os.listdir(tmp_dir)) == 0:
                    raise Exception("Code was not generated!")
                else:
                    inst.code_gen_dir = tmp_dir
                    for attribute in node.attribute:
                        if attribute.name == "code_gen_dir":
                            attribute.s = tmp_dir.encode("UTF-8")
            else:
                raise Exception("Code was not generated!")

        # there is already a code gen directory
        else:
            # check directory for files
            if os.path.isdir(code_gen_dir):
                if len(os.listdir(code_gen_dir)) == 0:
                    os.rmdir(code_gen_dir)
                    tmp_dir = tmp.mkdtemp(prefix="code_gen_" + str(node.op_type) + "_")
                    inst.tmp_dir = tmp_dir
                    inst.code_generation(context)
                    if os.path.isdir(tmp_dir):
                        if len(os.listdir(tmp_dir)) == 0:
                            raise Exception("Code was not generated!")
                        else:
                            inst.code_gen_dir = tmp_dir
                            for attribute in node.attribute:
                                if attribute.name == "code_gen_dir":
                                    attribute.s = tmp_dir.encode("UTF-8")
                    else:
                        raise Exception("Code was not generated!")
            else:
                inst.code_gen_dir = tmp_dir
                for attribute in node.attribute:
                    if attribute.name == "code_gen_dir":
                        attribute.s = tmp_dir.encode("UTF-8")

    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)
