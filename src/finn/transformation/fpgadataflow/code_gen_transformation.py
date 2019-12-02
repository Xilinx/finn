import os
import tempfile as tmp

import finn.custom_op.registry as registry
from finn.transformation import Transformation
from finn.core.utils import get_by_name


def code_gen_transformation(node, model):
    """Call custom implementation to generate code for single custom node
    and create folder that contains all the generated files"""
    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type](node)

        # get the path of the code generation directory
        code_gen_dir = inst.code_gen_dir
        code_gen_dir = code_gen_dir.s.decode("UTF-8")

        # parameter is empty
        if not code_gen_dir:
            tmp_dir = tmp.mkdtemp(prefix="code_gen_" + str(node.op_type) + "_")
            inst.tmp_dir = tmp_dir
            inst.code_generation(model)
            # check if directory exists
            if os.path.isdir(tmp_dir):
                if len(os.listdir(tmp_dir)) == 0:
                    raise Exception("Code was not generated!")
                else:
                    inst.code_gen_dir = tmp_dir
                    model.set_attribute(node, "code_gen_dir", tmp_dir)
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
                    inst.code_generation(model)
                    if os.path.isdir(tmp_dir):
                        if len(os.listdir(tmp_dir)) == 0:
                            raise Exception("Code was not generated!")
                        else:
                            inst.code_gen_dir = tmp_dir
                            model.set_attribute(node, "code_gen_dir", tmp_dir)
                    else:
                        raise Exception("Code was not generated!")
                # else: attribute is correctly set
            else:
                inst.code_gen_dir = tmp_dir
                model.set_attribute(node, "code_gen_dir", tmp_dir)

    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)


class CodeGen(Transformation):
    """Code generation for all nodes in model"""

    def apply(self, model):
        for node in model.graph.node:
            if node.domain == "finn":
                backend_attribute = get_by_name(node.attribute, "backend")
                backend_value = backend_attribute.s.decode('UTF-8')
                if backend_value == "fpgadataflow":
                    code_gen_transformation(node, model)
        return (model, False)
