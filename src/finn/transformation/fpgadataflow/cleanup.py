import os
import shutil

import finn.core.utils as util
import finn.custom_op.registry as registry
from finn.transformation import Transformation


class CleanUp(Transformation):
    """Remove any generated files for fpgadataflow nodes."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        for node in model.graph.node:
            op_type = node.op_type
            if node.domain == "finn":
                backend_attribute = util.get_by_name(node.attribute, "backend")
                backend_value = backend_attribute.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    try:
                        # lookup op_type in registry of CustomOps
                        inst = registry.custom_op[op_type](node)
                        code_gen_dir = inst.get_nodeattr("code_gen_dir_npysim")
                        if os.path.isdir(code_gen_dir):
                            shutil.rmtree(code_gen_dir)
                        inst.set_nodeattr("code_gen_dir_npysim", "")
                        inst.set_nodeattr("executable_path", "")
                    except KeyError:
                        # exception if op_type is not supported
                        raise Exception(
                            "Custom op_type %s is currently not supported." % op_type
                        )
        return (model, False)
