import finn.custom_op.registry as registry
import finn.util.basic as util
from finn.transformation import Transformation


class Compile(Transformation):
    """For every node: compile C++ code in node attribute "code_gen_dir_npysim" 
    and save path to executables in node attribute "executable_path".
    All nodes in the graph must have the fpgadataflow backend attribute.

    To use these executables, exec_mode must be set to "npysim" (using transformation 
    SetExecMode) and the model has to be executed using execute_onnx() from 
    finn.core.onnx_exec"""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        for node in model.graph.node:
            op_type = node.op_type
            if node.domain == "finn":
                backend_attribute = util.get_by_name(node.attribute, "backend")
                if backend_attribute is None:
                    continue
                backend_value = backend_attribute.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    try:
                        # lookup op_type in registry of CustomOps
                        inst = registry.custom_op[op_type](node)
                        # ensure that code is generated
                        assert inst.get_nodeattr("code_gen_dir_npysim") != ""
                        # call the compilation function for this node
                        inst.compile_singlenode_code()
                        # ensure that executable path is now set
                        assert inst.get_nodeattr("executable_path") != ""
                    except KeyError:
                        # exception if op_type is not supported
                        raise Exception(
                            "Custom op_type %s is currently not supported." % op_type
                        )
        return (model, False)
