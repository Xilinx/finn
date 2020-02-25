import os

import finn.custom_op.registry as registry
from finn.transformation import Transformation
from finn.util.basic import get_by_name, make_build_dir


def _codegen_single_node(node, model, fpgapart, clk):
    """Calls C++ code generation for one node. Resulting code can be used 
    to generate a Vivado IP block for the node."""

    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type](node)
        # get the path of the code generation directory
        code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
        # ensure that there is a directory
        if code_gen_dir == "" or not os.path.isdir(code_gen_dir):
            code_gen_dir = make_build_dir(
                prefix="code_gen_ipgen_" + str(node.op_type) + "_"
            )
            inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        # ensure that there is generated code inside the dir
        inst.code_generation_ipgen(model, fpgapart, clk)
    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)


class CodeGen_ipgen(Transformation):
    """Call custom implementation to generate code for single custom node
    and create folder that contains all the generated files.
    All nodes in the graph must have the fpgadataflow backend attribute and 
    transformation gets additional arguments:
    
    * fpgapart (string)
    
    * clk in ns (int)

    Outcome if succesful: Node attribute "code_gen_dir_ipgen" contains path to folder 
    that contains generated C++ code that can be used to generate a Vivado IP block. 
    The subsequent transformation is HLSSynth_IPGen"""

    def __init__(self, fpgapart, clk):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk = clk

    def apply(self, model):
        for node in model.graph.node:
            if node.domain == "finn":
                backend_attribute = get_by_name(node.attribute, "backend")
                if backend_attribute is None:
                    continue
                backend_value = backend_attribute.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    _codegen_single_node(node, model, self.fpgapart, self.clk)
        return (model, False)
