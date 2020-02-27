import finn.custom_op.registry as registry
import finn.util.basic as util
from finn.transformation import Transformation


class HLSSynth_IPGen(Transformation):
    """For each node: generate IP block from code in folder
    that is referenced in node attribute "code_gen_dir_ipgen"
    and save path of generated project in node attribute "ipgen_path".
    All nodes in the graph must have the fpgadataflow backend attribute.

    This transformation calls Vivado HLS for synthesis, so it will run for 
    some time (several minutes)"""


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
                        assert inst.get_nodeattr("code_gen_dir_ipgen") != "", """Node 
                        attribute "code_gen_dir_ipgen" is empty. Please run 
                        transformation CodeGen_ipgen first."""
                        # call the compilation function for this node
                        inst.ipgen_singlenode_code()
                        # ensure that executable path is now set
                        assert inst.get_nodeattr("ipgen_path") != "", """Transformation
                        HLSSynth_IPGen was not successful. Node attribute "ipgen_path" 
                        is empty."""
                    except KeyError:
                        # exception if op_type is not supported
                        raise Exception(
                            "Custom op_type %s is currently not supported." % op_type
                        )
        return (model, False)
