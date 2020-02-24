import finn.custom_op.registry as registry
import finn.util.basic as util
from finn.transformation import Transformation


class SetExecMode(Transformation):
    """Set attribute exec_mode in all fpgadataflow nodes to specify which 
    kind of execution should be used ("npysim" or "rtlsim")"""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

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
                        # set sim_mode accordingly to argument mode
                        inst.set_nodeattr("exec_mode", self.mode)
                        # ensure that sim_mode is now set
                        assert inst.get_nodeattr("exec_mode") != ""
                    except KeyError:
                        # exception if op_type is not supported
                        raise Exception(
                            "Custom op_type %s is currently not supported." % op_type
                        )
        return (model, False)
