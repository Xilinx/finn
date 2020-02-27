import os

import finn.custom_op.registry as registry
import finn.util.basic as util
from finn.transformation import Transformation


class ReplaceVerilogRelPaths(Transformation):
    """Convert ./ relative file paths to absolute ones for generated Verilog"""

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
                        # find the IP gen dir
                        ipgen_path = inst.get_nodeattr("ipgen_path")
                        if ipgen_path is not None and os.path.isdir(ipgen_path):
                            for dname, dirs, files in os.walk(ipgen_path):
                                for fname in files:
                                    if fname.endswith(".v"):
                                        fpath = os.path.join(dname, fname)
                                        with open(fpath, "r") as f:
                                            s = f.read()
                                        old = '$readmemh(".'
                                        new = '$readmemh("%s' % dname
                                        s = s.replace(old, new)
                                        with open(fpath, "w") as f:
                                            f.write(s)
                    except KeyError:
                        pass
        return (model, False)
