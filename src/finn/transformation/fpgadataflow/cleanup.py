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
        # delete PYNQ project, if any
        vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
        if vivado_pynq_proj_dir is not None and os.path.isdir(vivado_pynq_proj_dir):
            shutil.rmtree(vivado_pynq_proj_dir)
        model.set_metadata_prop("vivado_pynq_proj", "")
        # delete IP stitching project, if any
        ipstitch_path = model.get_metadata_prop("vivado_stitch_proj")
        if ipstitch_path is not None and os.path.isdir(ipstitch_path):
            shutil.rmtree(ipstitch_path)
        model.set_metadata_prop("vivado_stitch_proj", "")
        for node in model.graph.node:
            op_type = node.op_type
            if node.domain == "finn":
                backend_attribute = util.get_by_name(node.attribute, "backend")
                backend_value = backend_attribute.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    try:
                        # lookup op_type in registry of CustomOps
                        inst = registry.custom_op[op_type](node)
                        # delete code_gen_dir from npysim
                        code_gen_dir = inst.get_nodeattr("code_gen_dir_npysim")
                        if os.path.isdir(code_gen_dir):
                            shutil.rmtree(code_gen_dir)
                        inst.set_nodeattr("code_gen_dir_npysim", "")
                        inst.set_nodeattr("executable_path", "")
                        # delete code_gen_dir from ipgen and project folder
                        code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
                        ipgen_path = inst.get_nodeattr("ipgen_path")
                        if os.path.isdir(code_gen_dir):
                            shutil.rmtree(code_gen_dir)
                        if os.path.isdir(ipgen_path):
                            shutil.rmtree(ipgen_path)
                        inst.set_nodeattr("code_gen_dir_ipgen", "")
                        inst.set_nodeattr("ipgen_path", "")
                        # delete Java HotSpot Performance data log
                        for d_name in os.listdir("/tmp/"):
                            if "hsperfdata" in d_name:
                                shutil.rmtree("/tmp/" + str(d_name))

                    except KeyError:
                        # exception if op_type is not supported
                        raise Exception(
                            "Custom op_type %s is currently not supported." % op_type
                        )
        return (model, False)
