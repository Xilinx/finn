import os
import tempfile as tmp

import onnx

from finn.core.utils import get_by_name
from finn.transformation import Transformation


class CodeGen_ipstitch(Transformation):
    """Create a Vivado IP Block Design project from all the generated IPs of a
    graph. All nodes in the graph must have the fpgadataflow backend attribute,
    and the CodeGen_ipgen transformation must have been previously run on
    the graph."""

    def __init__(self, fpgapart, clk):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk = clk

    def apply(self, model):
        ip_dirs = []
        create_cmds = []
        # ensure that all nodes are fpgadataflow, and that IPs are generated
        for node in model.graph.node:
            assert node.domain == "finn"
            backend_attribute = get_by_name(node.attribute, "backend")
            assert backend_attribute is not None
            backend_value = backend_attribute.s.decode("UTF-8")
            assert backend_value == "fpgadataflow"
            ip_dir_attribute = get_by_name(node.attribute, "code_gen_dir_ipgen")
            assert ip_dir_attribute is not None
            ip_dir_value = ip_dir_attribute.s.decode("UTF-8")
            # TODO check for file presence instead?
            assert ip_dir_value != ""
            prj_name = "project_{}".format(node.name)
            ip_dirs += [ip_dir_value + "/%s/sol1/impl/ip" % prj_name]
            vlnv = "xilinx.com:hls:%s:1.0" % node.name
            inst_name = node.name
            create_cmd = "create_bd_cell -type ip -vlnv %s %s" % (vlnv, inst_name)
            create_cmds += [create_cmd]
        # create a temporary folder for the project
        vivado_proj = get_by_name(model.model.metadata_props, "vivado_proj", "key")
        if vivado_proj is None:
            vivado_proj = onnx.StringStringEntryProto()
            vivado_proj.key = "vivado_proj"
            vivado_proj.value = ""
            model.model.metadata_props.append(vivado_proj)
        if not os.path.isdir(vivado_proj.value):
            vivado_proj.value = tmp.mkdtemp(prefix="vivado_proj_")
        vivado_proj_dir = vivado_proj.value
        # start building the tcl script
        tcl = []
        # create vivado project
        tcl.append(
            "create_project %s %s -part %s"
            % ("finn_vivado_proj", vivado_proj_dir, self.fpgapart)
        )
        # add all the generated IP dirs to ip_repo_paths
        ip_dirs_str = " ".join(ip_dirs)
        tcl.append("set_property ip_repo_paths [%s] [current_project]" % ip_dirs_str)
        tcl.append("update_ip_catalog")
        # create block design and instantiate all layers
        tcl.append('create_bd_design "finn_design"')
        tcl.extend(create_cmds)
        # TODO connect streams between layers
        # TODO connect clock and reset to external port
        # TODO expose first in and last out
        return (model, False)
