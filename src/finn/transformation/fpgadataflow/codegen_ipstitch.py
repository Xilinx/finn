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

    def __init__(self, fpgapart):
        super().__init__()
        self.fpgapart = fpgapart

    def apply(self, model):
        ip_dirs = ["list"]
        create_cmds = []
        connect_cmds = []
        # ensure that all nodes are fpgadataflow, and that IPs are generated
        for node in model.graph.node:
            assert node.domain == "finn"
            backend_attribute = get_by_name(node.attribute, "backend")
            assert backend_attribute is not None
            backend_value = backend_attribute.s.decode("UTF-8")
            assert backend_value == "fpgadataflow"
            ip_dir_attribute = get_by_name(node.attribute, "ipgen_path")
            assert ip_dir_attribute is not None
            ip_dir_value = ip_dir_attribute.s.decode("UTF-8")
            ip_dir_value += "/sol1/impl/ip"
            assert os.path.isdir(ip_dir_value)
            ip_dirs += [ip_dir_value]
            vlnv = "xilinx.com:hls:%s:1.0" % node.name
            inst_name = node.name
            create_cmd = "create_bd_cell -type ip -vlnv %s %s" % (vlnv, inst_name)
            create_cmds += [create_cmd]
            # TODO nonlinear topologies: check this for all inputs
            my_producer = model.find_producer(node.input[0])
            if my_producer is None:
                # first node in graph
                # make clock and reset external
                connect_cmds.append(
                    "make_bd_pins_external [get_bd_pins %s/ap_clk]" % inst_name
                )
                connect_cmds.append(
                    "make_bd_pins_external [get_bd_pins %s/ap_rst_n]" % inst_name
                )
                # make input external
                connect_cmds.append(
                    "make_bd_intf_pins_external [get_bd_intf_pins %s/in0_V_V]"
                    % inst_name
                )
            else:
                # intermediate node
                # wire up global clock and reset
                connect_cmds.append(
                    "connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins %s/ap_rst_n]"
                    % inst_name
                )
                connect_cmds.append(
                    "connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins %s/ap_clk]"
                    % inst_name
                )
                # wire up input to previous output
                # TODO nonlinear topologies: loop over all inputs
                my_in_name = "%s/in0_V_V" % (inst_name)
                prev_out_name = "%s/out_V_V" % (my_producer.name)
                connect_cmds.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s] [get_bd_intf_pins %s]"
                    % (prev_out_name, my_in_name)
                )
            if model.find_consumer(node.output[0]) is None:
                # last node in graph
                # connect prev output to input
                # make output external
                connect_cmds.append(
                    "make_bd_intf_pins_external [get_bd_intf_pins %s/out_V_V]"
                    % inst_name
                )

        # create a temporary folder for the project
        vivado_proj = get_by_name(model.model.metadata_props, "vivado_proj", "key")
        if vivado_proj is None or not os.path.isdir(vivado_proj.value):
            vivado_proj = onnx.StringStringEntryProto()
            vivado_proj.key = "vivado_proj"
            vivado_proj.value = tmp.mkdtemp(prefix="vivado_proj_")
            model.model.metadata_props.append(vivado_proj)
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
        tcl.extend(connect_cmds)
        tcl.append("regenerate_bd_layout")
        tcl.append("validate_bd_design")
        tcl.append("save_bd_design")
        # TODO connect streams between layers
        # TODO connect clock and reset to external port
        # TODO expose first in and last out
        tcl_string = "\n".join(tcl) + "\n"
        with open(vivado_proj_dir + "/make_project.tcl", "w") as f:
            f.write(tcl_string)
        return (model, False)
