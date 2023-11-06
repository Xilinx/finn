import subprocess

from qonnx.transformation.base import Transformation
from distutils.dir_util import copy_tree

from finn.util.basic import make_build_dir

class SetupACCLInterface(Transformation):
    def __init__(self, ip_name="finn_design"):
        self.ip_name = ip_name

    def apply(self, model):
        vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        project_dir = make_build_dir("accl_ip")
        model.set_metadata_prop("accl_ip_dir", project_dir)
        copy_tree(vivado_stitch_proj_dir, project_dir)

        prjname = "finn_vivado_stitch_proj"

        tcl = []
        tcl.append(f"open_project {project_dir}/{prjname}.xpr")

        # TODO: Maybe we can avoid writing out the full path out here, seems a bit brittle
        tcl.append(f"open_bd_design {project_dir}/{prjname}.srcs/sources_1/bd/{self.ip_name}/{self.ip_name}.bd")

        has_accl_in = any(node.op_type == "ACCLIn" for node in model.graph.node)

        if has_accl_in:
            tcl.append("set_property name data_from_cclo [get_bd_intf_ports s_axis_0]")
            tcl.append("create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_0")
        else:
            tcl.append("create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 data_from_cclo")

        accl_out_node = None
        for node in model.graph.node:
            if node.op_type == "ACCLOut":
                accl_out_node = node
                break

        if accl_out_node is not None:
            tcl.append("set_property name data_to_cclo [get_bd_intf_ports m_axis_0]")
            tcl.append("create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_0")

            # TODO: In a case where we have multiple nodes that access this interface we
            # need to add an arbiter for these and the data streams.
            tcl += [
                "make_bd_intf_pins_external [get_bd_intf_pins {}/{}]".format(
                    accl_out_node.name,
                    pin_name
                )
                for pin_name in ["cmd_to_cclo", "sts_from_cclo", "s_axi_control"]
            ]

            tcl.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0")
            tcl.append("connect_bd_net [get_bd_pins xlconstant_0/dout] [get_bd_pins {}/wait_for_ack]".format(accl_out_node.name))
        else:
            tcl.append("create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 cmd_to_cclo")
            tcl.append("create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 sts_from_cclo")
            tcl.append("create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 data_to_cclo")

        tcl.append("save_bd_design")

        tcl_string = "\n".join(tcl) + "\n"

        tcl_file = vivado_stitch_proj_dir + "/setup_accl_interface.tcl"
        with open(tcl_file, "w") as f:
            f.write(tcl_string)

        subprocess.run([
            "vivado",
            "-mode", "batch",
            "-source", tcl_file,
        ])

        return model, False

