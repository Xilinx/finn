import subprocess

from qonnx.transformation.base import Transformation

class SetupACCLInterface(Transformation):
    def __init__(self, ip_name="finn_design"):
        self.ip_name = ip_name

    def apply(self, model):
        vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        prjname = "finn_vivado_stitch_proj"

        tcl = []
        tcl.append(f"open_project {vivado_stitch_proj_dir}/{prjname}.xpr")

        # TODO: Maybe we can avoid writing out the full path out here, seems a bit brittle
        tcl.append(f"open_bd_design {vivado_stitch_proj_dir}/{prjname}.srcs/sources_1/bd/{self.ip_name}/{self.ip_name}.bd")

        has_accl_input = bool(model.get_metadata_prop("has_accl_input"))
        has_accl_output = bool(model.get_metadata_prop("has_accl_output"))

        if has_accl_input:
            tcl.append('set_property name data_from_cclo [get_bd_intf_ports s_axis_0]')
            tcl.append('create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_0')
        else:
            tcl.append('create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 data_from_cclo')

        graph_out_names = [x.name for x in model.graph.output]
        assert len(graph_out_names) == 1, "Expected only one output at this point"
        final_node = model.find_producer(graph_out_names[0])

        if has_accl_output:
            tcl.append('set_property name data_to_cclo [get_bd_intf_ports m_axis_0]')
            tcl.append('create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_0')
            tcl += [
                'make_bd_intf_pins_external [get_bd_intf_pins {}/{}]'.format(
                    final_node.name,
                    pin_name
                )
                for pin_name in ["cmd_to_cclo", "sts_from_cclo"]
            ]
        else:
            tcl.append('create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 cmd_to_cclo')
            tcl.append('create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 sts_from_cclo')
            tcl.append('create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 data_to_cclo')

        tcl.append('save_bd_design')

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

