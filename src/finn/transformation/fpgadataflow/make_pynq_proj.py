import os
import subprocess
import tempfile as tmp

from finn.core.utils import get_by_name
from finn.transformation import Transformation


class MakePYNQProject(Transformation):
    """Create a PYNQ project (including the shell infrastructure) from the
    already-stitched IP block for this graph.
    All nodes in the graph must have the fpgadataflow backend attribute,
    and the CodeGen_ipstitch transformation must have been previously run on
    the graph.

    Outcome if successful: sets the vivado_pynq_proj attribute in the ONNX
    ModelProto's metadata_props field, with the created project dir as the
    value.
    """

    def __init__(self, platform):
        super().__init__()
        self.platform = platform

    def apply(self, model):
        pynq_shell_path = os.environ["PYNQSHELL_PATH"]
        if not os.path.isdir(pynq_shell_path):
            raise Exception("Ensure the PYNQ-HelloWorld utility repo is cloned.")
        ipstitch_path = model.get_metadata_prop("vivado_stitch_proj")
        if ipstitch_path is None or (not os.path.isdir(ipstitch_path)):
            raise Exception("No stitched IPI design found, apply CodeGen_ipstitch first.")

        # TODO extract the actual in-out bytes from graph
        in_bytes = 8
        out_bytes = 8
        in_if_name = "in0_V_V"
        out_if_name = "out_V_V"

        # create a temporary folder for the project
        vivado_pynq_proj_dir = tmp.mkdtemp(prefix="vivado_pynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        ip_config_tcl = """
variable config_ip_repo
variable config_ip_vlnv
variable config_ip_bytes_in
variable config_ip_bytes_out
variable config_ip_axis_name_in
variable config_ip_axis_name_out
variable config_ip_use_axilite
variable config_ip_project_dir

# for arguments involving paths below: use absolute paths or relative to the
# platform/overlay/bitstream folder
# where to create the project
set config_ip_project_dir %s
# IP repositories that the project depends on
set config_ip_repo %s

# non-path arguments
# VLNV of the IP block
set config_ip_vlnv xilinx.com:hls:resize_accel:1.0
# width of the AXI stream into the IP, in bytes
set config_ip_bytes_in %d
# width of the AXI stream out of the IP, in bytes
set config_ip_bytes_out %d
# the name of the input AXI stream interface
set config_ip_axis_name_in %s
# the name of the output AXI stream interface
set config_ip_axis_name_out %s
# whether the IP needs an AXI Lite interface for control
set config_ip_use_axilite 0
        """ % (
            vivado_pynq_proj_dir, ipstitch_path + "/ip", in_bytes, out_bytes,
            in_if_name, out_if_name
        )

        with open(vivado_pynq_proj_dir + "/ip_config.tcl", "w") as f:
            f.write(ip_config_tcl)
        # create a shell script and call Vivado
        make_project_sh = vivado_pynq_proj_dir + "/make_project.sh"
        working_dir = os.environ["PWD"]
        with open(make_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(pynq_shell_path))
            f.write("export platform=%s\n" % (self.platform))
            f.write("export ip_config=%s\n" % (vivado_pynq_proj_dir + "/ip_config.tcl"))
            f.write("make block_design\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        return (model, False)
