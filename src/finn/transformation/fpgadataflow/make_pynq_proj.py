import os
import subprocess

import numpy as np

from finn.core.utils import get_by_name, make_build_dir
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
            raise Exception(
                "No stitched IPI design found, apply CodeGen_ipstitch first."
            )
        vivado_stitch_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
        if vivado_stitch_vlnv is None:
            raise Exception(
                "No vlnv for stitched IP found, apply CodeGen_ipstitch first."
            )

        # collect list of all IP dirs
        ip_dirs = ["list"]
        for node in model.graph.node:
            ip_dir_attribute = get_by_name(node.attribute, "ipgen_path")
            assert ip_dir_attribute is not None
            ip_dir_value = ip_dir_attribute.s.decode("UTF-8")
            ip_dir_value += "/sol1/impl/ip"
            assert os.path.isdir(ip_dir_value)
            ip_dirs += [ip_dir_value]
        ip_dirs += [ipstitch_path + "/ip"]
        ip_dirs_str = "[%s]" % (" ".join(ip_dirs))

        # extract the actual in-out bytes from graph
        i_tensor_name = model.graph.input[0].name
        o_tensor_name = model.graph.output[0].name
        i_tensor_shape = model.get_tensor_shape(i_tensor_name)
        o_tensor_shape = model.get_tensor_shape(o_tensor_name)
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        i_bits = i_tensor_dt.bitwidth() * np.prod(i_tensor_shape)
        o_bits = o_tensor_dt.bitwidth() * np.prod(o_tensor_shape)
        # ensure i/o is padded to bytes
        assert i_bits % 8 == 0
        assert o_bits % 8 == 0
        in_bytes = i_bits / 8
        out_bytes = o_bits / 8
        in_if_name = "in0_V_V_0"
        out_if_name = "out_r_0"
        clk_name = "ap_clk_0"
        nrst_name = "ap_rst_n_0"

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_pynq_proj_")
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
variable config_output_products_dir

# for arguments involving paths below: use absolute paths or relative to the
# platform/overlay/bitstream folder
# where to create the project
set config_ip_project_dir %s
# IP repositories that the project depends on
set config_ip_repo %s
# where the produced bitfile and .hwh file will be placed
set config_output_products_dir %s

# non-path arguments
# VLNV of the IP block
set config_ip_vlnv %s
# width of the AXI stream into the IP, in bytes
set config_ip_bytes_in %d
# width of the AXI stream out of the IP, in bytes
set config_ip_bytes_out %d
# the name of the input AXI stream interface
set config_ip_axis_name_in %s
# the name of the output AXI stream interface
set config_ip_axis_name_out %s
# the name of the clock signal
set config_ip_clk_name %s
# the name of the active-low reset signal
set config_ip_nrst_name %s
# whether the IP needs an AXI Lite interface for control
set config_ip_use_axilite 0
        """ % (
            vivado_pynq_proj_dir,
            ip_dirs_str,
            vivado_pynq_proj_dir,
            vivado_stitch_vlnv,
            in_bytes,
            out_bytes,
            in_if_name,
            out_if_name,
            clk_name,
            nrst_name,
        )

        with open(vivado_pynq_proj_dir + "/ip_config.tcl", "w") as f:
            f.write(ip_config_tcl)
        # create a shell script for project creation and synthesis
        make_project_sh = vivado_pynq_proj_dir + "/make_project.sh"
        working_dir = os.environ["PWD"]
        with open(make_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(pynq_shell_path))
            f.write("export platform=%s\n" % (self.platform))
            f.write("export ip_config=%s\n" % (vivado_pynq_proj_dir + "/ip_config.tcl"))
            f.write("make block_design\n")
            f.write("cd {}\n".format(working_dir))
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        with open(synth_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(pynq_shell_path))
            f.write("export platform=%s\n" % (self.platform))
            f.write("export ip_config=%s\n" % (vivado_pynq_proj_dir + "/ip_config.tcl"))
            f.write("make bitstream\n")
            f.write("cd {}\n".format(working_dir))
        # call the project creation script
        # synthesis script will be called with a separate transformation
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        return (model, False)
