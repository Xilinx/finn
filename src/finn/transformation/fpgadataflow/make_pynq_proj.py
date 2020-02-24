import os
import subprocess

from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation
from finn.util.basic import get_by_name, make_build_dir, roundup_to_integer_multiple

from . import templates


class MakePYNQProject(Transformation):
    """Create a Vivado PYNQ overlay project (including the shell infrastructure)
    from the already-stitched IP block for this graph.
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

        # extract HLSCustomOp instances to get i/o stream widths
        i_tensor_name = model.graph.input[0].name
        o_tensor_name = model.graph.output[0].name
        first_node = getCustomOp(model.find_consumer(i_tensor_name))
        last_node = getCustomOp(model.find_producer(o_tensor_name))
        i_bits_per_cycle = first_node.get_instream_width()
        o_bits_per_cycle = last_node.get_outstream_width()
        # ensure i/o is padded to bytes
        i_bits_per_cycle_padded = roundup_to_integer_multiple(i_bits_per_cycle, 8)
        o_bits_per_cycle_padded = roundup_to_integer_multiple(o_bits_per_cycle, 8)
        assert i_bits_per_cycle_padded % 8 == 0
        assert i_bits_per_cycle_padded % 8 == 0
        in_bytes = i_bits_per_cycle_padded / 8
        out_bytes = o_bits_per_cycle_padded / 8
        in_if_name = "in0_V_V_0"
        out_if_name = "out_r_0"
        clk_name = "ap_clk_0"
        nrst_name = "ap_rst_n_0"
        vivado_ip_cache = os.getenv("VIVADO_IP_CACHE", default="")

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_pynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        ip_config_tcl = templates.ip_config_tcl_template % (
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
            vivado_ip_cache,
        )

        with open(vivado_pynq_proj_dir + "/ip_config.tcl", "w") as f:
            f.write(ip_config_tcl)
        # create a shell script for project creation and synthesis
        make_project_sh = vivado_pynq_proj_dir + "/make_project.sh"
        working_dir = os.environ["PWD"]
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        with open(make_project_sh, "w") as f:
            f.write(
                templates.call_pynqshell_makefile_template
                % (pynq_shell_path, self.platform, ipcfg, "block_design", working_dir)
            )
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        with open(synth_project_sh, "w") as f:
            f.write(
                templates.call_pynqshell_makefile_template
                % (pynq_shell_path, self.platform, ipcfg, "bitstream", working_dir)
            )
        # call the project creation script
        # synthesis script will be called with a separate transformation
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        return (model, False)
