# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import onnx.helper as helper
import os
import shutil
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


class TLastMarker_rtl(HLSCustomOp):
    """Node that adds AXI stream TLAST signals where needed. Its behavior
    is transparent in node-by-node execution, only visible in IP-stitched rtlsim or
    actual hardware. Number of transactions is configurable at runtime
    in this variant, and the configured number of transactions will
    persist after reset."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # number of (default) iterations until TLAST=1 is generated
            # can be reconfigured at runtime by writing the AXI lite regs
            "NumIters": ("i", True, 0),
            # width of input-output data streams, in bits
            "StreamWidth": ("i", True, 0),
            # width of individual element in stream, in bits
            "ElemWidth": ("i", True, 0),
            # factor to multiply spatial size with to get NumIters
            # only needed for runtime dynamic reconfig of FM sizes
            "SpatialSizeToIters": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # shape of input/output tensors
            "shape": ("ints", True, []),
            # attribute to save top module name - not user configurable
            "gen_top_module": ("s", False, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return tuple(self.get_nodeattr("shape"))

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        return ret

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def execute_node(self, context, graph):
        # TLastMarker's behavior is only visible when doing
        # rtlsim with stitched IP, since it marks the end
        # of the current image/input sample. when executing
        # inside FINN as a single node, this is not visible.
        # so here we simply return the input as output
        i_name = self.onnx_node.input[0]
        o_name = self.onnx_node.output[0]
        i_tensor = context[i_name]
        context[o_name] = i_tensor

    def get_number_output_values(self):
        return self.get_nodeattr("NumIters")

    def get_folded_input_shape(self, ind=0):
        stream_width = self.get_nodeattr("StreamWidth")
        elem_width = self.get_nodeattr("ElemWidth")
        n_packed_elems = stream_width // elem_width
        n_iters = self.get_nodeattr("NumIters")
        return (1, n_iters, n_packed_elems)

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def get_instream_width(self, ind=0):
        stream_width = self.get_nodeattr("StreamWidth")
        return stream_width

    def get_outstream_width(self, ind=0):
        stream_width = self.get_nodeattr("StreamWidth")
        return stream_width

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        stream_width = self.get_nodeattr("StreamWidth")
        intf_names["s_axis"] = [("in0_V", stream_width)]
        intf_names["m_axis"] = [("out_V", stream_width)]
        intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def make_shape_compatible_op(self, model):
        return helper.make_node(
            "Identity",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
        )

    def infer_node_datatype(self, model):
        # TLastMarker does not change datatype
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_template_values(self):
        stream_width = self.get_instream_width_padded()
        period_init = self.get_nodeattr("NumIters")
        # use 32 bits to cover for large number of stream bits
        period_bits = 32
        # always keep the period upon reset for now
        period_init_upon_reset = 0
        topname = self.get_verilog_top_module_name()

        code_gen_dict = {
            "DATA_WIDTH": int(stream_width),
            "PERIOD_BITS": int(period_bits),
            "PERIOD_INIT": int(period_init),
            "PERIOD_INIT_UPON_RESET": int(period_init_upon_reset),
            "TOP_MODULE_NAME": topname,
        }
        return code_gen_dict

    def get_dynamic_config_ccode(self):
        """Returns C code to generate register values to re-configure FM dimension.
        at runtime."""

        reg_ccode_template = """
void reconfigure_$LAYERNAME$(
    // base address for TLastMarker_rtl AXI lite interface
    unsigned int *reg_base,
    // spatial dimensions for input
    // dimY = height, dimX = width
    unsigned int dimY, unsigned int dimX
) {
    reg_base[0] = (dimY*dimX) * $SPATIAL_SIZE_TO_ITERS;
}
"""
        spatialSizeToIters = self.get_nodeattr("SpatialSizeToIters")
        layer_name = self.onnx_node.name
        reg_ccode = reg_ccode_template
        reg_ccode = reg_ccode.replace("$LAYERNAME$", layer_name)
        reg_ccode = reg_ccode.replace("$SPATIAL_SIZE_TO_ITERS$", str(spatialSizeToIters))

        return reg_ccode

    def get_dynamic_config(self, ifm_dims):
        """Returns a configuration dict to re-configure FM dimension
        during runtime."""

        spatial_dim = np.prod(ifm_dims)
        period = spatial_dim * self.get_nodeattr("SpatialSizeToIters")
        config = {
            "PERIOD": (0 * 4, period),
        }
        return config

    def generate_hdl(self):
        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/tlast_marker/hdl"
        template_path = rtlsrc + "/tlast_marker_wrapper.v"
        code_gen_dict = self.get_template_values()
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(
            os.path.join(code_gen_dir, self.get_verilog_top_module_name() + ".v"),
            "w",
        ) as f:
            f.write(template)

        sv_files = ["tlast_marker.sv"]
        for sv_file in sv_files:
            shutil.copy(rtlsrc + "/" + sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_all_verilog_paths(self):
        "Return list of all folders containing Verilog code for this node."

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]
        return verilog_paths

    def get_verilog_top_filename(self):
        "Return the Verilog top module filename for this node."

        verilog_file = "{}/{}.v".format(
            self.get_nodeattr("code_gen_dir_ipgen"), self.get_nodeattr("gen_top_module")
        )
        return verilog_file

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""
        # Modified to use generated (System-)Verilog instead of HLS output products

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]
        verilog_files = [
            "tlast_marker.sv",
            self.get_nodeattr("gen_top_module") + ".v",
        ]

        # build the Verilator emu library
        sim = PyVerilator.build(
            verilog_files,
            auto_eval=False,
            build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_verilog_top_module_name(),
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return sim

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "tlast_marker.sv",
            self.get_nodeattr("gen_top_module") + ".v",
        ]

        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]
        for f in sourcefiles:
            cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Normally: Generates C++ code and tcl script for IP generation.
        Here: Generates (System-)Verilog code for IP generation."""
        self.generate_hdl()

    def ipgen_singlenode_code(self):
        """Normally: Builds the bash script for IP generation."""
        pass

    def code_generation_cppsim(self, model):
        """Normally: Generates C++ code for simulation (cppsim)."""
        pass

    def compile_singlenode_code(self):
        pass

    def global_includes(self):
        pass

    def defines(self, var):
        pass

    def read_npy_data(self):
        pass

    def strm_decl(self):
        pass

    def docompute(self):
        pass

    def dataoutstrm(self):
        pass

    def save_as_npy(self):
        pass

    def blackboxfunction(self):
        pass

    def pragmas(self):
        pass

    def verify_node(self):
        pass
