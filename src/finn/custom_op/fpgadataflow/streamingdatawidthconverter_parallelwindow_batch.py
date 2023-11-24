# Copyright (c) 2020, Xilinx
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
# * Neither the name of FINN nor the names of its
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
import os
import shutil
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None

# does not do anything at the ONNX node-by-node level, and input-output
# tensor shapes are the same. performs data width conversion at the rtlsim level


class StreamingDataWidthConverter_rtl(HLSCustomOp):
    """Class that corresponds to finn-hlslib StreamingDataWidthConverter_ParallelWindow_Batch
    function. To be inserted between an RTL-SWG with parallel window mode enabled and a
    VVU."""

    def get_nodeattr_types(self):
        my_attrs = {
            # shape of input/output tensors
            "shape": ("ints", True, []),
            # bit width of input and output streams
            "inWidth": ("i", True, 0),
            "outWidth": ("i", True, 0),
            # FINN DataTypes for inputs/outputs
            "dataType": ("s", True, ""),
            "SIMD": ("i", True, 0),
            "PE": ("i", True, 0),
            "Channels": ("i", True, 0),
            "Kernel": ("ints", True, []),
            "Mode": ("s", False, ""),
            # attribute to save top module name - not user configurable
            "gen_top_module": ("s", False, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("dataType")]

    def get_normal_input_shape(self, ind=0):
        ishape = self.get_nodeattr("shape")
        return ishape

    def get_normal_output_shape(self, ind=0):
        oshape = self.get_nodeattr("shape")
        return oshape

    def get_folded_input_shape(self, ind=0):
        iwidth = self.get_nodeattr("inWidth")
        ishape = self.get_normal_input_shape()
        dummy_t = np.random.randn(*ishape)
        ibits = self.get_input_datatype().bitwidth()
        assert (
            iwidth % ibits == 0
        ), """DWC input width must be divisible by
        input element bitwidth"""
        ielems = int(iwidth // ibits)
        ichannels = ishape[-1]
        new_shape = []
        for i in ishape[:-1]:
            new_shape.append(i)
        new_shape.append(int(ichannels // ielems))
        new_shape.append(ielems)
        dummy_t = dummy_t.reshape(new_shape)
        return dummy_t.shape

    def get_folded_output_shape(self, ind=0):
        owidth = self.get_nodeattr("outWidth")
        oshape = self.get_normal_output_shape()
        dummy_t = np.random.randn(*oshape)
        obits = self.get_output_datatype().bitwidth()
        assert (
            owidth % obits == 0
        ), """DWC output width must be divisible by
        input element bitwidth"""
        oelems = int(owidth // obits)
        ochannels = oshape[-1]
        new_shape = []
        for i in oshape[:-1]:
            new_shape.append(i)
        new_shape.append(int(ochannels // oelems))
        new_shape.append(oelems)
        dummy_t = dummy_t.reshape(new_shape)

        return dummy_t.shape

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def get_instream_width(self, ind=0):
        in_width = self.get_nodeattr("inWidth")
        return in_width

    def get_outstream_width(self, ind=0):
        out_width = self.get_nodeattr("outWidth")
        return out_width

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == tuple(exp_ishape), "Unexpect input shape for StreamingDWC."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""StreamingDWC needs 1 data input""")

        return info_messages

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_shape = self.get_normal_input_shape()
        folded_ishape = self.get_folded_input_shape()

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert inp.shape == tuple(exp_shape), "Input shape does not match expected shape."

        if self.get_input_datatype() == DataType["BIPOLAR"]:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
            export_idt = DataType["BINARY"]
        else:
            export_idt = self.get_input_datatype()
        # reshape input into folded shape
        reshaped_input = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = reshaped_input.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            output = inp
            output = np.asarray([output], dtype=np.float32).reshape(*exp_shape)
            context[node.output[0]] = output

        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
            odt = export_idt
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )
            # load and reshape output
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(exp_shape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to "rtlsim" """.format(
                    mode
                )
            )
        # binary -> bipolar if needed
        if self.get_output_datatype() == DataType["BIPOLAR"]:
            out = context[node.output[0]]
            out = 2 * out - 1
            context[node.output[0]] = out
        assert context[node.output[0]].shape == tuple(
            exp_shape
        ), """Output
        shape doesn't match expected shape, should be same as input shape"""

    def prepare_codegen_default(self):
        template_path = os.environ["FINN_ROOT"] + "/finn-rtllib/dwc/dwc_axi_wrapper.v"

        code_gen_dict = {}
        code_gen_dict["$IN_WIDTH$"] = [str(self.get_nodeattr("inWidth"))]
        code_gen_dict["$OUT_WIDTH$"] = [str(self.get_nodeattr("outWidth"))]
        code_gen_dict["$SIMD$"] = [str(self.get_nodeattr("SIMD"))]
        code_gen_dict["$PE$"] = [str(self.get_nodeattr("PE"))]
        code_gen_dict["$CHANNELS$"] = [str(self.get_nodeattr("Channels"))]
        code_gen_dict["$KERNEL_PROD$"] = [str(np.prod(self.get_nodeattr("Kernel")))]
        code_gen_dict["$ACTIVATION_WIDTH$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$MODE$"] = [self.get_nodeattr("Mode")]

        return template_path, code_gen_dict

    def generate_hdl(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        template_path, code_gen_dict = self.prepare_codegen_default()
        # add general parameters to dictionary
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to template
        with open(template_path, "r") as f:
            template_wrapper = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"), "w"
        ) as f:
            f.write(template_wrapper)

        shutil.copy2(
            os.environ["FINN_ROOT"] + "/finn-rtllib/dwc/dwc_parallelwindow.sv", code_gen_dir
        )
        shutil.copy2(os.environ["FINN_ROOT"] + "/finn-rtllib/dwc/dwc_upsample.sv", code_gen_dir)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_all_verilog_paths(self):
        "Return list of all folders containing Verilog code for this node."

        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/dwc/")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        return [rtllib_dir, code_gen_dir]

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/dwc/")
        source_files = [
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            rtllib_dir + "dwc_parallelwindow.sv",
            rtllib_dir + "dwc_upsample.sv",
        ]

        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]

        for f in source_files:
            cmd.append("add_files -copy_to %s -norecurse %s" % (source_target, f))
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]

        return cmd

    def hls_sname(self):
        """Get the naming convention used by Vitis HLS for stream signals
        Example: the TDATA for a stream called "out" would be out_V_TDATA.
        """
        # despite being an RTL op, the interface naming here follows the HLS
        # style and has a V suffix for the stream interface names
        return "V"

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        return 0

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Path to (System-)Verilog files used by top-module & path to top-module
        verilog_paths = [code_gen_dir, os.environ["FINN_ROOT"] + "/finn-rtllib/dwc"]
        verilog_files = [self.get_nodeattr("gen_top_module") + "_wrapper.v"]

        # build the Verilator emu library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_verilog_top_module_name(),
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)

        return sim

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Normally: Generates C++ code and tcl script for IP generation.
        Here: Generates (System-)Verilog code for IP generation."""
        self.generate_hdl()

    def ipgen_singlenode_code(self):
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
