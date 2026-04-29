# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os

from finn.custom_op.fpgadataflow.requant import Requant
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_dsp_block, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class Requant_rtl(Requant, RTLBackend):
    """RTL backend for Requant operation using finn-rtllib/requant."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Requant.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _resolve_dsp_version(self, fpgapart):
        """Determine DSP version based on FPGA part."""
        dsp_block = get_dsp_block(fpgapart)
        match dsp_block:
            case "DSP58":
                return 3
            case "DSP48E2":
                return 2
            case _:
                return 1

    def generate_hdl(self, model, fpgapart, clk):
        """Generate RTL code for the requant operation."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        if code_gen_dir == "":
            code_gen_dir = make_build_dir("requant_rtl_ipgen_")
            self.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)

        # Get parameters
        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")
        cf = num_channels // pe  # Channel fold

        idt = self.get_input_datatype(0)
        odt = self.get_output_datatype()
        k = idt.bitwidth()  # Input precision
        n = odt.bitwidth()  # Output precision

        version = self._resolve_dsp_version(fpgapart)

        # Get scale and bias from model
        scale = self.get_scale(model)
        bias = self.get_bias(model)

        # Broadcast scalar scale/bias to all channels if needed
        if scale.size == 1:
            scale = np.full(num_channels, scale.item(), dtype=np.float32)
        if bias.size == 1:
            bias = np.full(num_channels, bias.item(), dtype=np.float32)

        # Reshape for PE interleaving: [PE][CF]
        # The RTL expects scales and biases in [PE][CF] layout
        scale_reshaped = scale.reshape(cf, pe).T  # [PE][CF]
        bias_reshaped = bias.reshape(cf, pe).T  # [PE][CF]

        # Format as SystemVerilog array literals
        def format_sv_array(arr):
            """Format 2D numpy array as SystemVerilog array literal."""
            lines = []
            for pe_idx in range(arr.shape[0]):
                # Use fixed-point notation with 6 decimal places (shortreal is 32-bit float)
                row = ", ".join(f"{float(v):.6f}" for v in arr[pe_idx])
                lines.append("'{" + row + "}")
            return "'{" + ", ".join(lines) + "}"

        scales_sv = format_sv_array(scale_reshaped)
        biases_sv = format_sv_array(bias_reshaped)

        # Calculate stream widths (byte-aligned)
        in_stream_width = ((pe * k + 7) // 8) * 8
        out_stream_width = ((pe * n + 7) // 8) * 8

        top_module_name = self.get_verilog_top_module_name()
        rtllib_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/requant/hdl/"

        # Generate SystemVerilog implementation module (with _impl suffix)
        sv_template_path = rtllib_dir + "requant_wrapper_template.sv"
        with open(sv_template_path, "r") as f:
            sv_template = f.read()

        sv_code = sv_template
        sv_code = sv_code.replace("$TOP_MODULE_NAME$", top_module_name)
        sv_code = sv_code.replace("$VERSION$", str(version))
        sv_code = sv_code.replace("$K$", str(k))
        sv_code = sv_code.replace("$N$", str(n))
        sv_code = sv_code.replace("$C$", str(num_channels))
        sv_code = sv_code.replace("$PE$", str(pe))
        sv_code = sv_code.replace("$SCALES$", scales_sv)
        sv_code = sv_code.replace("$BIASES$", biases_sv)
        sv_code = sv_code.replace("$IN_STREAM_WIDTH$", str(in_stream_width))
        sv_code = sv_code.replace("$OUT_STREAM_WIDTH$", str(out_stream_width))

        sv_output_path = os.path.join(code_gen_dir, top_module_name + "_impl.sv")
        with open(sv_output_path, "w") as f:
            f.write(sv_code)

        # Generate Verilog stub wrapper (for IP packaging - must be .v)
        v_template_path = rtllib_dir + "requant_wrapper_template.v"
        with open(v_template_path, "r") as f:
            v_template = f.read()

        v_code = v_template
        v_code = v_code.replace("$TOP_MODULE_NAME$", top_module_name)
        v_code = v_code.replace("$IN_STREAM_WIDTH$", str(in_stream_width))
        v_code = v_code.replace("$OUT_STREAM_WIDTH$", str(out_stream_width))

        v_output_path = os.path.join(code_gen_dir, top_module_name + ".v")
        with open(v_output_path, "w") as f:
            f.write(v_code)

        self.set_nodeattr("gen_top_module", top_module_name)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stitch ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        """Return list of RTL files needed for this node."""
        rtllib_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/requant/hdl/"
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        rtl_files = [
            rtllib_dir + "queue.sv",
            rtllib_dir + "requant.sv",
            rtllib_dir + "requant_axi.sv",
        ]

        # Add generated wrappers (Verilog stub + SystemVerilog impl)
        top_module = self.get_nodeattr("gen_top_module")
        if top_module == "":
            top_module = self.get_verilog_top_module_name()
        rtl_files.append(os.path.join(code_gen_dir, top_module + "_impl.sv"))
        rtl_files.append(os.path.join(code_gen_dir, top_module + ".v"))

        if abspath:
            return rtl_files
        else:
            return [os.path.basename(f) for f in rtl_files]

    def code_generation_ipi(self):
        sourcefiles = self.get_rtl_file_list(abspath=True)

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def execute_node(self, context, graph):
        """Execute the node, using RTL simulation if exec_mode is rtlsim."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim":
            # Custom RTL sim that only passes input 0 (data), not scale/bias
            # which are embedded as parameters in the generated HDL
            node = self.onnx_node
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

            # Only process input 0 (data tensor)
            inp = node.input[0]
            exp_ishape = tuple(self.get_normal_input_shape(0))
            folded_ishape = self.get_folded_input_shape(0)
            inp_val = context[inp]
            assert str(inp_val.dtype) == "float32", "Input datatype is not float32"
            assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
            export_idt = self.get_input_datatype(0)

            reshaped_input = inp_val.reshape(folded_ishape)
            np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)
            nbits = self.get_instream_width(0)
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )

            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)

            # Process output
            rtlsim_output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype(0)
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width(0)
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape(0)
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            # Load and reshape output
            exp_oshape = tuple(self.get_normal_output_shape(0))
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output

            assert (
                context[node.output[0]].shape == exp_oshape
            ), "Output shape doesn't match expected shape."
        else:
            # Use base class Python execution
            Requant.execute_node(self, context, graph)
