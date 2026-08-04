# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.pad1d import Pad1D
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


def _rtlsrc_dir():
    return os.environ["FINN_ROOT"] + "/finn-rtllib/pad1d/hdl"


class Pad1D_rtl(Pad1D, RTLBackend):
    """RTL implementation of Pad1D."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Pad1D.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _pack_value(self, value, dtype):
        bitwidth = dtype.bitwidth()
        if dtype == DataType["BIPOLAR"]:
            int_value = int((value + 1) // 2)
        else:
            if dtype.is_fixed_point():
                value = value / dtype.scale_factor()
            int_value = int(value)
            if int_value < 0:
                int_value += 1 << bitwidth
        return int_value & ((1 << bitwidth) - 1)

    def _get_pad_data_width(self, ind):
        dtype = self.get_input_datatype()
        pad_count = self._get_pad_count(ind)
        num_channels = self.get_nodeattr("NumChannels")
        return max(1, pad_count) * num_channels * dtype.bitwidth()

    def _get_pad_values(self, model, ind):
        pad_count = self._get_pad_count(ind)
        num_channels = self.get_nodeattr("NumChannels")
        if pad_count == 0:
            return np.zeros((1, 1, num_channels), dtype=np.float32)

        if len(self.onnx_node.input) <= ind:
            raise Exception(
                "Pad1D RTL generation requires a constant %s pad input."
                % self._get_pad_side_name(ind)
            )

        pad_values = model.get_initializer(self.onnx_node.input[ind])
        if pad_values is None:
            raise Exception(
                "Pad1D RTL generation requires a constant %s pad input."
                % self._get_pad_side_name(ind)
            )

        pad_values = np.asarray(pad_values, dtype=np.float32)
        self._validate_pad_shape(pad_values.shape, ind)
        if pad_values.shape[1] == 1 and pad_count > 1:
            pad_values = np.repeat(pad_values, pad_count, axis=1)
        return pad_values

    def _pack_pad_data(self, model, ind):
        dtype = self.get_input_datatype()
        bitwidth = dtype.bitwidth()
        if self._get_pad_count(ind) == 0:
            data_width = self._get_pad_data_width(ind)
            hex_digits = (data_width + 3) // 4
            return "%d'h%0*x" % (data_width, hex_digits, 0)

        pad_values = self._get_pad_values(model, ind)

        assert np.vectorize(dtype.allowed)(
            pad_values
        ).all(), "Pad1D %s pad values cannot be represented with %s" % (
            self._get_pad_side_name(ind),
            dtype.name,
        )

        packed = 0
        for i, value in enumerate(pad_values.flatten()):
            packed |= self._pack_value(value, dtype) << (i * bitwidth)
        data_width = self._get_pad_data_width(ind)
        hex_digits = (data_width + 3) // 4
        return "%d'h%0*x" % (data_width, hex_digits, packed)

    def generate_hdl(self, model, fpgapart, clk):
        simd = self.get_nodeattr("SIMD")
        num_channels = self.get_nodeattr("NumChannels")
        assert num_channels % simd == 0, "SIMD must divide NumChannels"

        rtlsrc = _rtlsrc_dir()
        template_path = rtlsrc + "/pad1d_template.v"
        with open(template_path, "r") as f:
            template = f.read()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        elem_width = self.get_input_datatype().bitwidth()
        fold_width = elem_width * simd
        code_gen_dict = {
            "TOP_MODULE_NAME": topname,
            "NUM_TOKENS": self.get_nodeattr("NumTokens"),
            "NUM_CHANNELS": num_channels,
            "SIMD": simd,
            "ELEM_WIDTH": elem_width,
            "PAD_LEFT": self.get_nodeattr("PadLeft"),
            "PAD_RIGHT": self.get_nodeattr("PadRight"),
            "FOLD_WIDTH": fold_width,
            "PAD_LEFT_DATA_WIDTH": self._get_pad_data_width(1),
            "PAD_RIGHT_DATA_WIDTH": self._get_pad_data_width(2),
            "PAD_LEFT_DATA": self._pack_pad_data(model, 1),
            "PAD_RIGHT_DATA": self._pack_pad_data(model, 2),
        }

        for key, value in code_gen_dict.items():
            template = template.replace("$%s$" % key, str(value))

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, topname + ".v"), "w") as f:
            f.write(template)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            return [
                os.path.join(_rtlsrc_dir(), "pad1d.sv"),
                os.path.join(
                    self.get_nodeattr("code_gen_dir_ipgen"),
                    self.get_nodeattr("gen_top_module") + ".v",
                ),
            ]
        return ["pad1d.sv", self.get_nodeattr("gen_top_module") + ".v"]

    def code_generation_ipi(self):
        sourcefiles = self.get_rtl_file_list(abspath=True)
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]
        for f in sourcefiles:
            cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            Pad1D.execute_node(self, context, graph)
        elif mode == "rtlsim":
            node = self.onnx_node
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

            # Only input 0 is streamed. Pad inputs are embedded as RTL parameters.
            inp = node.input[0]
            exp_ishape = tuple(self.get_normal_input_shape(0))
            folded_ishape = self.get_folded_input_shape(0)
            inp_val = context[inp]
            assert str(inp_val.dtype) == "float32", "Input datatype is not float32"
            assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
            export_idt = self.get_input_datatype(0)

            reshaped_input = inp_val.reshape(folded_ishape)
            input_path = os.path.join(code_gen_dir, "input_0.npy")
            np.save(input_path, reshaped_input)
            rtlsim_inp = npy_to_rtlsim_input(
                input_path,
                export_idt,
                self.get_instream_width(0),
            )
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)

            odt = self.get_output_datatype(0)
            output_path = os.path.join(code_gen_dir, "output.npy")
            rtlsim_output_to_npy(
                io_dict["outputs"]["out0"],
                output_path,
                odt,
                self.get_folded_output_shape(0),
                self.get_outstream_width(0),
                odt.bitwidth(),
            )
            output = np.load(output_path)
            exp_oshape = tuple(self.get_normal_output_shape(0))
            context[node.output[0]] = np.asarray(output, dtype=np.float32).reshape(exp_oshape)
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following values ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
