# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.quant import max_int, min_int

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.requant import Requant
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class Requant_hls(Requant, HLSBackend):
    """HLS backend for Requant operation.

    Computes: clip(round(x * scale + bias), min, max)

    Delegates to Requant_Batch from finn-hlslib (function-pipelined).
    Scale and bias are embedded as RequantParams in a generated header file.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Requant.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def get_exp_cycles(self):
        """Returns expected number of cycles for execution.

        Adds a constant offset to account for HLS pipeline initialization overhead.
        """
        hls_overhead = 10
        return Requant.get_exp_cycles(self) + hls_overhead

    def generate_params(self, model, path):
        """Generate RequantParams header with [PE][NF] layout."""
        code_gen_dir = path
        scale = self.get_scale(model)
        bias = self.get_bias(model)

        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        nf = num_channels // pe

        # Broadcast scalar values to all channels if needed
        if scale.size == 1:
            scale = np.full(num_channels, scale.item(), dtype=np.float32)
        if bias.size == 1:
            bias = np.full(num_channels, bias.item(), dtype=np.float32)

        # Reshape for PE interleaving: channel order is
        #   ch0=(nf=0,pe=0), ch1=(nf=0,pe=1), ..., ch_{PE-1}=(nf=0,pe=PE-1),
        #   ch_{PE}=(nf=1,pe=0), ...
        # Python arrays are [NF][PE], RequantParams needs [PE][NF] → transpose
        scale_nf_pe = scale.reshape(nf, pe)
        bias_nf_pe = bias.reshape(nf, pe)

        with open(os.path.join(code_gen_dir, "params.h"), "w") as f:
            f.write('#include "requant.hpp"\n\n')
            f.write(f"static RequantParams<{nf}, {pe}> const rq_params = {{\n")
            # scales[PE][NF]
            f.write("    .scales = {\n")
            for p in range(pe):
                vals = ", ".join(f"{scale_nf_pe[n, p]:.10f}f" for n in range(nf))
                f.write(f"        {{{vals}}},\n")
            f.write("    },\n")
            # biases[PE][NF]
            f.write("    .biases = {\n")
            for p in range(pe):
                vals = ", ".join(f"{bias_nf_pe[n, p]:.10f}f" for n in range(nf))
                f.write(f"        {{{vals}}},\n")
            f.write("    }\n")
            f.write("};\n")

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "requant.hpp"',
            '#include "params.h"',
        ]

    def defines(self, var):
        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")

        idt = self.get_input_datatype(0)
        odt = self.get_output_datatype()

        narrow = self.get_nodeattr("narrow")
        signed = 1 if odt.signed() else 0
        bitwidth = odt.bitwidth()
        min_val = min_int(signed, narrow, bitwidth)
        max_val = max_int(signed, narrow, bitwidth)

        self.code_gen_dict["$DEFINES$"] = [
            f"constexpr unsigned NumChannels1 = {num_channels};",
            f"constexpr unsigned PE1 = {pe};",
            f"constexpr int MIN_VAL = {min_val};",
            f"constexpr int MAX_VAL = {max_val};",
            f"using TI = {idt.get_hls_datatype_str()};",
            f"using TO = {odt.get_hls_datatype_str()};",
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "Requant_Batch<NumChannels1, PE1, MIN_VAL, MAX_VAL>"
            "(in0_V, out0_V, rq_params);"
        ]

    def blackboxfunction(self):
        pe = self.get_nodeattr("PE")
        idt = self.get_input_datatype(0).get_hls_datatype_str()
        odt = self.get_output_datatype().get_hls_datatype_str()
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void {}(hls::stream<tensor<{}, {}>> &in0_V, "
            "hls::stream<tensor<{}, {}>> &out0_V)".format(
                self.onnx_node.name, idt, pe, odt, pe
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            "#pragma HLS ARRAY_PARTITION variable=rq_params.scales complete dim=1",
            "#pragma HLS ARRAY_PARTITION variable=rq_params.biases complete dim=1",
        ]

    def ipgen_extra_directives(self):
        """Return a list of extra tcl directives for HLS synthesis."""
        return ["config_compile -pipeline_style flp"]

    def execute_node(self, context, graph):
        """Execute the node using cppsim or rtlsim.

        Custom implementation that only passes input 0 (data), since scale and
        bias are embedded as parameters in the generated code.
        """
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(f"Invalid value for attribute exec_mode! Is currently set to: {mode}")

        # Only process input 0 (data tensor), scale and bias are embedded
        inp = node.input[0]
        exp_ishape = tuple(self.get_normal_input_shape(0))
        folded_ishape = self.get_folded_input_shape(0)
        inp_val = context[inp]

        # Make sure the input has the right container datatype
        if inp_val.dtype not in [np.float32, np.float16]:
            warnings.warn(
                f"{node.name}: Changing input container datatype from "
                f"{inp_val.dtype} to {np.float32}"
            )
            inp_val = inp_val.astype(np.float32)

        assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
        export_idt = self.get_input_datatype(0)

        reshaped_input = inp_val.reshape(folded_ishape)
        reshaped_input = reshaped_input.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            self.exec_precompiled_singlenode_model()
            # load output npy file
            self.npy_to_dynamic_output(context)
            for o, outp in enumerate(node.output):
                exp_oshape = tuple(self.get_normal_output_shape(o))
                assert (
                    context[outp].shape == exp_oshape
                ), "cppsim did not produce expected output shape"
                if self.get_output_datatype(o) == DataType["BIPOLAR"]:
                    out = context[outp]
                    out = 2 * out - 1
                    context[outp] = out
        elif mode == "rtlsim":
            nbits = self.get_instream_width(0)
            rtlsim_inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)

            rtlsim_output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype(0)
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width(0)
            out_npy_path = f"{code_gen_dir}/output_0.npy"
            out_shape = self.get_folded_output_shape(0)
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            exp_oshape = tuple(self.get_normal_output_shape(0))
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output

            assert (
                context[node.output[0]].shape == exp_oshape
            ), "Output shape doesn't match expected shape."
