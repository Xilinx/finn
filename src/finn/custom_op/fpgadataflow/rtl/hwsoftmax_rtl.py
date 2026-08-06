############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
import os
import shutil

from finn.custom_op.fpgadataflow.hwsoftmax import HWSoftmax
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class HWSoftmax_rtl(HWSoftmax, RTLBackend):
    """RTL backend implementation for SoftMax kernel.
    Generates RTL code for hardware synthesis of SoftMax operations
    via the streaming FP32 softmaxf core (with optional integer
    input conversion via int_to_fp32)."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update(HWSoftmax.get_nodeattr_types(self))
        return my_attrs

    def _rtllib_files(self):
        return [
            "softmaxf.sv",
            "softmaxf_pkg.sv",
            "binopf.sv",
            "pwpolyf_pkg.sv",
            "pwpolyf.sv",
            "queue.sv",
            "int_to_fp32.sv",
        ]

    def generate_hdl(self, model, fpgapart, clk):
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/softmax_rtl/")
        template_path = rtllib_dir + "softmax_wrapper_template.v"
        simd = self.get_nodeattr("SIMD")
        topname = self.get_verilog_top_module_name()
        n = self.get_normal_input_shape()[-1]
        assert (
            n % simd == 0
        ), """Requirement N (last dim) divisable by SIMD is violated.
            Please set SIMD to a different value"""

        idt = self.get_input_datatype()
        width = idt.bitwidth()
        signed_flag = 1 if idt.signed() else 0
        fp32_pass = 1 if idt.name == "FLOAT32" else 0

        code_gen_dict = {
            "$N$": int(n),
            "$SIMD$": int(simd),
            "$WIDTH$": int(width),
            "$SIGNED$": int(signed_flag),
            "$FP32_PASSTHROUGH$": int(fp32_pass),
            "$TOP_MODULE_NAME$": topname,
        }

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key, value in code_gen_dict.items():
            template = template.replace(key, str(value))

        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v"),
            "w",
        ) as f:
            f.write(template)

        for sv_file in self._rtllib_files():
            shutil.copy(rtllib_dir + sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/softmax_rtl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [rtllib_dir + f for f in self._rtllib_files()]
        verilog_files.append(code_gen_dir + self.get_nodeattr("gen_top_module") + ".v")
        return verilog_files

    def code_generation_ipi(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = list(self._rtllib_files())
        sourcefiles.append(self.get_nodeattr("gen_top_module") + ".v")
        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def get_exp_cycles(self):
        # softmaxf is a fully-pipelined 4-stage RTL design with elastic queues
        # between stages. Once the pipeline fills, throughput is 1 beat/cycle.
        # Pipeline fill latency: max-tree + exp-poly + recip-NR + div-mul.
        # The queues (CREDIT_Y, CREDIT_S) absorb inter-stage latency, so
        # steady-state overhead is minimal.
        folded = self.get_folded_input_shape()
        n_beats = int(np.prod(folded[:-1]))
        n = self.get_normal_input_shape()[-1]
        simd = self.get_nodeattr("SIMD")
        beats_per_vec = max(1, n // simd)
        # Pipeline fill: ~50 cycles (tree + poly + recip + div latencies)
        return n_beats + beats_per_vec + 50

    def dsp_estimation(self, fpgapart=None):
        # softmaxf uses, per SIMD lane, one subtractor, a degree-2 exp
        # polynomial, and one final multiplier. Its two reciprocal iterations
        # each instantiate MUL/SUB/MUL, giving six shared DSPFP32 primitives.
        return 4 * self.get_nodeattr("SIMD") + 6

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            HWSoftmax.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
