############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
############################################################################

import math
import numpy as np
import os

from finn.custom_op.fpgadataflow.layernorm import LayerNorm
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import arith_rtl_files, fifo_rtl_files


class LayerNorm_rtl(LayerNorm, RTLBackend):
    """RTL backend implementation for LayerNorm kernel.
    Generates RTL code for hardware synthesis of LayerNorm operations.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update(LayerNorm.get_nodeattr_types(self))
        return my_attrs

    def generate_hdl(self, model, fpgapart, clk):
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/layernorm/")
        template_path = rtllib_dir + "layernorm_wrapper_template.v"
        simd = self.get_nodeattr("SIMD")
        topname = self.get_verilog_top_module_name()
        n = self.get_normal_input_shape()[-1]
        assert (
            n % simd == 0
        ), """Requirement N (last dim) divisable by SIMD is violated.
            Please set SIMD to a different value"""
        code_gen_dict = {
            "$N$": int(n),
            "$SIMD$": int(simd),
            "$NUM_RSQRT_REFINEMENTS$": self.get_nodeattr("numRsqrtRefinements"),
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

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/layernorm/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        return (
            [
                rtllib_dir + "layernorm.sv",
                rtllib_dir + "accuf.sv",
                rtllib_dir + "rsqrtf.sv",
                code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
            ]
            + arith_rtl_files(["binopf.sv"], abspath)
            + fifo_rtl_files(abspath)
        )

    def code_generation_ipi(self):
        cmd = []
        for f in self.get_rtl_file_list(abspath=True):
            cmd.append("add_files -norecurse %s" % f)
        cmd.append(
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        )
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            LayerNorm.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

    def _rsqrt_latency(self):
        """Latency allowance for rsqrtf, mirroring RSQRT_LATENCY of layernorm.sv:
        three 4-cycle DSP passes per Newton-Raphson iteration and, for the
        interleaving structures, 2 cycles of input skid and up to 3 cycles waiting
        for a slot. A single iteration keeps its historic flat 14 cycles."""
        refinements = self.get_nodeattr("numRsqrtRefinements")
        if refinements == 1:
            return 14
        nn = self.get_normal_input_shape()[-1] // self.get_nodeattr("SIMD")
        interleaved = 1 < nn < 12 * refinements - 3
        return 12 * refinements + (2 + 3 if interleaved else 0)

    def get_exp_cycles(self):
        """Mirrors the queue lengths of layernorm.sv, which is the ground truth for
        the latency of the statistics paths of both normalization steps."""
        simd = self.get_nodeattr("SIMD")
        idim = self.get_normal_input_shape()
        n = idim[-1]
        assert (
            n % simd == 0
        ), """Requirement N (last dim) divisable by SIMD is violated.
            Please set SIMD to a different value"""
        rsqrt_latency = self._rsqrt_latency()
        # SIMD adder tree + accumulation + decouple
        mean_latency = math.ceil(math.log2(simd)) * 2 + 4 + 3
        # ... + scaling by 1/N + rsqrt
        var_latency = mean_latency + 3 + rsqrt_latency
        val_queue_len_0 = n // simd + mean_latency
        val_queue_len_1 = n // simd + var_latency
        exp_cycles = val_queue_len_0 + val_queue_len_1 + np.prod(idim) // simd + 5

        return int(exp_cycles)
