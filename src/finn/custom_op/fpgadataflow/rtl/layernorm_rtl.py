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
import shutil

from finn.custom_op.fpgadataflow.layernorm import LayerNorm
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


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
        assert n // simd > 12, "N/SIMD must be larger than 12 for rsqrt throughput."
        code_gen_dict = {
            "$N$": int(n),
            "$SIMD$": int(simd),
            "$TOP_MODULE_NAME$": topname,
        }

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key in code_gen_dict:
            template = template.replace(key, str(code_gen_dict[key]))

        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v"),
            "w",
        ) as f:
            f.write(template)

        sv_files = ["layernorm.sv", "queue.sv", "accuf.sv", "binopf.sv", "rsqrtf.sv"]
        for sv_file in sv_files:
            shutil.copy(rtllib_dir + sv_file, code_gen_dir)
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

        verilog_files = [
            rtllib_dir + "layernorm.sv",
            rtllib_dir + "queue.sv",
            rtllib_dir + "accuf.sv",
            rtllib_dir + "binopf.sv",
            rtllib_dir + "rsqrtf.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]
        return verilog_files

    def code_generation_ipi(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "layernorm.sv",
            "queue.sv",
            "accuf.sv",
            "binopf.sv",
            "rsqrtf.sv",
        ]

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

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            LayerNorm.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

    def get_exp_cycles(self):
        simd = self.get_nodeattr("SIMD")
        idim = self.get_normal_input_shape()
        n = idim[-1]
        assert (
            n % simd == 0
        ), """Requirement N (last dim) divisable by SIMD is violated.
            Please set SIMD to a different value"""
        assert n // simd > 12, "N/SIMD must be larger than 12 for rsqrt throughput."

        val_queue_len_0 = n // simd + math.ceil(math.log2(simd)) * 2 + 7
        val_queue_len_1 = n // simd + math.ceil(math.log2(simd)) * 2 + 24
        exp_cycles = val_queue_len_0 + val_queue_len_1 + np.prod(idim) // simd + 5

        return int(exp_cycles)
