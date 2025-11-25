############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
import math
import os
import shutil
from qonnx.core.datatype import DataType
from typing import Optional

from finn.custom_op.fpgadataflow.inner_shuffle import InnerShuffle
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


def auto_size_simd(I_dim: int, SIMD: int) -> Optional[int]:
    """
    Return the smallest divisor d of I_dim such that d > SIMD.
    if no such divisor exists, return None.
    """
    if I_dim <= 0:
        raise ValueError("I_dim must be a positive integer")
    if SIMD < 0:
        raise ValueError("SIMD must be a non-negative integer")

    candidates = []
    limit = int(math.isqrt(I_dim))
    for a in range(1, limit + 1):
        if I_dim % a == 0:
            b = I_dim // a
            if a > SIMD:
                candidates.append(a)
            if b > SIMD:
                candidates.append(b)

    if not candidates:
        return None

    return min(candidates)


class InnerShuffle_rtl(InnerShuffle, RTLBackend):
    """CustomOp wrapper for the finn-rtllib inner_shuffle component."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        # check some constraints that it is a legal InnerShuffle
        I_dim = self.get_nodeattr("in_shape")[-2]
        SIMD = self.get_nodeattr("SIMD")
        if I_dim % SIMD != 0:
            new_simd = auto_size_simd(I_dim, SIMD)
            if new_simd is not None:
                self.set_nodeattr("SIMD", new_simd)
            else:
                raise RuntimeError("Unable to determine a new SIMD value for this transpose.")

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(InnerShuffle.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_template_values(self, idims, simd, dt):
        code_gen_dict = {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "I": idims[0],
            "J": idims[1],
            "SIMD": simd,
            "WIDTH": dt.bitwidth(),
            "STREAM_BITS": simd * dt.bitwidth(),
        }
        return code_gen_dict

    def generate_hdl(self, model, fpgapart, clk):
        rtlsrc = f'{os.environ["FINN_ROOT"]}/finn-rtllib/inner_shuffle'
        template_path = f"{rtlsrc}/inner_shuffle_template.v"
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        dt = DataType[self.get_nodeattr("data_type")]
        simd = self.get_nodeattr("SIMD")
        code_gen_dict = {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "I": self.get_nodeattr("in_shape")[-2],
            "J": self.get_nodeattr("in_shape")[-1],
            "SIMD": simd,
            "WIDTH": dt.bitwidth(),
            "STREAM_BITS": simd * dt.bitwidth(),
        }
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = f"${key_name}$"
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(os.path.join(code_gen_dir, f"{self.get_verilog_top_module_name()}.v"), "w") as f:
            f.write(template)

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        sv_files = ["inner_shuffle.sv", "skid.sv"]
        for sv_files in sv_files:
            shutil.copy(f"{rtlsrc}/{sv_files}", code_gen_dir)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = f"{self.get_nodeattr('code_gen_dir_ipgen')}/"
            rtllib_dir = f'{os.environ["FINN_ROOT"]}/finn-rtllib/inner_shuffle'
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        top_module = self.get_nodeattr("gen_top_module")
        return [
            f"{rtllib_dir}/inner_shuffle.sv",
            f"{rtllib_dir}/skid.sv",
            f"{code_gen_dir}{top_module}.v",
        ]

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        top_module = self.get_nodeattr("gen_top_module")
        sourcefiles = ["inner_shuffle.sv", "skid.sv", f"{top_module}.v"]
        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for vf in sourcefiles:
            cmd += [f"add_files -norecurse {vf}"]
        cmd += [f"create_bd_cell -type module -reference {top_module} {self.onnx_node.name}"]
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            InnerShuffle.execute_node(self, context, graph)
