############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
import os
import shutil
import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.elementwise_binary import ElementwiseBinaryOperation
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class ElementwiseBinary_rtl(ElementwiseBinaryOperation, RTLBackend):
    """Base CustomOp wrapper for the finn-rtllib eltwisef component."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(ElementwiseBinaryOperation.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def generate_hdl(self, model, fpgapart, clk):
        rtlsrc = f'{os.environ["FINN_ROOT"]}/finn-rtllib/eltwisef'
        template_path = f"{rtlsrc}/eltwisef_template.v"
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        dt = DataType[self.get_nodeattr("out_dtype")]
        pe = self.get_nodeattr("PE")
        
        op_name = self._get_rtl_op_name()
        
        code_gen_dict = {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "PE": pe,
            "OP": op_name,
            "B_SCALE": 1.0,
            "FORCE_BEHAVIORAL": 0,
            "STREAM_BITS": pe * 32,
        }
        
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = f"${key_name}$"
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(os.path.join(code_gen_dir, f"{self.get_verilog_top_module_name()}.v"), "w") as f:
            f.write(template)

        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        sv_files = ["eltwisef.sv", "binopf.sv", "queue.sv"]
        for sv_file in sv_files:
            shutil.copy(f"{rtlsrc}/{sv_file}", code_gen_dir)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = f"{self.get_nodeattr('code_gen_dir_ipgen')}/"
            rtllib_dir = f'{os.environ["FINN_ROOT"]}/finn-rtllib/eltwisef'
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        top_module = self.get_nodeattr("gen_top_module")
        return [
            f"{rtllib_dir}/eltwisef.sv",
            f"{rtllib_dir}/binopf.sv",
            f"{rtllib_dir}/queue.sv",
            f"{code_gen_dir}{top_module}.v",
        ]

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        top_module = self.get_nodeattr("gen_top_module")
        sourcefiles = ["eltwisef.sv", "binopf.sv", "queue.sv", f"{top_module}.v"]
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
            ElementwiseBinaryOperation.execute_node(self, context, graph)

    def _get_rtl_op_name(self):
        """Override in subclasses to return the correct RTL operation name."""
        raise NotImplementedError("Subclasses must implement _get_rtl_op_name")


class ElementwiseAdd_rtl(ElementwiseBinary_rtl):
    """RTL implementation of elementwise addition for FLOAT32."""
    
    _operation = "Add", np.add, "({0} + {1})", '"ADD"'
    
    def _get_rtl_op_name(self):
        return '"ADD"'


class ElementwiseSub_rtl(ElementwiseBinary_rtl):
    """RTL implementation of elementwise subtraction for FLOAT32."""
    
    _operation = "Sub", np.subtract, "({0} - {1})", '"SUB"'
    
    def _get_rtl_op_name(self):
        return '"SUB"'


class ElementwiseMul_rtl(ElementwiseBinary_rtl):
    """RTL implementation of elementwise multiplication for FLOAT32."""
    
    _operation = "Mul", np.multiply, "({0} * {1})", '"MUL"'
    
    def _get_rtl_op_name(self):
        return '"MUL"'
