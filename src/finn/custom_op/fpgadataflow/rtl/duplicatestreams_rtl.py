############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

import os

from finn.custom_op.fpgadataflow.duplicatestreams import DuplicateStreams
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class DuplicateStreams_rtl(DuplicateStreams, RTLBackend):
    """Buffered AXI-stream broadcaster for DuplicateStreams."""

    def get_nodeattr_types(self):
        return DuplicateStreams.get_nodeattr_types(self) | RTLBackend.get_nodeattr_types(self)

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            DuplicateStreams.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise ValueError('exec_mode must be either "cppsim" or "rtlsim"')

    def generate_hdl(self, model, fpgapart, clk):
        top_module = self.get_verilog_top_module_name()
        axi_width = self.get_instream_width_padded()
        num_outputs = self.get_nodeattr("NumOutputStreams")
        output_busif = ":".join(f"out{idx}_V" for idx in range(num_outputs))
        associated_busif = "in0_V:" + output_busif

        ports = [
            '\t(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)',
            (
                "\t(* X_INTERFACE_PARAMETER = "
                f'"ASSOCIATED_BUSIF {associated_busif}, ASSOCIATED_RESET ap_rst_n" *)'
            ),
            "\tinput ap_clk,",
            '\t(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)',
            "\tinput ap_rst_n,",
            "\toutput in0_V_TREADY,",
            "\tinput in0_V_TVALID,",
            "\tinput [AXI_BITS-1:0] in0_V_TDATA,",
        ]
        for idx in range(num_outputs):
            comma = "," if idx != num_outputs - 1 else ""
            ports += [
                f"\tinput out{idx}_V_TREADY,",
                f"\toutput out{idx}_V_TVALID,",
                f"\toutput [AXI_BITS-1:0] out{idx}_V_TDATA{comma}",
            ]

        ready_vector = ", ".join(f"out{idx}_V_TREADY" for idx in reversed(range(num_outputs)))
        output_assigns = []
        for idx in range(num_outputs):
            output_assigns += [
                f"\tassign out{idx}_V_TVALID = pending[{idx}];",
                f"\tassign out{idx}_V_TDATA = data_reg;",
            ]

        verilog = "\n".join(
            [
                f"module {top_module} #(",
                f"\tparameter integer AXI_BITS = {axi_width}",
                ")(",
                "\n".join(ports),
                ");",
                "",
                f"\treg [{num_outputs - 1}:0] pending;",
                "\treg [AXI_BITS-1:0] data_reg;",
                f"\twire [{num_outputs - 1}:0] output_ready = {{{ready_vector}}};",
                "\twire pending_outputs_done = &(~pending | output_ready);",
                "",
                "\tassign in0_V_TREADY = ~(|pending) | pending_outputs_done;",
                "\n".join(output_assigns),
                "",
                "\talways @(posedge ap_clk) begin",
                "\t\tif (!ap_rst_n) begin",
                f"\t\t\tpending <= {num_outputs}'b0;",
                "\t\t\tdata_reg <= {AXI_BITS{1'b0}};",
                "\t\tend else if (in0_V_TREADY) begin",
                "\t\t\tif (in0_V_TVALID) begin",
                f"\t\t\t\tpending <= {{{num_outputs}{{1'b1}}}};",
                "\t\t\t\tdata_reg <= in0_V_TDATA;",
                "\t\t\tend else begin",
                f"\t\t\t\tpending <= {num_outputs}'b0;",
                "\t\t\tend",
                "\t\tend else begin",
                "\t\t\tpending <= pending & ~output_ready;",
                "\t\tend",
                "\tend",
                "",
                "endmodule",
                "",
            ]
        )

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.set_nodeattr("gen_top_module", top_module)
        with open(os.path.join(code_gen_dir, top_module + ".v"), "w") as f:
            f.write(verilog)

        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") if abspath else ""
        return [os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v")]

    def code_generation_ipi(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        sourcefile = os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v")
        return [
            "add_files -norecurse %s" % sourcefile,
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name),
        ]
