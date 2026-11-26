############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
import numpy as np
import os
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow import elementwise_binary
from finn.custom_op.fpgadataflow.elementwise_binary import ElementwiseBinaryOperation
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import arith_rtl_files, fifo_rtl_files, roundup_to_integer_multiple
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)


class ElementwiseBinary_rtl(ElementwiseBinaryOperation, RTLBackend):
    """Base CustomOp wrapper for the finn-rtllib eltwise component.

    Supports float/float, int/float, float/int, and int/int data paths.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(ElementwiseBinaryOperation.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update(
            {
                # memory mode for the const value
                # has to be internal_decoupled for rtl elementwise ops
                "mem_mode": (
                    "s",
                    False,
                    "internal_decoupled",
                ),
                "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            }
        )
        return my_attrs

    def generate_hdl(self, model, fpgapart, clk):
        lhs_style = self.get_nodeattr("lhs_style")
        rhs_style = self.get_nodeattr("rhs_style")
        mlo = self.get_nodeattr("mlo_max_iter")
        has_const = lhs_style == "const" or rhs_style == "const"

        # When both inputs are dynamic (input/input), no weights to embed
        # MLO mode also allows rhs to be "input" for parameter streaming
        if has_const and not mlo:
            assert (
                rhs_style == "const" or lhs_style == "const"
            ), """Neither side is const and MLO is not enabled.
            Try setting the preferred_impl_style to hls or enabling MLO"""

        if has_const:
            assert (
                self.get_nodeattr("mem_mode") == "internal_decoupled"
            ), "Only internal_decoupled mode is supported for rtl elementwise ops with const inputs"

        # Resolve data types
        lhs_dtype = self.get_input_datatype(ind=0)
        rhs_dtype = self.get_input_datatype(ind=1)
        out_dtype = self.get_output_datatype(ind=0)

        lhs_float = lhs_dtype == DataType["FLOAT32"]
        rhs_float = rhs_dtype == DataType["FLOAT32"]
        both_int = not lhs_float and not rhs_float
        op_name = self._get_rtl_op_name()  # e.g. '"ADD"'

        # Validate type constraints mirroring RTL parameter validation
        if lhs_float:
            assert lhs_dtype == DataType["FLOAT32"], f"Float LHS must be FLOAT32, got {lhs_dtype}"
        if rhs_float:
            assert rhs_dtype == DataType["FLOAT32"], f"Float RHS must be FLOAT32, got {rhs_dtype}"
        if both_int:
            assert lhs_dtype.bitwidth() == rhs_dtype.bitwidth(), (
                f"Int/int path requires matching widths: "
                f"A_WIDTH={lhs_dtype.bitwidth()}, B_WIDTH={rhs_dtype.bitwidth()}"
            )
            assert (
                lhs_dtype.signed() == rhs_dtype.signed()
            ), "Int/int path requires matching signedness"
            if op_name == '"MUL"':
                max_w = 24 if lhs_dtype.signed() else 23
                assert (
                    lhs_dtype.bitwidth() <= max_w
                ), f"Int MUL width {lhs_dtype.bitwidth()} exceeds DSP58 capacity ({max_w})"

        # Validate output width matches RTL O_WIDTH derivation
        if both_int:
            if op_name == '"MUL"':
                expected_owidth = 2 * lhs_dtype.bitwidth()
            else:
                expected_owidth = lhs_dtype.bitwidth() + 1
            assert out_dtype.bitwidth() == expected_owidth, (
                f"Int/int {op_name} requires out_dtype bitwidth={expected_owidth}, "
                f"got {out_dtype.bitwidth()}"
            )
        else:
            assert (
                out_dtype.bitwidth() == 32
            ), f"Float path requires 32-bit output, got {out_dtype.bitwidth()}"

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        # Only generate weight params when there is a const input
        if has_const:
            self.generate_params(model, code_gen_dir)

        rtlsrc = f'{os.environ["FINN_ROOT"]}/finn-rtllib/eltwise'
        template_path = f"{rtlsrc}/eltwise_template.v"
        pe = self.get_nodeattr("PE")

        a_width = lhs_dtype.bitwidth()
        b_width = rhs_dtype.bitwidth()
        rep_dims, rep_coefs, _ = self._const_stream_nest()

        code_gen_dict = {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "PE": pe,
            "OP": op_name,
            "B_SCALE": 1.0,
            "A_FLOAT": 1 if lhs_float else 0,
            "B_FLOAT": 1 if rhs_float else 0,
            "A_WIDTH": a_width,
            "A_SIGNED": 1 if (not lhs_float and lhs_dtype.signed()) else 0,
            "B_WIDTH": b_width,
            "B_SIGNED": 1 if (not rhs_float and rhs_dtype.signed()) else 0,
            "B_REP_D": len(rep_dims),
            "B_REP_DIMS": "{%s}" % ", ".join("32'd%d" % d for d in rep_dims or [0]),
            "B_REP_COEFS": "{%s}" % ", ".join("32'd%d" % c for c in rep_coefs or [0]),
            "A_STREAM_BITS": pe * a_width,
            "B_STREAM_BITS": pe * b_width,
            "O_STREAM_BITS": pe * out_dtype.bitwidth(),
        }

        with open(template_path, "r") as f:
            template = f.read()
        for key_name, value in code_gen_dict.items():
            template = template.replace(f"${key_name}$", str(value))

        with open(os.path.join(code_gen_dir, f"{self.get_verilog_top_module_name()}.v"), "w") as f:
            f.write(template)

        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # Only generate memstream wrapper when there is a const input
        if has_const or mlo:
            self.generate_hdl_memstream(fpgapart, clk)

        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = f"{self.get_nodeattr('code_gen_dir_ipgen')}/"
            rtllib_dir = f'{os.environ["FINN_ROOT"]}/finn-rtllib/eltwise/'
            input_gen_dir = f'{os.environ["FINN_ROOT"]}/finn-rtllib/input_gen/hdl/'
        else:
            code_gen_dir = ""
            rtllib_dir = ""
            input_gen_dir = ""

        top_module = self.get_nodeattr("gen_top_module")
        return (
            [
                f"{input_gen_dir}input_gen.sv",
                f"{rtllib_dir}eltwise.sv",
                f"{rtllib_dir}binopi.sv",
                f"{code_gen_dir}{top_module}.v",
            ]
            + arith_rtl_files(["binopf.sv", "int_to_fp32.sv"], abspath)
            + fifo_rtl_files(abspath)
        )

    def get_verilog_top_module_intf_names(self):
        """
        Return the interface names for the Verilog top module.

        For RTL elementwise operations, this includes handling for MLO mode
        where the rhs parameter may be streamed as an input.
        """
        # Start collecting interface names in a dictionary starting with clock and reset
        intf_names = {"clk": ["ap_clk"], "rst": ["ap_rst_n"]}

        # AXI stream input interfaces
        intf_names["s_axis"] = []

        # If the left-hand-side is provided as runtime input interface names need to be inserted
        if self.lhs_style == "input":
            intf_names["s_axis"] += [("in0_V", self.get_instream_width_padded(ind=0))]

        # If the right-hand-side is provided as runtime input interface names need to be inserted
        # (input/input mode or MLO mode where adapt_for_loop_body changes rhs_style to "input")
        if self.rhs_style == "input":
            intf_names["s_axis"] += [("in1_V", self.get_instream_width_padded(ind=1))]

        # AXI stream output interfaces
        intf_names["m_axis"] = [("out0_V", self.get_outstream_width_padded(ind=0))]

        # No AXI-MM, AXI-Lite or protocol-less interfaces
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []

        # Return the interface name dictionary
        return intf_names

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        cmd = []
        node_name = self.onnx_node.name
        intf = self.get_verilog_top_module_intf_names()
        clk_name = intf["clk"][0]
        rst_name = intf["rst"][0]
        dout_name = intf["m_axis"][0][0]
        din_name = intf["s_axis"][0][0]
        mlo = self.get_nodeattr("mlo_max_iter")
        lhs_style = self.get_nodeattr("lhs_style")
        rhs_style = self.get_nodeattr("rhs_style")
        both_input = lhs_style == "input" and rhs_style == "input" and not mlo

        cmd.append("create_bd_cell -type hier %s" % node_name)
        cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
        cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
        cmd.append(
            "create_bd_intf_pin -mode Master "
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
        )
        cmd.append(
            "create_bd_intf_pin -mode Slave "
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
        )

        # When both inputs are dynamic or MLO streams rhs, expose in1_V
        if both_input or (mlo and rhs_style == "input"):
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/in1_V" % node_name
            )

        # instantiate the RTL block
        self.instantiate_ip(cmd)

        # connect elementwise core
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
            % (node_name, rst_name, node_name, node_name, rst_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
            % (node_name, clk_name, node_name, node_name, clk_name)
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
            "[get_bd_intf_pins %s/%s/%s]" % (node_name, din_name, node_name, node_name, din_name)
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
            "[get_bd_intf_pins %s/%s/%s]" % (node_name, dout_name, node_name, node_name, dout_name)
        )

        if both_input:
            # input/input mode: wire in1_V directly to eltwise core, no memstream
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/in1_V] "
                "[get_bd_intf_pins %s/%s/in1_V]" % (node_name, node_name, node_name)
            )
        else:
            # const or MLO mode: use memstream wrapper
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1

            axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
            ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
            file_suffix = "_memstream_wrapper.v"

            strm_tmpl = None
            for fname in os.listdir(code_gen_dir):
                if fname.endswith(file_suffix):
                    strm_tmpl = fname

            if strm_tmpl is None:
                raise Exception(f"No memstream wrapper found in {code_gen_dir}")

            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                os.path.join(code_gen_dir, strm_tmpl),
                axi_dir + "axilite.sv",
                ms_rtllib_dir + "memstream_axi.sv",
                ms_rtllib_dir + "memstream.sv",
            ]
            for f in sourcefiles:
                cmd.append("add_files -norecurse %s" % f)
            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (strm_tmpl_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            # MLO mode: connect external in1_V to memstream input
            if mlo and rhs_style == "input":
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/in1_V] "
                    "[get_bd_intf_pins %s/%s/s_axis_0]" % (node_name, node_name, strm_inst)
                )
            # Connect memstream output to core input
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/in1_V]" % (node_name, strm_inst, node_name, node_name)
            )
            if runtime_writable:
                axilite_name = intf["axilite"][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s" % (node_name, axilite_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, axilite_name, node_name, strm_inst, axilite_name)
                )
                cmd.append("assign_bd_address")
        cmd.append("save_bd_design")
        return cmd

    def instantiate_ip(self, cmd):
        node_name = self.onnx_node.name
        top_module = self.get_nodeattr("gen_top_module")

        for f in self.get_rtl_file_list(abspath=True):
            cmd.append("add_files -norecurse %s" % f)

        # Always create the core inside the hierarchical wrapper
        cmd.append(
            "create_bd_cell -type hier -reference %s /%s/%s" % (top_module, node_name, node_name)
        )

    def get_tav_io_dict(self):
        n_inps = np.prod(self.get_folded_input_shape(0)[:-1])
        # fewer than beats if the core replays them itself
        n_rhs = self.calc_wmem() * self.calc_wmem_reps()
        return {
            "inputs": {
                "in0": [i for i in range(n_inps)],
                "in1": [i for i in range(n_rhs)],
            },
            "outputs": {"out0": []},
        }

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim":
            node = self.onnx_node
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            lhs = context[node.input[0]]
            rhs = context[node.input[1]]

            assert lhs.shape == self.get_normal_input_shape(
                ind=0
            ), f"Input shape mismatch for {node.input[0]}"
            if self.rhs_style != "const":
                assert rhs.shape == self.get_normal_input_shape(
                    ind=1
                ), f"Input shape mismatch for {node.input[1]}"

            out_shape = self.get_normal_output_shape(ind=0)

            if self.lhs_style == "const":
                lhs = np.broadcast_to(lhs, out_shape)
                lhs = lhs.reshape(self.get_folded_output_shape(ind=0))
            else:
                lhs = lhs.reshape(self.get_folded_input_shape(ind=0))

            lhs_filename = os.path.join(code_gen_dir, "input_0.npy")
            rhs_filename = os.path.join(code_gen_dir, "input_1.npy")

            if self.rhs_style == "const":
                # as the const streamer: the words of the const, replayed as a whole
                self.make_weight_file(rhs, "decoupled_npy", rhs_filename)
                rhs = np.tile(np.load(rhs_filename)[0], (self.calc_wmem_reps(), 1))
            else:
                rhs = rhs.reshape(self.get_folded_input_shape(ind=1))

            np.save(lhs_filename, lhs)
            np.save(rhs_filename, rhs)

            io_dict = {"inputs": {}, "outputs": {"out0": []}}
            lhs_dtype = self.get_input_datatype(ind=0)
            lhs_width = self.get_instream_width(ind=0)
            rhs_dtype = self.get_input_datatype(ind=1)
            rhs_width = self.get_instream_width(ind=1)

            mem_mode = self.get_nodeattr("mem_mode")
            mlo = self.get_nodeattr("mlo_max_iter")
            lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
            # MLO mode: stream RHS when it's marked as input (from adapt_for_loop_body)
            rhs_decoupled = (self.rhs_style == "const" and mem_mode == "internal_decoupled") or (
                self.rhs_style == "input" and mlo
            )

            if self.lhs_style == "input" or lhs_decoupled:
                io_dict["inputs"]["in0"] = npy_to_rtlsim_input(lhs_filename, lhs_dtype, lhs_width)
            if self.rhs_style == "input" or rhs_decoupled:
                io_dict["inputs"]["in1"] = npy_to_rtlsim_input(rhs_filename, rhs_dtype, rhs_width)

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)

            out = io_dict["outputs"]["out0"]
            dtype = self.get_output_datatype(ind=0)
            width = self.get_outstream_width(ind=0)
            shape = self.get_folded_output_shape(ind=0)
            filename = os.path.join(code_gen_dir, "output_0.npy")
            rtlsim_output_to_npy(out, filename, dtype, shape, width, dtype.bitwidth())
            out = np.load(filename)
            out = out.reshape(self.get_normal_output_shape(ind=0))
            if dtype.is_integer():
                context[node.output[0]] = out
            else:
                context[node.output[0]] = out.astype(np.float32)
        else:
            ElementwiseBinaryOperation.execute_node(self, context, graph)

    def generate_params(self, model, code_gen_dir):
        weights = model.get_initializer(self.onnx_node.input[1])
        if weights is not None:
            self.make_weight_file(weights, "decoupled_npy", f"{code_gen_dir}/input_1.npy")
            self.make_weight_file(weights, "decoupled_verilog_dat", f"{code_gen_dir}/memblock.dat")

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        folded_weight_shape = self.get_folded_input_shape(1)
        weight_tensor = weights.reshape(folded_weight_shape).copy()

        # When broadcasting the last axis (rhs_shape[-1]==1), replicate the
        # scalar value across PE lanes so memstream provides PE values per cycle
        if self.broadcast_last_axis and weight_tensor.shape[-1] == 1:
            weight_tensor = np.tile(
                weight_tensor, (1,) * (len(weight_tensor.shape) - 1) + (self.pe,)
            )

        # Reshape to compact 3D form for decoupled modes
        weight_tensor = weight_tensor.reshape(1, -1, weight_tensor.shape[-1]).copy()

        if weight_file_mode == "decoupled_npy":
            np.save(weight_file_name, weight_tensor)
            return

        # decoupled_verilog_dat
        export_wdt = self.get_input_datatype(1)
        weight_width = self.get_instream_width(1)
        weight_width_padded = roundup_to_integer_multiple(weight_width, 4)

        weight_tensor_hex = pack_innermost_dim_as_hex_string(
            weight_tensor,
            export_wdt,
            weight_width_padded,
            reverse_inner=True,
            prefix="",
        )

        weight_stream = weight_tensor_hex.flatten()
        with open(weight_file_name, "w") as f:
            for val in weight_stream:
                f.write(val + "\n")

    def calc_wmem(self):
        return int(super().calc_wmem())

    def _const_stream_nest(self):
        """Return (dims, coefs, reps) of the const (or loop parameter) stream.

        The memstream stores the words of the const, PE elements each, and cycles through
        them. The data needs a word on every beat:

        * Outer axes the const broadcasts over, e.g. N and T for a [C] const on [N, T, C],
          replay all of its words, reps times.
        * The other axes are the levels of the loop nest (dims, coefs) the eltwise core
          replays the words of a pass by (eltwise.sv B_REP_DIMS, B_REP_COEFS). An axis the
          const broadcasts over has coefficient 0: it replays the words of the axes below it,
          e.g. dims (C, H, W), coefs (W, 0, 1) for a (C, 1, W) const on (C, H, W), or holds
          each word if there are none, e.g. dims (C, H, W), coefs (1, 0, 0) for a (C, 1, 1)
          const.
        * Without such an axis the nest is empty, the words are used as they are stored.
        """
        # Beats per axis, the last axis is folded by PE (asserts that PE divides it)
        out_beats = [int(b) for b in self.get_folded_output_shape(ind=0)[:-1]]
        const_shape = [1] * (len(out_beats) - len(self.rhs_shape)) + list(self.rhs_shape)
        # Words of the const per axis, a single one along an axis it broadcasts over
        const_beats = [1 if c == 1 else b for c, b in zip(const_shape, out_beats)]
        first = next((i for i, c in enumerate(const_beats) if c != 1), len(out_beats))
        reps = int(np.prod(out_beats[:first]))
        dims = out_beats[first:]
        # Stride of each axis in the words of the const
        coefs = [
            0 if c < b else int(np.prod(const_beats[i + 1 :]))
            for i, (c, b) in enumerate(zip(const_beats, out_beats))
        ][first:]
        if 0 not in coefs:
            dims, coefs = [], []
        return dims, coefs, reps

    def calc_wmem_reps(self):
        """Return how many times the words of the const stream, replayed by the loop nest of
        _const_stream_nest(), are replayed as a whole. Regular single-set memstreams cycle on
        their own, FINNLoop multi-set memstreams use this value as the stream-tap repetition
        count for each loop iteration."""
        return self._const_stream_nest()[2]

    def calc_numInputVectors(self):
        folded_lhs = self.get_folded_input_shape(0)
        if len(folded_lhs) >= 2:
            return list(folded_lhs[:-1])
        return [1]

    def _get_rtl_op_name(self):
        """Override in subclasses to return the correct RTL operation name."""
        raise NotImplementedError("Subclasses must implement _get_rtl_op_name")


class ElementwiseAdd_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseAdd):
    """RTL implementation of elementwise addition."""

    def _get_rtl_op_name(self):
        return '"ADD"'


class ElementwiseSub_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseSub):
    """RTL implementation of elementwise subtraction."""

    def _get_rtl_op_name(self):
        return '"SUB"'


class ElementwiseMul_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseMul):
    """RTL implementation of elementwise multiplication."""

    def _get_rtl_op_name(self):
        return '"MUL"'
