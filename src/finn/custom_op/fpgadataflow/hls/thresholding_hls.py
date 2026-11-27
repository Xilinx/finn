# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import os
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.util.data_packing import (
    hls_datatype_str,
    npy_to_rtlsim_input,
    rtlsim_output_to_npy,
)

# ONNX i/o tensor shape assumptions for Thresholding:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the threshold tensor, shape (NumChannels, n_thres_steps)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


class Thresholding_hls(Thresholding, HLSBackend):
    """Thresholding by the pipelined binary search of finn-hlslib (thresholding<>
    in thresholding.hpp), the HLS counterpart of Thresholding_rtl: per-stage
    threshold memories inside the kernel, optionally runtime-writeable through
    AXI-lite in the address layout of Thresholding_rtl."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def get_cfg_addr_bits(self, model):
        """Width of a threshold address of the configuration stream: channel fold,
        PE and threshold index fields (thresholding_addr_bits() of the kernel),
        at least one bit."""
        pe = self.get_nodeattr("PE")
        c = self.get_kernel_channels(model)
        n = self.get_nodeattr("numSteps")

        def clog2(x):
            return math.ceil(math.log2(x))

        return max(1, clog2(c // pe) + clog2(pe) + clog2(n))

    def get_kernel_channels(self, model):
        """Channels with own thresholds: NumChannels, or PE for thresholds shared
        by all channels (a single row), which are then stored once per PE."""
        if model.get_tensor_shape(self.onnx_node.input[1])[0] == 1:
            return self.get_nodeattr("PE")
        return self.get_nodeattr("NumChannels")

    def _check_types(self):
        idt = self.get_input_datatype(0)
        tdt = self.get_input_datatype(1)
        if idt.is_integer() and not tdt.is_integer():
            raise ValueError(
                "Thresholds must be converted to integers for integer inputs "
                "using RoundAndClipThresholds transform before code generation."
            )
        if not idt.is_integer() and tdt.is_integer():
            raise ValueError("Floating-point inputs and integer thresholds are not supported.")
        if (idt.is_fixed_point() or tdt.is_fixed_point()) and (
            idt.is_fixed_point() != tdt.is_fixed_point() or idt.scale_factor() != tdt.scale_factor()
        ):
            raise ValueError(
                "Fixed-point inputs need fixed-point thresholds of the same scale "
                "(see RoundAndClipThresholds)."
            )
        odt = self.get_output_datatype()
        if odt != DataType["BIPOLAR"] and odt.bitwidth() < self.get_min_output_bits():
            raise ValueError(
                f"{self.onnx_node.name}: outputDataType {odt.name} is too narrow for "
                f"{self.get_nodeattr('numSteps')} thresholds with ActVal "
                f"{self.get_nodeattr('ActVal')}."
            )
        if self.get_nodeattr("depth_trigger_uram"):
            raise ValueError(
                f"{self.onnx_node.name}: Thresholding_hls cannot place thresholds in "
                "URAM, which HLS cannot initialize; use Thresholding_rtl."
            )
        if self.get_nodeattr("mlo_max_iter"):
            raise ValueError(f"{self.onnx_node.name}: MLO thresholding requires Thresholding_rtl.")

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates c++ code and tcl script for ip generation, and the AXI-lite
        bridge for runtime-writeable thresholds."""
        super().code_generation_ipgen(model, fpgapart, clk)
        if self.get_nodeattr("runtime_writeable_weights"):
            self.generate_axilite_bridge(model)

    def get_axilite_bridge_name(self):
        return self.onnx_node.name + "_axilite"

    def generate_axilite_bridge(self, model):
        """Fill the Verilog wrapper of axilite_stream that turns AXI-lite accesses
        into the kernel's configuration requests and responses."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        template_path = os.path.join(
            os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/axilite_stream_template.v"
        )
        with open(template_path, "r") as f:
            template = f.read()
        addr_bits = self.get_cfg_addr_bits(model)
        values = {
            "$MODULE_NAME$": self.get_axilite_bridge_name(),
            "$ADDR_WIDTH$": str(addr_bits + 2),
            "$IP_DATA_WIDTH$": str(self.get_input_datatype(1).bitwidth()),
            "$IP_ADDR_WIDTH$": str(addr_bits),
        }
        for key, value in values.items():
            template = template.replace(key, value)
        with open(os.path.join(code_gen_dir, self.get_axilite_bridge_name() + ".v"), "w") as f:
            f.write(template)

    def _hls_value(self, dtype):
        """Return a formatter of threshold values as C++ initializer literals."""
        if dtype.is_integer():
            return lambda v: "%dLL" % int(v)
        if dtype.is_fixed_point():
            return lambda v: "%.17g" % float(v)

        def fmt_float(v):
            v = float(v)
            if math.isinf(v):
                return "INFINITY" if v > 0 else "-INFINITY"
            return "%.9ef" % v

        return fmt_float

    def generate_params(self, model, path):
        """Write thresh.h with the initial contents of the per-stage threshold
        memories as the specializations thresh_init<S> expected by thresholding<>."""
        self._check_types()
        thresholds = model.get_initializer(self.onnx_node.input[1])
        tdt = self.get_input_datatype(1)
        assert np.vectorize(tdt.allowed)(
            thresholds
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        if thresholds.shape[0] != 1:
            assert (
                np.diff(thresholds, axis=-1) >= 0
            ).all(), f"{self.onnx_node.name}: Thresholds must be sorted per channel."
        fmt = self._hls_value(tdt)
        tt = hls_datatype_str(tdt)
        pe = self.get_nodeattr("PE")
        lines = [
            "// Generated by FINN: thresholds of %s per search stage, see the"
            % self.onnx_node.name,
            "// INIT parameter of thresholding<> in finn-hlslib thresholding.hpp",
            "#include <cmath>",
            "static constexpr unsigned  C1 = %d;" % self.get_kernel_channels(model),
            "template<unsigned  S> struct thresh_init;",
        ]
        for s, stage in enumerate(self.get_stage_thresholds(thresholds)):
            rom = "stage_rom<%s, %d, %d>" % (tt, pe, stage.shape[1])
            data = ",\n".join("{" + ",".join(fmt(v) for v in row) + "}" for row in stage)
            lines += [
                "template<> struct thresh_init<%d> {" % s,
                "\tstatic %s const &data() {" % rom,
                "\t\tstatic %s const  d = {{\n%s\n\t\t}};" % (rom, data),
                "\t\treturn  d;",
                "\t}",
                "};",
            ]
        with open(os.path.join(path, "thresh.h"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce the runtime weight file (weight_file_mode "decoupled_runtime"),
        which the driver writes over AXI-lite as is."""
        if weight_file_mode != "decoupled_runtime":
            raise Exception("Unknown weight_file_mode")
        self.make_runtime_weight_file(weights, weight_file_name)

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        # only the data input is streamed, the thresholds are in the kernel; the
        # npy container must match the element type read by cppsim (half for FLOAT16)
        container = np.float16 if self.get_input_datatype(0) == DataType["FLOAT16"] else np.float32
        inp = context[node.input[0]].astype(container)
        np.save(
            os.path.join(code_gen_dir, "input_0.npy"),
            inp.reshape(self.get_folded_input_shape()).copy(),
        )
        odt = self.get_output_datatype()
        if mode == "cppsim":
            self.exec_precompiled_singlenode_model()
            self.npy_to_dynamic_output(context)
            if odt == DataType["BIPOLAR"]:
                context[node.output[0]] = 2 * context[node.output[0]] - 1
        else:
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir),
                self.get_input_datatype(0),
                self.get_instream_width(0),
            )
            io_dict = {"inputs": {"in0": rtlsim_inp}, "outputs": {"out0": []}}
            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)
            out_npy_path = "{}/output_0.npy".format(code_gen_dir)
            rtlsim_output_to_npy(
                io_dict["outputs"]["out0"],
                out_npy_path,
                odt,
                self.get_folded_output_shape(),
                self.get_outstream_width(),
                odt.bitwidth(),
            )
            output = np.load(out_npy_path)
            context[node.output[0]] = np.asarray([output], dtype=np.float32).reshape(
                *self.get_normal_output_shape()
            )
        assert (
            context[node.output[0]].shape == self.get_normal_output_shape()
        ), "Output shape is not as expected"

    def _hls_types(self):
        ti = hls_datatype_str(self.get_input_datatype(0))
        tt = hls_datatype_str(self.get_input_datatype(1))
        odt = self.get_output_datatype()
        # bipolar outputs are exported as binary
        to = hls_datatype_str(DataType["BINARY"] if odt == DataType["BIPOLAR"] else odt)
        return ti, tt, to

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "thresholding.hpp"', '#include "thresh.h"']

    def defines(self, var):
        ti, tt, to = self._hls_types()
        self.code_gen_dict["$DEFINES$"] = [
            "static constexpr unsigned  N1 = %d;" % self.get_nodeattr("numSteps"),
            "static constexpr unsigned  PE1 = %d;" % self.get_nodeattr("PE"),
            "static constexpr int  BIAS1 = %d;" % self.get_nodeattr("ActVal"),
            "static constexpr unsigned  DEPTH_TRIGGER_BRAM1 = %d;"
            % self.get_nodeattr("depth_trigger_bram"),
            "using  TI1 = %s;" % ti,
            "using  TT1 = %s;" % tt,
            "using  TO1 = %s;" % to,
            # C1, the channels with own thresholds, comes with thresh.h
            "using  Cfg1 = thresholding_cfg<N1, C1, PE1, TT1>;",
        ]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "half" if self._hls_types()[0] == "half" else "float"
        self.code_gen_dict["$READNPYDATA$"] = [
            'npy2tensorstream<%s>("%s/input_0.npy", in0_V);' % (npy_type, code_gen_dir)
        ]

    def strm_decl(self):
        pe = self.get_nodeattr("PE")
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            'hls::stream<tensor<TI1, %d>> in0_V ("in0_V");' % pe,
            'hls::stream<tensor<TO1, %d>> out0_V ("out0_V");' % pe,
            # buffer stream for freerunning timeout template
            'hls::stream<tensor<TO1, %d>> strm0 ("strm0");' % pe,
        ]
        if self.get_nodeattr("runtime_writeable_weights"):
            self.code_gen_dict["$STREAMDECLARATIONS$"] += [
                'hls::stream<Cfg1> cfg_V ("cfg_V");',
                'hls::stream<TT1> rsp_V ("rsp_V");',
            ]

    def docompute(self):
        streams = "in0_V, out0_V"
        if self.get_nodeattr("runtime_writeable_weights"):
            streams += ", cfg_V, rsp_V"
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "thresholding<N1, C1, TT1, thresh_init, BIAS1, 1, DEPTH_TRIGGER_BRAM1>(%s);" % streams
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        shape_cpp_str = str(self.get_folded_output_shape()).replace("(", "{").replace(")", "}")
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'tensorstream2npy<float>(strm0, %s, "%s/output_0.npy");' % (shape_cpp_str, code_gen_dir)
        ]

    def blackboxfunction(self):
        pe = self.get_nodeattr("PE")
        ports = [
            "hls::stream<tensor<TI1, %d>> &in0_V" % pe,
            "hls::stream<tensor<TO1, %d>> &out0_V" % pe,
        ]
        if self.get_nodeattr("runtime_writeable_weights"):
            ports += ["hls::stream<Cfg1> &cfg_V", "hls::stream<TT1> &rsp_V"]
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(%s)" % (self.onnx_node.name, ", ".join(ports))
        ]

    def pragmas(self):
        # The kernel is a stalling free-running pipeline: its input register
        # slices are dispensable, the output slice is required for AXI-Stream
        # compliance (see thresholding.hpp).
        pragmas = [
            "#pragma HLS interface axis port=in0_V register_mode=off",
            "#pragma HLS interface axis port=out0_V",
        ]
        if self.get_nodeattr("runtime_writeable_weights"):
            pragmas += [
                "#pragma HLS interface axis port=cfg_V register_mode=off",
                "#pragma HLS aggregate variable=cfg_V compact=bit",
                "#pragma HLS interface axis port=rsp_V",
            ]
        pragmas += [
            "#pragma HLS interface ap_ctrl_none port=return",
            "#pragma HLS dataflow disable_start_propagation",
        ]
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def code_generation_ipi(self):
        if not self.get_nodeattr("runtime_writeable_weights"):
            return super().code_generation_ipi()
        # A hierarchy of the HLS kernel and the AXI-lite bridge feeding its
        # configuration stream, with the ports of Thresholding_rtl.
        node_name = self.onnx_node.name
        intf = self.get_verilog_top_module_intf_names()
        clk_name = intf["clk"][0]
        rst_name = intf["rst"][0]
        dout_name = intf["m_axis"][0][0]
        din_name = intf["s_axis"][0][0]
        axilite_name = intf["axilite"][0]
        bridge = self.get_axilite_bridge_name()
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
        cmd = [
            "create_bd_cell -type hier %s" % node_name,
            "create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name),
            "create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name),
            "create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s"
            % (node_name, dout_name),
            "create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s"
            % (node_name, din_name),
            "create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s"
            % (node_name, axilite_name),
            "create_bd_cell -type ip -vlnv %s /%s/%s"
            % (self.get_nodeattr("ip_vlnv"), node_name, node_name),
        ]
        for f in [
            axi_dir + "axilite.sv",
            axi_dir + "axilite_stream.sv",
            os.path.join(code_gen_dir, bridge + ".v"),
        ]:
            cmd.append("add_files -norecurse %s" % f)
        cmd.append("create_bd_cell -type module -reference %s /%s/%s" % (bridge, node_name, bridge))
        for cell in [node_name, bridge]:
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, clk_name, node_name, cell, clk_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, rst_name, node_name, cell, rst_name)
            )
        for outer, cell, inner in [
            (din_name, node_name, din_name),
            (dout_name, node_name, dout_name),
            (axilite_name, bridge, "s_axilite"),
        ]:
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] [get_bd_intf_pins %s/%s/%s]"
                % (node_name, outer, node_name, cell, inner)
            )
        cmd += [
            "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_req] "
            "[get_bd_intf_pins %s/%s/cfg_V]" % (node_name, bridge, node_name, node_name),
            "connect_bd_intf_net [get_bd_intf_pins %s/%s/rsp_V] "
            "[get_bd_intf_pins %s/%s/s_axis_rsp]" % (node_name, node_name, node_name, bridge),
            "save_bd_design",
        ]
        return cmd

    def get_op_and_param_counts(self):
        weight_bits = self.get_input_datatype(1).bitwidth()
        thres_count = self.get_nodeattr("NumChannels") * self.get_nodeattr("numSteps")
        # thresholds are called weights in this layer
        return {"param_threshold_%db" % weight_bits: thres_count}

    def get_tav_io_dict(self):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        return {"inputs": {"in0": [0 for i in range(n_inps)]}, "outputs": {"out0": []}}
