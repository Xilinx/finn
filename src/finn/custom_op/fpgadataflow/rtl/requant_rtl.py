# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.requant import Requant
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_dsp_block, make_build_dir, roundup_to_integer_multiple
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


def _twos(value, width):
    """Return the ``width``-bit two's-complement representation of ``value``."""
    return int(value) & ((1 << width) - 1)


def _clog2(n):
    """Ceil(log2(n)) matching Verilog ``$clog2`` for n >= 1."""
    return (n - 1).bit_length()


class Requant_rtl(Requant, RTLBackend):
    """RTL backend for Requant operation using finn-rtllib/requant."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Requant.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update(
            {
                # memory mode for the scale/bias parameters
                # internal_embedded: constant-folded into the datapath (default)
                # internal_decoupled: streamed from two on-chip memstreams
                "mem_mode": (
                    "s",
                    False,
                    "internal_embedded",
                    {"internal_embedded", "internal_decoupled"},
                ),
                "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            }
        )
        return my_attrs

    def adapt_for_loop_body(self, input_types):
        """Adapt Requant_rtl for loop body (MLO) execution.

        Requant's default ``mem_mode`` is ``internal_embedded``, which
        constant-folds scale/bias into the SV datapath and therefore cannot vary
        per loop iteration. When LoopRolling flags this node for MLO (it sets
        ``mlo_max_iter`` on the consumer of each per-iteration PARAMETER input),
        switch to ``internal_decoupled`` so scale+bias are streamed from the two
        memstreams instead (mirrors MVAU_rtl's embedded->external_mem switch, but
        with Requant's own decoupled target mode). Gate on the per-node
        ``mlo_max_iter`` signal rather than the positional ``input_types``.
        """
        if self.get_nodeattr("mlo_max_iter") > 0:
            self.set_nodeattr("mem_mode", "internal_decoupled")

    def _mlo_set_bits_padded(self):
        """Byte-padded width of the per-iteration set-select stream.

        Must match the FINNLoop stream_tap ``$DATA_WIDTH$``
        (finn_loop.py:548-550) so that the tap's ``m_axis`` connects cleanly to
        the ``in1_V``/``in2_V`` set-select slave pins.
        """
        iteration = self.get_nodeattr("mlo_max_iter")
        data_width = DataType.get_smallest_possible(iteration).bitwidth()
        return roundup_to_integer_multiple(data_width, 8)

    def get_verilog_top_module_intf_names(self):
        """Interface names for the Requant top module.

        Base case exposes only the activation stream ``in0_V`` and output
        ``out0_V`` (scale/bias are internal to the decoupled hierarchy). In MLO
        mode two extra set-select slave pins ``in1_V`` (scale) and ``in2_V``
        (bias) are appended in graph-input order; each is driven by a FINNLoop
        stream_tap and selects the active parameter set in its memstream.
        """
        intf_names = {"clk": ["ap_clk"], "rst": ["ap_rst_n"]}
        intf_names["s_axis"] = [("in0_V", self.get_instream_width_padded(0))]
        if (
            self.get_nodeattr("mlo_max_iter") > 0
            and self.get_nodeattr("mem_mode") == "internal_decoupled"
        ):
            set_bits_padded = self._mlo_set_bits_padded()
            intf_names["s_axis"] += [
                ("in1_V", set_bits_padded),
                ("in2_V", set_bits_padded),
            ]
        intf_names["m_axis"] = [("out0_V", self.get_outstream_width_padded(0))]
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        return intf_names

    def _resolve_dsp_version(self, fpgapart):
        """Determine DSP version based on FPGA part."""
        dsp_block = get_dsp_block(fpgapart)
        match dsp_block:
            case "DSP58":
                return 3
            case "DSP48E2":
                return 2
            case _:
                return 1

    def _derive_decoupled_widths(self, model, fpgapart):
        """Derive the fixed-point stream layout for decoupled mode.

        Returns a dict with the decomposed params plus the per-lane and
        byte-aligned stream widths for the scale and bias parameter streams and
        the input/output data streams. All widths are consistent with the
        localparam derivation in ``requant_axi_decoupled.sv``.
        """
        version = self._resolve_dsp_version(fpgapart)
        params = self.decompose_params(model, version)
        s_width = params["s_width"]
        x_width = params["x_width"]
        tap_min = params["tap_min"]
        tap_max = params["tap_max"]

        pe = self.get_nodeattr("PE")
        k = self.get_input_datatype(0).bitwidth()
        n = self.get_output_datatype().bitwidth()

        # In MLO mode the scale stream WIDTH and the core TAP_MIN/TAP_MAX are
        # baked once at generate_hdl time but must cover *every* loop iteration's
        # params. Since a valid tap is structurally bounded to
        # 0 <= tap <= s_width + x_width + 1 - n (see decompose_params, purely
        # datatype/version-derived), size the window to that worst case instead
        # of the per-layer [min, max]. This makes the layout iteration-invariant
        # and leaf-local (no cross-iteration knowledge needed).
        if self.get_nodeattr("mlo_max_iter") > 0:
            tap_min = 0
            tap_max = s_width + x_width + 1 - n

        bias_w = s_width + x_width
        tap_range = tap_max - tap_min + 1
        tap_bits = max(1, _clog2(tap_range)) if tap_range > 1 else 1
        scale_lane_w = s_width + tap_bits

        info = dict(params)
        info.update(
            {
                "version": version,
                "tap_min": tap_min,
                "tap_max": tap_max,
                "bias_w": bias_w,
                "tap_bits": tap_bits,
                "scale_lane_w": scale_lane_w,
                "in_stream_width": roundup_to_integer_multiple(pe * k, 8),
                "out_stream_width": roundup_to_integer_multiple(pe * n, 8),
                "scale_stream_width": roundup_to_integer_multiple(pe * scale_lane_w, 8),
                "bias_stream_width": roundup_to_integer_multiple(pe * bias_w, 8),
            }
        )
        return info

    def _pack_param_words(self, info):
        """Pack the decomposed params into per-fold stream words.

        Lane 0 (pe=0) occupies the least significant bits, matching the RTL
        ``s_scale_tdata[0+:PE*SCALE_LANE_W]`` / ``core_sdat`` unpacking. Returns
        two lists (scale_words, bias_words) of ``CF`` Python integers each.
        """
        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")
        cf = num_channels // pe

        scale = info["scale"]
        bias = info["bias"]
        tap = info["tap"]
        tap_min = info["tap_min"]
        s_width = info["s_width"]
        tap_bits = info["tap_bits"]
        scale_lane_w = info["scale_lane_w"]
        bias_w = info["bias_w"]

        scale_words = []
        bias_words = []
        for c in range(cf):
            s_word = 0
            b_word = 0
            for p in range(pe):
                t_off = int(tap[p][c]) - tap_min
                s_lane = (_twos(t_off, tap_bits) << s_width) | _twos(scale[p][c], s_width)
                s_word |= s_lane << (p * scale_lane_w)
                b_word |= _twos(bias[p][c], bias_w) << (p * bias_w)
            scale_words.append(s_word)
            bias_words.append(b_word)
        return scale_words, bias_words

    @staticmethod
    def _write_memblock(path, words, stream_width):
        """Write memstream init .dat: one hex word per line, zero-padded."""
        hex_digits = stream_width // 4
        with open(path, "w") as f:
            for w in words:
                f.write("{:0{}x}\n".format(int(w), hex_digits))

    def generate_params(self, model, path, fpgapart=None):
        """Emit the two decoupled memstream init files for the current params.

        Writes ``scale_memblock.dat`` and ``bias_memblock.dat`` into ``path`` for
        the scale/bias currently attached to this node in ``model``. Used both by
        standalone decoupled generation and, per loop iteration, by
        ``FINNLoop.generate_params`` (which sets the per-iteration initializers
        before calling this). ``fpgapart`` is required to resolve the DSP version
        so the fixed-point layout matches the elaborated core. Returns the packed
        ``(scale_words, bias_words)`` for reuse by callers.
        """
        assert fpgapart is not None, "Requant_rtl.generate_params requires fpgapart"
        info = self._derive_decoupled_widths(model, fpgapart)
        scale_words, bias_words = self._pack_param_words(info)
        self._write_memblock(
            os.path.join(path, "scale_memblock.dat"),
            scale_words,
            info["scale_stream_width"],
        )
        self._write_memblock(
            os.path.join(path, "bias_memblock.dat"),
            bias_words,
            info["bias_stream_width"],
        )
        return scale_words, bias_words

    def generate_hdl(self, model, fpgapart, clk):
        """Generate RTL code for the requant operation."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        if code_gen_dir == "":
            code_gen_dir = make_build_dir("requant_rtl_ipgen_")
            self.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            self._generate_hdl_embedded(model, fpgapart, clk, code_gen_dir)
        elif mem_mode == "internal_decoupled":
            self._generate_hdl_decoupled(model, fpgapart, clk, code_gen_dir)
        else:
            raise ValueError(f"{self.onnx_node.name}: unsupported mem_mode {mem_mode}.")

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stitch ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def _generate_hdl_embedded(self, model, fpgapart, clk, code_gen_dir):
        """Embedded mode: constant-fold scale/bias into the SV datapath."""
        # Get parameters
        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")
        cf = num_channels // pe  # Channel fold

        idt = self.get_input_datatype(0)
        odt = self.get_output_datatype()
        k = idt.bitwidth()  # Input precision
        n = odt.bitwidth()  # Output precision

        version = self._resolve_dsp_version(fpgapart)

        # Get scale and bias from model
        scale = self.get_scale(model)
        bias = self.get_bias(model)

        # Broadcast scalar scale/bias to all channels if needed
        if scale.size == 1:
            scale = np.full(num_channels, scale.item(), dtype=np.float32)
        if bias.size == 1:
            bias = np.full(num_channels, bias.item(), dtype=np.float32)

        # Reshape for PE interleaving: [PE][CF]
        # The RTL expects scales and biases in [PE][CF] layout
        scale_reshaped = scale.reshape(cf, pe).T  # [PE][CF]
        bias_reshaped = bias.reshape(cf, pe).T  # [PE][CF]

        # Format as SystemVerilog array literals
        def format_sv_array(arr):
            """Format 2D numpy array as SystemVerilog array literal."""
            lines = []
            for pe_idx in range(arr.shape[0]):
                # Use fixed-point notation with 6 decimal places (shortreal is 32-bit float)
                row = ", ".join(f"{float(v):.6f}" for v in arr[pe_idx])
                lines.append("'{" + row + "}")
            return "'{" + ", ".join(lines) + "}"

        scales_sv = format_sv_array(scale_reshaped)
        biases_sv = format_sv_array(bias_reshaped)

        # Calculate stream widths (byte-aligned)
        in_stream_width = ((pe * k + 7) // 8) * 8
        out_stream_width = ((pe * n + 7) // 8) * 8

        top_module_name = self.get_verilog_top_module_name()
        rtllib_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/requant/hdl/"

        # Generate SystemVerilog implementation module (with _impl suffix)
        sv_template_path = rtllib_dir + "requant_wrapper_template.sv"
        with open(sv_template_path, "r") as f:
            sv_template = f.read()

        sv_code = sv_template
        sv_code = sv_code.replace("$TOP_MODULE_NAME$", top_module_name)
        sv_code = sv_code.replace("$VERSION$", str(version))
        sv_code = sv_code.replace("$K$", str(k))
        sv_code = sv_code.replace("$N$", str(n))
        sv_code = sv_code.replace("$C$", str(num_channels))
        sv_code = sv_code.replace("$PE$", str(pe))
        sv_code = sv_code.replace("$SCALES$", scales_sv)
        sv_code = sv_code.replace("$BIASES$", biases_sv)
        sv_code = sv_code.replace("$IN_STREAM_WIDTH$", str(in_stream_width))
        sv_code = sv_code.replace("$OUT_STREAM_WIDTH$", str(out_stream_width))

        sv_output_path = os.path.join(code_gen_dir, top_module_name + "_impl.sv")
        with open(sv_output_path, "w") as f:
            f.write(sv_code)

        # Generate Verilog stub wrapper (for IP packaging - must be .v)
        v_template_path = rtllib_dir + "requant_wrapper_template.v"
        with open(v_template_path, "r") as f:
            v_template = f.read()

        v_code = v_template
        v_code = v_code.replace("$TOP_MODULE_NAME$", top_module_name)
        v_code = v_code.replace("$IN_STREAM_WIDTH$", str(in_stream_width))
        v_code = v_code.replace("$OUT_STREAM_WIDTH$", str(out_stream_width))

        v_output_path = os.path.join(code_gen_dir, top_module_name + ".v")
        with open(v_output_path, "w") as f:
            f.write(v_code)

        self.set_nodeattr("gen_top_module", top_module_name)

    def _generate_hdl_decoupled(self, model, fpgapart, clk, code_gen_dir):
        """Decoupled mode: stream scale/bias from two memstreams.

        The float->fixed-point decomposition is done in Python
        (``decompose_params``); the resulting per-channel words are packed and
        emitted as two memstream init files, and the compute core is elaborated
        with the worst-case TAP_MIN/TAP_MAX window instead of embedded params.
        """
        info = self._derive_decoupled_widths(model, fpgapart)

        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")
        cf = num_channels // pe
        k = self.get_input_datatype(0).bitwidth()
        n = self.get_output_datatype().bitwidth()

        top_module_name = self.get_verilog_top_module_name()
        rtllib_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/requant/hdl/"

        subst = {
            "$TOP_MODULE_NAME$": top_module_name,
            "$VERSION$": str(info["version"]),
            "$K$": str(k),
            "$N$": str(n),
            "$C$": str(num_channels),
            "$PE$": str(pe),
            "$TAP_MIN$": str(info["tap_min"]),
            "$TAP_MAX$": str(info["tap_max"]),
            "$IN_STREAM_WIDTH$": str(info["in_stream_width"]),
            "$OUT_STREAM_WIDTH$": str(info["out_stream_width"]),
            "$SCALE_STREAM_WIDTH$": str(info["scale_stream_width"]),
            "$BIAS_STREAM_WIDTH$": str(info["bias_stream_width"]),
        }

        # Verilog wrapper (.v) for IP packaging
        with open(rtllib_dir + "requant_wrapper_decoupled_template.v", "r") as f:
            v_code = f.read()
        for key, val in subst.items():
            v_code = v_code.replace(key, val)
        with open(os.path.join(code_gen_dir, top_module_name + ".v"), "w") as f:
            f.write(v_code)

        self.set_nodeattr("gen_top_module", top_module_name)

        # Emit memstream init files (.dat) via the shared generate_params path,
        # plus the packed words as .npy for standalone rtlsim.
        scale_words, bias_words = self.generate_params(model, code_gen_dir, fpgapart)
        np.save(
            os.path.join(code_gen_dir, "scale_words.npy"),
            np.array(scale_words, dtype=object),
        )
        np.save(
            os.path.join(code_gen_dir, "bias_words.npy"),
            np.array(bias_words, dtype=object),
        )

        # Emit the two memstream wrappers (scale + bias)
        node_name = self.onnx_node.name
        self.generate_hdl_memstream(
            fpgapart,
            name=node_name + "_scale",
            depth=cf,
            width=info["scale_stream_width"],
            init_file=os.path.join(code_gen_dir, "scale_memblock.dat"),
            ram_style="auto",
        )
        self.generate_hdl_memstream(
            fpgapart,
            name=node_name + "_bias",
            depth=cf,
            width=info["bias_stream_width"],
            init_file=os.path.join(code_gen_dir, "bias_memblock.dat"),
            ram_style="auto",
        )

    def get_rtl_file_list(self, abspath=False):
        """Return list of RTL files needed for this node."""
        rtllib_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/requant/hdl/"
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        top_module = self.get_nodeattr("gen_top_module")
        if top_module == "":
            top_module = self.get_verilog_top_module_name()

        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            rtl_files = [
                rtllib_dir + "queue.sv",
                rtllib_dir + "requant_decoupled.sv",
                rtllib_dir + "requant_axi_decoupled.sv",
                # generated Verilog wrapper (directly instantiates the SV core)
                os.path.join(code_gen_dir, top_module + ".v"),
            ]
        else:
            rtl_files = [
                rtllib_dir + "queue.sv",
                rtllib_dir + "requant.sv",
                rtllib_dir + "requant_axi.sv",
                # generated SystemVerilog impl + Verilog stub wrapper
                os.path.join(code_gen_dir, top_module + "_impl.sv"),
                os.path.join(code_gen_dir, top_module + ".v"),
            ]

        if abspath:
            return rtl_files
        else:
            return [os.path.basename(f) for f in rtl_files]

    def code_generation_ipi(self):
        """Return the Vivado IPI Tcl commands that instantiate this node."""
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            return self._code_generation_ipi_decoupled()

        sourcefiles = self.get_rtl_file_list(abspath=True)

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def _code_generation_ipi_decoupled(self):
        """Instantiate the decoupled core plus the scale/bias memstreams."""
        node_name = self.onnx_node.name
        top_module = self.get_nodeattr("gen_top_module")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        source_target = "./ip/verilog/rtl_ops/%s" % node_name

        cmd = ["file mkdir %s" % source_target]

        # Hierarchy with external data in/out pins (params are internal)
        cmd.append("create_bd_cell -type hier %s" % node_name)
        cmd.append("create_bd_pin -dir I -type clk /%s/ap_clk" % node_name)
        cmd.append("create_bd_pin -dir I -type rst /%s/ap_rst_n" % node_name)
        cmd.append(
            "create_bd_intf_pin -mode Master "
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/out0_V" % node_name
        )
        cmd.append(
            "create_bd_intf_pin -mode Slave "
            "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/in0_V" % node_name
        )

        # MLO: expose per-iteration set-select slave pins for scale (in1_V) and
        # bias (in2_V). These carry the memstream set index driven by the
        # FINNLoop stream-tap graph; standalone (SETS=1) leaves them absent.
        mlo = self.get_nodeattr("mlo_max_iter") > 0
        if mlo:
            for ext_pin in ["in1_V", "in2_V"]:
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, ext_pin)
                )

        # Compute core
        for f in self.get_rtl_file_list(abspath=True):
            cmd.append("add_files -copy_to %s -norecurse %s" % (source_target, f))
        cmd.append(
            "create_bd_cell -type module -reference %s /%s/%s" % (top_module, node_name, node_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins %s/%s/ap_clk]"
            % (node_name, node_name, node_name)
        )
        cmd.append(
            "connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins %s/%s/ap_rst_n]"
            % (node_name, node_name, node_name)
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/in0_V] "
            "[get_bd_intf_pins %s/%s/in0_V]" % (node_name, node_name, node_name)
        )
        cmd.append(
            "connect_bd_intf_net [get_bd_intf_pins %s/out0_V] "
            "[get_bd_intf_pins %s/%s/out0_V]" % (node_name, node_name, node_name)
        )

        # Shared memstream sources
        axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
        ms_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
        for f in [axi_dir + "axilite.sv", ms_dir + "memstream_axi.sv", ms_dir + "memstream.sv"]:
            cmd.append("add_files -copy_to %s -norecurse %s" % (source_target, f))

        # Scale + bias memstreamers, each feeding the matching core param port.
        # In MLO mode the external set-select pin (in1_V/in2_V) drives the
        # memstreamer's s_axis_0; standalone leaves s_axis_0 unwired (SETS=1).
        for suffix, core_port, ext_pin in [
            ("_scale", "s_scale_V", "in1_V"),
            ("_bias", "s_bias_V", "in2_V"),
        ]:
            # Discover the generated wrapper by suffix rather than reconstructing
            # from node_name: inside a FINNLoop body the node is renamed between
            # generate_hdl (files carry the loop-prefixed name) and ipi time, so
            # the module name lives in gen_top_module / the filename, not in
            # self.onnx_node.name (mirrors MVAU matrixvectoractivation.py:1176).
            file_suffix = suffix + "_memstream_wrapper.v"
            wrapper_fname = None
            for fname in os.listdir(code_gen_dir):
                if fname.endswith(file_suffix):
                    wrapper_fname = fname
            assert wrapper_fname is not None, "Requant decoupled: could not find %s in %s" % (
                file_suffix,
                code_gen_dir,
            )
            wrapper_file = os.path.join(code_gen_dir, wrapper_fname)
            cmd.append("add_files -copy_to %s -norecurse %s" % (source_target, wrapper_file))
            strm_mod = wrapper_fname[:-2]
            strm_inst = node_name + suffix + "_wstrm"
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s" % (strm_mod, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins %s/%s/ap_clk]"
                % (node_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins %s/%s/ap_clk2x]"
                % (node_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins %s/%s/ap_rst_n]"
                % (node_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, strm_inst, node_name, node_name, core_port)
            )
            if mlo:
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/s_axis_0]" % (node_name, ext_pin, node_name, strm_inst)
                )

        cmd.append("save_bd_design")
        return cmd

    def execute_node(self, context, graph):
        """Execute the node, using RTL simulation if exec_mode is rtlsim."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim":
            node = self.onnx_node
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            decoupled = self.get_nodeattr("mem_mode") == "internal_decoupled"

            # Process input 0 (data tensor)
            inp = node.input[0]
            exp_ishape = tuple(self.get_normal_input_shape(0))
            folded_ishape = self.get_folded_input_shape(0)
            inp_val = context[inp]
            assert str(inp_val.dtype) == "float32", "Input datatype is not float32"
            assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
            export_idt = self.get_input_datatype(0)

            reshaped_input = inp_val.reshape(folded_ishape)
            np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)
            nbits = self.get_instream_width(0)
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )

            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }

            if decoupled:
                # A SETS>1 (MLO) memstream needs a per-iteration set-select
                # index on s_axis_0, which the standalone .npy streaming path
                # cannot supply. Such nodes must be executed via the stitched
                # FINNLoop rtlsim, which drives the set index in hardware.
                assert self.get_nodeattr("mlo_max_iter") == 0, (
                    "%s: standalone rtlsim cannot drive the set-select index; "
                    "a SETS>1 (MLO) Requant must be executed via FINNLoop." % node.name
                )
                # Feed the two decomposed parameter streams alongside the data.
                # The memstream cycles CF words; replicate per input vector.
                scale_words = list(
                    np.load(os.path.join(code_gen_dir, "scale_words.npy"), allow_pickle=True)
                )
                bias_words = list(
                    np.load(os.path.join(code_gen_dir, "bias_words.npy"), allow_pickle=True)
                )
                num_vectors = int(np.prod(self.get_nodeattr("numInputVectors")))
                io_dict["inputs"]["s_scale"] = [int(w) for w in scale_words] * num_vectors
                io_dict["inputs"]["s_bias"] = [int(w) for w in bias_words] * num_vectors

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)

            # Process output
            rtlsim_output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype(0)
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width(0)
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape(0)
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            # Load and reshape output
            exp_oshape = tuple(self.get_normal_output_shape(0))
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output

            assert (
                context[node.output[0]].shape == exp_oshape
            ), "Output shape doesn't match expected shape."
        else:
            # Use base class Python execution
            Requant.execute_node(self, context, graph)
