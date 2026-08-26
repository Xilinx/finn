# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.quant import max_int, min_int

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Requant(HWCustomOp):
    """Abstraction layer for HW implementation of Requantization.

    Requantization computes: clip(round(x * scale + bias), min, max)

    This is an alternative to Thresholding for cases where the thresholds
    are uniformly spaced. Instead of comparing against N thresholds, we
    compute the output directly using a multiply-add operation.

    Inputs:
        input[0]: Data tensor to requantize
        input[1]: Scale tensor (per-channel or scalar, stored as initializer)
        input[2]: Bias tensor (per-channel or scalar, stored as initializer)
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # parallelization; channels processed per cycle
            "PE": ("i", False, 1),
            # number of channels
            "NumChannels": ("i", True, 0),
            # FINN DataTypes for inputs, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # Whether to use narrow range (1) or full range (0)
            # Note: RTL backend only supports narrow=0 and unsigned output
            "narrow": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_scale(self, model):
        """Get scale tensor from model initializer (input[1])."""
        if len(self.onnx_node.input) > 1:
            scale = model.get_initializer(self.onnx_node.input[1])
            if scale is not None:
                return scale.flatten()
        # Default: scale = 1.0
        return np.array([1.0], dtype=np.float32)

    def get_bias(self, model):
        """Get bias tensor from model initializer (input[2])."""
        if len(self.onnx_node.input) > 2:
            bias = model.get_initializer(self.onnx_node.input[2])
            if bias is not None:
                return bias.flatten()
        # Default: bias = 0.0
        return np.array([0.0], dtype=np.float32)

    def is_per_channel(self, model):
        """Check if scale/bias are per-channel (vs per-tensor)."""
        scale = self.get_scale(model)
        bias = self.get_bias(model)
        return scale.size > 1 or bias.size > 1

    @staticmethod
    def _derive_mul_widths(version, k):
        """Python port of ``derive_MUL_WIDTHS`` in finn-rtllib/requant/hdl/requant.sv.

        Returns (s, x): the multiplier operand widths implied by the DSP
        ``version`` and the input precision ``k``."""
        if version == 1:
            aw, bw = 25, 18
        elif version == 2:
            aw, bw = 27, 18
        elif version == 3:
            aw, bw = 27, 24
        else:
            raise ValueError(f"Unsupported DSP VERSION {version}.")
        if k <= bw:
            return 25, k
        return bw, (aw if aw < k else k)

    def decompose_params(self, model, version):
        """Port of ``derive_PARAMS`` in finn-rtllib/requant/hdl/requant.sv.

        Decomposes the float32 scale/bias per channel into the fixed-point
        parameters the RTL datapath consumes: an integer ``scale`` mantissa, an
        aligned integer ``bias`` (with the round-half-up constant folded in) and
        a per-channel ``tap`` shift amount. This is the single source of truth
        for the float->fixed-point conversion, used both by the embedded codegen
        and by the decoupled memstream param generation.

        Returns a dict with numpy int arrays shaped ``[PE][CF]``:
        ``scale``, ``bias``, ``tap`` plus scalars ``tap_min``, ``tap_max`` and
        the operand widths ``s_width`` (scale), ``x_width`` (data).
        Raises ``ValueError`` for the same out-of-range conditions the RTL
        asserts (``$error``/``$finish``), so failures surface at build time.
        """
        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")
        cf = num_channels // pe

        idt = self.get_input_datatype(0)
        odt = self.get_output_datatype()
        k = idt.bitwidth()
        n = odt.bitwidth()

        s_width, x_width = self._derive_mul_widths(version, k)

        scale = self.get_scale(model)
        bias = self.get_bias(model)
        if scale.size == 1:
            scale = np.full(num_channels, scale.item(), dtype=np.float32)
        if bias.size == 1:
            bias = np.full(num_channels, bias.item(), dtype=np.float32)
        # [PE][CF] layout, matching requant_rtl.generate_hdl
        scale = scale.astype(np.float32).reshape(cf, pe).T
        bias = bias.astype(np.float32).reshape(cf, pe).T

        scale_fixed = np.zeros((pe, cf), dtype=object)
        bias_fixed = np.zeros((pe, cf), dtype=object)
        tap_arr = np.zeros((pe, cf), dtype=object)

        for p in range(pe):
            for c in range(cf):
                sc_val = scale[p][c]
                bs_val = bias[p][c]
                s_bits = int(np.float32(sc_val).view(np.uint32))
                b_bits = int(np.float32(bs_val).view(np.uint32))

                # Decomposed scale: sm = {1'b1, mant[22:0], 1'b0} (25 bits, *2^-24)
                sm = (1 << 24) | ((s_bits & 0x7FFFFF) << 1)
                se = ((s_bits >> 23) & 0xFF) - 127
                sign_s = (s_bits >> 31) & 1

                # Decomposed bias: bm = {1'b1, mant[22:0]} (24 bits, *2^-23)
                bm = (1 << 23) | (b_bits & 0x7FFFFF)
                be = ((b_bits >> 23) & 0xFF) - 127
                sign_b = (b_bits >> 31) & 1

                tap = (s_width - 2) - se
                if tap < 0:
                    raise ValueError(
                        f"{self.onnx_node.name}: Scale {sc_val} is too large "
                        f"for the output precision."
                    )
                if s_width + x_width + 1 < tap + n:
                    raise ValueError(
                        f"{self.onnx_node.name}: Scale {sc_val} is too small "
                        f"for the output precision."
                    )

                # Mantissa into product datapath
                sm >>= 25 - s_width
                scale_mag = ((sm >> 1) & ((1 << 24) - 1)) + (sm & 1)
                p_scale = -scale_mag if sign_s else scale_mag

                # Align bias mantissa into product datapath
                if be > (x_width - n):
                    raise ValueError(
                        f"{self.onnx_node.name}: Bias {bs_val} overflows the " f"output range."
                    )
                shift = be - (23 - tap)
                if shift < -24:
                    bias_mag = 0
                elif shift < 0:
                    bias_mag = bm >> (-shift)
                else:
                    bias_mag = bm << shift
                p_bias = -bias_mag if sign_b else bias_mag
                # Rounding (round-half-up constant folded into the bias)
                if tap > 0:
                    p_bias += 1 << (tap - 1)
                elif shift < 0:
                    p_bias += (bm >> (-shift - 1)) & 1

                scale_fixed[p][c] = p_scale
                bias_fixed[p][c] = p_bias
                tap_arr[p][c] = tap

        tap_flat = [int(tap_arr[p][c]) for p in range(pe) for c in range(cf)]
        return {
            "scale": scale_fixed,
            "bias": bias_fixed,
            "tap": tap_arr,
            "tap_min": min(tap_flat),
            "tap_max": max(tap_flat),
            "s_width": s_width,
            "x_width": x_width,
        }

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype().name),
                str(idt.name),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            return DataType[self.get_nodeattr("inputDataType")]
        else:
            # Scale and bias are float
            return DataType["FLOAT32"]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_normal_input_shape(self, ind=0):
        """Returns input shape in format [N, H, W, C] or [N, C]."""
        num_input_vecs = self.get_nodeattr("numInputVectors")
        num_channels = self.get_nodeattr("NumChannels")
        return tuple(num_input_vecs + [num_channels])

    def get_normal_output_shape(self, ind=0):
        """Returns output shape."""
        return self.get_normal_input_shape(0)

    def get_folded_input_shape(self, ind=0):
        """Returns folded input shape."""
        if ind == 0:
            normal_shape = self.get_normal_input_shape(0)
            pe = self.get_nodeattr("PE")
            num_channels = self.get_nodeattr("NumChannels")
            fold = num_channels // pe
            return tuple(list(normal_shape[:-1]) + [fold, pe])
        else:
            return self.get_normal_input_shape(ind)

    def get_folded_output_shape(self, ind=0):
        """Returns folded output shape."""
        return self.get_folded_input_shape(0)

    def get_exp_cycles(self):
        """Returns expected number of cycles for execution."""
        return self.get_number_output_values()

    def execute_node(self, context, graph):
        """Execute the requant operation."""
        node = self.onnx_node
        x = context[node.input[0]]

        # Get scale and bias
        scale = (
            self.get_scale(graph.model)
            if hasattr(graph, "model")
            else context.get(node.input[1], np.array([1.0]))
        )
        bias = (
            self.get_bias(graph.model)
            if hasattr(graph, "model")
            else context.get(node.input[2], np.array([0.0]))
        )

        # Get output range from output datatype
        odt = self.get_output_datatype()
        narrow = self.get_nodeattr("narrow")
        signed = 1 if odt.signed() else 0
        bitwidth = odt.bitwidth()
        min_val = min_int(signed, narrow, bitwidth)
        max_val = max_int(signed, narrow, bitwidth)

        # Apply requantization: clip(round(x * scale + bias), min, max)
        # Use floor(x + 0.5) for round-half-up, not np.round which uses banker's rounding
        x_scaled = x * scale + bias
        x_rounded = np.floor(x_scaled + 0.5)
        x_clipped = np.clip(x_rounded, min_val, max_val)

        context[node.output[0]] = x_clipped.astype(np.float32)

    def get_instream_width(self, ind=0):
        """Returns input stream width."""
        if ind == 0:
            pe = self.get_nodeattr("PE")
            idt = self.get_input_datatype(0)
            return pe * idt.bitwidth()
        else:
            # Scale and bias (inputs 1, 2) are embedded, not streamed
            return 0

    def get_outstream_width(self, ind=0):
        """Returns output stream width."""
        pe = self.get_nodeattr("PE")
        odt = self.get_output_datatype()
        return pe * odt.bitwidth()
