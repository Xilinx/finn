############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
import warnings
from qonnx.core.datatype import DataType
from typing import Optional

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.input_gen import input_gen_buffer_depth


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


class _NestSim:
    """Simulate the Nest<R, W, N, C, V...> HLS template from input_gen.hpp.

    Models the read-pointer and free-pointer update logic of the HLS reorder
    buffer, including loop termination and counter reset behavior.
    """

    def __init__(self, R, W, *rest):
        self.R = R
        self.W = W
        self.is_terminal = len(rest) == 0
        if self.is_terminal:
            self._rp_rewind = 0
            self._fp_rewind = 0
            self.max_rp_retract = 0
        else:
            self.N = rest[0]
            self.C = rest[1]
            # As in finn-hlslib input_gen.hpp: the inner loop may advance the free
            # pointer in lockstep if its furthest advance, (N-1)*C plus its own
            # (hypothetical) free-pointer rewind, never carries it beyond W.
            self.R_INNER = (
                R
                and (self.C > 0)
                and ((self.N - 1) * self.C + _NestSim(True, self.C, *rest[2:])._fp_rewind <= W)
            )
            self.inner = _NestSim(self.R_INNER, self.C, *rest[2:])
            self._rp_rewind = (self.N - 1) * self.C + self.inner._rp_rewind
            self._fp_rewind = (self.N - 1) * self.C + self.inner._fp_rewind if self.R_INNER else 0
            self.terminal_rp_inc = W - self._rp_rewind
            self.cnt = self.N - 2
            self.max_rp_retract = max(-self.terminal_rp_inc, self.inner.max_rp_retract)

    def tick(self):
        if self.is_terminal:
            return self.W, (self.W if self.R else 0), True
        rp_inc, fp_inc, term = self.inner.tick()
        if term:
            if self.cnt < 0:
                rp_inc = self.terminal_rp_inc
                if self.R:
                    fp_inc = self.W - self._fp_rewind
                self.cnt = self.N - 2
                return rp_inc, fp_inc, True
            else:
                self.cnt -= 1
                return rp_inc, fp_inc, False
        return rp_inc, fp_inc, False


class OuterShuffle(HWCustomOp):
    """Abstraction layer for HW OuterShuffle (rearrange and transpose) layers.
    Only permutations that do not effect the inner most dimensions are feasible"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        # SIMD must divide the innermost axis, which an outer shuffle cannot move.
        last_dim = self.get_nodeattr("transpose_in_shape")[-1]
        SIMD = self.get_nodeattr("SIMD")
        if last_dim % SIMD != 0:
            new_simd = auto_size_simd(last_dim, SIMD)
            if new_simd is not None:
                self.set_nodeattr("SIMD", new_simd)
            else:
                raise RuntimeError("Unable to determine a new SIMD value for this transpose.")

    def get_nodeattr_types(self):
        my_attrs = {
            "data_type": ("s", True, ""),
            "transpose_in_shape": ("ints", True, []),
            "in_shape": ("ints", True, []),
            "transpose_out_shape": ("ints", True, []),
            "out_shape": ("ints", True, []),
            "loop_coeffs": ("ints", True, []),
            "perm": ("ints", True, []),
            "SIMD": ("i", False, 1),
            "NumChannels": ("i", False, 128),
            "original_node_name": ("s", False, ""),  # Track original shuffle name for SIMD config
            "original_simd": ("i", False, 1),  # Track original shuffle SIMD for config export
            # Storage of the input_gen reorder buffer. "auto" leaves the choice to
            # HLS, which binds buffers deeper than 262144 words to URAM by itself.
            # Affects resources only, not the II (see get_input_gen_ii).
            "ram_style": ("s", False, "auto", {"auto", "block", "distributed", "ultra"}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("in_shape")

    def get_normal_output_shape(self, ind=0):
        return self.get_nodeattr("out_shape")

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        input_reshaped = input_data.reshape(self.get_nodeattr("transpose_in_shape"))
        transposed = np.transpose(input_reshaped, axes=self.get_nodeattr("perm"))
        output_reshaped = transposed.reshape(self.get_nodeattr("out_shape"))
        context[node.output[0]] = output_reshaped

    def get_input_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            warn_str = (
                f"data_type changing for {node.name}: {str(self.get_input_datatype())} -> {str(dt)}"
            )
            warnings.warn(warn_str)
        self.set_nodeattr("data_type", dt.name)
        model.set_tensor_datatype(node.output[0], dt)

    def verify_node(self):
        raise NotImplementedError("This function is not yet immplemented.")

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_output_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_oshape[-1] % simd == 0, "SIMD must divide into the innermost output dimension"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "SIMD must divide into the innermost input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_input_gen_nest(self):
        """The loop nest the HLS kernel instantiates input_gen<> with.

        Returns (num_words, extents, coeffs): words per frame after SIMD folding,
        the output loop extents and their input-side coefficients, derived from
        transpose_in_shape, perm and SIMD exactly like the HLS backend's docompute.
        """
        simd = self.get_nodeattr("SIMD")
        in_shape = list(self.get_nodeattr("transpose_in_shape"))
        perm = list(self.get_nodeattr("perm"))
        extents = [in_shape[p] for p in perm]
        adjusted = in_shape + [1]
        input_strides = [int(np.prod(adjusted[i + 1 :])) for i in range(len(in_shape))]
        coeffs = [input_strides[p] for p in perm]
        extents[-1] = int(extents[-1] / simd)
        coeffs = [1 if x == 1 else int(x / simd) for x in coeffs]
        return int(np.prod(extents)), extents, coeffs

    def get_buffer_depth(self):
        """Depth (BUF_SIZE, in words) of the input_gen reorder buffer, as the
        stall-free sizing of finn-hlslib input_gen() dimensions it."""
        num_words, extents, coeffs = self.get_input_gen_nest()
        return input_gen_buffer_depth(num_words, extents, coeffs)

    def get_input_gen_ii(self):
        """Pipeline II the HLS kernel achieves for its reorder buffer.

        Always 1: buffers up to 262144 words read with latency 1, and deeper
        ones, which HLS maps to a latency-pipelined memory cascade (URAM), are
        read ahead into a small queue by input_gen() (finn-hlslib 5f0aec0+), so
        the cascade latency no longer enters the loop recurrence. Before that,
        deep buffers dropped to II=3 (Vivado 2024.2) or 1 + depth/131072
        (2025.1+). Explicit storage (ram_style) does not change the II either.
        """
        return 1

    def get_exp_cycles(self):
        """Estimate cycles by simulating the input_gen HLS pipeline.

        Derives the loop nest from transpose_in_shape, perm and SIMD
        (get_input_gen_nest), the buffer depth from the stall-free sizing of
        finn-hlslib input_gen() (get_buffer_depth) and its pipeline II
        (get_input_gen_ii).

        The HLS pipeline has three stall sources:
        1. WP_DELAY (=4): write-pointer pipeline latency before reads begin
        2. Read stalls: consumer waits for data (rp >= wp_delayed)
        3. Write stalls: producer blocked by full buffer (wp - fp >= buf_size)
        """
        total_elems, extents, coeffs = self.get_input_gen_nest()
        interleaved = [int(item) for pair in zip(extents, coeffs) for item in pair]
        nest = _NestSim(True, total_elems, *tuple(interleaved))
        WP_DELAY = 4
        buf_size = self.get_buffer_depth()
        pipeline_ii = self.get_input_gen_ii()

        # Simulate the input_gen pipeline at II=1.
        # Models the wp delay pipeline, finite buffer backpressure,
        # and the Nest-driven read pointer pattern.
        wp = [0] * WP_DELAY
        rp = 0
        fp = 0
        ovld = False
        input_consumed = 0
        output_produced = 0
        cycle = 0

        while output_produced < total_elems and cycle < total_elems * 10:
            cycle += 1

            # Shift write pointer delay pipeline
            for i in range(WP_DELAY - 1, 0, -1):
                wp[i] = wp[i - 1]

            # Write into buffer if space available
            if wp[0] - fp < buf_size and input_consumed < total_elems:
                wp[0] += 1
                input_consumed += 1

            # Drain output buffer
            if ovld:
                output_produced += 1
                ovld = False

            # Refill output buffer via Nest tick
            if not ovld and rp < wp[WP_DELAY - 1]:
                rp_inc, fp_inc, _ = nest.tick()
                rp += rp_inc
                fp += fp_inc
                ovld = True

        if ovld:
            output_produced += 1

        return cycle * pipeline_ii
