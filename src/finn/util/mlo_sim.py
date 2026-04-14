# Copyright (C) 2025, Advanced Micro Devices, Inc.
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


# This module contains helpers for handling the MLO rtlsimulation. It instantiates
# aximm simulation tasks for handling the aximm interfaces.

import numpy as np
import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import Callable

from finn import xsi

SimEngine = xsi.SimEngine if xsi.is_available() else None


def is_mlo(model: ModelWrapper) -> bool:
    """Returns True if the model is an MLO model, false otherwise"""
    for node in model.graph.node:
        if node.op_type == "FINNLoop":
            return True
    return False


def dat_file_to_numpy_array(file_path):
    byte_values = []

    with open(file_path, "r") as file:
        for line in file:
            hex_string = line.strip()
            for i in range(len(hex_string) - 2, -1, -2):
                byte = hex_string[i : i + 2]
                byte_values.append(int(byte, 16))
            if len(hex_string) % 2 == 1:  # Dealing when we have a leftover nibble
                byte_values.append(int(hex_string[-1], 16))
    byte_array = np.array(byte_values, dtype=np.uint8)

    return byte_array


def mlo_prehook_func_factory(node) -> Callable[[SimEngine], None]:
    """Factory that will construct a prehook function to
    setup the axi memory mapped interfaces for MLO validation.
    """

    # Get the FINNLoop
    finnloop_op = getCustomOp(node)

    finnloop_body = finnloop_op.get_nodeattr("body")

    mvau_hbm_weights = {}
    extern_idx = 0
    for idx, lb_inp in enumerate(finnloop_body.graph.input):
        downstream = finnloop_body.find_consumer(lb_inp.name)
        if downstream.op_type.startswith("MVAU"):
            mvau_hbm_weights[idx] = {}
            mvau_hbm_weights[idx]["name"] = lb_inp.name
            code_gen_dir = finnloop_op.get_nodeattr('code_gen_dir_ipgen')
            npy_file = f"{code_gen_dir}/input1_MVAU_rtl_id_{idx}.npy"
            datfile = f"{code_gen_dir}/memblock_MVAU_rtl_id_{idx}.dat"
            mvau_op = getCustomOp(downstream)
            mh = mvau_op.get_nodeattr("MH")
            mw = mvau_op.get_nodeattr("MW")
            wdt_width = mvau_op.get_input_datatype(1).bitwidth()
            # Must match RTL LAYER_OFFS: align to AXI bus width (256 bits = 32 bytes)
            axi_bytes = 32
            raw_layer_bytes = (mh * mw * wdt_width + 7) // 8
            layer_offs = (raw_layer_bytes + axi_bytes - 1) & ~(axi_bytes - 1)
            if os.path.isfile(npy_file):
                weight_npy = np.load(npy_file)
                # Pack npy values into byte array for AXI-MM
                # Memory byte order matches npy_to_rtlsim_input packing
                tinner = weight_npy.shape[-1]
                words_per_iter = raw_layer_bytes // tinner
                flat = weight_npy.reshape(-1, tinner)
                n_iters = len(flat) // words_per_iter
                weight_bytes = []
                for it in range(n_iters):
                    for row in flat[it * words_per_iter:(it + 1) * words_per_iter]:
                        for val in row:
                            weight_bytes.append(int(val) & 0xFF)
                    # Pad to layer_offs boundary
                    pad = layer_offs - raw_layer_bytes
                    weight_bytes.extend([0] * pad)
                mvau_hbm_weights[idx]["value"] = np.array(weight_bytes, dtype=np.uint8)
            else:
                mvau_hbm_weights[idx]["value"] = dat_file_to_numpy_array(datfile)
            mvau_hbm_weights[idx]["extern_idx"] = extern_idx
            mvau_hbm_weights[idx]["extern_name"] = f"m_axi_MVAU_id_{idx}"
            extern_idx += 1

    def mlo_rtlsim_prehook(sim):
        sim.aximm_queue("m_axi_hbm")
        for name, intf in mvau_hbm_weights.items():
            sim.aximm_ro_image(intf["extern_name"], 0, intf["value"].flatten())

    return mlo_rtlsim_prehook
