# Copyright (c) 2020, Xilinx
# Copyright (C) 2026, Advanced Micro Devices, Inc.
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

from qonnx.transformation.base import Transformation

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.vivado import parse_ooc_synth_results


class SynthOutOfContext(Transformation):
    """Compatibility wrapper for the historical standalone OOC synth transform.

    The OOC flow now lives in CreateStitchedIP via ``run_pnr=True`` so that DCP,
    timing, and utilization reports are generated from the same Vivado project.
    """

    def __init__(self, part, clk_period_ns, clk_name="ap_clk"):
        super().__init__()
        self.part = part
        self.clk_period_ns = clk_period_ns
        self.clk_name = clk_name

    def apply(self, model):
        assert self.clk_name == "ap_clk", "Only the default ap_clk OOC clock is supported."
        model = model.transform(CreateStitchedIP(self.part, self.clk_period_ns, run_pnr=True))
        vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        ret = parse_ooc_synth_results(vivado_stitch_proj_dir)
        if ret is not None and "BRAM" not in ret:
            ret["BRAM"] = ret.get("BRAM_36K", 0) + ret.get("BRAM_18K", 0) / 2
        model.set_metadata_prop("res_total_ooc_synth", str(ret))
        return (model, False)
