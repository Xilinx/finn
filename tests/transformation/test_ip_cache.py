# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import pytest

import finn.util.create as create
from finn.core.datatype import DataType
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.cache_ip import CacheIP
from finn.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

@pytest.mark.parametrize("bitwidth", [DataType.INT2])
def test_ip_cache(bitwidth):
    w = bitwidth
    a = bitwidth
    layer_spec = [
        {"mw": 10, "mh": 10, "simd": 10, "pe": 10, "idt": a, "wdt": w, "act": None, "mem_mode": "decoupled"},
        {"mw": 10, "mh": 10, "simd": 10, "pe": 10, "idt": a, "wdt": w, "act": None, "mem_mode": "decoupled"},
        {"mw": 10, "mh": 10, "simd": 10, "pe": 10, "idt": a, "wdt": w, "act": None, "mem_mode": "decoupled"},
        {"mw": 10, "mh": 10, "simd": 5, "pe": 10, "idt": a, "wdt": w, "act": a, "mem_mode": "decoupled"},
        {"mw": 10, "mh": 10, "simd": 10, "pe": 10, "idt": a, "wdt": w, "act": a},
    ]

    model = create.hls_random_mlp_maker(layer_spec)

    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(CacheIP())
    model = model.transform(HLSSynthIP(ip_cache_file=model.get_metadata_prop("ip_cache")))
    model = model.transform(CreateStitchedIP("xc7z020clg400-1", 5))
