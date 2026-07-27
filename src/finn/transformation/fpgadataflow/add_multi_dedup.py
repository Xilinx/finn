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

import os
from qonnx.custom_op.registry import getCustomOp

# add_multi.sv is now a single parametric module (it wraps the elaboration-
# scheduled add_multi_sv), so every node's copy is byte-identical. These helpers
# stay a stable hook (prepare_ip / synth_ooc) that rewrites the canonical body.


def build_unified_add_multi_body(model):
    """Return the canonical add_multi.sv body (the rtllib source verbatim)."""
    rtllib_template = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/add_multi.sv")
    with open(rtllib_template, "r") as f:
        return f.read()


def generate_unified_add_multi(model, build_dir):
    """Write the canonical add_multi.sv into ``build_dir`` (flat OOC synth dir)."""
    with open(os.path.join(build_dir, "add_multi.sv"), "w") as f:
        f.write(build_unified_add_multi_body(model))


def unify_add_multi_per_node(model):
    """Overwrite each MVAU_rtl node's add_multi.sv with the canonical body.
    VVAU_rtl targets the DSP58 core and never uses the add_multi path."""
    unified = build_unified_add_multi_body(model)
    for node in model.graph.node:
        if node.op_type == "MVAU_rtl":
            code_gen_dir = getCustomOp(node).get_nodeattr("code_gen_dir_ipgen")
            if code_gen_dir and os.path.isdir(code_gen_dir):
                with open(os.path.join(code_gen_dir, "add_multi.sv"), "w") as f:
                    f.write(unified)
