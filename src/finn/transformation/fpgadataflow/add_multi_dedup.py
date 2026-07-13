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

# All per-node add_multi.sv copies share one module name, so a flat-namespace
# compile (whole-design xsi sim, stitched IP, OOC synth) keeps only the last
# one. Unifying every copy to the superset of all CATCH_COMP specs keeps
# it valid for every node.


def build_unified_add_multi_body(model):
    """Return add_multi.sv text whose CATCH_COMP entries are the union of all
    MVAU_rtl nodes' ``add_multi_comp_specs``."""
    all_specs = set()
    for node in model.graph.node:
        if node.op_type == "MVAU_rtl":
            specs_str = getCustomOp(node).get_nodeattr("add_multi_comp_specs")
            if specs_str:
                for spec in specs_str.split(";"):
                    n, w, d = map(int, spec.split(","))
                    all_specs.add((n, w, d))

    rtllib_template = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/add_multi.sv")
    with open(rtllib_template, "r") as f:
        template = f.read()

    if all_specs:
        entries = "\n".join(f"\t`CATCH_COMP({n},{w},{d})" for n, w, d in sorted(all_specs)) + "\n"
    else:
        entries = ""

    marker = "\t// FINN_GENERATED_COMP_ENTRIES\n"
    if marker not in template:
        raise RuntimeError(
            "FINN_GENERATED_COMP_ENTRIES marker not found in finn-rtllib/mvu/add_multi.sv!"
        )
    return template.replace(marker, entries + marker)


def generate_unified_add_multi(model, build_dir):
    """Write the unified add_multi.sv into ``build_dir`` (used by OOC synth,
    which flattens all sources into one directory)."""
    with open(os.path.join(build_dir, "add_multi.sv"), "w") as f:
        f.write(build_unified_add_multi_body(model))


def unify_add_multi_per_node(model):
    """Overwrite each MVAU_rtl node's add_multi.sv with the unified body. Only
    MVAU_rtl uses the LUT compressor (add_multi) path; VVAU_rtl targets the
    DSP58 core and never instantiates add_multi compressors."""
    unified = build_unified_add_multi_body(model)
    for node in model.graph.node:
        if node.op_type == "MVAU_rtl":
            code_gen_dir = getCustomOp(node).get_nodeattr("code_gen_dir_ipgen")
            if code_gen_dir and os.path.isdir(code_gen_dir):
                with open(os.path.join(code_gen_dir, "add_multi.sv"), "w") as f:
                    f.write(unified)
