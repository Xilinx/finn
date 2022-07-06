# Copyright (c) 2020, Xilinx
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
import qonnx.custom_op.registry as registry
from qonnx.transformation.base import Transformation

from finn.util.fpgadataflow import is_fpgadataflow_node


class ReplaceVerilogRelPaths(Transformation):
    """Convert ./ relative file paths to absolute ones for generated Verilog"""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        for node in model.graph.node:
            if is_fpgadataflow_node(node) is True:
                try:
                    # lookup op_type in registry of CustomOps
                    inst = registry.getCustomOp(node)
                    # find the IP gen dir
                    ipgen_path = inst.get_nodeattr("ipgen_path")
                    if ipgen_path is not None and os.path.isdir(ipgen_path):
                        for dname, dirs, files in os.walk(ipgen_path):
                            for fname in files:
                                if fname.endswith(".v"):
                                    fpath = os.path.join(dname, fname)
                                    with open(fpath, "r") as f:
                                        s = f.read()
                                    old = '$readmemh(".'
                                    new = '$readmemh("%s' % dname
                                    s = s.replace(old, new)
                                    old = '"./'
                                    new = '"%s/' % dname
                                    s = s.replace(old, new)
                                    with open(fpath, "w") as f:
                                        f.write(s)
                except KeyError:
                    pass
        return (model, False)
