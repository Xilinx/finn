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

import finn.custom_op.registry as registry
from finn.transformation import Transformation
from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import is_fpgadataflow_node


def _codegen_single_node(node, model, fpgapart, clk):
    """Calls C++ code generation for one node. Resulting code can be used
    to generate a Vivado IP block for the node."""

    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type](node)
        # get the path of the code generation directory
        code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
        # ensure that there is a directory
        if code_gen_dir == "" or not os.path.isdir(code_gen_dir):
            code_gen_dir = make_build_dir(
                prefix="code_gen_ipgen_" + str(node.op_type) + "_"
            )
            inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        # ensure that there is generated code inside the dir
        inst.code_generation_ipgen(model, fpgapart, clk)
    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)


class CodeGen_ipgen(Transformation):
    """Call custom implementation to generate code for single custom node
    and create folder that contains all the generated files.
    All nodes in the graph must have the fpgadataflow backend attribute and
    transformation gets additional arguments:

    * fpgapart (string)

    * clk in ns (int)

    Outcome if succesful: Node attribute "code_gen_dir_ipgen" contains path to folder
    that contains generated C++ code that can be used to generate a Vivado IP block.
    The subsequent transformation is HLSSynth_IPGen"""

    def __init__(self, fpgapart, clk):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk = clk

    def apply(self, model):
        for node in model.graph.node:
            if is_fpgadataflow_node(node) is True:
                _codegen_single_node(node, model, self.fpgapart, self.clk)
        return (model, False)
