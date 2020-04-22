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

import finn.custom_op.registry as registry
import finn.util.basic as util
from finn.transformation import NodeLocalTransformation


class HLSSynth_IPGen(NodeLocalTransformation):
    """For each node: generate IP block from code in folder
    that is referenced in node attribute "code_gen_dir_ipgen"
    and save path of generated project in node attribute "ipgen_path".
    All nodes in the graph must have the fpgadataflow backend attribute.

    This transformation calls Vivado HLS for synthesis, so it will run for
    some time (several minutes)
    
    * NUM_DEFAULT_WORKERS (int or None) number of parallel workers. Default is 1. None: Use all available cores.
    """

    def __init__(self, NUM_DEFAULT_WORKERS=1):
        super().__init__(NUM_DEFAULT_WORKERS=NUM_DEFAULT_WORKERS)
    
    def applyNodeLocal(self, node):
        op_type = node.op_type
        if node.domain == "finn":
            backend_attribute = util.get_by_name(node.attribute, "backend")
            if backend_attribute is None:
                return (node, False)
            backend_value = backend_attribute.s.decode("UTF-8")
            if backend_value == "fpgadataflow":
                try:
                    # lookup op_type in registry of CustomOps
                    inst = registry.custom_op[op_type](node)
                    # ensure that code is generated
                    assert (
                        inst.get_nodeattr("code_gen_dir_ipgen") != ""
                    ), """Node
                    attribute "code_gen_dir_ipgen" is empty. Please run
                    transformation CodeGen_ipgen first."""
                    # call the compilation function for this node
                    inst.ipgen_singlenode_code()
                    # ensure that executable path is now set
                    assert (
                        inst.get_nodeattr("ipgen_path") != ""
                    ), """Transformation
                    HLSSynth_IPGen was not successful. Node attribute "ipgen_path"
                    is empty."""
                except KeyError:
                    # exception if op_type is not supported
                    raise Exception(
                        "Custom op_type %s is currently not supported." % op_type
                    )
        
        return (node, False)
