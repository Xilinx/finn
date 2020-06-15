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
from finn.transformation import Transformation
from finn.util.basic import get_num_default_workers
import multiprocessing as mp
import copy

def _codegen_single_node(node, model):
    """Calls C++ code generation for one node. Resulting code can be used
    to simulate node using cppsim."""

    op_type = node.op_type
    try: 
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type](node)
        # get the path of the code generation directory
        code_gen_dir = inst.get_nodeattr("code_gen_dir_cppsim")
        # ensure that there is a directory
        if code_gen_dir == "" or not os.path.isdir(code_gen_dir):
            code_gen_dir = make_build_dir(
                prefix="code_gen_cppsim_" + str(node.name) + "_"
            )
            inst.set_nodeattr("code_gen_dir_cppsim", code_gen_dir)
        # ensure that there is generated code inside the dir
        inst.code_generation_cppsim(model)
    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)


class PrepareCppSim(Transformation):
    """Call custom implementation to generate code for single custom node
    and create folder that contains all the generated files.
    All nodes in the graph must have the fpgadataflow backend attribute.

    Outcome if succesful: Node attribute "code_gen_dir_cppsim" contains path to folder
    that contains generated C++ code that can be used to simulate node using cppsim.
    The subsequent transformation is CompileCppSim"""

    def __init__(self, num_workers=None):
        super().__init__()
        if num_workers is None:
            self._num_workers = get_num_default_workers()
        else:
            self._num_workers = num_workers
        assert self._num_workers >= 0, "Number of workers must be nonnegative."
        if self._num_workers == 0:
            self._num_workers = mp.cpu_count()

    def prepareCppSim_node(self, node):
        print(node.name)
        if is_fpgadataflow_node(node) is True:
            _codegen_single_node(node, self.model)
        return (node, False)


    def apply(self, model):
        # Remove old nodes from the current model
        self.model = copy.deepcopy(model)
        old_nodes = []
        for i in range(len(model.graph.node)):
            old_nodes.append(model.graph.node.pop())

        # Execute transformation in parallel
        with mp.Pool(self._num_workers) as p:
            new_nodes_and_bool = p.map(self.prepareCppSim_node, old_nodes, chunksize=1)

        # extract nodes and check if the transformation needs to run again
        # Note: .pop() had initially reversed the node order
        run_again = False
        for node, run in reversed(new_nodes_and_bool):
            # Reattach new nodes to old model
            model.graph.node.append(node)
            if run is True:
                run_again = True

        return (model, run_again)