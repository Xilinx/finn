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

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


class PrepareRTLSim(NodeLocalTransformation):
    """For a graph with generated RTL sources (after HLSSynth_IPGen), create a
    Verilator emulation library for each node to prepare for rtlsim
    execution and set the rtlsim_so property to the path to the generated
    emulation library.

    To use these libraries, exec_mode must be set to "rtlsim" (using
    SetExecMode) and the model has to be executed using execute_onnx() from
    finn.core.onnx_exec

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(self, num_workers=None):
        super().__init__(num_workers=num_workers)

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if node.domain == "finn":
            backend_attribute = util.get_by_name(node.attribute, "backend")
            if backend_attribute is not None:
                backend_value = backend_attribute.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    try:
                        # lookup op_type in registry of CustomOps
                        inst = registry.custom_op[op_type](node)
                        # ensure that code is generated
                        assert (
                            inst.get_nodeattr("code_gen_dir_ipgen") != ""
                        ), """Node
                        attribute "code_gen_dir_ipgen" is not set. Please run
                        HLSSynth_IPGen first."""
                        if PyVerilator is None:
                            raise ImportError(
                                "Installation of PyVerilator is required."
                            )
                        inst.prepare_rtlsim()
                        # ensure that executable path is now set
                        assert (
                            inst.get_nodeattr("rtlsim_so") != ""
                        ), "Failed to prepare RTLSim, no rtlsim_so attribute found."
                    except KeyError:
                        # exception if op_type is not supported
                        raise Exception(
                            "Custom op_type %s is currently not supported." % op_type
                        )
        return (node, False)
