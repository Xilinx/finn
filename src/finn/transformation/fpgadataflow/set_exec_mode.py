# Copyright (C) 2020, Xilinx, Inc.
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

import qonnx.custom_op.registry as registry
from qonnx.transformation.base import Transformation

from finn.util.fpgadataflow import is_hls_node, is_rtl_node


class SetExecMode(Transformation):
    """Set attribute exec_mode in all fpgadataflow nodes to specify which
    kind of execution should be used ("cppsim" or "rtlsim").
    Note that RTL components do not support cppsim.
    For now, only a model consisting of 100% of HLS layers can be executed
    using cppsim."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def apply(self, model):
        mode = self.mode
        # if "cppsim" selected, check if model does not contain RTL layers
        if mode == "cppsim" and any(is_rtl_node(node) for node in model.graph.node):
            raise Exception(
                """Model contains RTL layers,
               cppsim can only be used on models consisting of HLS layers
               and non fpgadataflow nodes."""
            )
        else:
            for node in model.graph.node:
                op_type = node.op_type
                if is_hls_node(node) or is_rtl_node(node):
                    try:
                        # lookup op_type in registry of CustomOps
                        inst = registry.getCustomOp(node)
                        # set sim_mode accordingly to argument mode
                        inst.set_nodeattr("exec_mode", mode)
                        # ensure that sim_mode is now set
                        assert (
                            inst.get_nodeattr("exec_mode") != ""
                        ), """Transformation
                            was not successful. Node attribute "exec_mode" is not set"""
                    except KeyError:
                        # exception if op_type is not supported
                        raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (model, False)
