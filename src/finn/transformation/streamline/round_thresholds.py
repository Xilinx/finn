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

import numpy as np

from finn.transformation import Transformation


class RoundAndClipThresholds(Transformation):
    """For MultiThreshold nodes operating on integer inputs, round up
    thresholds values to the nearest integer. Additionally, if the input
    is unsigned, sets negative thresholds to zero."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "MultiThreshold":
                idtype = model.get_tensor_datatype(n.input[0])
                T = model.get_initializer(n.input[1])
                Tnew = np.ceil(T)
                if idtype.is_integer() and (T != Tnew).any():
                    # round up the thresholds to nearest integer
                    model.set_initializer(n.input[1], Tnew)
                    # use same datatype as inputs for thresholds
                    model.set_tensor_datatype(n.input[1], idtype)
                    graph_modified = True
                if idtype.is_integer() and not idtype.signed() and (Tnew < 0).any():
                    # clip any negative thresholds
                    Tnew = np.clip(Tnew, 0, None)
                    model.set_initializer(n.input[1], Tnew)
                    # use same datatype as inputs for thresholds
                    model.set_tensor_datatype(n.input[1], idtype)
                    graph_modified = True
        return (model, graph_modified)
