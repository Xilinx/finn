# Copyright (c) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
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

from __future__ import annotations

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import Iterable

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw


class RaiseScalarToRank1(Transformation):
    """Lift all scalar tensors in the model to rank-1 tensors.

    Scalars in ONNX are represented with an empty shape. Downstream FINN
    transformations often expect tensors to have at least rank 1. This
    transformation scans all tensors that have shape information attached and
    ensures scalars are reshaped to have shape ``[1]`` while keeping any
    initializer data consistent.
    """

    def __init__(self):
        super().__init__()

    def _tensor_names(self, model: ModelWrapper) -> Iterable[str]:
        graph = model.graph
        tensors = [vi.name for vi in graph.value_info]
        tensors += [inp.name for inp in graph.input]
        tensors += [out.name for out in graph.output]
        # Preserve order but remove duplicates
        seen = set()
        for name in tensors:
            if name not in seen:
                seen.add(name)
                yield name

    def apply(self, model: ModelWrapper):
        graph_modified = False
        for tensor_name in self._tensor_names(model):
            tensor_shape = model.get_tensor_shape(tensor_name)
            if tensor_shape is None:
                continue
            if len(tensor_shape) == 0:
                to_hw.lift_to_rank1(tensor_name, model)
                graph_modified = True
        return (model, graph_modified)
