# Copyright (c) 2021, Xilinx
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


from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class ExternalizeParams(Transformation):
    """Create top-level graph inputs for IODMAs serving layers where weights are
    marked as external using mem_mode="external"."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph_modified = False

        def filter_fc_extw(x):
            if x.op_type == "IODMA":
                burst_mode = get_by_name(x.attribute, "burstMode")
                if burst_mode is not None:
                    burst_mode = burst_mode.s.decode("UTF-8")
                    return burst_mode == "wrap"

        dma_extw_nodes = list(filter(filter_fc_extw, model.graph.node))

        for dma_extw in dma_extw_nodes:
            extw_tensor_name = dma_extw.input[0]
            extw_tensor_name_out = dma_extw.output[0]
            if extw_tensor_name in [x.name for x in model.graph.input]:
                continue
            else:
                extw_vi = model.get_tensor_valueinfo(extw_tensor_name)
                assert extw_vi is not None
                model.graph.value_info.remove(extw_vi)
                model.graph.input.append(extw_vi)
                iodma_init = model.get_initializer(extw_vi.name)
                assert iodma_init is not None
                # remove output-side initializer to get correct dataflow partitioning
                model.graph.initializer.remove(
                    [
                        x
                        for x in model.graph.initializer
                        if x.name == extw_tensor_name_out
                    ][0]
                )
                graph_modified = True

        return (model, graph_modified)
