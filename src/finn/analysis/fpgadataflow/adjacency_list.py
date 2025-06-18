# Copyright (C) 2025, Advanced Micro Devices, Inc.
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
from collections import defaultdict, deque


def adjacency_list(model):
    """Returns adjacency list of MLO param nodes and their connectivity in the model."""
    mlo_optypes = ["MVAU_hls", "MVAU_rtl", "Thresholding_rtl"]
    graph = model.graph

    full_graph = defaultdict(list)
    # Build full DAG across all nodes
    for node in graph.node:
        for input_tensor in node.input:
            producer = model.find_producer(input_tensor)
            if producer:
                full_graph[producer.name].append(node.name)
            elif input_tensor in [input.name for input in graph.input]:
                full_graph[input_tensor].append(node.name)
        for output_tensor in node.output:
            producer = model.find_producer(output_tensor)
            if producer and output_tensor in [output.name for output in graph.output]:
                full_graph[producer.name].append(output_tensor)

    streamtap_nodes = [node.name for node in graph.node if (node.op_type in mlo_optypes)]
    graph_inputs = [input.name for input in graph.input]
    graph_outputs = [output.name for output in graph.output]

    relevant_nodes = streamtap_nodes + graph_inputs + graph_outputs
    streamtap_adjacency = defaultdict(list)

    for node in relevant_nodes:
        visited = set()
        queue = deque(full_graph.get(node, []))
        while queue:
            source = node
            sink = queue.popleft()
            if sink in visited:
                continue
            visited.add(sink)

            if sink in relevant_nodes:
                if sink in graph_outputs:
                    sink = f"__OUTPUT{graph_outputs.index(sink)}__"
                if source in graph_inputs:
                    source = f"__INPUT{graph_inputs.index(source)}__"
                streamtap_adjacency[source].append(sink)
            else:
                queue.extend(full_graph.get(sink, []))

    return dict(streamtap_adjacency)
