# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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
from collections import defaultdict, deque
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)


def hls_random_mlp_maker(layer_spec):
    """Create an MLP of given specification using HLSCustomOp instances.
    Generate random weights/thresholds of appropriate size."""
    ret = []
    for lyr in layer_spec:
        idt = lyr["idt"]
        wdt = lyr["wdt"]
        mw = lyr["mw"]
        mh = lyr["mh"]
        act = lyr["act"]
        lyr["W"] = gen_finn_dt_tensor(wdt, (mw, mh))
        if act is None:
            # no activation, produce accumulators
            T = None
            tdt = None
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                odt = DataType["UINT32"]
            else:
                odt = DataType["INT32"]
        else:
            odt = act
            (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
            n_steps = act.get_num_possible_values() - 1
            T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
            # provide non-decreasing thresholds
            T = np.sort(T, axis=1)
            # generate thresholds for activation
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                tdt = DataType["UINT32"]
                # bias thresholds to be positive
                T = np.ceil((T + mw) / 2)
                assert (T >= 0).all()
            else:
                tdt = DataType["INT32"]
        lyr["T"] = T
        lyr["tdt"] = tdt
        lyr["odt"] = odt
        ret.append(lyr)

    return hls_mlp_maker(ret)


def hls_mlp_maker(layer_spec):
    """Create an MLP of given specification using DotProd + optional
    MultiThreshold HWCustomOp instances."""

    current_in_name = ""
    current_out_name = ""
    i = 0

    graph = helper.make_graph(nodes=[], name="mlp", inputs=[], outputs=[])

    model = qonnx_make_model(graph, producer_name="finn")
    model = ModelWrapper(model)

    for lyr in layer_spec:
        current_W_name = "W_%d" % i
        current_T_name = "T_%d" % i
        current_in_name = "act_%d" % i
        current_out_name = "act_%d" % (i + 1)

        W = lyr["W"]
        (mw, mh) = W.shape
        T = lyr["T"]
        pe = lyr["pe"]
        simd = lyr["simd"]
        wdt = lyr["wdt"]
        idt = lyr["idt"]
        tdt = lyr["tdt"]
        odt = lyr["odt"]

        if i == 0:
            global_in = helper.make_tensor_value_info(current_in_name, TensorProto.FLOAT, [1, mw])
            model.graph.input.append(global_in)

        if i == len(layer_spec) - 1:
            global_out = helper.make_tensor_value_info(current_out_name, TensorProto.FLOAT, [1, mh])
            model.graph.output.append(global_out)

        # DotProd always outputs accumulator type
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            dp_odt = DataType["UINT32"]
        else:
            dp_odt = DataType["INT32"]

        if T is not None:
            # DotProd → MultiThreshold chain
            dp_out_name = "dp_%d_out" % i
        else:
            dp_out_name = current_out_name

        dp_node = helper.make_node(
            "DotProd",
            [current_in_name, current_W_name],
            [dp_out_name],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            IFMChannels=mw,
            OFMChannels=mh,
            IFMDim=[1, 1],
            OFMDim=[1, 1],
            KernelSize=[1, 1],
            Stride=[1, 1],
            Dilation=[1, 1],
            Padding=[0, 0, 0, 0],
            depthwise=0,
            SIMD=simd,
            PE=pe,
            inputDataType=idt.name,
            weightDataType=wdt.name,
            outputDataType=dp_odt.name,
            mem_mode="internal_decoupled",
        )
        model.graph.node.append(dp_node)
        model.set_tensor_datatype(current_in_name, idt)
        model.set_tensor_datatype(dp_out_name, dp_odt)
        model.set_tensor_datatype(current_W_name, wdt)
        model.set_initializer(current_W_name, W)

        if T is not None:
            # Add MultiThreshold node after DotProd
            mt_node = helper.make_node(
                "MultiThreshold",
                [dp_out_name, current_T_name],
                [current_out_name],
                domain="qonnx.custom_op.general",
                out_dtype=odt.name,
            )
            model.graph.node.append(mt_node)
            model.set_tensor_datatype(current_out_name, odt)
            model.set_tensor_datatype(current_T_name, tdt)
            model.set_initializer(current_T_name, T)
        else:
            model.set_tensor_datatype(current_out_name, dp_odt)

        i += 1

    return model


def adjacency_list(model, filter_function):
    """Returns adjacency list of nodes based on filter function."""
    graph = model.graph

    full_graph = defaultdict(list)
    # Build full DAG across all nodes
    for node in graph.node:
        for input_tensor in node.input:
            producer = model.find_producer(input_tensor)
            if producer:
                full_graph[producer.name].append(node.name)
            elif (
                hasattr(graph, "input")
                and graph.input
                and input_tensor in [input.name for input in graph.input]
            ):
                full_graph[input_tensor].append(node.name)
        for output_tensor in node.output:
            producer = model.find_producer(output_tensor)
            if (
                producer
                and hasattr(graph, "output")
                and graph.output
                and output_tensor in [output.name for output in graph.output]
            ):
                full_graph[producer.name].append(output_tensor)

    # Apply filtering logic
    if not callable(filter_function):
        raise ValueError("filter_function must be callable")
    filter_nodes = [node.name for node in graph.node if filter_function(node)]
    graph_inputs = (
        [input.name for input in graph.input] if hasattr(graph, "input") and graph.input else []
    )
    graph_outputs = (
        [output.name for output in graph.output]
        if hasattr(graph, "output") and graph.output
        else []
    )

    relevant_nodes = filter_nodes + graph_inputs + graph_outputs
    filtered_adjacency = defaultdict(list)

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
                filtered_adjacency[source].append(sink)
            else:
                queue.extend(full_graph.get(sink, []))

    return dict(filtered_adjacency)
