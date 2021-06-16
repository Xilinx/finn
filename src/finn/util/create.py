# Copyright (c) 2020 Xilinx, Inc.
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
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import calculate_signed_dot_prod_range, gen_finn_dt_tensor


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
            if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
                odt = DataType.UINT32
            else:
                odt = DataType.INT32
        else:
            odt = act
            (min, max) = calculate_signed_dot_prod_range(idt, wdt, mw)
            n_steps = act.get_num_possible_values() - 1
            T = np.random.randint(min, max - 1, (mh, n_steps)).astype(np.float32)
            # provide non-decreasing thresholds
            T = np.sort(T, axis=1)
            # generate thresholds for activation
            if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
                tdt = DataType.UINT32
                # bias thresholds to be positive
                T = np.ceil((T + mw) / 2)
                assert (T >= 0).all()
            else:
                tdt = DataType.INT32
        lyr["T"] = T
        lyr["tdt"] = tdt
        lyr["odt"] = odt
        ret.append(lyr)

    return hls_mlp_maker(ret)


def hls_mlp_maker(layer_spec):
    """Create an MLP of given specification using HLSCustomOp instances."""

    current_in_name = ""
    current_out_name = ""
    i = 0

    graph = helper.make_graph(nodes=[], name="mlp", inputs=[], outputs=[])

    model = helper.make_model(graph, producer_name="finn")
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
        if "mem_mode" in lyr.keys():
            mmode = lyr["mem_mode"]
        else:
            mmode = "const"

        if i == 0:
            global_in = helper.make_tensor_value_info(
                current_in_name, TensorProto.FLOAT, [1, mw]
            )
            model.graph.input.append(global_in)

        if i == len(layer_spec) - 1:
            global_out = helper.make_tensor_value_info(
                current_out_name, TensorProto.FLOAT, [1, mh]
            )
            model.graph.output.append(global_out)

        # there are two ways to implement bipolar weights and inputs for
        # StreamingFC:
        # - specify their datatypes as such
        # - specify their datatypes as BINARY as use binaryXnorMode
        if wdt == DataType.BIPOLAR and idt == DataType.BIPOLAR:
            # we'll internally convert weights/inputs to binary and specify the
            # datatypes as such, and also set the binaryXnorMode attribute to 1
            export_wdt = DataType.BINARY
            export_idt = DataType.BINARY
            binary_xnor_mode = 1
        else:
            export_wdt = wdt
            export_idt = idt
            binary_xnor_mode = 0

        if T is not None:
            no_act = 0
            node_inp_list = [current_in_name, current_W_name, current_T_name]
            if odt == DataType.BIPOLAR:
                actval = 0
            else:
                actval = odt.min()
        else:
            # no thresholds
            node_inp_list = [current_in_name, current_W_name]
            actval = 0
            no_act = 1
        FCLayer_node = helper.make_node(
            "StreamingFCLayer_Batch",
            node_inp_list,
            [current_out_name],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            MW=mw,
            MH=mh,
            SIMD=simd,
            PE=pe,
            inputDataType=export_idt.name,
            weightDataType=export_wdt.name,
            outputDataType=odt.name,
            ActVal=actval,
            binaryXnorMode=binary_xnor_mode,
            noActivation=no_act,
            mem_mode=mmode,
        )

        model.graph.node.append(FCLayer_node)
        model.set_tensor_datatype(current_in_name, idt)
        model.set_tensor_datatype(current_out_name, odt)
        model.set_tensor_datatype(current_W_name, wdt)
        if binary_xnor_mode:
            # convert bipolar to binary
            model.set_initializer(current_W_name, (W + 1) / 2)
        else:
            model.set_initializer(current_W_name, W)
        if T is not None:
            model.set_tensor_datatype(current_T_name, tdt)
            model.set_initializer(current_T_name, T)
        i += 1

    return model
