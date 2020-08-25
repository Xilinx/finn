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

from onnx import TensorProto, helper

from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveUniqueNodeNames


def check_two_dict_for_equality(dict1, dict2):
    for key in dict1:
        assert key in dict2, "Key: {} is not in both dictionaries".format(key)
        assert (
            dict1[key] == dict2[key]
        ), """Values for key {} are not the same
        in both dictionaries""".format(
            key
        )

    return True


def test_res_estimate():
    mw = mh = 4
    simd = 1
    pe = 1
    idt = DataType.INT2
    wdt = DataType.INT2
    odt = DataType.INT2
    actval = odt.min()

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    node_inp_list = ["inp", "weights", "thresh"]

    FCLayer_node = helper.make_node(
        "StreamingFCLayer_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=0,
        noActivation=0,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)

    model = model.transform(GiveUniqueNodeNames())
    prod_resource_estimation = model.analysis(res_estimation)
    expect_resource_estimation = {
        "StreamingFCLayer_Batch_0": {
            "BRAM_18K": 0,
            "BRAM_efficiency": 1,
            "LUT": 357,
            "DSP": 0,
            "URAM": 0,
        }
    }

    assert check_two_dict_for_equality(
        prod_resource_estimation, expect_resource_estimation
    ), """The produced output of
    the res_estimation analysis pass is not equal to the expected one"""

    prod_resource_estimation = model.analysis(res_estimation_complete)
    expect_resource_estimation = {
        "StreamingFCLayer_Batch_0": [
            {"BRAM_18K": 0, "BRAM_efficiency": 1, "LUT": 352, "DSP": 1, "URAM": 0},
            {"BRAM_18K": 0, "BRAM_efficiency": 1, "LUT": 357, "DSP": 0, "URAM": 0},
        ]
    }

    assert check_two_dict_for_equality(
        prod_resource_estimation, expect_resource_estimation
    ), """The produced output of
    the res_estimation_complete analysis pass is not equal to the expected one"""
