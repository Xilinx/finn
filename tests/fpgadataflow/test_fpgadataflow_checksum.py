# Copyright (c) 2022, Xilinx, Inc.
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

import pytest

import numpy as np
from onnx import TensorProto, helper
from pyverilator.util.axi_utils import axilite_read, axilite_write
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_hook import InsertHook
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5


def create_two_fc_model():
    # create a model with two MatrixVectorActivation instances
    wdt = DataType["INT2"]
    idt = DataType["INT32"]
    odt = DataType["INT32"]
    m = 4
    actval = 0
    no_act = 1
    binary_xnor_mode = 0
    pe = 2
    simd = 2

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, m])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, m])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, m])

    fc0 = helper.make_node(
        "MatrixVectorActivation",
        ["inp", "w0"],
        ["mid"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
        mem_mode="decoupled",
    )

    fc1 = helper.make_node(
        "MatrixVectorActivation",
        ["mid", "w1"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=m,
        MH=m,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
        mem_mode="decoupled",
    )

    graph = helper.make_graph(
        nodes=[fc0, fc1],
        name="fclayer_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[mid],
    )

    model = helper.make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("mid", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("w0", wdt)
    model.set_tensor_datatype("w1", wdt)

    # generate weights
    w0 = np.eye(m, dtype=np.float32)
    w1 = np.eye(m, dtype=np.float32)
    model.set_initializer("w0", w0)
    model.set_initializer("w1", w1)

    return model


@pytest.mark.fpgadataflow
def test_fpgadataflow_checksum():
    # use a graph consisting of two fc layers to test
    # checksum node insertion
    model = create_two_fc_model()

    # set checksum output hook
    for n in model.graph.node:
        n0 = getCustomOp(n)
        n0.set_nodeattr("output_hook", "checksum")

    model = model.transform(InsertHook())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())

    assert (
        len(model.get_nodes_by_op_type("CheckSum")) == 2
    ), """Insertion of
        checksum layers was unsuccessful"""

    # to verify the functionality of the checksum layer
    # cppsim and rtlsim will be compared

    x = gen_finn_dt_tensor(DataType["INT32"], (1, 4))

    # cppsim
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    inp = {"global_in": x}
    y_cppsim = oxe.execute_onnx(model, inp, return_full_exec_context=True)
    checksum0_cppsim = y_cppsim["CheckSum_0_out1"]
    checksum1_cppsim = y_cppsim["CheckSum_1_out1"]

    # in this test case scenario the checksums are equal
    assert checksum0_cppsim == checksum1_cppsim, "CheckSums are not equal"

    # rtlsim
    model = model.transform(InsertFIFO(True))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")

    # define function to read out the checksums from axilite
    checksums = []
    drain = []

    def read_checksum_and_drain(sim):
        chk_addr = 16
        drain_addr = 32
        for i in range(len(model.get_nodes_by_op_type("CheckSum"))):
            axi_name = "s_axi_checksum_{}_".format(i)
            checksums.append(axilite_read(sim, chk_addr, basename=axi_name))
            drain.append(axilite_read(sim, drain_addr, basename=axi_name))

    drain_value = False

    def write_drain(sim):
        addr = 32
        for i in range(len(model.get_nodes_by_op_type("CheckSum"))):
            axi_name = "s_axi_checksum_{}_".format(i)
            axilite_write(sim, addr, drain_value, basename=axi_name)

    rtlsim_exec(model, inp, pre_hook=write_drain, post_hook=read_checksum_and_drain)
    checksum0_rtlsim = int(checksums[0])
    checksum1_rtlsim = int(checksums[1])
    checksum0_drain = int(drain[0])
    checksum1_drain = int(drain[1])

    assert (
        checksum0_rtlsim == checksum0_cppsim
    ), """The first checksums do not
        match in cppsim vs. rtlsim"""
    assert (
        checksum1_rtlsim == checksum1_cppsim
    ), """The second checksums do not
        match in cppsim vs. rtlsim"""

    assert (
        checksum0_drain == 0
    ), "Drain read doesn't match drain write for first checksum"
    assert (
        checksum1_drain == 0
    ), "Drain read doesn't match drain write for second checksum"

    # TODO: test for drain set to true
