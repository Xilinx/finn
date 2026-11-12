# Copyright (C) 2020, Xilinx, Inc.
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

import pytest

import copy
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.create import hls_random_mlp_maker
from finn.util.test import make_runtime_weight_stream

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_runtime_weights_single_layer():
    idt = DataType["UINT32"]
    wdt = DataType["UINT4"]
    act = None
    mw = 64
    mh = 32
    pe = 4
    simd = 16
    layer_spec = {
        "idt": idt,
        "wdt": wdt,
        "mw": mw,
        "mh": mh,
        "act": act,
        "pe": pe,
        "simd": simd,
    }
    layer_spec_list = [layer_spec]
    model = hls_random_mlp_maker(layer_spec_list)
    model = model.transform(SpecializeLayers(test_fpga_part))
    fcl = model.get_nodes_by_op_type("MVAU_hls")[0]
    op_inst = getCustomOp(fcl)
    op_inst.set_nodeattr("mem_mode", "internal_decoupled")
    op_inst.set_nodeattr("runtime_writeable_weights", 1)
    old_weights = model.get_initializer(fcl.input[1])
    old_weight_stream = make_runtime_weight_stream(op_inst, old_weights)
    model = model.transform(InsertFIFO(True))
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")
    in_tensor = np.asarray(range(mw), dtype=np.float32)
    # add two copies of the input tensor as the first one is just used to
    # "flush out" the pipeline (as mvau already starts receiving old weights while
    # we read/write new ones and reads seem to cause a disturbance too)
    in_tensor = np.tile(in_tensor, (2, 1))
    exec_ctx = {"act_0": in_tensor}
    extracted_weight_stream = []

    def read_weights(sim):
        addr = 0
        read_handles = []
        addresses = []
        for i in range(len(old_weight_stream)):
            addresses.append(addr)
            addr += 4
        read_handles.append(sim.read_axilite("s_axilite_0", iter(addresses)))
        sim.run()
        for addr in addresses:
            extracted_weight_stream.append(int(read_handles[0][addr], 16))

    rtlsim_exec(model, exec_ctx, pre_hook=read_weights)
    assert extracted_weight_stream == old_weight_stream
    y = exec_ctx["act_1"]
    # only use second batch element in output; first will be invalid due to
    # old weights (see above)
    assert (y[1] == np.dot(in_tensor[1], old_weights)).all()

    new_weights = gen_finn_dt_tensor(wdt, (mw, mh))
    new_weight_stream = make_runtime_weight_stream(op_inst, new_weights)

    def write_weights(sim):
        addr = 0
        writes = []
        for nw in new_weight_stream:
            # convert value to hex value and without '0x' prefix
            hex_val = format(nw, "x")
            writes.append((addr, hex_val))
            addr += 4
        sim.write_axilite("s_axilite_0", iter(writes))
        sim.run()

    rtlsim_exec(model, exec_ctx, pre_hook=write_weights)
    y = exec_ctx["act_1"]
    # only use second batch element in output; first will be invalid due to
    # old weights (see above)
    assert (y[1] == np.dot(in_tensor[1], new_weights)).all()


# SIMD > 1 (but < K*K) so that the element order within a weight word matters
@pytest.mark.parametrize(
    "impl_style, part", [("hls", test_fpga_part), ("rtl", "xcvc1902-vsva2197-2MP-e-S")]
)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_runtime_weights_vvau(impl_style, part):
    idt = DataType["INT4"]
    wdt = DataType["INT4"]
    odt = DataType["INT32"]
    k = 3
    dim = 4
    channels = 4
    pe = 2
    simd = 3
    in_shape = [1, dim, dim, k * k * channels]
    out_shape = [1, dim, dim, channels]
    vvau_node = helper.make_node(
        "VVAU",
        ["inp", "weights"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=pe,
        SIMD=simd,
        Dim=[dim, dim],
        Channels=channels,
        Kernel=[k, k],
        resType="dsp" if impl_style == "rtl" else "lut",
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        noActivation=1,
        mem_mode="internal_decoupled",
        runtime_writeable_weights=1,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(
        nodes=[vvau_node],
        name="vvau_graph",
        inputs=[helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="vvau-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    old_weights = gen_finn_dt_tensor(wdt, (channels, 1, k, k))
    model.set_initializer("weights", old_weights)
    # the HW op executed in Python serves as reference for old and new weights
    ref_model = copy.deepcopy(model)

    def expected_output(x, weights):
        ref_model.set_initializer("weights", weights)
        return oxe.execute_onnx(ref_model, {"inp": x})["outp"]

    model = model.transform(SpecializeLayers(part))
    vvau = model.get_nodes_by_op_type("VVAU_" + impl_style)[0]
    op_inst = getCustomOp(vvau)
    old_weight_stream = make_runtime_weight_stream(op_inst, old_weights)
    model = model.transform(InsertFIFO(True))
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    # two copies of the input, the first one only flushes out the pipeline
    # (see test_runtime_weights_single_layer)
    in_tensor = np.tile(gen_finn_dt_tensor(idt, in_shape), (2, 1, 1, 1))
    exec_ctx = {inp_name: in_tensor}
    extracted_weight_stream = []

    def read_weights(sim):
        addresses = [4 * i for i in range(len(old_weight_stream))]
        read_handle = sim.read_axilite("s_axilite_0", iter(addresses))
        sim.run()
        for addr in addresses:
            extracted_weight_stream.append(int(read_handle[addr], 16))

    rtlsim_exec(model, exec_ctx, pre_hook=read_weights)
    assert extracted_weight_stream == old_weight_stream
    y = exec_ctx[out_name]
    assert (y[1] == expected_output(in_tensor[1:2], old_weights)[0]).all()

    new_weights = gen_finn_dt_tensor(wdt, (channels, 1, k, k))
    new_weight_stream = make_runtime_weight_stream(op_inst, new_weights)

    def write_weights(sim):
        writes = [(4 * i, format(nw, "x")) for i, nw in enumerate(new_weight_stream)]
        sim.write_axilite("s_axilite_0", iter(writes))
        sim.run()

    rtlsim_exec(model, exec_ctx, pre_hook=write_weights)
    y = exec_ctx[out_name]
    assert (y[1] == expected_output(in_tensor[1:2], new_weights)[0]).all()
