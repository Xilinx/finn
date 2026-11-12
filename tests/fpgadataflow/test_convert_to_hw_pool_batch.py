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

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.test import tree_model_test


def make_single_maxpool_modelwrapper(
    k, stride, pad, ifm_ch, ifm_dim, ofm_dim, idt, use_1d=False, ceil_mode=0
):
    odt = idt
    if use_1d:
        ishape = [1, ifm_ch, 1, ifm_dim]
        oshape = [1, ifm_ch, 1, ofm_dim]
        kshape = [1, k]
        pads = [0, pad, 0, pad]
        strides = [1, stride]
    else:
        ishape = [1, ifm_ch, ifm_dim, ifm_dim]
        oshape = [1, ifm_ch, ofm_dim, ofm_dim]
        kshape = [k, k]
        pads = [pad, pad, pad, pad]
        strides = [stride, stride]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)
    mp_node = helper.make_node(
        "MaxPool",
        ["inp"],
        ["outp"],
        kernel_shape=kshape,
        pads=pads,
        strides=strides,
        ceil_mode=ceil_mode,
    )
    graph = helper.make_graph(nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="mp-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model = model.transform(InferShapes())

    return model


def make_single_quantavpool_modelwrapper(k, stride, ifm_ch, ifm_dim, ofm_dim, idt, odt):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ifm_ch, ofm_dim, ofm_dim])

    mp_node = helper.make_node(
        "QuantAvgPool2d",
        ["inp"],
        ["outp"],
        domain="qonnx.custom_op.general",
        stride=stride,
        kernel=k,
        ibits=idt.bitwidth(),
        obits=odt.bitwidth(),
        signed=1 if idt.signed() else 0,
        data_layout="NCHW",
    )
    graph = helper.make_graph(nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="mp-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model = model.transform(InferShapes())

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# input datatype
@pytest.mark.parametrize("idt", [DataType["UINT4"], DataType["INT4"], DataType["INT8"]])
# output datatype
@pytest.mark.parametrize("odt", [DataType["UINT4"], DataType["INT4"]])
# pool configuration:                   ( k,stride, pad, ifm_dim )
@pytest.mark.parametrize("pool_config", [(7, 7, 0, 7), (3, 2, 1, 5)])
# input channels
@pytest.mark.parametrize("ifm_ch", [1, 4])
# number of out channel computed in parallel
@pytest.mark.parametrize("pe", [1, 2, 4])
# pool type
@pytest.mark.parametrize("op_type", ["QuantAvgPool2d", "MaxPool", "MaxPool1D"])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_convert_to_hw_pool(idt, odt, pool_config, ifm_ch, pe, op_type, exec_mode):
    k, stride, pad, ifm_dim = pool_config

    if ifm_ch % pe != 0:
        pytest.skip("ifm_ch%pe != 0. Skipping")

    ofm_dim = int(((ifm_dim + 2 * pad - k) / stride) + 1)

    ishape = (1, ifm_ch, ifm_dim, ifm_dim)
    use_1d = False
    if op_type == "MaxPool1D":
        use_1d = True
        ishape = (1, ifm_ch, 1, ifm_dim)
        op_type = "MaxPool"

    x = gen_finn_dt_tensor(idt, ishape)
    # prepare input data
    input_dict = prepare_inputs(x)
    if op_type == "MaxPool":
        if idt != odt:
            pytest.skip("Skipping Maxpool with idt != odt")

        model = make_single_maxpool_modelwrapper(
            k, stride, pad, ifm_ch, ifm_dim, ofm_dim, idt, use_1d
        )
    elif op_type == "QuantAvgPool2d":
        if pad != 0:
            pytest.skip("No padding support for QuantAvgPool2d. Skipping")

        if idt.signed() != odt.signed():
            pytest.skip("Skipping QuantAvgPool2d with idt.signed() != odt.signed()")
        model = make_single_quantavpool_modelwrapper(k, stride, ifm_ch, ifm_dim, ofm_dim, idt, odt)
    else:
        assert False, "{} is not a supported op_type".format(op_type)

    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    new_model = model.transform(to_hw.InferPool())
    new_model = new_model.transform(GiveUniqueNodeNames())
    y_produced = oxe.execute_onnx(new_model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
    new_model = new_model.transform(SpecializeLayers("xc7z020clg400-1"))

    # Folding — only Pool PE needs to be set (no separate SWG node)
    for n in new_model.graph.node:
        if n.op_type.startswith("Pool"):
            inst = getCustomOp(n)
            inst.set_nodeattr("PE", pe)

    # MinimizeAccumulatorWidth - for QuantAvgPool this should reduce AccumBits
    if op_type == "QuantAvgPool2d":
        pool_node = [n for n in new_model.graph.node if n.op_type.startswith("Pool")][0]
        pool_inst = getCustomOp(pool_node)
        initial_accum = pool_inst.get_nodeattr("AccumBits")
        new_model = new_model.transform(MinimizeAccumulatorWidth())
        new_accum = pool_inst.get_nodeattr("AccumBits")
        assert new_accum <= initial_accum

    # Consolidated Pool: Transpose → Pool → Transpose (3 nodes)
    assert len(new_model.graph.node) == 3
    assert new_model.graph.node[0].op_type == "Transpose"
    assert new_model.graph.node[1].op_type.startswith("Pool")
    assert new_model.graph.node[2].op_type == "Transpose"

    if exec_mode == "cppsim":
        new_model = new_model.transform(SetExecMode("cppsim"))
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # execute new_model
    y_produced = oxe.execute_onnx(new_model, input_dict)["outp"]
    assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        node = new_model.get_nodes_by_op_type("Pool_hls")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = new_model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)


# ceil_mode=1 MaxPool geometries.  InferPool absorbs any ceil extension
# (where the drop-rule output exceeds the floor output) into extra trailing
# padding within the consolidated Pool node.  All three cases below must
# build and match the onnxruntime reference:
#   (k, stride, pad, ifm_dim):   drop  floor   how it builds
#     (2, 2, 0, 8):                4     4      ceil_mode is a no-op
#     (2, 2, 1, 9):                5     5      pad absorbed into Pool
#     (2, 2, 0, 7):                4     3      pad-to-ceil 7->8 (bottom/right +1)
@pytest.mark.parametrize(
    "ceil_config",
    [
        (2, 2, 0, 8),
        (2, 2, 1, 9),
        (2, 2, 0, 7),
    ],
)
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_convert_to_hw_maxpool_ceil_mode(ceil_config, exec_mode):
    k, stride, pad, ifm_dim = ceil_config
    idt = DataType["UINT4"]
    ifm_ch = 4
    pe = 2

    ofm_dim = compute_pool_output_dim(ifm_dim, k, stride, pad, ceil_mode=1)
    model = make_single_maxpool_modelwrapper(
        k, stride, pad, ifm_ch, ifm_dim, ofm_dim, idt, ceil_mode=1
    )

    x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_dim, ifm_dim))
    input_dict = prepare_inputs(x)
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    new_model = model.transform(to_hw.InferPool())
    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(SpecializeLayers("xc7z020clg400-1"))

    for n in new_model.graph.node:
        if n.op_type.startswith("Pool"):
            getCustomOp(n).set_nodeattr("PE", pe)

    if exec_mode == "cppsim":
        new_model = new_model.transform(SetExecMode("cppsim"))
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    y_produced = oxe.execute_onnx(new_model, input_dict)["outp"]
    assert (y_produced == y_expected).all()


# Non-square MaxPool geometries: the fused Pool node carries independent H/W
# for feature map, kernel, stride and padding ([H_begin, W_begin, H_end, W_end]).
#   (ifm_h, ifm_w, kh, kw, sh, sw, pads)          exercises
nonsquare_configs = [
    (6, 10, 2, 2, 2, 2, [0, 0, 0, 0]),  # non-square feature map
    (8, 9, 2, 3, 2, 3, [0, 0, 0, 0]),  # asymmetric kernel + stride
    (7, 5, 3, 2, 2, 1, [1, 0, 1, 1]),  # asymmetric padding, overlapping windows
    (9, 7, 2, 3, 2, 3, [0, 0, 0, 0]),  # floor remainder: unused last row and column
    (5, 8, 3, 3, 2, 2, [1, 1, 0, 0]),  # leading-only padding, floor remainder
]


@pytest.mark.parametrize("nonsquare_config", nonsquare_configs)
# (idt, pe): signed input exercises the negative pad value; PE=1 vs PE>1 the
# scalar vs broadcasting tensor<> pad value construction
@pytest.mark.parametrize("idt_pe", [(DataType["UINT4"], 2), (DataType["INT4"], 1)])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_convert_to_hw_maxpool_nonsquare(nonsquare_config, idt_pe, exec_mode):
    ifm_h, ifm_w, kh, kw, sh, sw, pads = nonsquare_config
    idt, pe = idt_pe
    ifm_ch = 4
    ofm_h = (ifm_h + pads[0] + pads[2] - kh) // sh + 1
    ofm_w = (ifm_w + pads[1] + pads[3] - kw) // sw + 1

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_h, ifm_w])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ifm_ch, ofm_h, ofm_w])
    mp_node = helper.make_node(
        "MaxPool", ["inp"], ["outp"], kernel_shape=[kh, kw], pads=pads, strides=[sh, sw]
    )
    graph = helper.make_graph(nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mp-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    model = model.transform(InferShapes())

    x = gen_finn_dt_tensor(idt, (1, ifm_ch, ifm_h, ifm_w))
    input_dict = prepare_inputs(x)
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    new_model = model.transform(to_hw.InferPool())
    new_model = new_model.transform(GiveUniqueNodeNames())
    pool_nodes = new_model.get_nodes_by_op_type("Pool")
    assert len(pool_nodes) == 1, "non-square MaxPool was not converted"
    pool_inst = getCustomOp(pool_nodes[0])
    assert pool_inst.get_nodeattr("IFMDim") == [ifm_h, ifm_w]
    assert pool_inst.get_nodeattr("OutImgDims") == [ofm_h, ofm_w]
    y_produced = oxe.execute_onnx(new_model, input_dict)["outp"]
    assert (y_produced == y_expected).all(), "python execution mismatch"

    new_model = new_model.transform(SpecializeLayers("xc7z020clg400-1"))
    for n in new_model.graph.node:
        if n.op_type.startswith("Pool"):
            getCustomOp(n).set_nodeattr("PE", pe)

    if exec_mode == "cppsim":
        new_model = new_model.transform(SetExecMode("cppsim"))
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    y_produced = oxe.execute_onnx(new_model, input_dict)["outp"]
    assert (y_produced == y_expected).all(), exec_mode + " mismatch"

    if exec_mode == "rtlsim":
        node = new_model.get_nodes_by_op_type("Pool_hls")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = new_model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10), (exp_cycles, cycles_rtlsim)


# Multi-frame streaming through the Pool IP.  Geometries leave input pixels
# outside every window (floor remainder), which the fused
# pad<>/input_gen<> must still consume and discard, or the next frame is
# misaligned.  BatchSize > 1 makes node-level rtlsim stream all frames
# back-to-back through a single IP instance.
#   (ifm_h, ifm_w, kh, kw, sh, sw, pads, idt, pe)
multiframe_configs = [
    (7, 7, 2, 2, 2, 2, [0, 0, 0, 0], DataType["INT8"], 4),  # odd dims: last row/col unused
    (10, 8, 3, 3, 2, 2, [0, 0, 0, 0], DataType["INT4"], 1),  # overlap + unused row/column
    (5, 8, 3, 3, 2, 2, [1, 1, 0, 0], DataType["UINT4"], 2),  # padding + floor remainder
]


@pytest.mark.parametrize("multiframe_config", multiframe_configs)
@pytest.mark.parametrize("n_frames", [3])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_pool_multiframe_rtlsim(multiframe_config, n_frames):
    ifm_h, ifm_w, kh, kw, sh, sw, pads, idt, pe = multiframe_config
    ifm_ch = 4
    fpga_part, clk_ns = "xc7z020clg400-1", 5
    ofm_h = (ifm_h + pads[0] + pads[2] - kh) // sh + 1
    ofm_w = (ifm_w + pads[1] + pads[3] - kw) // sw + 1

    # NCHW MaxPool (supports uneven padding, as produced by ceil_mode)
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_h, ifm_w])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ifm_ch, ofm_h, ofm_w])
    mp_node = helper.make_node(
        "MaxPool", ["inp"], ["outp"], kernel_shape=[kh, kw], pads=pads, strides=[sh, sw]
    )
    graph = helper.make_graph(nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mp-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    model = model.transform(InferShapes())

    # per-frame reference from the original (pre-HW) node
    x = gen_finn_dt_tensor(idt, (n_frames, ifm_ch, ifm_h, ifm_w))
    y_expected = np.concatenate(
        [oxe.execute_onnx(model, {"inp": x[i : i + 1]})["outp"] for i in range(n_frames)]
    )

    model = model.transform(to_hw.InferPool())
    assert [n.op_type for n in model.graph.node] == ["Transpose", "Pool", "Transpose"]
    model = model.transform(SpecializeLayers(fpga_part))
    inst = getCustomOp(model.get_nodes_by_op_type("Pool_hls")[0])
    inst.set_nodeattr("PE", pe)
    inst.set_nodeattr("BatchSize", n_frames)
    # stream all frames through one IP instance: batch every tensor in the graph
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        shape = model.get_tensor_shape(vi.name)
        model.set_tensor_shape(vi.name, [n_frames] + list(shape[1:]))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    y_produced = oxe.execute_onnx(model, {"inp": x})["outp"]
    for i in range(n_frames):
        assert (y_produced[i] == y_expected[i]).all(), "frame %d mismatch" % i


@pytest.mark.parametrize("kernel_stride", [((1, 2), (2, 2)), ((2, 1), (2, 2)), ((1, 2), (2, 3))])
@pytest.mark.fpgadataflow
def test_pool_kernel_lt_stride_not_converted(kernel_stride):
    """Pooling with kernel < stride (in either dimension) skips input that no
    window reads. Its efficient support is shelved, so InferPool must leave
    such nodes untouched."""
    (kh, kw), (sh, sw) = kernel_stride
    ifm_ch, ifm_h, ifm_w = 4, 8, 8
    ofm_h, ofm_w = (ifm_h - kh) // sh + 1, (ifm_w - kw) // sw + 1
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_h, ifm_w])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ifm_ch, ofm_h, ofm_w])
    mp_node = helper.make_node(
        "MaxPool", ["inp"], ["outp"], kernel_shape=[kh, kw], pads=[0, 0, 0, 0], strides=[sh, sw]
    )
    graph = helper.make_graph(nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mp-model"))
    model.set_tensor_datatype("inp", DataType["UINT4"])
    model.set_tensor_datatype("outp", DataType["UINT4"])
    model = model.transform(InferShapes())
    model = model.transform(to_hw.InferPool())
    assert [n.op_type for n in model.graph.node] == ["MaxPool"]


# input datatype
@pytest.mark.parametrize("idt", [DataType["UINT4"]])
# output datatype
@pytest.mark.parametrize("odt", [DataType["UINT4"]])
# pool configuration:                   ( k,stride, pad, ifm_dim )
@pytest.mark.parametrize("pool_config", [(2, 1, 0, 512)])
# input channels
@pytest.mark.parametrize("ifm_ch", [32])
# number of out channel computed in parallel
@pytest.mark.parametrize("pe", [32])
# pool type
@pytest.mark.parametrize("op_type", ["MaxPool1D"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_analytical_characterization_pool(idt, odt, pool_config, ifm_ch, pe, op_type):
    k, stride, pad, ifm_dim = pool_config

    if ifm_ch % pe != 0:
        pytest.skip("ifm_ch%pe != 0. Skipping")

    if pad != 0 and idt.signed():
        pytest.skip("No support for pal_val != 0. Skipping")

    np.random.seed(0)

    part = "xc7z020clg400-1"

    ofm_dim = int(((ifm_dim + 2 * pad - k) / stride) + 1)

    ishape = (1, ifm_ch, ifm_dim, ifm_dim)
    use_1d = False
    if op_type == "MaxPool1D":
        use_1d = True
        ishape = (1, ifm_ch, 1, ifm_dim)
        op_type = "MaxPool"

    if op_type == "MaxPool":
        if idt != odt:
            pytest.skip("Skipping Maxpool with idt != odt")

        model = make_single_maxpool_modelwrapper(
            k, stride, pad, ifm_ch, ifm_dim, ofm_dim, idt, use_1d
        )
    elif op_type == "QuantAvgPool2d":
        if pad != 0:
            pytest.skip("No padding support for QuantAvgPool2d. Skipping")

        if idt.signed() != odt.signed():
            pytest.skip("Skipping QuantAvgPool2d with idt.signed() != odt.signed()")
        model = make_single_quantavpool_modelwrapper(k, stride, ifm_ch, ifm_dim, ofm_dim, idt, odt)
    else:
        assert False, "{} is not a supported op_type".format(op_type)

    model = model.transform(to_hw.InferPool())

    # Folding
    for n in model.graph.node:
        if n.op_type.startswith("Pool"):
            inst = getCustomOp(n)

            ishape = inst.get_folded_input_shape()
            oshape = inst.get_folded_output_shape()

            inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
            outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)

            graph = helper.make_graph(nodes=[n], name="mp_graph", inputs=[inp], outputs=[outp])
            model = qonnx_make_model(graph, producer_name="mp-model")
            model = ModelWrapper(model)
            model.set_tensor_datatype("inp", idt)
            model.set_tensor_datatype("outp", odt)
            model = model.transform(InferShapes())

            inst.set_nodeattr("PE", pe)
            model = model.transform(SpecializeLayers(part))

    node_details = ("Pool", op_type, k, ifm_ch, ifm_dim, ofm_dim, pe, idt)

    target_clk_ns = 4

    # TAV tolerances are fractions -- of the tokens moved and of the frame
    # length -- with a floor for the fixed part of the error (wind-up, adder
    # tree depth, the cycle rtlsim's period rounds away). The floors are the
    # flat budgets this test used before the split, so nothing that passed
    # then fails now; the fractions are what govern once the frame is long
    # enough to exceed them.
    max_allowed_volume_frac = 0.2
    volume_const = 5000
    max_allowed_length_frac = 0.2
    length_const = 5000

    assert tree_model_test(
        model,
        node_details,
        part,
        target_clk_ns,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    ), "characterized TAV does not match RTLsim'd one!"


# Fused Pool_hls tree model (finn.util.input_gen.pool_schedule) vs. rtlsim with
# tight tolerances. The TAV length differs from rtlsim's by its fill/drain
# artifact (rtlsim takes total cycles / frames as the period), up to ~13% for a
# global pool; the token counts must match within a few tokens.
#   (ifm_h, ifm_w, ch, kh, kw, sh, sw, pads, pe)
tree_model_pool_configs = [
    (8, 8, 4, 2, 2, 2, 2, [0, 0, 0, 0], 1),  # exact tiling
    (6, 10, 4, 2, 2, 2, 2, [0, 0, 0, 0], 2),  # non-square
    (9, 9, 4, 3, 3, 1, 1, [0, 0, 0, 0], 2),  # overlap, stride 1
    (7, 7, 4, 2, 2, 2, 2, [0, 0, 0, 0], 1),  # floor remainder (input-bound)
    (5, 5, 4, 3, 3, 2, 2, [1, 1, 1, 1], 2),  # padding
    (7, 5, 4, 3, 2, 2, 1, [1, 0, 1, 1], 2),  # asymmetric padding
    (7, 7, 16, 7, 7, 7, 7, [0, 0, 0, 0], 4),  # global
    (12, 9, 3, 5, 3, 2, 1, [0, 0, 0, 0], 3),  # overlap + remainder
]


@pytest.mark.parametrize("config", tree_model_pool_configs)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_analytical_characterization_pool_fused(config):
    ifm_h, ifm_w, ch, kh, kw, sh, sw, pads, pe = config
    idt = DataType["UINT4"]
    part, target_clk_ns = "xczu7ev-ffvc1156-2-e", 5.0
    ofm_h = (ifm_h + pads[0] + pads[2] - kh) // sh + 1
    ofm_w = (ifm_w + pads[1] + pads[3] - kw) // sw + 1
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch, ifm_h, ifm_w])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch, ofm_h, ofm_w])
    mp_node = helper.make_node(
        "MaxPool", ["inp"], ["outp"], kernel_shape=[kh, kw], pads=pads, strides=[sh, sw]
    )
    graph = helper.make_graph(nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mp-model"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    model = model.transform(InferShapes())
    model = model.transform(to_hw.InferPool())

    # single-node model around the (NHWC) Pool node
    pool = model.get_nodes_by_op_type("Pool")[0]
    getCustomOp(pool).set_nodeattr("PE", pe)
    i = helper.make_tensor_value_info(
        pool.input[0], TensorProto.FLOAT, model.get_tensor_shape(pool.input[0])
    )
    o = helper.make_tensor_value_info(
        pool.output[0], TensorProto.FLOAT, model.get_tensor_shape(pool.output[0])
    )
    single = ModelWrapper(qonnx_make_model(helper.make_graph([pool], "pool", [i], [o])))
    single.set_tensor_datatype(pool.input[0], idt)
    single.set_tensor_datatype(pool.output[0], idt)
    single = single.transform(InferShapes())

    assert tree_model_test(
        single,
        ("Pool fused",) + tuple(map(str, config)),
        part,
        target_clk_ns,
        max_allowed_volume_frac=0.0,
        max_allowed_length_frac=0.15,
        volume_const=3,
        length_const=8,
    ), "characterized TAV does not match RTLsim'd one!"


@pytest.mark.fpgadataflow
def test_pool_maxpool_infer_datatype_follows_input():
    """MaxPool's InputDataType/OutputDataType should track the producer's actual
    output datatype instead of staying stale (see issue #1694), since a stale
    InputDataType makes the HLS backend generate an interface width that no
    longer matches the AXI stream actually feeding it."""
    idt = DataType["UINT8"]
    model = make_single_maxpool_modelwrapper(
        k=2, stride=2, pad=0, ifm_ch=4, ifm_dim=4, ofm_dim=2, idt=idt
    )
    model = model.transform(to_hw.InferPool())
    pool_node = model.get_nodes_by_op_type("Pool")[0]
    inst = getCustomOp(pool_node)
    assert inst.get_input_datatype() == idt
    assert inst.get_output_datatype() == idt

    # simulate the producer's output datatype narrowing after conversion, as
    # e.g. MinimizeAccumulatorWidth would do on an upstream node
    narrower_idt = DataType["UINT4"]
    model.set_tensor_datatype(pool_node.input[0], narrower_idt)
    inst.infer_node_datatype(model)

    assert inst.get_input_datatype() == narrower_idt
    assert inst.get_output_datatype() == narrower_idt
    assert model.get_tensor_datatype(pool_node.output[0]) == narrower_idt
    pe = inst.get_nodeattr("PE")
    assert inst.get_instream_width() == narrower_idt.bitwidth() * pe
    assert inst.get_outstream_width() == narrower_idt.bitwidth() * pe
