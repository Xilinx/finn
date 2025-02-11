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


import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.streamline import Streamline
from qonnx.util.range_analysis import RangeInfo
from warnings import warn

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import verify_step
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds


def step_aggregate_scale_bias(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Use scaled-int range analysis to compute aggregated scale and bias."""

    current_irange = dict()
    if cfg.input_range_info is None:
        # try to derive from input datatype annotations
        inp_dtypes = [model.get_tensor_datatype(x.name) for x in model.graph.input]
        all_int_inps = all([x.is_integer() for x in inp_dtypes])
        assert (
            all_int_inps
        ), "This step needs either input_range_info or integer input datatype annotations."
        for inp in model.graph.input:
            iname = inp.name
            idt = model.get_tensor_datatype(iname)
            current_minmax = (idt.min(), idt.max())
            unit_scale = np.asarray(1.0, dtype=np.float32)
            zero_bias = np.asarray(1.0, dtype=np.float32)
            current_irange[iname] = RangeInfo(
                shape=model.get_tensor_shape(iname),
                range=current_minmax,
                int_range=current_minmax,
                scale=unit_scale,
                bias=zero_bias,
            )
    else:
        if isinstance(cfg.input_range_info, list):
            for ind, inp in enumerate(model.graph.input):
                current_irange[inp.name] = cfg.input_range_info[ind]
    model = model.transform(Streamline(irange=current_irange))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())

    mac_intensive_optypes = ["Conv", "MatMul"]
    for node in model.graph.node:
        if node.op_type in mac_intensive_optypes:
            i0_dt = model.get_tensor_datatype(node.input[0])
            i1_dt = model.get_tensor_datatype(node.input[1])
            if not (i0_dt.is_integer() and i1_dt.is_integer()):
                warn(
                    "MAC-intensive node %s has non-integer inputs: (%s, %s)"
                    % (node.name, i0_dt.get_canonical_name(), i1_dt.get_canonical_name())
                )

    step_name = "step_aggregate_scale_bias"
    if step_name in cfg._resolve_verification_steps():
        verify_step(model, cfg, step_name, need_parent=False)

    return model


def step_lower_convs_to_matmul(model: ModelWrapper, cfg: DataflowBuildConfig):
    need_lowering = len(model.get_nodes_by_op_type("Conv")) > 0
    if need_lowering:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
    # TODO add assertions/checks:
    # - no Conv nodes left in graph

    step_name = "step_lower_convs_to_matmul"
    if step_name in cfg._resolve_verification_steps():
        verify_step(model, cfg, step_name, need_parent=False)
    return model


def step_convert_to_channels_last(model: ModelWrapper, cfg: DataflowBuildConfig):
    # TODO better check for channels-last requirement?
    need_channels_last = len(model.get_nodes_by_op_type("Transpose")) > 0
    if need_channels_last:
        model = model.transform(ConvertToChannelsLastAndClean())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
    # TODO add assertions/checks - how should these look?
    step_name = "step_convert_to_channels_last"
    if step_name in cfg._resolve_verification_steps():
        verify_step(model, cfg, step_name, need_parent=False)
    return model


def step_convert_to_thresholds(model: ModelWrapper, cfg: DataflowBuildConfig):
    # TODO to be replaced by the new threshold conversion methodology when it's ready
    model = model.transform(InferDataLayouts())
    model = model.transform(
        ConvertQONNXtoFINN(
            filter_function=default_filter_function_generator(
                max_multithreshold_bit_width=cfg.max_multithreshold_bit_width
            )
        )
    )
    model = model.transform(RemoveIdentityOps())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.FactorOutMulSignMagnitude())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.Absorb1BitMulIntoMatMul())
    model = model.transform(absorb.Absorb1BitMulIntoConv())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataTypes())

    step_name = "step_convert_to_thresholds"
    if step_name in cfg._resolve_verification_steps():
        verify_step(model, cfg, step_name, need_parent=False)
    return model


def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert eligible nodes to `HWCustomOp` subclasses that represent HW
    layers. Which nodes and particular configurations can be converted to HW
    is limited, see the source code of the `convert_to_hw` module for more.
    In the end am empty json file is created which can be used to set user specific
    preferred implementation styles for each node."""

    if cfg.standalone_thresholds:
        # doing this first causes all threshold layers to be standalone
        model = model.transform(to_hw.InferThresholdingLayer())
    # needed for depthwise convs
    model = model.transform(to_hw.InferVectorVectorActivation())
    # needed for bipolar MatMul layers
    model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
    # needed for non-bipolar MatMul layers
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    # TopK to LabelSelect
    model = model.transform(to_hw.InferLabelSelectLayer())
    # input quantization (if any) as standalone threshold
    model = model.transform(to_hw.InferThresholdingLayer())
    # typical layers for convnets / vision
    model = model.transform(to_hw.InferStreamingMaxPool())
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    # standalone elementwise ops, activations and quantizers
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(to_hw.InferReLUAsElementwiseMax())
    model = model.transform(to_hw.InferQuantAsFloat2Int())
    # DuplicateStreams for forking outputs
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    # get rid of Tranpose -> Tranpose identity seq
    # TODO this should not be necessary after chans-last conversion
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    # TODO replace by passthrough inference to keep shape semantics intact
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())

    step_name = "step_convert_to_hw"
    if step_name in cfg._resolve_verification_steps():
        verify_step(model, cfg, step_name, need_parent=False)

    return model
