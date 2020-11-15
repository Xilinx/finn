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

from finn.core.modelwrapper import ModelWrapper
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import (
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)


class ShellFlowType(Enum):
    VIVADO_ZYNQ = auto()
    VITIS_ALVEO = auto()


class DataflowOutputType(Enum):
    STITCHED_IP = auto()
    BITFILE = auto()
    PYNQ_DRIVER = auto()


class ComputeEngineMemMode(Enum):
    CONST = "const"
    DECOUPLED = "decoupled"


@dataclass
class DataflowBuildConfig:
    output_dir: str
    folding_config_file: str
    hls_clk_period_ns: float
    synth_clk_period_ns: float
    board: str
    shell_flow_type: ShellFlowType
    generate_outputs: List[DataflowOutputType]
    fpga_part: str = None
    default_mem_mode: ComputeEngineMemMode = ComputeEngineMemMode.DECOUPLED
    vitis_platform: str = None
    save_intermediate_models: bool = False
    enable_debug: bool = False


def tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    return model


def streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    # move past any reshapes to be able to streamline input scaling
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    need_lowering = len(model.get_nodes_by_op_type("Conv")) > 0
    if need_lowering:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    # absorb final add-mul nodes into TopK
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    return model


def convert_to_hls(model: ModelWrapper, cfg: DataflowBuildConfig):
    mem_mode = cfg.default_mem_mode
    # needed for bipolar MatMul layers
    model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    # needed for non-bipolar MatMul layers
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    # TopK to LabelSelect
    model = model.transform(to_hls.InferLabelSelectLayer())
    # input quantization (if any) to standalone thresholding
    # TODO call first if standalone thresholding is desired
    model = model.transform(to_hls.InferThresholdingLayer())
    # needed for convolutions -- TODO always exec?
    need_conv = len(model.get_nodes_by_op_type("Im2Col")) > 0
    if need_conv:
        model = model.transform(to_hls.InferConvInpGen())
        model = model.transform(to_hls.InferStreamingMaxPool())
        model = model.transform(RemoveCNVtoFCFlatten())
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    return model


def build_dataflow(
    model,
    output_dir,
    folding_config_dict,
    hls_clock_setting,
    synth_clock_setting,
    board,
    flow_type,
    fpga_part=None,
    generate_outputs=["bitfile", "driver", "stitched_ip"],
    save_intermediate_models=False,
    enable_debug=False,
):
    # load the model if specified as filename
    if type(model) == str:
        model = ModelWrapper(model)
    assert type(model) is ModelWrapper
    # create the output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    transform_steps = [tidy_up, streamline, convert_to_hls]
    step_no = 0
    for transform_step in transform_steps:
        model = transform_step(model)
        chkpt_name = "%d_%s.onnx" % (step_no, transform_step.__name__)
        if save_intermediate_models:
            model.save("%s/%s" % (output_dir, chkpt_name))
        step_no += 1
