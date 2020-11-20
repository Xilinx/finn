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
from dataclasses_json import dataclass_json
from enum import Enum
from typing import List, Optional

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
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
import time
from shutil import copy, copytree
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.util.basic import pynq_part_map, alveo_part_map
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.custom_op.registry import getCustomOp
import clize


class ShellFlowType(str, Enum):
    VIVADO_ZYNQ = "vivado_zynq"
    VITIS_ALVEO = "vitis_alveo"


class DataflowOutputType(str, Enum):
    STITCHED_IP = "stitched_ip"
    BITFILE = "bitfile"
    PYNQ_DRIVER = "pynq_driver"


class ComputeEngineMemMode(str, Enum):
    CONST = "const"
    DECOUPLED = "decoupled"


@dataclass_json
@dataclass
class DataflowBuildConfig:
    "Build configuration to be passed to the build_dataflow function."

    output_dir: str
    folding_config_file: str
    synth_clk_period_ns: float
    board: str
    shell_flow_type: ShellFlowType
    generate_outputs: List[DataflowOutputType]
    fpga_part: Optional[str] = None
    auto_fifo_depths: Optional[bool] = True
    hls_clk_period_ns: Optional[float] = None
    default_mem_mode: Optional[ComputeEngineMemMode] = ComputeEngineMemMode.DECOUPLED
    vitis_platform: Optional[str] = None
    vitis_floorplan_file: Optional[str] = None
    save_intermediate_models: Optional[bool] = False
    enable_debug: Optional[bool] = False
    from_step_num: Optional[int] = None
    to_step_num: Optional[int] = None

    def resolve_hls_clk_period(self):
        if self.hls_clk_period_ns is None:
            # use same clk for synth and hls if not explicitly specified
            return self.synth_clk_period_ns
        else:
            return self.hls_clk_period_ns

    def resolve_driver_platform(self):
        if self.shell_flow_type == ShellFlowType.VIVADO_ZYNQ:
            return "zynq-iodma"
        elif self.shell_flow_type == ShellFlowType.VITIS_ALVEO:
            return "alveo"
        else:
            raise Exception(
                "Couldn't resolve driver platform for " + str(self.shell_flow_type)
            )

    def resolve_fpga_part(self):
        if self.fpga_part is None:
            # lookup from part map if not specified
            if self.shell_flow_type == ShellFlowType.VIVADO_ZYNQ:
                return pynq_part_map[self.board]
            elif self.shell_flow_type == ShellFlowType.VITIS_ALVEO:
                return alveo_part_map[self.board]
            else:
                raise Exception("Couldn't resolve fpga_part for " + self.board)
        else:
            # return as-is when explicitly specified
            return self.fpga_part


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
    mem_mode = cfg.default_mem_mode.value
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


def create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig):
    parent_model = model.transform(CreateDataflowPartition())
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 1, "Only a single StreamingDataflowPartition supported."
    sdp_node = sdp_nodes[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    model = ModelWrapper(dataflow_model_filename)
    return model


def apply_folding_config(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(ApplyConfig(cfg.folding_config_file))
    return model


def hls_ipgen(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(
        PrepareIP(cfg.resolve_fpga_part(), cfg.resolve_hls_clk_period())
    )
    model = model.transform(HLSSynthIP())
    return model


def auto_set_fifo_depths(model: ModelWrapper, cfg: DataflowBuildConfig):
    if cfg.auto_fifo_depths:
        model = model.transform(
            InsertAndSetFIFODepths(
                cfg.resolve_fpga_part(), cfg.resolve_hls_clk_period()
            )
        )
        model = model.transform(
            PrepareIP(cfg.resolve_fpga_part(), cfg.resolve_hls_clk_period())
        )
        model = model.transform(HLSSynthIP())
    return model


def create_stitched_ip(model: ModelWrapper, cfg: DataflowBuildConfig):
    if DataflowOutputType.STITCHED_IP in cfg.generate_outputs:
        stitched_ip_dir = cfg.output_dir + "/stitched_ip"
        model = model.transform(
            CreateStitchedIP(cfg.resolve_fpga_part(), cfg.synth_clk_period_ns)
        )
        # TODO copy all ip sources into output dir? as zip?
        copytree(model.get_metadata_prop("vivado_stitch_proj"), stitched_ip_dir)
        print("Vivado stitched IP written into " + stitched_ip_dir)
    return model


def make_pynq_driver(model: ModelWrapper, cfg: DataflowBuildConfig):
    if DataflowOutputType.PYNQ_DRIVER in cfg.generate_outputs:
        driver_dir = cfg.output_dir + "/driver"
        model = model.transform(MakePYNQDriver(cfg.resolve_driver_platform()))
        copytree(model.get_metadata_prop("pynq_driver_dir"), driver_dir)
        print("PYNQ Python driver written into " + driver_dir)
    return model


def synthesize_bitfile(model: ModelWrapper, cfg: DataflowBuildConfig):
    if DataflowOutputType.BITFILE in cfg.generate_outputs:
        bitfile_dir = cfg.output_dir + "/bitfile"
        os.makedirs(bitfile_dir)
        if cfg.shell_flow_type == ShellFlowType.VIVADO_ZYNQ:
            model = model.transform(
                ZynqBuild(cfg.board, cfg.synth_clk_period_ns, cfg.enable_debug)
            )
            copy(model.get_metadata_prop("bitfile"), bitfile_dir + "/finn-accel.bit")
            copy(model.get_metadata_prop("hw_handoff"), bitfile_dir + "/finn-accel.hwh")
        elif cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
            model = model.transform(
                VitisBuild(
                    cfg.resolve_fpga_part(),
                    cfg.synth_clk_period_ns,
                    cfg.vitis_platform,
                    enable_debug=cfg.enable_debug,
                    floorplan_file=cfg.vitis_floorplan_file,
                )
            )
            copy(model.get_metadata_prop("bitfile"), bitfile_dir + "/finn-accel.xclbin")
        else:
            raise Exception("Unrecognized shell_flow_type: " + str(cfg.shell_flow_type))
        print("Bitfile written into " + bitfile_dir)

    return model


def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    try:
        model = ModelWrapper(model_filename)
        assert type(model) is ModelWrapper
        print("Building dataflow accelerator from " + model_filename)
        print("Outputs will be generated at " + cfg.output_dir)
        # create the output dir if it doesn't exist
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        transform_steps = [
            tidy_up,
            streamline,
            convert_to_hls,
            create_dataflow_partition,
            apply_folding_config,
            hls_ipgen,
            auto_set_fifo_depths,
            create_stitched_ip,
            make_pynq_driver,
            synthesize_bitfile,
        ]
        step_num = 0
        time_per_step = dict()
        for transform_step in transform_steps:
            if cfg.from_step_num is not None and step_num < cfg.from_step_num:
                step_num += 1
                continue
            if cfg.to_step_num is not None and step_num > cfg.to_step_num:
                step_num += 1
                continue
            step_name = transform_step.__name__
            print(
                "Running step: %s [%d/%d]" % (step_name, step_num, len(transform_steps))
            )
            step_start = time.time()
            model = transform_step(model, cfg)
            step_end = time.time()
            time_per_step[step_name] = step_end - step_start
            chkpt_name = "%d_%s.onnx" % (step_num, step_name)
            if cfg.save_intermediate_models:
                intermediate_model_dir = cfg.output_dir + "/intermediate_models"
                if not os.path.exists(intermediate_model_dir):
                    os.makedirs(intermediate_model_dir)
                model.save("%s/%s" % (intermediate_model_dir, chkpt_name))
            step_num += 1
        with open(cfg.output_dir + "/time_per_step.txt", "w") as f:
            f.write(str(time_per_step))
        print("Completed successfully")
        return 0
    except Exception as inst:
        print("Build failed:")
        print(type(inst))
        print(inst.args)
        print(inst)
        return -1


def build_dataflow_directory(path_to_cfg_dir: str):
    """Best-effort build a dataflow accelerator from the specified directory.

    :param path_to_cfg_dir: Directory containing the model and build config

    The specified directory path_to_cfg_dir must contain the following files:

    * model.onnx : ONNX model to be converted to dataflow accelerator
    * dataflow_build_config.json : JSON file with build configuration

    """
    assert os.path.isdir(path_to_cfg_dir), "Directory not found: " + path_to_cfg_dir
    onnx_filename = path_to_cfg_dir + "/model.onnx"
    json_filename = path_to_cfg_dir + "/dataflow_build_config.json"
    assert os.path.isfile(onnx_filename), "ONNX not found: " + onnx_filename
    assert os.path.isfile(json_filename), "Build config not found: " + json_filename
    with open(json_filename, "r") as f:
        json_str = f.read()
    build_cfg = DataflowBuildConfig.from_json(json_str)
    old_wd = os.getcwd()
    # change into build dir to resolve relative paths
    os.chdir(path_to_cfg_dir)
    ret = build_dataflow_cfg(onnx_filename, build_cfg)
    os.chdir(old_wd)
    return ret


def main():
    clize.run(build_dataflow_directory)


if __name__ == "__main__":
    main()
