# Copyright (c) 2022 Xilinx, Inc.
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

import pkg_resources as pk

import numpy as np
from qonnx.custom_op.registry import getCustomOp

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.derive_characteristic import DeriveCharacteristic
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.util.basic import make_build_dir


def custom_step_fifosize(model, cfg):
    # TODO convert to NodeLocalTransformation
    def accumulate_char_fxn(chrc):
        p = len(chrc)
        ret = []
        for t in range(2 * p):
            if t == 0:
                ret.append(chrc[0])
            else:
                ret.append(ret[-1] + chrc[t % p])
        return ret

    # TODO handle chrc for input and output nodes
    all_act_tensors = [x.name for x in model.graph.value_info]
    for tensor_nm in all_act_tensors:
        # generate accumulated characteristic functions
        prod = getCustomOp(model.find_producer(tensor_nm))
        prod_chrc = prod.get_nodeattr("io_characteristic")
        prod_chrc = np.asarray(prod_chrc, dtype=np.uint8).reshape(2, -1)[1]
        prod_chrc = accumulate_char_fxn(prod_chrc)
        cons = getCustomOp(model.find_consumer(tensor_nm))
        cons_chrc = cons.get_nodeattr("io_characteristic")
        cons_chrc = np.asarray(cons_chrc, dtype=np.uint8).reshape(2, -1)[0]
        cons_chrc = accumulate_char_fxn(cons_chrc)
        # TODO find minimum phase shift

    for node in model.graph.node:
        inst = getCustomOp(node)
        chrc = inst.get_nodeattr("io_characteristic")
        chrc = np.asarray(chrc, dtype=np.uint8).reshape(2, -1)

    return model


def custom_step_fifocharacterize(model, cfg):
    model = model.transform(PrepareRTLSim())
    period = model.analysis(dataflow_performance)["max_cycles"] + 10
    model = model.transform(DeriveCharacteristic(period))
    return model


def test_end2end_fifosizing():
    chkpt_name = pk.resource_filename("finn.qnn-data", "build_dataflow/model.onnx")
    tmp_output_dir = make_build_dir("build_fifosizing_")
    # tmp_output_dir = "/tmp/finn_dev_maltanar/build_fifosizing_5mt0o6s_"
    steps = build_cfg.default_build_dataflow_steps
    steps = steps[:10]
    steps.append(custom_step_fifocharacterize)
    # steps.append(custom_step_fifosize)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        auto_fifo_depths=False,
        target_fps=10000,
        synth_clk_period_ns=10.0,
        board="Pynq-Z1",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[],
        steps=steps,
        default_mem_mode=build_cfg.ComputeEngineMemMode.CONST,
        start_step="custom_step_fifocharacterize",
    )
    build.build_dataflow_cfg(chkpt_name, cfg)
