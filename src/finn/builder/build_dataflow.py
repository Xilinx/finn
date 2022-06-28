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

import clize
import json
import logging
import os
import pdb  # NOQA
import sys
import time
import traceback
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    default_build_dataflow_steps,
)
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup


# adapted from https://stackoverflow.com/a/39215961
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def resolve_build_steps(cfg: DataflowBuildConfig, partial: bool = True):
    steps = cfg.steps
    if steps is None:
        steps = default_build_dataflow_steps
    steps_as_fxns = []
    for transform_step in steps:
        if type(transform_step) is str:
            # lookup step function from step name
            steps_as_fxns.append(build_dataflow_step_lookup[transform_step])
        elif callable(transform_step):
            # treat step as function to be called as-is
            steps_as_fxns.append(transform_step)
        else:
            raise Exception("Could not resolve build step: " + str(transform_step))
    if partial:
        step_names = list(map(lambda x: x.__name__, steps_as_fxns))
        if cfg.start_step is None:
            start_ind = 0
        else:
            start_ind = step_names.index(cfg.start_step)
        if cfg.stop_step is None:
            stop_ind = len(step_names) - 1
        else:
            stop_ind = step_names.index(cfg.stop_step)
        steps_as_fxns = steps_as_fxns[start_ind : (stop_ind + 1)]

    return steps_as_fxns


def resolve_step_filename(
    step_name: str, cfg: DataflowBuildConfig, step_delta: int = 0
):
    step_names = list(
        map(lambda x: x.__name__, resolve_build_steps(cfg, partial=False))
    )
    assert step_name in step_names, "start_step %s not found" + step_name
    step_no = step_names.index(step_name) + step_delta
    assert step_no >= 0, "Invalid step+delta combination"
    assert step_no < len(step_names), "Invalid step+delta combination"
    filename = cfg.output_dir + "/intermediate_models/"
    filename += "%s.onnx" % (step_names[step_no])
    return filename


def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    """Best-effort build a dataflow accelerator using the given configuration.

    :param model_filename: ONNX model filename to build
    :param cfg: Build configuration
    """
    # if start_step is specified, override the input model
    if cfg.start_step is None:
        print("Building dataflow accelerator from " + model_filename)
        model = ModelWrapper(model_filename)
    else:
        intermediate_model_filename = resolve_step_filename(cfg.start_step, cfg, -1)
        print(
            "Building dataflow accelerator from intermediate checkpoint"
            + intermediate_model_filename
        )
        model = ModelWrapper(intermediate_model_filename)
    assert type(model) is ModelWrapper
    finn_build_dir = os.environ["FINN_BUILD_DIR"]

    print("Intermediate outputs will be generated in " + finn_build_dir)
    print("Final outputs will be generated in " + cfg.output_dir)
    print("Build log is at " + cfg.output_dir + "/build_dataflow.log")
    # create the output dir if it doesn't exist
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    step_num = 1
    time_per_step = dict()
    build_dataflow_steps = resolve_build_steps(cfg)
    # set up logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(message)s",
        filename=cfg.output_dir + "/build_dataflow.log",
        filemode="a",
    )
    log = logging.getLogger("build_dataflow")
    stdout_logger = StreamToLogger(log, logging.INFO)
    stderr_logger = StreamToLogger(log, logging.ERROR)
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    for transform_step in build_dataflow_steps:
        try:
            step_name = transform_step.__name__
            print(
                "Running step: %s [%d/%d]"
                % (step_name, step_num, len(build_dataflow_steps))
            )
            # redirect output to logfile
            sys.stdout = stdout_logger
            sys.stderr = stderr_logger
            print(
                "Running step: %s [%d/%d]"
                % (step_name, step_num, len(build_dataflow_steps))
            )
            # run the step
            step_start = time.time()
            model = transform_step(model, cfg)
            step_end = time.time()
            # restore stdout/stderr
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            time_per_step[step_name] = step_end - step_start
            chkpt_name = "%s.onnx" % (step_name)
            if cfg.save_intermediate_models:
                intermediate_model_dir = cfg.output_dir + "/intermediate_models"
                if not os.path.exists(intermediate_model_dir):
                    os.makedirs(intermediate_model_dir)
                model.save("%s/%s" % (intermediate_model_dir, chkpt_name))
            step_num += 1
        except:  # noqa
            # restore stdout/stderr
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            # print exception info and traceback
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            # start postmortem debug if configured
            if cfg.enable_build_pdb_debug:
                pdb.post_mortem(tb)
            else:
                print("enable_build_pdb_debug not set in build config, exiting...")
            print("Build failed")
            return -1

    with open(cfg.output_dir + "/time_per_step.json", "w") as f:
        json.dump(time_per_step, f, indent=2)
    print("Completed successfully")
    return 0


def build_dataflow_directory(path_to_cfg_dir: str):
    """Best-effort build a dataflow accelerator from the specified directory.

    :param path_to_cfg_dir: Directory containing the model and build config

    The specified directory path_to_cfg_dir must contain the following files:

    * model.onnx : ONNX model to be converted to dataflow accelerator
    * dataflow_build_config.json : JSON file with build configuration

    """
    # get absolute path
    path_to_cfg_dir = os.path.abspath(path_to_cfg_dir)
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
    """Entry point for dataflow builds. Invokes `build_dataflow_directory` using
    command line arguments"""
    clize.run(build_dataflow_directory)


if __name__ == "__main__":
    main()
