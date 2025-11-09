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


def resolve_build_steps(cfg: DataflowBuildConfig, partial: bool = True):
    steps = cfg.steps
    if steps is None:
        steps = default_build_dataflow_steps
    steps_as_fxns = []
    for transform_step in steps:
        if callable(transform_step):
            # treat step as function to be called as-is
            steps_as_fxns.append(transform_step)
        elif type(transform_step) is str:
            # lookup step function from step name
            steps_as_fxns.append(build_dataflow_step_lookup[transform_step])
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


def resolve_step_filename(step_name: str, cfg: DataflowBuildConfig, step_delta: int = 0):
    step_names = list(map(lambda x: x.__name__, resolve_build_steps(cfg, partial=False)))
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
    # Set up builder logger for user-facing status messages
    builder_log = logging.getLogger("finn.builder")

    # if start_step is specified, override the input model
    if cfg.start_step is None:
        builder_log.debug("Building dataflow accelerator from " + model_filename)
        model = ModelWrapper(model_filename)
    else:
        intermediate_model_filename = resolve_step_filename(cfg.start_step, cfg, -1)
        builder_log.debug(
            "Building dataflow accelerator from intermediate checkpoint "
            + intermediate_model_filename
        )
        model = ModelWrapper(intermediate_model_filename)
    assert type(model) is ModelWrapper
    finn_build_dir = os.environ["FINN_BUILD_DIR"]

    builder_log.debug("Intermediate outputs will be generated in " + finn_build_dir)
    builder_log.debug("Final outputs will be generated in " + cfg.output_dir)
    builder_log.debug("Build log is at " + cfg.output_dir + "/build_dataflow.log")
    # create the output dir if it doesn't exist
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    step_num = 1
    time_per_step = dict()
    build_dataflow_steps = resolve_build_steps(cfg)

    # Set up root logger with file handler for audit trail
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        filename=cfg.output_dir + "/build_dataflow.log",
        filemode="a",
    )

    # Configure finn.builder logger (progress messages) - controlled by show_progress
    builder_logger = logging.getLogger("finn.builder")
    builder_logger.setLevel(logging.INFO)
    if cfg.show_progress:
        # Show progress messages on console with clean formatting
        builder_console = logging.StreamHandler(sys.stdout)
        builder_console.setFormatter(logging.Formatter("%(message)s"))
        builder_logger.addHandler(builder_console)
    # Add file handler for audit trail (match root logger format for consistency)
    builder_file = logging.FileHandler(cfg.output_dir + "/build_dataflow.log", mode="a")
    builder_file.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s: %(message)s")
    )
    builder_logger.addHandler(builder_file)
    # Don't propagate to finn parent (we handle both console and file locally)
    builder_logger.propagate = False

    # Configure finn tool loggers (subprocess output) - controlled by verbose
    finn_logger = logging.getLogger("finn")
    finn_logger.setLevel(logging.DEBUG)  # Permissive parent (children can filter)

    # Add console handler if verbose mode
    if cfg.verbose:
        finn_console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
        finn_console_handler.setFormatter(console_formatter)
        finn_console_handler.setLevel(logging.ERROR)
        finn_logger.addHandler(finn_console_handler)

    # Always propagate to file (via root logger)
    finn_logger.propagate = True

    # Apply subprocess log level overrides (console and file independently)
    # Collect all categories from both configs
    all_categories = set()
    if cfg.subprocess_console_levels:
        all_categories.update(cfg.subprocess_console_levels.keys())
    if cfg.subprocess_log_levels:
        all_categories.update(cfg.subprocess_log_levels.keys())

    configured_logger_names = []
    for category in all_categories:
        logger_name = f"finn.{category}"
        configured_logger_names.append(logger_name)
        subprocess_logger = logging.getLogger(logger_name)

        # Determine console level (default: ERROR - minimize console spam)
        console_level = (cfg.subprocess_console_levels or {}).get(category, logging.ERROR)
        # Determine file level (default: DEBUG for comprehensive audit trail)
        file_level = (cfg.subprocess_log_levels or {}).get(category, logging.DEBUG)

        # Set logger level to minimum needed by active destinations
        # When verbose=False, console_level is irrelevant (no console handler exists)
        if cfg.verbose:
            subprocess_logger.setLevel(min(console_level, file_level))
        else:
            subprocess_logger.setLevel(file_level)

        # Add child-specific console handler (when verbose)
        if cfg.verbose:
            child_console_handler = logging.StreamHandler(sys.stdout)
            child_console_handler.setFormatter(console_formatter)
            child_console_handler.setLevel(console_level)
            subprocess_logger.addHandler(child_console_handler)

        # Always propagate to root for file logging
        subprocess_logger.propagate = True

    # Add filter to parent console handler to exclude configured children
    # (prevents duplication for any children that DO propagate)
    if cfg.verbose and configured_logger_names:

        class ExcludeConfiguredLoggersFilter(logging.Filter):
            def filter(self, record):
                # Block messages from configured subprocess loggers
                return not any(record.name.startswith(name) for name in configured_logger_names)

        finn_console_handler.addFilter(ExcludeConfiguredLoggersFilter())

    for transform_step in build_dataflow_steps:
        try:
            step_name = transform_step.__name__
            builder_log.info(
                "Running step: %s [%d/%d]" % (step_name, step_num, len(build_dataflow_steps))
            )
            # run the step
            step_start = time.time()
            model = transform_step(model, cfg)
            step_end = time.time()
            time_per_step[step_name] = step_end - step_start
            chkpt_name = "%s.onnx" % (step_name)
            if cfg.save_intermediate_models:
                intermediate_model_dir = cfg.output_dir + "/intermediate_models"
                if not os.path.exists(intermediate_model_dir):
                    os.makedirs(intermediate_model_dir)
                model.save("%s/%s" % (intermediate_model_dir, chkpt_name))
            step_num += 1
        except:  # noqa
            # print exception info and traceback
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            # start postmortem debug if configured
            if cfg.enable_build_pdb_debug:
                pdb.post_mortem(tb)
            else:
                builder_log.error("enable_build_pdb_debug not set in build config, exiting...")
            builder_log.error("Build failed")
            return -1

    with open(cfg.output_dir + "/time_per_step.json", "w") as f:
        json.dump(time_per_step, f, indent=2)
    builder_log.info("Completed successfully")
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
