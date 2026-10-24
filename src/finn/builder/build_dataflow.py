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
import time
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_checks import (
    format_report,
    run_all_config_checks,
    save_report,
)
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    default_build_dataflow_steps,
)
from finn.builder.build_dataflow_phases import build_dataflow_phase_lookup
from finn.builder.build_dataflow_steps import (
    _maybe_enable_verify_behavioral,
    build_dataflow_step_lookup,
)
from finn.util.build_log import log_run, running_step

# Not __name__: run as a script, this module is __main__, outside the finn logger tree.
log = logging.getLogger("finn.builder.build_dataflow")


def _step_index(step_names, step_name, option):
    if step_name not in step_names:
        raise ValueError(
            "%s %r not found, the steps of this build are: %s"
            % (option, step_name, ", ".join(step_names))
        )
    return step_names.index(step_name)


def resolve_build_steps(cfg: DataflowBuildConfig, partial: bool = True):
    """Resolve build steps from config, supporting both phases and fine-grained steps.

    Note: When using phase-based builds with start_step/stop_step, specify phase names
    (e.g., start_step="phase_build_hardware") rather than fine-grained step names.
    Phases save intermediate models for each internal step, so checkpoints like
    step_hw_ipgen.onnx will exist, but the build loop operates at the phase level.
    """
    steps = cfg.steps
    if steps is None:
        steps = default_build_dataflow_steps

    # Merge phase and step lookup dictionaries
    all_steps = {
        **build_dataflow_step_lookup,
        **build_dataflow_phase_lookup,
    }

    steps_as_fxns = []
    for transform_step in steps:
        step_name = None

        # Get step function and name
        if type(transform_step) is str:
            step_name = transform_step
            if transform_step in all_steps:
                step_fn = all_steps[transform_step]
            else:
                raise ValueError(f"Unknown step or phase: {transform_step}")
        elif callable(transform_step):
            step_fn = transform_step
            step_name = getattr(transform_step, "__name__", None)
        else:
            raise ValueError(f"Invalid step type: {type(transform_step)}")

        # Inject steps BEFORE this step
        if step_name and step_name in cfg.inject_steps_before:
            for injected_step in cfg.inject_steps_before[step_name]:
                steps_as_fxns.append(injected_step)

        # Add the main step
        steps_as_fxns.append(step_fn)

        # Inject steps AFTER this step
        if step_name and step_name in cfg.inject_steps_after:
            for injected_step in cfg.inject_steps_after[step_name]:
                steps_as_fxns.append(injected_step)

    if partial:
        step_names = list(map(lambda x: x.__name__, steps_as_fxns))
        if cfg.start_step is None:
            start_ind = 0
        else:
            start_ind = _step_index(step_names, cfg.start_step, "start_step")
        if cfg.stop_step is None:
            stop_ind = len(step_names) - 1
        else:
            stop_ind = _step_index(step_names, cfg.stop_step, "stop_step")
        steps_as_fxns = steps_as_fxns[start_ind : (stop_ind + 1)]

    return steps_as_fxns


def resolve_step_filename(step_name: str, cfg: DataflowBuildConfig, step_delta: int = 0):
    step_names = list(map(lambda x: x.__name__, resolve_build_steps(cfg, partial=False)))
    step_no = _step_index(step_names, step_name, "start_step") + step_delta
    assert step_no >= 0, "Invalid step+delta combination"
    assert step_no < len(step_names), "Invalid step+delta combination"
    filename = cfg.output_dir + "/intermediate_models/"
    filename += "%s.onnx" % (step_names[step_no])
    return filename


def _run_build_steps(model, cfg, build_dataflow_steps):
    """Run the resolved build steps in order.

    Each step runs under running_step, so the records of the finn loggers and the
    Python warnings that it produces go to build_dataflow.log, stamped with the
    name of the configured step or phase.

    Returns 0 on success and -1 on the first failing step.
    """
    step_num = 1
    time_per_step = dict()
    for transform_step in build_dataflow_steps:
        step_name = transform_step.__name__
        with running_step(step_name):
            try:
                log.info("Running step: %s [%d/%d]", step_name, step_num, len(build_dataflow_steps))
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
            except Exception as e:
                log.error("Step %s failed:", step_name, exc_info=True)
                # start postmortem debug if configured
                if cfg.enable_build_pdb_debug:
                    pdb.post_mortem(e.__traceback__)
                else:
                    log.info("enable_build_pdb_debug not set in build config, exiting...")
                log.info("Build failed")
                return -1

    with open(cfg.output_dir + "/time_per_step.json", "w") as f:
        json.dump(time_per_step, f, indent=2)
    log.info("Completed successfully")
    return 0


def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    """Best-effort build a dataflow accelerator using the given configuration.

    The build is logged to build_dataflow.log in the output directory; see
    finn.util.build_log for what reaches the console.

    :param model_filename: ONNX model filename to build
    :param cfg: Build configuration
    """
    # create the output dir if it doesn't exist
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    with log_run(cfg.output_dir + "/build_dataflow.log", runner=log.name, verbose=cfg.verbose):
        return _build_dataflow_cfg(model_filename, cfg)


def _build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    # if start_step is specified, override the input model
    if cfg.start_step is None:
        log.info("Building dataflow accelerator from " + model_filename)
        model = ModelWrapper(model_filename)
    else:
        intermediate_model_filename = resolve_step_filename(cfg.start_step, cfg, -1)
        log.info(
            "Building dataflow accelerator from intermediate checkpoint "
            + intermediate_model_filename
        )
        model = ModelWrapper(intermediate_model_filename)
    assert type(model) is ModelWrapper
    finn_build_dir = os.environ["FINN_BUILD_DIR"]

    log.info("Intermediate outputs will be generated in " + finn_build_dir)
    log.info("Final outputs will be generated in " + cfg.output_dir)
    log.info("Build log is at " + cfg.output_dir + "/build_dataflow.log")

    _maybe_enable_verify_behavioral(cfg)

    # Run configuration checks
    config_report = run_all_config_checks(cfg)
    log.info(format_report(config_report))
    report_path = save_report(config_report, cfg.output_dir)
    log.info(f"Configuration check report saved to: {report_path}")

    if config_report.has_errors():
        if cfg.mute_config_assertions is True:
            log.warning(
                "Configuration errors detected but muted by mute_config_assertions=True. "
                "Build may fail or produce unexpected results."
            )
        else:
            raise AssertionError(
                "Configuration check failed with errors. "
                "Fix the issues above or set mute_config_assertions=True to proceed."
            )

    build_dataflow_steps = resolve_build_steps(cfg)
    return _run_build_steps(model, cfg, build_dataflow_steps)


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
    try:
        return build_dataflow_cfg(onnx_filename, build_cfg)
    finally:
        os.chdir(old_wd)


def main():
    """Entry point for dataflow builds. Invokes `build_dataflow_directory` using
    command line arguments. Exits with a nonzero status if the build failed."""

    def cli(path_to_cfg_dir: str):
        """Best-effort build a dataflow accelerator from the specified directory.

        :param path_to_cfg_dir: Directory containing the model and build config

        The specified directory path_to_cfg_dir must contain the following files:

        * model.onnx : ONNX model to be converted to dataflow accelerator
        * dataflow_build_config.json : JSON file with build configuration

        """
        if build_dataflow_directory(path_to_cfg_dir):
            raise SystemExit(1)

    clize.run(cli)


if __name__ == "__main__":
    main()
